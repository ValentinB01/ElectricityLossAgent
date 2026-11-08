from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset_builders import MODELS, prepare
from src.evaluation.evaluator import combine_predictions, evaluate_run, prediction_table
from src.models.cnn1d_model import CNN1DClassifier
from src.models.hybrid_model import HybridClassifier
from src.training.config import load_training_config, validate_training_config
from src.training.datasets import ConsumerDataset, PreparedDataStore
from src.training.experiment_tracker import choose_device, seed_everything, write_json
from src.training.train_neural import load_network, train_neural
from src.training.train_trees import positive_class_weight, train_tree


@pytest.fixture
def training_config(tmp_path):
    c = load_training_config()
    c['experiment'].update(output_dir=str(tmp_path / 'outputs'), model_dir=str(tmp_path / 'experiments'), cpu_threads=2, device='cpu')
    for name in ('lightgbm', 'xgboost'):
        c['models'][name]['params']['n_estimators'] = 6
        c['models'][name]['early_stopping_rounds'] = 2
    for name in ('cnn1d', 'hybrid'):
        c['models'][name].update(max_epochs=1, batch_size=16, channels=[4, 8, 8])
    c['visualization'].update(dpi=80, examples_per_category=1, save_svg=False)
    return c


@pytest.fixture
def prepared(synthetic_csv):
    _, config = synthetic_csv
    prepare(config, list(MODELS))
    return PreparedDataStore(config['data']['output_dir'])


def test_float32_score_threshold_matches_metrics_and_csv(tmp_path):
    from types import SimpleNamespace
    bundle = SimpleNamespace(ids=np.array(['a']), y=np.array([1]))
    probability = np.array([.45], dtype=np.float32)
    frame = prediction_table(bundle, probability, {'threshold': .45})
    path = tmp_path / 'predictions.csv'
    frame.to_csv(path, index=False)
    restored = pd.read_csv(path, float_precision='round_trip')
    assert frame.prediction_optimized.iloc[0] == 0
    assert restored.probability.iloc[0] < .45
    assert restored.prediction_optimized.iloc[0] == (restored.probability.iloc[0] >= .45)


def test_probability_shapes_cnn_order_and_hybrid_alignment(prepared, training_config):
    cnn, hybrid = prepared.load('cnn1d', 'train'), prepared.load('hybrid', 'train')
    np.testing.assert_array_equal(cnn.ids, hybrid.ids)
    sequence, features, target = ConsumerDataset(hybrid)[0]
    assert sequence.shape == (2, 1035) and features.shape == (160,)
    np.testing.assert_array_equal(sequence.numpy().T, hybrid.sequence[hybrid.order[0]])
    for name, model in [('cnn1d', CNN1DClassifier(training_config['models']['cnn1d'])),
                         ('hybrid', HybridClassifier(160, training_config['models']['hybrid']))]:
        model.eval()
        logits = model(sequence[None], features[None])
        probability = torch.sigmoid(logits)
        assert probability.shape == (1,) and 0 <= probability.item() <= 1
    # Singleton training batch is supported without dropping the final consumer.
    HybridClassifier(160, training_config['models']['hybrid']).train()(sequence[None], features[None])


def test_id_realignment_and_corrupt_branch_rejection(prepared):
    reference = prepared.load('lightgbm', 'train')
    path = prepared.root / 'xgboost/train.parquet'
    pd.read_parquet(path).iloc[::-1].to_parquet(path, index=False)
    reordered = prepared.load('xgboost', 'train')
    pd.testing.assert_frame_equal(reference.tabular(), reordered.tabular())
    path = prepared.root / 'hybrid/X_features_train.npy'
    array = np.load(path)
    np.save(path, array[::-1])
    with pytest.raises(ValueError, match='branches'):
        prepared.load('hybrid', 'train')


def test_tree_imbalance_and_checkpoint_serialization(prepared, training_config, tmp_path):
    assert positive_class_weight([0, 0, 0, 1]) == 3
    for name in ('lightgbm', 'xgboost'):
        train, val = prepared.load(name, 'train'), prepared.load(name, 'val')
        result = train_tree(name, train, val, training_config, tmp_path / name)
        assert len(result['candidates']) == 2 and result['early_stopping_split'] == 'val'
        model = joblib.load(result['model_path'])
        p1, p2 = model.predict_proba(val.tabular())[:, 1], model.predict_proba(val.tabular())[:, 1]
        np.testing.assert_array_equal(p1, p2)
        assert np.isfinite(p1).all()
        assert not {'FLAG', 'CONS_NO', 'split'} & set(train.feature_names)
        with pytest.raises(ValueError, match='train fitting'):
            train_tree(name, train, replace(val, split='test'), training_config, tmp_path / (name + '_bad'))


def test_neural_checkpoint_and_test_guard(prepared, training_config, tmp_path):
    for name in ('cnn1d', 'hybrid'):
        train, val = prepared.load(name, 'train'), prepared.load(name, 'val')
        result = train_neural(name, train, val, training_config, tmp_path / name)
        model, checkpoint = load_network(result['model_path'])
        assert checkpoint['epoch'] == 1
        sequence, features, _ = ConsumerDataset(val)[0]
        assert model(sequence[None], features[None]).shape == (1,)
        with pytest.raises(ValueError, match='never test'):
            train_neural(name, train, replace(val, split='test'), training_config, tmp_path / (name + '_bad'))
    frozen = tmp_path / 'not_frozen'
    write_json(frozen / 'run.json', {'all_selections_frozen': False})
    with pytest.raises(ValueError, match='frozen'):
        evaluate_run(frozen)


def test_combined_prediction_alignment():
    a = pd.DataFrame({'CONS_NO': ['a', 'b'], 'y_true': [0, 1], 'probability': [.1, .9]})
    b = pd.DataFrame({'CONS_NO': ['b', 'a'], 'y_true': [1, 0], 'probability': [.8, .2]})
    combined = combine_predictions({'lightgbm': a, 'xgboost': b})
    assert combined.xgboost_score.tolist() == [.2, .8]
    b.loc[0, 'y_true'] = 0
    with pytest.raises(ValueError, match='labels'):
        combine_predictions({'lightgbm': a, 'xgboost': b})
def test_actual_processed_data_integration_if_available():
    config = load_training_config()
    path = Path(config['data']['processed_dir'])
    if not (path / 'shared/run_manifest.json').exists():
        pytest.skip('Real processed data is unavailable')
    audit = PreparedDataStore(path).alignment_audit()
    assert audit['passed'] and audit['splits']['train']['consumers'] == 29660


def test_synthetic_end_to_end_plots_dashboard_and_tracking(prepared, training_config):
    from src.training.train import run_training
    training_config['data']['processed_dir'] = str(prepared.root)
    training_config['visualization']['generate_consumer_examples'] = False
    run = run_training(training_config, list(MODELS), evaluate=True, visualize=True, smoke=True)
    manifest = json.loads((run / 'run.json').read_text())
    assert manifest['diagnostic'] and manifest['all_selections_frozen'] and manifest['evaluation_complete']
    assert len(manifest['models']) == 4
    assert (run / 'figures/comparison/roc_curves.png').is_file()
    assert (run / 'figures/comparison/confusion_matrices.png').is_file()
    page = (run / 'dashboard/model_comparison.html').read_text(encoding='utf-8')
    assert 'DIAGNOSTIC SUBSET' in page and 'plotly.js' in page
    assert '<script src="https://' not in page
    comparison = pd.read_csv(run / 'metrics/model_comparison.csv')
    assert len(comparison) == 4 and comparison.diagnostic.all()
    diagnostic_predictions = pd.read_csv(run / 'predictions/lightgbm_test.csv', dtype={'CONS_NO': str})
    actual_validation = set(prepared.assignments.loc[prepared.assignments.split == 'val', 'CONS_NO'])
    assert set(diagnostic_predictions.CONS_NO).issubset(actual_validation)
    assert not set(diagnostic_predictions.CONS_NO) & set(prepared.assignments.loc[prepared.assignments.split == 'test', 'CONS_NO'])
    assert not (Path(training_config['experiment']['output_dir']) / 'latest_run.json').exists()
