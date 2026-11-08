"""Model contracts tested without the training orchestration or prepared data."""
import joblib
import numpy as np
import pytest
import torch

from src.models.cnn1d_model import CNN1DClassifier
from src.models.hybrid_model import HybridClassifier
from src.models.lightgbm_model import create_classifier as lightgbm_classifier
from src.models.xgboost_model import create_classifier as xgboost_classifier


@pytest.mark.parametrize('name', ['lightgbm', 'xgboost'])
def test_tree_probability_contract_and_serialization(name, tmp_path):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 4))
    target = (features[:, 0] > 0).astype(int)
    params = {'n_estimators': 8, 'learning_rate': .2}
    if name == 'lightgbm':
        model = lightgbm_classifier({**params, 'min_child_samples': 2}, 42, 1, 3.)
        model.fit(features[:60], target[:60])
        assert model.get_params()['metric'] == 'average_precision'
    else:
        model = xgboost_classifier({**params, 'max_depth': 2, 'tree_method': 'hist'}, 42, 1, 3., 2)
        model.fit(features[:60], target[:60], eval_set=[(features[60:], target[60:])], verbose=False)
        assert model.get_params()['eval_metric'] == 'aucpr'
        assert model.get_params()['early_stopping_rounds'] == 2
    assert model.get_params()['scale_pos_weight'] == 3.
    assert model.get_params()['random_state'] == 42
    probability = model.predict_proba(features[60:])
    assert probability.shape == (20, 2) and np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.)
    path = tmp_path / f'{name}.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(features[60:]), probability)


@pytest.mark.parametrize('name', ['cnn1d', 'hybrid'])
def test_neural_logits_gradients_and_state_roundtrip(name):
    torch.manual_seed(42)
    config = {'channels': [4, 8, 8], 'kernels': [7, 5, 3], 'dropout': .3}
    def build():
        return CNN1DClassifier(config) if name == 'cnn1d' else HybridClassifier(160, config)
    model = build()
    sequence = torch.randn(2, 2, 1035, requires_grad=True)
    features = torch.randn(2, 160, requires_grad=True)
    logits = model(sequence, features)
    assert logits.shape == (2,) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.])).backward()
    assert sequence.grad is not None and torch.isfinite(sequence.grad).all() and sequence.grad.abs().sum() > 0
    if name == 'hybrid':
        assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    model.eval()
    restored = build().eval()
    restored.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(restored(sequence, features), model(sequence, features))
        assert torch.sigmoid(model(sequence[:1], features[:1])).shape == (1,)


def test_hybrid_singleton_training_batch_preserves_running_statistics():
    model = HybridClassifier(160, {'channels': [4, 8, 8], 'kernels': [7, 5, 3], 'dropout': .3}).train()
    normalization = model.statistics[1]
    running_mean, running_var = normalization.running_mean.clone(), normalization.running_var.clone()
    logits = model(torch.randn(1, 2, 1035), torch.randn(1, 160))
    logits.sum().backward()
    assert logits.shape == (1,) and torch.isfinite(logits).all()
    torch.testing.assert_close(normalization.running_mean, running_mean)
    torch.testing.assert_close(normalization.running_var, running_var)
    assert normalization.weight.grad is not None
