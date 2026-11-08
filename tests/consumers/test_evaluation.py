"""Evaluation primitives work without model/training modules or prepared data."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.consumers.evaluation.bootstrap import bootstrap_run, paired_bootstrap
from src.consumers.evaluation.metrics import classification_metrics, top_k_metrics, validate_scores
from src.consumers.evaluation.thresholds import select_threshold


@pytest.fixture
def threshold_config():
    return {'threshold_start': .01, 'threshold_stop': .99, 'threshold_step': .01,
            'threshold_objective': 'f1', 'minimum_precision': .70,
            'optimize_threshold': True, 'default_threshold': .5}


def test_metrics_known_confusion_and_degenerate_classes():
    m = classification_metrics([0, 0, 1, 1], [.1, .6, .4, .9])
    assert m['confusion_matrix'] == [[1, 1], [1, 1]]
    assert m['precision'] == m['recall'] == m['f1'] == m['f2'] == .5
    assert m['accuracy'] == m['specificity'] == m['false_positive_rate'] == .5
    assert m['mcc'] == 0
    assert m['brier_score'] == pytest.approx((.01 + .36 + .36 + .01) / 4)
    assert m['average_precision'] == pytest.approx((1 + 2/3) / 2)
    assert m['roc_auc'] == .75
    assert classification_metrics([0, 0], [.1, .2])['average_precision'] is None
    assert classification_metrics([1, 1], [.1, .2])['roc_auc'] is None


@pytest.mark.parametrize('scores', [[.1, np.nan], [-.1, .2], [.1, 1.1], [[.1], [.2]]])
def test_invalid_scores_rejected(scores):
    with pytest.raises(ValueError):
        validate_scores([0, 1], scores)


def test_threshold_selection_and_precision_constraint(threshold_config):
    selected, grid = select_threshold([0, 0, 1, 1], [.1, .2, .3, .4], threshold_config)
    assert selected['fit_split'] == 'val' and selected['threshold'] == .3
    assert len(grid) == 99
    repeat, _ = select_threshold([0, 0, 1, 1], [.1, .2, .3, .4], threshold_config)
    assert repeat == selected
    constrained = {**threshold_config, 'threshold_objective': 'recall_at_precision', 'minimum_precision': .99}
    feasible, _ = select_threshold([0, 0, 1, 1], [.1, .2, .3, .4], constrained)
    assert feasible['threshold'] == .3
    infeasible, _ = select_threshold([0, 1], [.9, .1], constrained)
    assert infeasible['threshold'] is None and 'infeasible' in infeasible['status']
    fixed, _ = select_threshold([0, 0, 1, 1], [.1, .2, .3, .4],
                                {**threshold_config, 'optimize_threshold': False})
    assert fixed['threshold'] == .5 and fixed['status'] == 'optimization_disabled'


def test_top_k_ceil_and_id_ties():
    result = top_k_metrics([0, 1, 1, 0], [.2, .9, .8, .1], [25, 50, 100])
    assert result[0]['precision_at_k'] == 1 and result[0]['recall_at_k'] == .5
    assert result[1]['precision_at_k'] == result[1]['recall_at_k'] == 1
    assert result[2]['k'] == 4 and result[2]['precision_at_k'] == .5
    tied = top_k_metrics([1, 0], [.5, .5], [50], ['b', 'a'])
    assert tied[0]['true_positives'] == 0


def test_paired_bootstrap_reproducibility_and_identical_models():
    kwargs = dict(y=[0, 0, 1, 1], scores={'lightgbm': [.1, .2, .8, .9], 'hybrid': [.1, .2, .8, .9]},
                  thresholds={'lightgbm': .5, 'hybrid': .5}, iterations=20, seed=42)
    result = paired_bootstrap(**kwargs)
    assert paired_bootstrap(**kwargs) == result
    assert result['paired_differences']['hybrid_minus_lightgbm']['average_precision'] == [0., 0.]
    assert result['models']['lightgbm']['roc_auc'] == [1., 1.]


def test_bootstrap_saved_prediction_wrapper(tmp_path):
    manifest = {'models': {'lightgbm': {'threshold_selection': {'threshold': .5}},
                           'hybrid': {'threshold_selection': {'threshold': .5}}},
                'config': {'evaluation': {'bootstrap_iterations': 10}, 'experiment': {'seed': 42}}}
    (tmp_path / 'run.json').write_text(json.dumps(manifest), encoding='utf-8')
    (tmp_path / 'predictions').mkdir()
    pd.DataFrame({'CONS_NO': ['a', 'b', 'c', 'd'], 'y_true': [0, 0, 1, 1],
                  'lightgbm_score': [.1, .2, .8, .9], 'hybrid_score': [.1, .2, .8, .9]}).to_csv(
                      tmp_path / 'predictions/all_models_test.csv', index=False)
    result = bootstrap_run(tmp_path)
    saved = json.loads((tmp_path / 'metrics/bootstrap.json').read_text(encoding='utf-8'))
    assert saved == result and result['iterations'] == 10


def test_imports_without_training_or_model_dependencies():
    code = """
import importlib.abc
import sys
class ExcludeLaterCommits(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch' or fullname.startswith(('src.consumers.training', 'src.consumers.models', 'src.consumers.visualization')):
            raise ModuleNotFoundError('Unavailable in evaluation-only commit: ' + fullname, name=fullname)
sys.meta_path.insert(0, ExcludeLaterCommits())
from src.consumers.evaluation.metrics import classification_metrics
from src.consumers.evaluation.thresholds import select_threshold
from src.consumers.evaluation.bootstrap import paired_bootstrap, bootstrap_run
assert classification_metrics([0, 1], [.1, .9])['average_precision'] == 1.
assert not any(n.startswith(('src.consumers.training', 'src.consumers.models', 'src.consumers.visualization')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
