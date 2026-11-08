from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.consumers.evaluation.metrics import classification_metrics
from src.consumers.training.config import load_training_config
from src.consumers.visualization.baselines import fit_baselines


def bundle(split, x, y):
    return SimpleNamespace(split=split, y=np.asarray(y), tabular=lambda: pd.DataFrame(x, columns=['a', 'b']))


def test_baselines_train_only_and_constant_reference_metrics():
    train = bundle('train', [[0, np.nan], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7]], [0, 0, 0, 0, 1, 1])
    val = bundle('val', [[0, 1], [1e8, np.nan], [5, 6], [7, 8]], [0, 0, 1, 1])
    settings = {'decision_stump': {'max_depth': 1, 'random_state': 42},
                'logistic_regression': {'C': 1., 'max_iter': 1000, 'random_state': 42}}
    models = fit_baselines(train, val, settings, load_training_config()['evaluation'])
    linear = models['logistic_regression']['model']
    # The enormous validation value must not affect learned train medians or scaling.
    assert linear.named_steps['imputer'].statistics_[0] == 2.5
    assert linear.named_steps['scaler'].mean_[0] == pytest.approx(16 / 6)
    medians = linear.named_steps['imputer'].statistics_.copy()
    linear.predict_proba(val.tabular())
    np.testing.assert_array_equal(linear.named_steps['imputer'].statistics_, medians)
    for name in ('always_normal', 'train_prevalence'):
        item = models[name]
        p = item['model'].predict_proba(val.tabular())[:, 1]
        m = classification_metrics(val.y, p, item['selection']['threshold'])
        assert m['average_precision'] == .5 and m['roc_auc'] == .5
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    majority = models['always_normal']
    m = classification_metrics(val.y, majority['validation_probability'], .5)
    assert m['f1'] == 0 and m['accuracy'] == .5
    with pytest.raises(ValueError, match='never test'):
        fit_baselines(train, bundle('test', [[1, 2]], [0]), settings, load_training_config()['evaluation'])
