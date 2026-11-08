from __future__ import annotations

import math
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def validate_scores(y, probability) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y), np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError('Labels and probabilities must be nonempty aligned vectors')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Expected binary labels and finite scores in [0,1]')
    return y.astype(np.int8), p


def classification_metrics(y, probability, threshold: float = .5) -> dict:
    y, p = validate_scores(y, probability)
    if not 0 <= threshold <= 1:
        raise ValueError('Classification threshold must be in [0,1]')
    prediction = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = [int(x) for x in confusion_matrix(y, prediction, labels=[0, 1]).ravel()]
    both = len(np.unique(y)) == 2
    return {
        'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
        'roc_auc': float(roc_auc_score(y, p)) if both else None,
        'accuracy': float(accuracy_score(y, prediction)),
        'balanced_accuracy': float(balanced_accuracy_score(y, prediction)) if both else None,
        'precision': float(precision_score(y, prediction, zero_division=0)),
        'recall': float(recall_score(y, prediction, zero_division=0)),
        'f1': float(f1_score(y, prediction, zero_division=0)),
        'f2': float(fbeta_score(y, prediction, beta=2, zero_division=0)),
        'mcc': float(matthews_corrcoef(y, prediction)) if len(np.unique(np.r_[y, prediction])) > 1 else 0.,
        'specificity': tn / (tn + fp) if tn + fp else None,
        'false_positive_rate': fp / (tn + fp) if tn + fp else None,
        'false_negative_rate': fn / (tp + fn) if tp + fn else None,
        'brier_score': float(np.mean((p - y)**2)),
        'true_positives': tp, 'true_negatives': tn, 'false_positives': fp, 'false_negatives': fn,
        'confusion_matrix': [[tn, fp], [fn, tp]], 'threshold': float(threshold),
    }


def top_k_metrics(y, probability, percentages=(1, 5, 10, 20), ids=None) -> list[dict]:
    y, p = validate_scores(y, probability)
    # IDs break equal-score ties; no label enters the ranking.
    keys = np.arange(len(y)).astype(str) if ids is None else np.asarray(ids, dtype=str)
    order = np.lexsort((keys, -p))
    result = []
    for percentage in percentages:
        k = max(1, min(len(y), math.ceil(len(y) * percentage / 100)))
        positives = int(y[order[:k]].sum())
        result.append({'percentage': percentage, 'k': k, 'true_positives': positives,
                       'precision_at_k': positives / k,
                       'recall_at_k': positives / int(y.sum()) if y.sum() else None})
    return result
