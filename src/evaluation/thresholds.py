from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from .metrics import validate_scores


def select_threshold(y_validation, validation_scores, config: dict) -> tuple[dict, pd.DataFrame]:
    y, p = validate_scores(y_validation, validation_scores)
    grid = np.arange(config['threshold_start'], config['threshold_stop'] + config['threshold_step'] / 2,
                     config['threshold_step'])
    grid = np.unique(np.round(grid[(grid >= config['threshold_start']) & (grid <= config['threshold_stop'] + 1e-10)], 10))
    table = pd.DataFrame([{'threshold': float(t),
                            'precision': precision_score(y, p >= t, zero_division=0),
                            'recall': recall_score(y, p >= t, zero_division=0),
                            'f1': f1_score(y, p >= t, zero_division=0),
                            'predicted_positives': int((p >= t).sum())} for t in grid])
    objective = config['threshold_objective']
    status = 'selected'
    if not config['optimize_threshold']:
        selected = float(config['default_threshold'])
        status = 'optimization_disabled'
    elif objective == 'recall_at_precision':
        feasible = table[(table.precision >= config['minimum_precision']) & (table.predicted_positives > 0)]
        if feasible.empty:
            # Explicit abstention: no switch to a different objective.
            return {'threshold': None, 'objective': objective, 'fit_split': 'val',
                    'status': 'infeasible_minimum_precision', 'minimum_precision': config['minimum_precision']}, table
        selected = float(feasible.sort_values(['recall', 'precision', 'threshold'], ascending=[False, False, False]).iloc[0].threshold)
    else:
        # Ties favor higher precision, then a higher threshold (deterministic).
        selected = float(table.sort_values(['f1', 'precision', 'threshold'], ascending=[False, False, False]).iloc[0].threshold)
    return {'threshold': selected, 'objective': objective, 'fit_split': 'val', 'status': status,
            'minimum_precision': config['minimum_precision'], 'grid': grid.tolist()}, table
