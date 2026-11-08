from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODELS = ('lightgbm', 'xgboost', 'cnn1d', 'hybrid')


def load_training_config(path: str | Path | None = None) -> dict:
    default = ROOT / 'configs/training.yaml'
    with default.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    selected = Path(path).resolve() if path else default
    if selected != default:
        with selected.open(encoding='utf-8') as handle:
            supplied = yaml.safe_load(handle)

        def merge(target, changes):
            if not isinstance(changes, dict):
                raise ValueError('Configuration must be a mapping')
            for key, value in changes.items():
                if key not in target:
                    raise ValueError(f'Unknown training configuration key: {key}')
                if isinstance(target[key], dict):
                    merge(target[key], value)
                else:
                    target[key] = value
        merge(config, supplied)
    for section, key in [('data', 'processed_dir'), ('experiment', 'output_dir'), ('experiment', 'model_dir')]:
        config[section][key] = str((selected.parent / config[section][key]).resolve())
    validate_training_config(config)
    return config


def validate_training_config(c: dict) -> None:
    if c['experiment']['device'] not in ('auto', 'cpu', 'cuda', 'mps'):
        raise ValueError('device must be auto, cpu, cuda or mps')
    if not isinstance(c['experiment']['seed'], int) or not 0 <= c['experiment']['seed'] < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    if c['experiment']['cpu_threads'] < 1 or c['experiment']['num_workers'] < 0:
        raise ValueError('Invalid CPU thread/worker count')
    e = c['evaluation']
    if e['primary_metric'] != 'average_precision' or e['default_threshold'] != .5:
        raise ValueError('Use Average Precision for selection and 0.5 for the fixed baseline')
    if e['threshold_objective'] not in ('f1', 'recall_at_precision'):
        raise ValueError('Unknown threshold objective')
    if not 0 < e['minimum_precision'] <= 1 or not 0 < e['threshold_start'] <= e['threshold_stop'] < 1 or e['threshold_step'] <= 0:
        raise ValueError('Invalid threshold grid or minimum precision')
    if any(not 0 < x <= 100 for x in e['top_k_percentages']):
        raise ValueError('top_k_percentages must be in (0,100]')
    if e['bootstrap_iterations'] < 1:
        raise ValueError('bootstrap_iterations must be positive')
    for name, model in c['models'].items():
        if name in ('lightgbm', 'xgboost'):
            if model['early_stopping_rounds'] < 1 or model['params']['n_estimators'] < 1 or model['tuning_trials'] < 1:
                raise ValueError('Tree rounds, patience and trial counts must be positive')
            for parameter, spec in model['tuning_space'].items():
                if len(spec) != 4 or spec[0] not in ('int', 'float') or spec[1] > spec[2] or spec[3] and spec[1] <= 0:
                    raise ValueError(f'Invalid tuning range for {name}.{parameter}')
        else:
            for key in ('batch_size', 'max_epochs', 'early_stopping_patience', 'scheduler_patience'):
                if model[key] < 1:
                    raise ValueError(f'{name}.{key} must be positive')
            if not 0 <= model['dropout'] < 1 or model['learning_rate'] <= 0 or model['gradient_clip'] <= 0:
                raise ValueError('Invalid neural dropout, learning rate or gradient clipping')
            if len(model['channels']) != 3 or len(model['kernels']) != 3 or any(k < 1 or k % 2 == 0 for k in model['kernels']):
                raise ValueError('CNN needs three positive channel widths and three odd kernels')
            if any(ch < 1 for ch in model['channels']):
                raise ValueError('CNN channels must be positive')
    if c['visualization']['dpi'] < 72 or c['visualization']['examples_per_category'] < 1:
        raise ValueError('Invalid visualization settings')


def diagnostic_config(config: dict) -> dict:
    c = deepcopy(config)
    c['experiment']['diagnostic'] = True
    for name, model in c['models'].items():
        if name in ('lightgbm', 'xgboost'):
            model.update(tuning_enabled=False, early_stopping_rounds=3)
            model['params']['n_estimators'] = 8
        else:
            model.update(max_epochs=2, batch_size=64, early_stopping_patience=2)
    c['evaluation']['bootstrap_enabled'] = False
    c['visualization']['examples_per_category'] = 1
    return c
