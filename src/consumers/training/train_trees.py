from __future__ import annotations

from copy import deepcopy
import inspect
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.consumers.evaluation.metrics import classification_metrics
from .datasets import SplitData
from .experiment_tracker import write_json

LOGGER = logging.getLogger(__name__)


def positive_class_weight(labels) -> float:
    labels = np.asarray(labels)
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if not positives or not negatives:
        raise ValueError('Training data must include both classes')
    return negatives / positives


def _fit(name, params, positive_weight, seed, threads, patience, train, val):
    if train.split != 'train' or val.split != 'val':
        raise ValueError('Only train fitting and val early stopping are allowed')
    if name == 'lightgbm':
        from lightgbm import early_stopping, record_evaluation
        from src.consumers.models.lightgbm_model import create_classifier
        model = create_classifier(params, seed, threads, positive_weight)
        history = {}
        evaluation = {'eval_X': (train.tabular(), val.tabular()), 'eval_y': (train.y, val.y)} if 'eval_X' in inspect.signature(model.fit).parameters else {'eval_set': [(train.tabular(), train.y), (val.tabular(), val.y)]}
        model.fit(train.tabular(), train.y, **evaluation,
                  eval_names=['train', 'val'], callbacks=[early_stopping(patience, first_metric_only=True, verbose=False),
                                                         record_evaluation(history)])
        frame = pd.DataFrame({'iteration': np.arange(len(history['val']['average_precision'])) + 1,
                              'train_average_precision': history['train']['average_precision'],
                              'val_average_precision': history['val']['average_precision']})
        best = model.best_iteration_
    else:
        from src.consumers.models.xgboost_model import create_classifier
        model = create_classifier(params, seed, threads, positive_weight, patience)
        model.fit(train.tabular(), train.y, eval_set=[(train.tabular(), train.y), (val.tabular(), val.y)], verbose=False)
        history = model.evals_result()
        frame = pd.DataFrame({'iteration': np.arange(len(history['validation_1']['aucpr'])) + 1,
                              'train_aucpr': history['validation_0']['aucpr'],
                              'val_aucpr': history['validation_1']['aucpr']})
        best = model.best_iteration + 1
    return model, frame, int(best)


def _suggest(trial, space, base):
    params = deepcopy(base)
    for name, (kind, lower, upper, log) in space.items():
        params[name] = trial.suggest_int(name, lower, upper, log=log) if kind == 'int' else trial.suggest_float(name, lower, upper, log=log)
    return params


def train_tree(name: str, train: SplitData, val: SplitData, config: dict, folder: Path) -> dict:
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    settings = config['models'][name]
    seed, threads = config['experiment']['seed'], config['experiment']['cpu_threads']
    ratio = positive_class_weight(train.y)
    choices = [False, True] if settings['compare_class_weights'] else [settings['weighted_loss']]
    candidates = []
    best_model, best_history, best_result = None, None, None
    for weighted in choices:
        weight = ratio if weighted else 1.0
        params = deepcopy(settings['params'])
        label = 'weighted' if weighted else 'unweighted'
        candidate_folder = folder / label
        candidate_folder.mkdir()
        if settings['tuning_enabled']:
            import optuna
            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=seed))
            def objective(trial):
                candidate = _suggest(trial, settings['tuning_space'], settings['params'])
                model, _, _ = _fit(name, candidate, weight, seed, threads, settings['early_stopping_rounds'], train, val)
                return float(average_precision_score(val.y, model.predict_proba(val.tabular())[:, 1]))
            study.optimize(objective, n_trials=settings['tuning_trials'], show_progress_bar=True)
            params.update(study.best_params)
            study.trials_dataframe().to_csv(candidate_folder / 'optuna_trials.csv', index=False)
        LOGGER.info('%s: fitting %s, training-derived positive weight %.4f', name, label, weight)
        fit_start = time.perf_counter()
        model, history, best_round = _fit(name, params, weight, seed, threads, settings['early_stopping_rounds'], train, val)
        fit_time = time.perf_counter() - fit_start
        score = model.predict_proba(val.tabular())[:, 1]
        metrics = classification_metrics(val.y, score)
        record = {'candidate': label, 'weighted': weighted, 'positive_weight': weight,
                  'validation_average_precision': metrics['average_precision'], 'best_iteration': best_round,
                  'fit_time_seconds': fit_time, 'params': params}
        history.to_csv(candidate_folder / 'history.csv', index=False)
        write_json(candidate_folder / 'validation_metrics.json', metrics)
        write_json(candidate_folder / 'configuration.json', record)
        candidates.append(record)
        if best_result is None or record['validation_average_precision'] > best_result['validation_average_precision']:
            best_model, best_history, best_result = model, history, record
        LOGGER.info('%s %s: validation AP %.5f, selected round %d', name, label, metrics['average_precision'], best_round)
    joblib.dump(best_model, folder / 'model.joblib')
    if name == 'lightgbm':
        best_model.booster_.save_model(str(folder / 'model.txt'), num_iteration=best_result['best_iteration'])
        importance = best_model.booster_.feature_importance(importance_type='gain')
    else:
        best_model.save_model(folder / 'model.ubj')
        gain = best_model.get_booster().get_score(importance_type='gain')
        importance = [gain.get(name, 0.) for name in train.feature_names]
    pd.DataFrame({'feature': train.feature_names, 'gain': importance}).sort_values('gain', ascending=False).to_csv(folder / 'feature_importance.csv', index=False)
    best_history.to_csv(folder / 'history.csv', index=False)
    write_json(folder / 'feature_names.json', train.feature_names)
    result = {**best_result, 'model': name, 'model_path': str((folder / 'model.joblib').resolve()),
              'experiment_dir': str(folder.resolve()), 'candidates': candidates,
              'training_time_seconds': time.perf_counter() - start,
              'training_time_scope': 'all weighting candidates and optional tuning, including serialization',
              'parameter_count': None, 'model_size_bytes': (folder / 'model.joblib').stat().st_size,
              'peak_gpu_memory_bytes': None, 'device': 'cpu',
              'early_stopping_split': 'val', 'fit_split': 'train',
              'early_stopping_metric': 'average_precision' if name == 'lightgbm' else 'xgboost aucpr (not sklearn AP)'}
    write_json(folder / 'training_result.json', result)
    return result
