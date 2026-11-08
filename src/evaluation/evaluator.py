"""Evaluate frozen checkpoints; selection is never performed in this module."""
from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.training.config import load_training_config
from src.training.datasets import PreparedDataStore
from src.training.experiment_tracker import append_registry, read_json, write_json, seed_everything
from src.training.inference import predict_frozen_model
from .metrics import classification_metrics, top_k_metrics

LOGGER = logging.getLogger(__name__)


def prediction_table(bundle, scores, selection):
    # Match metric/threshold calculations and preserve float32 scores losslessly in CSV.
    scores = np.asarray(scores, dtype=np.float64)
    threshold = selection['threshold']
    frame = pd.DataFrame({'CONS_NO': bundle.ids, 'y_true': bundle.y, 'probability': scores,
                          'prediction_05': (scores >= .5).astype(int),
                          'prediction_optimized': (scores >= threshold).astype(int) if threshold is not None else pd.NA,
                          'optimized_threshold': threshold})
    return frame


def combine_predictions(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = None
    for name, frame in predictions.items():
        if not frame.CONS_NO.is_unique:
            raise ValueError('Duplicate consumer IDs in predictions')
        current = frame[['CONS_NO', 'y_true', 'probability']].rename(columns={'probability': f'{name}_score'})
        if combined is None:
            combined = current.copy()
        else:
            if set(combined.CONS_NO) != set(current.CONS_NO):
                raise ValueError('Prediction populations differ')
            current = current.set_index('CONS_NO').loc[combined.CONS_NO]
            if not np.array_equal(current.y_true, combined.y_true):
                raise ValueError('Ground-truth labels disagree across models')
            combined[f'{name}_score'] = current[f'{name}_score'].to_numpy()
    return combined


def locate_run(config: dict, explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    pointer = Path(config['experiment']['output_dir']) / 'latest_run.json'
    if not pointer.is_file():
        raise FileNotFoundError('No completed full training selection. Run python -m src.training.train --all first.')
    return Path(read_json(pointer)['run_dir'])


def evaluate_run(run_dir: str | Path, publish: bool = True) -> pd.DataFrame:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'run.json')
    if not manifest['all_selections_frozen'] or set(manifest['models']) != set(manifest['requested_models']):
        raise ValueError('All requested models and validation thresholds must be frozen before test evaluation')
    config = manifest['config']
    seed_everything(config['experiment']['seed'], config['experiment']['cpu_threads'])
    store = PreparedDataStore(config['data']['processed_dir'])
    if store.manifest['run_fingerprint'] != manifest['alignment_audit']['preprocessing_fingerprint']:
        raise ValueError('Processed data changed since model selection')
    metrics_dir, predictions_dir = run_dir / 'metrics', run_dir / 'predictions'
    metrics_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)
    rows, predictions, records = [], {}, {}
    for name, result in manifest['models'].items():
        if not manifest['diagnostic']:
            # Re-export validation scores from the frozen checkpoint, without selecting
            # a new threshold. This also preserves neural float32 values exactly in CSV.
            validation = store.load(name, 'val')
            validation_probability, _ = predict_frozen_model(result, validation, config)
            prediction_table(validation, validation_probability, result['threshold_selection']).to_csv(
                predictions_dir / f'{name}_val.csv', index=False)
        LOGGER.info('Evaluating frozen %s checkpoint on test consumers', name)
        if manifest['diagnostic']:
            # Diagnostics use validation consumers, never the final test population.
            bundle = store.load(name, 'val')
            used = pd.read_csv(predictions_dir / f'{name}_val.csv', dtype={'CONS_NO': str}).CONS_NO
            remaining = np.flatnonzero(~np.isin(bundle.ids, used))
            if len(remaining):
                bundle = replace(bundle, ids=bundle.ids[remaining], y=bundle.y[remaining], order=bundle.order[remaining])
            bundle = bundle.sample(min(128, len(bundle.y)), config['experiment']['seed'] + 1)
        else:
            bundle = store.load(name, 'test')
        probability, duration = predict_frozen_model(result, bundle, config)
        selection = result['threshold_selection']
        frame = prediction_table(bundle, probability, selection)
        frame.to_csv(predictions_dir / f'{name}_test.csv', index=False)
        predictions[name] = frame
        fixed = classification_metrics(bundle.y, probability, .5)
        optimized = classification_metrics(bundle.y, probability, selection['threshold']) if selection['threshold'] is not None else None
        record = {'model': name, 'split': 'diagnostic_validation' if manifest['diagnostic'] else 'test', 'diagnostic': manifest['diagnostic'],
                  'fixed_05': fixed, 'optimized': optimized, 'threshold_selection': selection,
                  'top_k': top_k_metrics(bundle.y, probability, config['evaluation']['top_k_percentages'], bundle.ids)}
        records[name] = record
        write_json(metrics_dir / f'{name}_test.json', record)
        write_json(Path(result['experiment_dir']) / 'test_metrics.json', record)
        result['inference_time_seconds'] = duration
        result['inference_seconds_per_1000'] = duration / len(bundle.y) * 1000
        base = optimized if optimized is not None else {key: None for key in fixed}
        row = {'model': name, **{k: v for k, v in base.items() if k not in ('confusion_matrix', 'threshold')},
               'optimized_threshold': selection['threshold'], 'threshold_status': selection['status'],
               **{f'fixed_05_{k}': v for k, v in fixed.items() if k not in ('confusion_matrix', 'threshold')},
               'validation_average_precision': result['validation_average_precision'],
               'training_time_seconds': result['training_time_seconds'], 'inference_time_seconds': duration,
               'inference_seconds_per_1000': result['inference_seconds_per_1000'],
               'parameter_count': result['parameter_count'], 'model_size_bytes': result['model_size_bytes'],
               'peak_gpu_memory_bytes': result['peak_gpu_memory_bytes'], 'device': result['device'],
               'run_id': manifest['run_id'], 'seed': config['experiment']['seed'], 'split': 'test',
               'diagnostic': manifest['diagnostic']}
        # Threshold-independent metrics remain defined even if the precision constraint is infeasible.
        for key in ('average_precision', 'roc_auc', 'brier_score'):
            row[key] = fixed[key]
        rows.append(row)
        LOGGER.info('%s test: AP %.5f, ROC-AUC %.5f, threshold %s', name, fixed['average_precision'], fixed['roc_auc'], selection['threshold'])
    combined = combine_predictions(predictions)
    combined.to_csv(predictions_dir / 'all_models_test.csv', index=False)
    results = pd.DataFrame(rows)
    results.to_csv(metrics_dir / 'model_comparison.csv', index=False)
    json_rows = results.astype(object).where(results.notna(), None).to_dict('records')
    write_json(metrics_dir / 'model_comparison.json', json_rows)
    manifest['evaluation_complete'] = True
    write_json(run_dir / 'run.json', manifest)
    if publish and not manifest['diagnostic']:
        root = Path(config['experiment']['output_dir'])
        for directory in ('predictions', 'metrics'):
            (root / directory).mkdir(parents=True, exist_ok=True)
            for path in (run_dir / directory).glob('*'):
                if path.is_file():
                    shutil.copy2(path, root / directory / path.name)
        # Evaluation reruns update the same registry records, not append duplicates.
        registry = root / 'metrics/experiments.csv'
        if registry.exists():
            previous = pd.read_csv(registry)
            previous.loc[previous.run_id != manifest['run_id']].to_csv(registry, index=False)
        append_registry(root, results.to_dict('records'))
    if config['evaluation']['bootstrap_enabled']:
        from .bootstrap import bootstrap_run
        bootstrap_run(run_dir)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    evaluate_run(locate_run(load_training_config(args.config), args.run_dir))


if __name__ == '__main__':
    main()
