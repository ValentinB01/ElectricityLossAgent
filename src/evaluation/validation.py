"""Verify completed experiment artifacts, independently recomputing saved metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.training.config import load_training_config
from src.training.datasets import PreparedDataStore
from src.training.experiment_tracker import read_json, write_json
from src.training.train_neural import load_network
from .evaluator import combine_predictions, locate_run
from .metrics import classification_metrics, validate_scores
from .thresholds import select_threshold


def validate_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'run.json')
    if not manifest['all_selections_frozen'] or not manifest['evaluation_complete']:
        raise ValueError('Experiment selection and test evaluation must be complete')
    if manifest['diagnostic']:
        raise ValueError('This final-population verifier is for full runs, not diagnostic subsets')
    store = PreparedDataStore(manifest['config']['data']['processed_dir'])
    selected = max(manifest['models'], key=lambda n: manifest['models'][n]['validation_average_precision'])
    if selected != manifest['best_model_by_validation_ap']:
        raise ValueError('Best-model selection must use validation AP only')
    frames = {}
    for name, result in manifest['models'].items():
        if result['fit_split'] != 'train' or result['early_stopping_split'] != 'val':
            raise ValueError('Unexpected fit/early-stopping population')
        if name in ('lightgbm', 'xgboost'):
            joblib.load(result['model_path'])
        else:
            load_network(result['model_path'])
        val = pd.read_csv(run_dir / 'predictions' / f'{name}_val.csv', dtype={'CONS_NO': str}, float_precision='round_trip')
        test = pd.read_csv(run_dir / 'predictions' / f'{name}_test.csv', dtype={'CONS_NO': str}, float_precision='round_trip')
        if set(val.CONS_NO) & set(test.CONS_NO):
            raise ValueError('Validation and test IDs overlap')
        for split, predictions in [('val', val), ('test', test)]:
            assignment = store.assignments.loc[store.assignments.split == split]
            if not np.array_equal(predictions.CONS_NO, assignment.CONS_NO) or not np.array_equal(predictions.y_true, assignment.FLAG):
                raise ValueError(f'{name}/{split} IDs or labels disagree with shared assignments')
            validate_scores(predictions.y_true, predictions.probability)
        selection, _ = select_threshold(val.y_true, val.probability, manifest['config']['evaluation'])
        if selection['threshold'] != result['threshold_selection']['threshold']:
            raise ValueError(f'{name} frozen threshold does not reproduce from validation predictions')
        metrics = read_json(run_dir / 'metrics' / f'{name}_test.json')
        for setting, threshold, column in [('fixed_05', .5, 'prediction_05'),
                                            ('optimized', selection['threshold'], 'prediction_optimized')]:
            if threshold is None:
                continue
            calculated = classification_metrics(test.y_true, test.probability, threshold)
            if not np.array_equal(test[column], (test.probability >= threshold).astype(int)):
                raise ValueError(f'{name} saved binary predictions disagree with their probability/threshold')
            for key, value in calculated.items():
                expected = metrics[setting][key]
                if isinstance(value, list):
                    if value != expected:
                        raise ValueError('Confusion matrix differs from saved predictions')
                elif value is None:
                    if expected is not None:
                        raise ValueError('Undefined metric was not represented explicitly')
                elif not np.isclose(value, expected, atol=1e-10, rtol=1e-10):
                    raise ValueError(f'{name} {setting}.{key} does not match saved predictions')
        frames[name] = test
    combined = combine_predictions(frames)
    saved = pd.read_csv(run_dir / 'predictions/all_models_test.csv', dtype={'CONS_NO': str}, float_precision='round_trip')
    pd.testing.assert_frame_equal(combined, saved, check_dtype=False, atol=1e-12, rtol=1e-12)
    expected = ['roc_curves', 'precision_recall_curves', 'confusion_matrices', 'confusion_matrices_normalized',
                'metrics_comparison', 'classification_errors', 'probability_distributions', 'computational_cost',
                'top_k_detection', 'model_agreement', 'improvement_over_baseline']
    settings = manifest['config']['visualization']
    if settings['generate_static']:
        for filename in expected:
            for extension, enabled in [('png', settings['save_png']), ('svg', settings['save_svg'])]:
                if enabled and not (run_dir / 'figures/comparison' / f'{filename}.{extension}').is_file():
                    raise ValueError(f'Missing required figure: {filename}.{extension}')
        for name in manifest['models']:
            for filename in ['threshold_analysis', 'training_curves'] + (['feature_importance'] if name in ('lightgbm', 'xgboost') else []):
                extension = 'png' if settings['save_png'] else 'svg'
                if not (run_dir / 'figures' / name / f'{filename}.{extension}').is_file():
                    raise ValueError(f'Missing model figure: {name}/{filename}')
    if settings['generate_interactive']:
        page = (run_dir / 'dashboard/model_comparison.html').read_text(encoding='utf-8')
        if 'plotly.js' not in page or '<script src="https://' in page:
            raise ValueError('Dashboard must embed Plotly and work without an external CDN')
    result = {'passed': True, 'models': list(frames), 'test_consumers_per_model': len(saved),
              'best_model_selected_by_validation_ap': selected,
              'checks': ['all checkpoints load', 'disjoint validation/test populations', 'all saved IDs/labels aligned',
                         'thresholds reproduce from validation', 'binary predictions reproduce from scores',
                         'all test metrics independently recomputed', 'combined predictions match',
                         'required static figures exist', 'standalone dashboard contains embedded Plotly'],
              'development_limitation': 'First diagnostic run exposed 128 test consumers; no training settings were changed using those scores.'}
    write_json(run_dir / 'verification.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    print(validate_run(locate_run(load_training_config(args.config), args.run_dir)))


if __name__ == '__main__':
    main()
