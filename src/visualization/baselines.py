"""Fit simple baselines and compare with frozen predictions on identical consumers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import yaml

from src.evaluation.evaluator import locate_run, prediction_table
from src.evaluation.metrics import classification_metrics
from src.evaluation.thresholds import select_threshold
from src.training.config import ROOT, load_training_config
from src.training.datasets import PreparedDataStore
from src.training.experiment_tracker import read_json, write_json
from .plots import COLORS, LABELS, plt

LOGGER = logging.getLogger(__name__)
BASELINES = {
    'always_normal': ('Mereu normal', '#A3ABB8'),
    'train_prevalence': ('Prevalență constantă', '#778397'),
    'decision_stump': ('Arbore de adâncime 1', '#526176'),
    'logistic_regression': ('Regresie logistică', '#25354D'),
}


def fit_baselines(train, val, settings: dict, evaluation: dict):
    if train.split != 'train' or val.split != 'val':
        raise ValueError('Baseline fitting requires train; threshold selection requires val, never test')
    x_train, x_val = train.tabular(), val.tabular()
    models = {
        'always_normal': DummyClassifier(strategy='most_frequent'),
        'train_prevalence': DummyClassifier(strategy='prior'),
        'decision_stump': Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('classifier', DecisionTreeClassifier(**settings['decision_stump'])),
        ]),
        'logistic_regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True)),
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(**settings['logistic_regression'])),
        ]),
    }
    fitted = {}
    for name, model in models.items():
        LOGGER.info('Fitting baseline %s using %d training consumers', name, len(train.y))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x_train, train.y)
        probability = model.predict_proba(x_val)[:, 1]
        if name == 'always_normal':
            selection = {'threshold': .5, 'objective': 'fixed_always_normal_rule', 'fit_split': None,
                         'status': 'fixed_by_baseline_definition'}
            grid = pd.DataFrame()
        else:
            selection, grid = select_threshold(val.y, probability, evaluation)
        fitted[name] = {'model': model, 'selection': selection, 'grid': grid,
                        'validation_probability': probability}
    return fitted


def make_figure(table: pd.DataFrame, y: np.ndarray, output: Path, dpi=300):
    names = table.model.tolist()
    labels = [LABELS[n] if n in LABELS else BASELINES[n][0] for n in names]
    colors = [COLORS[n] if n in COLORS else BASELINES[n][1] for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8))
    for ax, metric, title in zip(axes, ['average_precision', 'roc_auc', 'f1'],
                                ['Average Precision (AP)', 'ROC-AUC', 'F1 la pragul selectat']):
        values = table[metric].to_numpy(dtype=float)
        bars = ax.barh(np.arange(len(names)), values, color=colors, height=.66)
        ax.set_yticks(np.arange(len(names)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.04)
        ax.set_title(title, fontsize=13, pad=14, fontweight='bold')
        ax.set_xlabel('Scor (mai mare = mai bun)')
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#DFE5ED', alpha=.7)
        ax.axhline(3.5, color='#B4BECC', linewidth=.9)
        for spine in ('top', 'right', 'left'):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=8, labelsize=10)
        for bar, value in zip(bars, values):
            ax.text(value + .012, bar.get_y() + bar.get_height()/2,
                    f'{value:.4f}'.replace('.', ','), va='center', fontsize=10)
        if metric == 'average_precision':
            ax.axvline(float(y.mean()), ls='--', lw=1.3, color='#9E3542', alpha=.85)
    fig.suptitle('Comparația modelelor cu baseline-uri simple', fontsize=19, fontweight='bold', y=.97)
    fig.text(.5, .90, f'Același set de test: {len(y):,} consumatori, {int(y.sum()):,} etichetați furt. Modele în culori; baseline-uri în gri.',
             ha='center', fontsize=11, color='#4D5D72')
    fig.text(.04, .075, f'Linia punctată: AP fără discriminare = prevalența din test ({y.mean():.2%}). AP nu este accuracy.', fontsize=10, color='#4D5D72')
    fig.text(.04, .043, 'F1: prag ales pe validare, cu excepția regulii fixe „mereu normal”. Baseline-urile se antrenează numai pe train.', fontsize=10, color='#4D5D72')
    fig.subplots_adjust(top=.82, bottom=.19, left=.12, right=.98, wspace=.77)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(output.with_suffix('.' + extension), dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def run_comparison(run_dir: Path, settings_path: Path):
    manifest = read_json(run_dir / 'run.json')
    if manifest['diagnostic'] or not manifest['evaluation_complete']:
        raise ValueError('Baselines require an evaluated full-population experiment')
    with settings_path.open(encoding='utf-8') as handle:
        settings = yaml.safe_load(handle)
    store = PreparedDataStore(manifest['config']['data']['processed_dir'])
    if store.manifest['run_fingerprint'] != manifest['alignment_audit']['preprocessing_fingerprint']:
        raise ValueError('Dataset generation differs from the frozen model experiment')
    train, val = store.load('lightgbm', 'train'), store.load('lightgbm', 'val')
    fitted = fit_baselines(train, val, settings, manifest['config']['evaluation'])
    # Test predictions are calculated only after every baseline and threshold is frozen.
    test = store.load('lightgbm', 'test')
    output_root = Path(manifest['config']['experiment']['output_dir'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = output_root / 'baselines' / stamp
    archive.mkdir(parents=True, exist_ok=False)
    rows, records = [], {}
    for name in ('lightgbm', 'xgboost', 'cnn1d', 'hybrid'):
        if name not in manifest['models']:
            raise ValueError('The comparison requires all four frozen models')
        frame = pd.read_csv(run_dir / 'predictions' / f'{name}_test.csv', dtype={'CONS_NO': str}, float_precision='round_trip')
        if not frame.CONS_NO.is_unique or set(frame.CONS_NO) != set(test.ids):
            raise ValueError('Model and baseline test populations differ')
        frame = frame.set_index('CONS_NO').loc[test.ids]
        if not np.array_equal(frame.y_true, test.y):
            raise ValueError('Model and baseline labels differ after ID alignment')
        threshold = manifest['models'][name]['threshold_selection']['threshold']
        if threshold is None:
            raise ValueError('Selected model threshold is unavailable; cannot compare optimized F1')
        metrics = classification_metrics(test.y, frame.probability, threshold)
        records[name] = {'metrics': metrics, 'source': 'existing frozen model; no retraining'}
        rows.append({'model': name, 'group': 'trained_model', **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}})
    for name, item in fitted.items():
        selection = item['selection']
        if selection['threshold'] is None:
            raise ValueError(f'No feasible validation threshold for baseline {name}')
        score = item['model'].predict_proba(test.tabular())[:, 1]
        metrics = classification_metrics(test.y, score, selection['threshold'])
        fixed = classification_metrics(test.y, score, .5)
        rows.append({'model': name, 'group': 'baseline', **{k: v for k, v in metrics.items() if k != 'confusion_matrix'}})
        records[name] = {'metrics': metrics, 'fixed_05': fixed, 'threshold_selection': selection}
        prediction_table(test, score, selection).to_csv(archive / f'{name}_test.csv', index=False)
        prediction_table(val, item['validation_probability'], selection).to_csv(archive / f'{name}_val.csv', index=False)
        if len(item['grid']):
            item['grid'].to_csv(archive / f'{name}_threshold_grid.csv', index=False)
        joblib.dump(item['model'], archive / f'{name}.joblib')
        restored = joblib.load(archive / f'{name}.joblib')
        np.testing.assert_array_equal(score, restored.predict_proba(test.tabular())[:, 1])
    table = pd.DataFrame(rows)
    table.to_csv(archive / 'models_vs_baselines.csv', index=False)
    write_json(archive / 'metadata.json', {
        'source_run': str(run_dir), 'settings': settings, 'baseline_fit_split': 'train',
        'threshold_fit_split': 'val', 'test_consumers': len(test.ids), 'test_positives': int(test.y.sum()),
        'train_prevalence': float(train.y.mean()), 'test_prevalence': float(test.y.mean()),
        'feature_names': train.feature_names, 'results': records,
        'limitations': ['Existing test set has already been evaluated during model development.',
                       'Descriptive comparison on one split; no confidence intervals or claim of significance.',
                       'Most-frequent and constant-prior scores have no ranking discrimination; their AP is test prevalence.',
                       'Always-normal F1 uses its fixed decision rule; other thresholds follow validation selection.'],
    })
    figure = archive / 'models_vs_baselines'
    make_figure(table, test.y, figure, settings['figure']['dpi'])
    for extension in ('png', 'svg'):
        target = output_root / 'figures/comparison' / f'models_vs_baselines.{extension}'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure.with_suffix('.' + extension), target)
    shutil.copy2(archive / 'models_vs_baselines.csv', output_root / 'metrics/models_vs_baselines.csv')
    print(table[['model', 'average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'threshold']].to_string(index=False))
    print(f'Baseline archive: {archive}')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/baselines.yaml')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run_comparison(locate_run(load_training_config(), args.run_dir), args.config)


if __name__ == '__main__':
    main()
