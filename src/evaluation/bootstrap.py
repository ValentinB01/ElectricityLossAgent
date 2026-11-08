"""Optional stratified paired consumer bootstrap, without any model refitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def paired_bootstrap(y, scores: dict, thresholds: dict, iterations=1000, seed=42) -> dict:
    y = np.asarray(y)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if not len(positives) or not len(negatives):
        raise ValueError('Stratified bootstrap needs both classes')
    rng = np.random.default_rng(seed)
    samples = {name: {metric: [] for metric in ('average_precision', 'roc_auc', 'f1', 'recall')} for name in scores}
    for _ in range(iterations):
        ix = np.r_[rng.choice(negatives, len(negatives), replace=True), rng.choice(positives, len(positives), replace=True)]
        target = y[ix]
        for name, probability in scores.items():
            p = np.asarray(probability)[ix]
            values = {'average_precision': average_precision_score(target, p), 'roc_auc': roc_auc_score(target, p)}
            threshold = thresholds[name]
            if threshold is not None:
                prediction = p >= threshold
                tp = int((prediction & (target == 1)).sum())
                fp = int((prediction & (target == 0)).sum())
                fn = len(positives) - tp
                values.update(f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0., recall=tp / len(positives))
            for metric, value in values.items():
                samples[name][metric].append(float(value))
    def interval(values):
        return [float(x) for x in np.quantile(values, [.025, .975])] if len(values) else None
    differences = {}
    pairs = [('hybrid', 'lightgbm'), ('hybrid', 'xgboost'), ('cnn1d', 'lightgbm'), ('xgboost', 'lightgbm')]
    for a, b in pairs:
        if a in scores and b in scores:
            differences[f'{a}_minus_{b}'] = {metric: interval(np.asarray(samples[a][metric]) - np.asarray(samples[b][metric]))
                                            for metric in samples[a] if len(samples[a][metric]) == len(samples[b][metric])}
    return {'iterations': iterations, 'seed': seed, 'sampling': 'stratified paired consumer indices',
            'confidence_level': .95, 'models': {n: {m: interval(v) for m, v in by_metric.items()} for n, by_metric in samples.items()},
            'paired_differences': differences,
            'interpretation': 'Descriptive percentile intervals for frozen test predictions; no selection or retraining.'}


def bootstrap_run(run_dir: Path):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / 'run.json').read_text(encoding='utf-8'))
    frame = pd.read_csv(run_dir / 'predictions/all_models_test.csv', dtype={'CONS_NO': str})
    config = manifest['config']
    result = paired_bootstrap(frame.y_true.to_numpy(), {n: frame[f'{n}_score'].to_numpy() for n in manifest['models']},
                              {n: r['threshold_selection']['threshold'] for n, r in manifest['models'].items()},
                              config['evaluation']['bootstrap_iterations'], config['experiment']['seed'])
    output = run_dir / 'metrics/bootstrap.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result
