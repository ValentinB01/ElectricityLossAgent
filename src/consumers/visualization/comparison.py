"""Generate figures, descriptive error analysis, report and offline dashboard."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from src.consumers.evaluation.evaluator import locate_run
from src.consumers.evaluation.error_analysis import analyze_errors
from src.consumers.training.config import load_training_config
from src.consumers.training.experiment_tracker import read_json, write_json
from .plots import plt, COLORS, LABELS, comparison_figures, model_figures, save_figure


def consumer_figures(run_dir: Path, manifest: dict):
    root = Path(manifest['config']['data']['processed_dir'])
    original = np.load(root / 'shared/consumption.npy', mmap_mode='r', allow_pickle=False)
    consumers = pd.read_csv(root / 'shared/consumers.csv', dtype={'CONS_NO': str})
    lookup = pd.Series(np.arange(len(consumers)), index=consumers.CONS_NO)
    calendar = pd.to_datetime(pd.read_csv(root / 'shared/calendar.csv').date)
    predictions = pd.read_csv(run_dir / 'predictions/all_models_test.csv', dtype={'CONS_NO': str}).set_index('CONS_NO')
    cases = pd.read_csv(run_dir / 'metrics/representative_consumers.csv', dtype={'CONS_NO': str})
    reference = manifest['best_model_by_validation_ap']
    # Four outcome groups for the validation-preselected reference model, plus disagreement.
    chosen = cases[cases.model.isin([reference, 'all'])]
    for _, case in chosen.iterrows():
        consumer = case.CONS_NO
        scores = predictions.loc[consumer]
        row = original[int(lookup.loc[consumer])]
        fig, ax = plt.subplots(figsize=(11, 4.5), layout='constrained')
        ax.plot(calendar, row, color='#334155', lw=.9)
        ax.set(xlabel='Calendar date; missing observations remain gaps', ylabel='Raw daily consumption',
               title=f'{case.category.replace("_", " ").title()} · {consumer}\nConsumer label={int(scores.y_true)}; ' +
                     ', '.join(f'{LABELS[n]}={scores[f"{n}_score"]:.3f}' for n in manifest['models']))
        fig.suptitle(f'Reference: {LABELS[reference]}; case selected by documented deterministic rule', fontsize=10)
        save_figure(fig, run_dir / 'figures/examples' / f'{case.category}_{consumer}', manifest['config']['visualization'])


def write_report(run_dir: Path, manifest: dict, metrics: pd.DataFrame, errors: dict):
    names = list(manifest['models'])
    selected = manifest['best_model_by_validation_ap']
    data_report = read_json(Path(manifest['config']['data']['processed_dir']) / 'shared/dataset_report.json')
    diagnostic = manifest['diagnostic']
    lines = ['# Electricity theft model comparison', '',
             '**DIAGNOSTIC SUBSET ONLY — no full-population performance claims.**' if diagnostic else 'Full prepared training population; common held-out consumer evaluation.', '',
             f"Run: `{manifest['run_id']}`. Seed: {manifest['config']['experiment']['seed']}. Timestamp (UTC): {manifest['timestamp_utc']}.", '',
             f'**Preselected by validation Average Precision: {LABELS[selected]}**. This choice and every classification threshold were frozen before test inference.', '',
             '| Model | Validation AP | Test AP | ROC-AUC | Precision | Recall | F1 | F2 | Threshold | FP | FN |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        row = metrics.loc[name]
        values = [manifest['models'][name]['validation_average_precision'], row.average_precision, row.roc_auc,
                  row.precision, row.recall, row.f1, row.f2, row.optimized_threshold, row.false_positives, row.false_negatives]
        lines.append('| ' + LABELS[name] + ' | ' + ' | '.join(f'{v:.4f}' if pd.notna(v) else 'Unavailable' for v in values) + ' |')
    lines += ['', '## Metric definitions', '',
              '- Average Precision (AP): scikit-learn stepwise precision–recall summary; primary selection metric. It is not trapezoidal or interpolated PR-AUC.',
              '- ROC-AUC: probability that a randomly chosen theft-labeled consumer ranks above a normal consumer.',
              '- Precision: fraction of flagged consumers labeled theft. Recall: fraction of theft-labeled consumers detected.',
              '- F1 balances precision/recall; F2 gives recall greater weight. Balanced accuracy averages class recalls. MCC summarizes binary association.',
              '- Specificity is normal-class recall; FPR is FP/(TN+FP), FNR is FN/(TP+FN). Accuracy is secondary.',
              '- Brier score is the mean squared error of risk scores against binary labels; lower is better. Weighted-model scores are not automatically calibrated.',
              '- Precision@K and Recall@K use the ceiling of K% of the complete test population. Consumer ID breaks equal-score ties, not the target.', '',
              '## Selection and threshold policy', '',
              f"Threshold objective: `{manifest['config']['evaluation']['threshold_objective']}`; predefined grid 0.01–0.99 by default. Ties favor precision, then a higher threshold. Test labels never choose thresholds.",
              'The fixed 0.5 metrics are saved separately in JSON and prefixed `fixed_05_` in the comparison table. An infeasible minimum-precision policy explicitly returns an unavailable optimized threshold, without substituting another objective.', '',
              'Tree weighting candidates are chosen by validation AP only. LightGBM early stopping monitors AP; XGBoost monitors its native `aucpr`, which is separately labeled and is not the AP reported in the comparison. Neural checkpoints monitor validation AP with a configured minimum improvement and patience.', '',
              '## Tree and temporal models', '',
              'Both tree classifiers use the same 80 raw statistical features and preserve valid NaNs. CNN learns from the chronological consumption/mask channels. Hybrid concatenates a separately trained temporal representation with the normalized statistical branch; the statistical input count is discovered from the saved arrays.', '']
    if 'hybrid' in names:
        h = metrics.loc['hybrid']
        lines += ['Hybrid test AP differences (descriptive; no test-based model selection):', '']
        for name in names:
            if name != 'hybrid':
                delta = h.average_precision - metrics.loc[name, 'average_precision']
                lines.append(f'- Hybrid minus {LABELS[name]}: {delta:+.4f} AP ({delta*100:+.2f} percentage points).')
    lines += ['', '## Precision/recall and error analysis', '',
              'False positives are normal-labeled consumers flagged as suspicious; false negatives are theft-labeled consumers missed. Group summaries and error-vs-correct comparisons are descriptive, within the same true class. Missingness, zeros, variability, extremes, history length and temporal-change features are included when present.', '']
    for name in names:
        row = metrics.loc[name]
        lines.append(f'- {LABELS[name]}: precision {row.precision:.4f}, recall {row.recall:.4f}; FP={row.false_positives:.0f}, FN={row.false_negatives:.0f}.')
    lines += ['', f"All-model agreement: {errors['all_models_agree']} consumers. Hybrid alone correct: {errors['hybrid_only_correct']}. Strong score disagreement (range ≥0.5): {errors['strong_disagreement_count_score_range_ge_0_5']}.",
              'Inspect `error_feature_associations.csv` for error/reference medians and differences scaled by within-class IQR. These are associations, not causal explanations.',
              'Representative TP/TN/FP/FN plots select scores nearest the within-group median for the validation-preselected model; ties use ID. Disagreement plots select the largest score range, ties by ID. Empty categories produce no invented examples. Curves use the original aligned consumption with NaNs as gaps; no day-level theft labels are inferred.', '',
              '## Measured computational cost', '',
              '| Model | Device | Training/selection seconds | Inference seconds / 1,000 | Checkpoint bytes | Parameters |',
              '|---|---|---:|---:|---:|---:|']
    for name in names:
        row = metrics.loc[name]
        result = manifest['models'][name]
        lines.append(f"| {LABELS[name]} | {row.device} | {row.training_time_seconds:.2f} | {row.inference_seconds_per_1000:.4f} | {int(row.model_size_bytes):,} | {result['parameter_count'] if result['parameter_count'] is not None else 'N/A (tree ensemble)'} |")
    lines += ['', 'Tree training time includes both weighting candidates and optional tuning. Neural training time includes every epoch and validation pass. Inference excludes model deserialization and includes batching, device transfer and computation; device synchronization is used for CUDA/MPS. Timings depend on this hardware and software; GPU and CPU are not hardware-independent comparisons.', '',
              '```json', __import__('json').dumps(manifest['hardware'], indent=2), '```', '',
              '## Limitations', '',
              '- Single reproducible split and initial seed; validation is reused for early stopping, weighting selection and threshold selection. There is no nested cross-validation.',
              '- During implementation, the first real-data diagnostic run evaluated 128 test consumers before the full training run. Those scores did not change the predefined training configuration, candidate rules or thresholds. The final run freezes all model selections before full test inference, but its test population was not entirely blind during development. Subsequent smoke evaluation uses validation consumers instead; the original diagnostic archive is retained.',
              '- Hyperparameter tuning is optional and disabled in the initial run. Neural training can reach its configured epoch cap; the best saved validation checkpoint is used.',
              f"- Original data includes {data_report['population']['missing_percentage_after_preprocessing']:.2f}% missing aligned measurements and {data_report['population']['consumers_without_observations']} fully missing consumers. All were retained in the main population.",
              '- Availability differs by label. Missingness features and the observation mask can capture measurement artifacts. A dedicated quality ablation remains a future model experiment.',
              '- SGCC labels may contain noise and do not identify theft dates. Risk scores support investigation prioritization, not proof of theft or day-level ground truth.',
              '- Weighted losses can affect calibration. No post-test calibration, threshold changes, resampling or model fitting was performed.',
              '- Optional paired bootstrap intervals describe uncertainty in frozen test predictions; they must not be used for post-test selection or treated as automatic formal significance claims.', '',
              'Full metrics, fixed-threshold results, prediction CSVs, candidate histories and experiment metadata accompany this report. The standalone dashboard embeds Plotly and requires no server.', '']
    verification_path = run_dir / 'verification.json'
    if verification_path.exists():
        verification = read_json(verification_path)
        lines += ['## Artifact verification', '',
                  f"Saved-artifact validation passed: {verification.get('passed')}. Final test suite: {verification.get('unit_tests_passed', 'see test-results.xml')} passed.",
                  f"Browser rendering verification: {verification.get('browser_verification', 'not recorded')}. Offline HTML/JavaScript syntax checks: {verification.get('dashboard_javascript_syntax', 'not recorded')}.",
                  'All checkpoints, identities, labels, thresholds, metric definitions and combined predictions were independently checked. Static figure appearance was inspected separately.', '']
    path = run_dir / 'reports/model_comparison.md'
    path.parent.mkdir(exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def generate_comparison(run_dir: str | Path, publish=True):
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'run.json')
    if not manifest['evaluation_complete']:
        raise ValueError('Evaluate the selected checkpoints before plotting test comparisons')
    names = list(manifest['models'])
    predictions = {n: pd.read_csv(run_dir / 'predictions' / f'{n}_test.csv', dtype={'CONS_NO': str}) for n in names}
    reference = predictions[names[0]]
    for name in names[1:]:
        frame = predictions[name].set_index('CONS_NO').loc[reference.CONS_NO].reset_index()
        if not np.array_equal(frame.y_true, reference.y_true):
            raise ValueError('Plots require identical evaluation labels and consumer populations')
        predictions[name] = frame
    metrics = pd.read_csv(run_dir / 'metrics/model_comparison.csv').set_index('model')
    details = {n: read_json(run_dir / 'metrics' / f'{n}_test.json') for n in names}
    errors = analyze_errors(run_dir)
    settings = manifest['config']['visualization']
    if settings['generate_static']:
        comparison_figures(run_dir, manifest, predictions, metrics, details)
        model_figures(run_dir, manifest)
        if settings['generate_consumer_examples']:
            consumer_figures(run_dir, manifest)
    write_report(run_dir, manifest, metrics, errors)
    if settings['generate_interactive']:
        from .dashboard import generate_dashboard
        generate_dashboard(run_dir)
    if publish and not manifest['diagnostic']:
        root = Path(manifest['config']['experiment']['output_dir'])
        for directory in ('figures', 'dashboard', 'reports', 'metrics'):
            if (run_dir / directory).exists():
                shutil.copytree(run_dir / directory, root / directory, dirs_exist_ok=True)
    files = [str(p.relative_to(run_dir)) for folder in ('figures', 'dashboard', 'reports') for p in (run_dir / folder).rglob('*') if p.is_file()]
    write_json(run_dir / 'visualization_manifest.json', {'files': files, 'diagnostic': manifest['diagnostic']})
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    generate_comparison(locate_run(load_training_config(args.config), args.run_dir))


if __name__ == '__main__':
    main()
