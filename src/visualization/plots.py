"""Publication figures from frozen predictions; no model/threshold optimization."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[2] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

COLORS = {'lightgbm': '#2563EB', 'xgboost': '#E67E22', 'cnn1d': '#198754', 'hybrid': '#7C3AED'}
LABELS = {'lightgbm': 'LightGBM', 'xgboost': 'XGBoost', 'cnn1d': 'CNN 1D', 'hybrid': 'Hybrid'}


def configure_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 12,
                         'axes.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.grid': True, 'grid.alpha': .18,
                         'svg.fonttype': 'none', 'legend.frameon': False})


def save_figure(fig, path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if config['save_png']:
        fig.savefig(path.with_suffix('.png'), dpi=config['dpi'], bbox_inches='tight')
    if config['save_svg']:
        fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def bars(ax, names, values, ylabel, title, percent=False):
    values = np.asarray(values, dtype=float)
    columns = ax.bar([LABELS[n] for n in names], values, color=[COLORS[n] for n in names], width=.65)
    ax.set(ylabel=ylabel, title=title)
    ax.tick_params(axis='x', rotation=12)
    for column, value in zip(columns, values):
        if np.isfinite(value):
            ax.annotate(f'{value:.3f}' if abs(value) <= 1 and percent else f'{value:.2f}' if percent else f'{value:,.0f}',
                        (column.get_x() + column.get_width() / 2, value), ha='center', xytext=(0, 4),
                        textcoords='offset points', fontsize=9)
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() >= 0:
        ax.set_ylim(0, max(.05, finite.max() * 1.2))


def comparison_figures(run_dir: Path, manifest: dict, predictions: dict, metrics: pd.DataFrame, details: dict):
    configure_style()
    settings = manifest['config']['visualization']
    folder = run_dir / 'figures/comparison'
    names = list(predictions)
    scores = {n: predictions[n].probability.to_numpy() for n in names}
    y = predictions[names[0]].y_true.to_numpy()
    def metric(n, key):
        value = metrics.loc[n, key]
        return float(value) if pd.notna(value) else np.nan
    fig, ax = plt.subplots(figsize=(7.5, 6), layout='constrained')
    for name in names:
        fpr, tpr, _ = roc_curve(y, scores[name])
        ax.plot(fpr, tpr, color=COLORS[name], lw=2, label=f'{LABELS[name]} (AUC {metric(name,"roc_auc"):.3f})')
    ax.plot([0, 1], [0, 1], '--', color='#6B7280', label='Random classifier')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curves — held-out consumers', xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc='lower right')
    save_figure(fig, folder / 'roc_curves', settings)
    fig, ax = plt.subplots(figsize=(7.5, 6), layout='constrained')
    for name in names:
        precision, recall, _ = precision_recall_curve(y, scores[name])
        ax.step(recall, precision, where='post', color=COLORS[name], lw=2,
                label=f'{LABELS[name]} (AP {metric(name,"average_precision"):.3f})')
    ax.axhline(y.mean(), ls='--', color='#6B7280', label=f'Prevalence ({y.mean():.2%})')
    ax.set(xlabel='Recall', ylabel='Precision', title='Precision–recall curves — Average Precision (AP)', xlim=(0, 1), ylim=(0, 1.02))
    ax.legend()
    save_figure(fig, folder / 'precision_recall_curves', settings)
    matrices = {n: np.asarray(details[n]['optimized']['confusion_matrix']) if details[n]['optimized'] else None for n in names}
    for normalized in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
        maximum = 1 if normalized else max([m.max() for m in matrices.values() if m is not None] or [1])
        image = None
        for ax, name in zip(axes.flat, names):
            matrix = matrices[name]
            if matrix is None:
                ax.text(.5, .5, 'Validation precision constraint infeasible', ha='center', wrap=True)
                continue
            values = matrix / matrix.sum(axis=1, keepdims=True) if normalized else matrix
            image = ax.imshow(values, cmap='Blues', vmin=0, vmax=maximum)
            for i in range(2):
                for j in range(2):
                    label = [['TN', 'FP'], ['FN', 'TP']][i][j]
                    ax.text(j, i, f'{label}\n{values[i,j]:.1%}' if normalized else f'{label}\n{values[i,j]:,}',
                            ha='center', va='center', color='white' if values[i, j] > maximum * .5 else '#17253B', fontsize=12)
            ax.grid(False)
            ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Normal', 'Theft'],
                   yticklabels=[f'Normal (n={int((y==0).sum())})', f'Theft (n={int((y==1).sum())})'],
                   xlabel='Predicted class', ylabel='True class',
                   title=f'{LABELS[name]} · validation threshold {manifest["models"][name]["threshold_selection"]["threshold"]:.2f}')
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        fig.suptitle('Confusion matrices — fraction within each true class' if normalized else 'Confusion matrices — consumer counts', fontsize=14)
        if image is not None:
            fig.colorbar(image, ax=axes, label='Fraction of true-class consumers' if normalized else 'Consumers', shrink=.75)
        save_figure(fig, folder / ('confusion_matrices_normalized' if normalized else 'confusion_matrices'), settings)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout='constrained')
    for ax, key, label in zip(axes.flat, ['average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'f2'],
                              ['Average Precision', 'ROC-AUC', 'Precision', 'Recall', 'F1', 'F2']):
        bars(ax, names, [metric(n, key) for n in names], 'Score', f'{label}' + (' (threshold independent)' if key in ('average_precision', 'roc_auc') else ' (validation threshold)'), True)
    save_figure(fig, folder / 'metrics_comparison', settings)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    x = np.arange(len(names))
    for ax, keys, label in [(axes[0], ['false_positives', 'false_negatives'], 'Consumers'),
                            (axes[1], ['false_positive_rate', 'false_negative_rate'], 'Fraction of true-class consumers')]:
        for offset, key, hatch in [(-.18, keys[0], ''), (.18, keys[1], '//')]:
            ax.bar(x + offset, [metric(n, key) for n in names], width=.36, color=[COLORS[n] for n in names],
                   hatch=hatch, alpha=.85, label='False positives' if offset < 0 else 'False negatives')
        ax.set(xticks=x, xticklabels=[LABELS[n] for n in names], ylabel=label,
               title='Normal consumers incorrectly flagged / theft consumers missed')
        ax.tick_params(axis='x', rotation=12)
        ax.legend()
    save_figure(fig, folder / 'classification_errors', settings)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
    for ax, name in zip(axes.flat, names):
        for target, style, label in [(0, 'step', 'Normal'), (1, 'stepfilled', 'Theft-labeled')]:
            ax.hist(scores[name][y == target], bins=np.linspace(0, 1, 31), density=True, histtype=style,
                    alpha=.85 if target == 0 else .35, color=COLORS[name], label=label)
        ax.set(title=LABELS[name], xlabel='Raw model risk score (uncalibrated)', ylabel='Within-class density', xlim=(0, 1))
        ax.legend()
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    save_figure(fig, folder / 'probability_distributions', settings)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    bars(axes[0], names, [metric(n, 'training_time_seconds') for n in names], 'Seconds', 'Training and selection time', True)
    bars(axes[1], names, [metric(n, 'inference_seconds_per_1000') for n in names], 'Seconds per 1,000 consumers', 'Inference on recorded hardware', True)
    fig.suptitle('Measured computational cost · trees on CPU; neural device recorded per experiment')
    save_figure(fig, folder / 'computational_cost', settings)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for name in names:
        top = details[name]['top_k']
        for ax, key in zip(axes, ['precision_at_k', 'recall_at_k']):
            ax.plot([r['percentage'] for r in top], [r[key] for r in top], marker='o', color=COLORS[name], label=LABELS[name])
    for ax, title in zip(axes, ['Precision@K', 'Recall@K']):
        ax.set(title=title, xlabel='Highest-risk consumers selected (%)', ylabel='Fraction', ylim=(0, 1.02))
        ax.legend()
    save_figure(fig, folder / 'top_k_detection', settings)
    feasible = [n for n in names if manifest['models'][n]['threshold_selection']['threshold'] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    if feasible:
        binary = np.column_stack([scores[n] >= manifest['models'][n]['threshold_selection']['threshold'] for n in feasible])
        votes = binary.sum(axis=1)
        vote_counts = np.bincount(votes, minlength=len(feasible) + 1)
        disagreement = np.array([[np.mean(binary[:, i] != binary[:, j]) for j in range(len(feasible))] for i in range(len(feasible))])
        axes[0].bar(np.arange(len(vote_counts)), vote_counts, color='#64748B')
        axes[0].set(xticks=np.arange(len(vote_counts)), xlabel='Number of models predicting theft', ylabel='Consumers', title='Agreement at feasible frozen thresholds')
        image = axes[1].imshow(disagreement, cmap='Purples', vmin=0, vmax=1)
        axes[1].grid(False)
        axes[1].set(xticks=range(len(feasible)), yticks=range(len(feasible)), xticklabels=[LABELS[n] for n in feasible],
                    yticklabels=[LABELS[n] for n in feasible], title='Pairwise prediction disagreement')
        for i in range(len(feasible)):
            for j in range(len(feasible)):
                axes[1].text(j, i, f'{disagreement[i,j]:.1%}', ha='center', va='center')
        fig.colorbar(image, ax=axes[1], label='Fraction of consumers')
    else:
        for ax in axes:
            ax.text(.5, .5, 'No feasible validation threshold\nfor optimized classification', ha='center', va='center')
            ax.set_axis_off()
    save_figure(fig, folder / 'model_agreement', settings)
    if 'lightgbm' in names:
        others = [n for n in names if n != 'lightgbm']
        fig, axes = plt.subplots(1, 5, figsize=(17, 4.5), layout='constrained')
        for ax, key in zip(axes, ['average_precision', 'roc_auc', 'f1', 'recall', 'false_positive_rate']):
            values = [(metric(n, key) - metric('lightgbm', key)) * 100 for n in others]
            ax.bar([LABELS[n] for n in others], values, color=[COLORS[n] for n in others])
            ax.axhline(0, color='black', lw=.8)
            ax.set(title=f'{key.replace("_", " ").title()}\n' + ('Lower is better' if key == 'false_positive_rate' else 'Higher is better'), ylabel='Difference vs LightGBM (percentage points)')
            ax.tick_params(axis='x', rotation=30)
        save_figure(fig, folder / 'improvement_over_baseline', settings)


def model_figures(run_dir, manifest):
    settings = manifest['config']['visualization']
    for name, result in manifest['models'].items():
        source, folder = Path(result['experiment_dir']), run_dir / 'figures' / name
        grid = pd.read_csv(source / 'threshold_grid.csv')
        fig, ax = plt.subplots(figsize=(7.5, 5), layout='constrained')
        for key, style in [('precision', '-'), ('recall', '--'), ('f1', ':')]:
            ax.plot(grid.threshold, grid[key], style, label=key.title(), lw=2)
        ax.axvline(.5, color='#64748B', ls='--', label='Fixed 0.5')
        threshold = result['threshold_selection']['threshold']
        if threshold is not None:
            ax.axvline(threshold, color=COLORS[name], label=f'Selected {threshold:.2f}')
        ax.set(xlabel='Classification threshold', ylabel='Validation metric', ylim=(0, 1.02),
               title=f'{LABELS[name]} — validation-only threshold selection')
        ax.legend()
        save_figure(fig, folder / 'threshold_analysis', settings)
        history = pd.read_csv(source / 'history.csv')
        if name in ('cnn1d', 'hybrid'):
            fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
            axes[0].plot(history.epoch, history.train_loss, label='Train')
            axes[0].plot(history.epoch, history.val_loss, label='Validation')
            axes[0].set(title='Binary cross-entropy loss', ylabel='Loss')
            axes[0].legend()
            axes[1].plot(history.epoch, history.val_average_precision, color=COLORS[name])
            axes[1].set(title='Validation Average Precision', ylabel='AP')
            axes[2].plot(history.epoch, history.learning_rate, color=COLORS[name])
            axes[2].set(title='Learning rate', ylabel='Learning rate', yscale='log')
            for ax in axes:
                ax.axvline(result['best_epoch'], ls='--', color='#64748B', label='Best epoch')
                ax.set_xlabel('Epoch')
        else:
            fig, ax = plt.subplots(figsize=(7.5, 5), layout='constrained')
            for column in [c for c in history if c != 'iteration']:
                ax.plot(history.iteration, history[column], label=column.replace('_', ' '))
            ax.axvline(result['best_iteration'], ls='--', color='#64748B', label='Selected round')
            ax.set(xlabel='Boosting iteration', ylabel='Average Precision' if name == 'lightgbm' else 'XGBoost aucpr (not Average Precision)',
                   title=f'{LABELS[name]} — boosting learning curve')
            ax.legend()
            importance = pd.read_csv(source / 'feature_importance.csv').head(20).iloc[::-1]
            fi, fax = plt.subplots(figsize=(9, 7), layout='constrained')
            fax.barh(importance.feature, importance.gain, color=COLORS[name])
            fax.set(xlabel='LightGBM total split gain' if name == 'lightgbm' else 'XGBoost mean split gain',
                    title=f'{LABELS[name]} — top 20 features (gain; not causal)')
            save_figure(fi, folder / 'feature_importance', settings)
        save_figure(fig, folder / 'training_curves', settings)
