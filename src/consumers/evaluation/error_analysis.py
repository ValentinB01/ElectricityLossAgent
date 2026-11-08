"""Descriptive error groups and deterministic representative consumers."""
from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd

from src.consumers.training.experiment_tracker import read_json, write_json

QUALITY_COLUMNS = ['missing_ratio', 'zero_ratio', 'coefficient_of_variation', 'max_consumption',
                   'observed_days', 'mean_consumption', 'std_consumption', 'longest_missing_streak',
                   'mean_daily_absolute_change', 'sudden_drop_ratio']


def analyze_errors(run_dir: Path) -> dict:
    manifest = read_json(run_dir / 'run.json')
    combined = pd.read_csv(run_dir / 'predictions/all_models_test.csv', dtype={'CONS_NO': str})
    source = Path(manifest['config']['data']['processed_dir'])
    features = pd.read_parquet(source / 'shared/engineered_features.parquet')
    available = [c for c in QUALITY_COLUMNS if c in features]
    detail = combined.merge(features[['CONS_NO', *available]], on='CONS_NO', how='left', validate='one_to_one')
    names = list(manifest['models'])
    binary, summaries, associations, representatives = {}, [], [], []
    count = manifest['config']['visualization']['examples_per_category']
    for name in names:
        threshold = manifest['models'][name]['threshold_selection']['threshold']
        if threshold is None:
            continue
        prediction = detail[f'{name}_score'].to_numpy() >= threshold
        binary[name] = prediction
        detail[f'{name}_prediction'] = prediction.astype(int)
        actual = detail.y_true.to_numpy()
        masks = {'true_positive': (actual == 1) & prediction, 'true_negative': (actual == 0) & ~prediction,
                 'false_positive': (actual == 0) & prediction, 'false_negative': (actual == 1) & ~prediction}
        for category, mask in masks.items():
            group = detail.loc[mask]
            summary = {'model': name, 'category': category, 'consumers': int(mask.sum())}
            for column in available:
                summary[f'{column}_median'] = float(group[column].median()) if group[column].notna().any() else None
            summaries.append(summary)
            # Select typical scores within each error group; stable ID breaks ties.
            if not group.empty:
                median_score = group[f'{name}_score'].median()
                chosen = group.assign(distance=(group[f'{name}_score'] - median_score).abs()).sort_values(['distance', 'CONS_NO']).head(count)
                representatives += [{'model': name, 'category': category, 'CONS_NO': c, 'selection_rule': 'closest to within-group median score, then ID'}
                                    for c in chosen.CONS_NO]
        for category, correct_category, label in [('false_positive', 'true_negative', 0), ('false_negative', 'true_positive', 1)]:
            error = detail.loc[masks[category]]
            correct = detail.loc[masks[correct_category]]
            for column in available:
                a, b = error[column].dropna(), correct[column].dropna()
                pooled = detail.loc[detail.y_true == label, column].dropna()
                iqr = pooled.quantile(.75) - pooled.quantile(.25) if len(pooled) else np.nan
                associations.append({'model': name, 'error_category': category, 'reference_category': correct_category,
                                     'feature': column, 'error_count': len(a), 'reference_count': len(b),
                                     'error_median': float(a.median()) if len(a) else None,
                                     'reference_median': float(b.median()) if len(b) else None,
                                     'median_difference_over_class_iqr': float((a.median() - b.median()) / iqr)
                                     if len(a) and len(b) and pd.notna(iqr) and iqr > 0 else None})
    scores = detail[[f'{n}_score' for n in names]].to_numpy()
    for a, b in combinations(names, 2):
        detail[f'{a}_minus_{b}_score'] = detail[f'{a}_score'] - detail[f'{b}_score']
    detail['score_range'] = scores.max(axis=1) - scores.min(axis=1)
    if binary:
        votes = np.column_stack(list(binary.values())).sum(axis=1)
        detail['positive_votes'] = votes
        detail['all_models_agree'] = (votes == 0) | (votes == len(binary))
        if 'hybrid' in binary and len(binary) > 1:
            hybrid_correct = binary['hybrid'] == detail.y_true.to_numpy()
            other_wrong = np.column_stack([p != detail.y_true.to_numpy() for n, p in binary.items() if n != 'hybrid']).all(axis=1)
            detail['hybrid_only_correct'] = hybrid_correct & other_wrong
        pairwise = pd.DataFrame({a: {b: float(np.mean(binary[a] != binary[b])) for b in binary} for a in binary})
        pairwise.to_csv(run_dir / 'metrics/pairwise_disagreement.csv', index_label='model')
    largest = detail.sort_values(['score_range', 'CONS_NO'], ascending=[False, True]).head(count)
    representatives += [{'model': 'all', 'category': 'high_disagreement', 'CONS_NO': c, 'selection_rule': 'largest max-minus-min score, then ID'}
                        for c in largest.CONS_NO]
    output = run_dir / 'metrics'
    detail.to_csv(output / 'consumer_error_analysis.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'error_group_summaries.csv', index=False)
    pd.DataFrame(associations).to_csv(output / 'error_feature_associations.csv', index=False)
    pd.DataFrame(representatives).to_csv(output / 'representative_consumers.csv', index=False)
    detail.sort_values(['score_range', 'CONS_NO'], ascending=[False, True]).to_csv(output / 'model_disagreements.csv', index=False)
    summary = {'summary_rows': summaries, 'representative_cases': len(representatives),
               'all_models_agree': int(detail.all_models_agree.sum()) if 'all_models_agree' in detail else None,
               'hybrid_only_correct': int(detail.hybrid_only_correct.sum()) if 'hybrid_only_correct' in detail else None,
               'exactly_three_models_agree': int(np.isin(detail.positive_votes, [1, 3]).sum()) if len(binary) == 4 else None,
               'two_models_predict_theft': int((detail.positive_votes == 2).sum()) if len(binary) == 4 else None,
               'strong_disagreement_count_score_range_ge_0_5': int((detail.score_range >= .5).sum()),
               'interpretation': 'Descriptive within-true-class comparisons, no causal attribution or daily theft dates.',
               'quality_columns_available': available}
    write_json(output / 'error_analysis.json', summary)
    return summary
