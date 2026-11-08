"""Offline Plotly dashboard from one evaluated, frozen experiment."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve

from src.evaluation.evaluator import locate_run
from src.training.config import load_training_config
from src.training.experiment_tracker import read_json
from .plots import COLORS, LABELS


def generate_dashboard(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / 'run.json')
    if not manifest['evaluation_complete']:
        raise ValueError('Evaluate frozen checkpoints before generating the dashboard')
    config, names = manifest['config'], list(manifest['models'])
    metrics = pd.read_csv(run_dir / 'metrics/model_comparison.csv').set_index('model')
    combined = pd.read_csv(run_dir / 'predictions/all_models_test.csv', dtype={'CONS_NO': str})
    y = combined.y_true.to_numpy()
    records = {n: read_json(run_dir / 'metrics' / f'{n}_test.json') for n in names}
    plots = []
    def add(title, fig, x=None, ylabel=None):
        fig.update_layout(template='plotly_white', title=title, font=dict(family='Arial', size=13),
                          margin=dict(l=65, r=35, t=75, b=65), height=460,
                          legend=dict(orientation='h', y=-.2), hovermode='closest')
        if x:
            fig.update_xaxes(title=x)
        if ylabel:
            fig.update_yaxes(title=ylabel)
        plots.append(fig.to_html(full_html=False, include_plotlyjs=True if not plots else False,
                                 config={'responsive': True, 'displaylogo': False, 'toImageButtonOptions': {'format': 'svg'}}))
    roc, pr = go.Figure(), go.Figure()
    for name in names:
        score = combined[f'{name}_score'].to_numpy()
        fpr, tpr, _ = roc_curve(y, score)
        precision, recall, _ = precision_recall_curve(y, score)
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=COLORS[name]),
                                name=f'{LABELS[name]} AUC={metrics.loc[name,"roc_auc"]:.3f}'))
        pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', line=dict(color=COLORS[name], shape='hv'),
                               name=f'{LABELS[name]} AP={metrics.loc[name,"average_precision"]:.3f}'))
    roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='gray', dash='dash'), name='Random'))
    pr.add_hline(y=float(y.mean()), line_dash='dash', annotation_text=f'Prevalence {y.mean():.2%}')
    add('ROC curves — held-out consumers', roc, 'False positive rate', 'True positive rate')
    add('Precision–recall curves — Average Precision, not interpolated PR-AUC', pr, 'Recall', 'Precision')
    fig = go.Figure()
    keys = ['average_precision', 'roc_auc', 'precision', 'recall', 'f1', 'f2']
    for name in names:
        fig.add_bar(x=['AP', 'ROC-AUC', 'Precision', 'Recall', 'F1', 'F2'], y=[metrics.loc[name, k] for k in keys],
                    name=LABELS[name], marker_color=COLORS[name])
    fig.update_layout(barmode='group')
    add('Test metrics — classification uses frozen validation thresholds', fig, ylabel='Score')
    fig = go.Figure()
    for name in names:
        fig.add_bar(x=['False positives', 'False negatives'], y=[metrics.loc[name, 'false_positives'], metrics.loc[name, 'false_negatives']],
                    name=LABELS[name], marker_color=COLORS[name])
    fig.update_layout(barmode='group')
    add('Normal consumers flagged / theft-labeled consumers missed', fig, ylabel='Consumers')
    fig = make_subplots(rows=2, cols=2, subplot_titles=[LABELS[n] for n in names])
    for i, name in enumerate(names):
        for label, opacity in [(0, .5), (1, .8)]:
            fig.add_trace(go.Histogram(x=combined.loc[combined.y_true == label, f'{name}_score'], histnorm='probability density',
                                       xbins=dict(start=0, end=1, size=1/30), opacity=opacity, marker_color=COLORS[name],
                                       name=f'{LABELS[name]} — {"normal" if label==0 else "theft"}'), row=i//2+1, col=i%2+1)
    fig.update_layout(barmode='overlay')
    add('Within-class score distributions — raw uncalibrated risk scores', fig, 'Model score', 'Density')
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Precision@K', 'Recall@K'])
    for name in names:
        for j, key in enumerate(['precision_at_k', 'recall_at_k']):
            fig.add_trace(go.Scatter(x=[r['percentage'] for r in records[name]['top_k']], y=[r[key] for r in records[name]['top_k']],
                                     mode='lines+markers', line=dict(color=COLORS[name]), name=LABELS[name], showlegend=j==0), row=1, col=j+1)
    add('Investigation capacity — complete held-out population', fig, 'Top-ranked consumers (%)', 'Fraction')
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Training/selection time', 'Inference per 1,000 consumers'])
    for j, key in enumerate(['training_time_seconds', 'inference_seconds_per_1000']):
        fig.add_trace(go.Bar(x=[LABELS[n] for n in names], y=[metrics.loc[n, key] for n in names],
                             marker_color=[COLORS[n] for n in names], showlegend=False), row=1, col=j+1)
    add('Measured runtime on the recorded hardware', fig, ylabel='Seconds')
    for name in names:
        result = manifest['models'][name]
        source = Path(result['experiment_dir'])
        history = pd.read_csv(source / 'history.csv')
        fig = go.Figure()
        step = 'epoch' if 'epoch' in history else 'iteration'
        columns = ['train_loss', 'val_loss', 'val_average_precision'] if step == 'epoch' else [c for c in history if c != step]
        if step == 'epoch':
            fig = make_subplots(specs=[[{'secondary_y': True}]])
        for column in columns:
            trace = go.Scatter(x=history[step], y=history[column], mode='lines', name=column.replace('_', ' '))
            if step == 'epoch':
                fig.add_trace(trace, secondary_y=column == 'val_average_precision')
            else:
                fig.add_trace(trace)
        best = result.get('best_epoch', result.get('best_iteration'))
        fig.add_vline(x=best, line_dash='dash', annotation_text=f'Selected {step}: {best}')
        add(f'{LABELS[name]} training history — validation selection', fig, step.title())
        grid = pd.read_csv(source / 'threshold_grid.csv')
        fig = go.Figure([go.Scatter(x=grid.threshold, y=grid[k], name=k.title()) for k in ['precision', 'recall', 'f1']])
        fig.add_vline(x=.5, line_dash='dash', annotation_text='Fixed 0.5')
        if result['threshold_selection']['threshold'] is not None:
            fig.add_vline(x=result['threshold_selection']['threshold'], line_color=COLORS[name], annotation_text='Selected on validation')
        add(f'{LABELS[name]} — validation threshold analysis', fig, 'Threshold', 'Validation score')
        if name in ('lightgbm', 'xgboost'):
            importance = pd.read_csv(source / 'feature_importance.csv').head(20).iloc[::-1]
            fig = go.Figure(go.Bar(x=importance.gain, y=importance.feature, orientation='h', marker_color=COLORS[name]))
            fig.update_layout(margin=dict(l=230))
            add(f'{LABELS[name]} — top 20 gain features (model behavior, not causation)', fig,
                'Total split gain' if name == 'lightgbm' else 'Mean split gain')
    selected = manifest['best_model_by_validation_ap']
    metric_table = '<table><thead><tr><th>Model</th><th>Validation AP</th><th>Test AP</th><th>Test ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1</th><th>Frozen threshold</th></tr></thead><tbody>'
    for name in names:
        row = metrics.loc[name]
        values = [manifest['models'][name]['validation_average_precision'], row.average_precision, row.roc_auc, row.precision, row.recall, row.f1, row.optimized_threshold]
        metric_table += '<tr><td>' + LABELS[name] + '</td>' + ''.join(f'<td>{x:.4f}</td>' if pd.notna(x) else '<td>Unavailable</td>' for x in values) + '</tr>'
    metric_table += '</tbody></table>'
    limit = config['visualization']['dashboard_table_limit']
    cases = combined.sort_values('CONS_NO').head(limit).copy()
    for name in names:
        threshold = manifest['models'][name]['threshold_selection']['threshold']
        cases[f'{name}_prediction'] = (cases[f'{name}_score'] >= threshold).astype(int) if threshold is not None else None
    prediction_columns = [f'{n}_prediction' for n in names]
    cases['agreement'] = cases[prediction_columns].nunique(axis=1) == 1
    if any(manifest['models'][n]['threshold_selection']['threshold'] is None for n in names):
        cases['agreement'] = None
    case_json = cases.to_json(orient='records').replace('<', '\\u003c')
    title = 'DIAGNOSTIC SUBSET — not full-dataset performance' if manifest['diagnostic'] else 'SGCC electricity theft model comparison'
    page = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>body{{margin:0;background:#f2f5fa;color:#17253b;font:15px Arial,sans-serif}}main{{max-width:1250px;margin:auto;padding:28px}}h1{{font-size:30px}}p{{line-height:1.6}}section{{background:white;border:1px solid #dce3ec;border-radius:10px;padding:22px;margin:18px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #e5eaf0}}th:first-child,td:first-child{{text-align:left}}th{{background:#eef2f7}}.muted{{color:#55657a}}input,select,button{{font:inherit;padding:9px;margin:4px;border:1px solid #bcc9da;border-radius:5px}}.plotly-graph-div{{width:100%!important}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{main{{padding:12px}}section{{padding:10px}}}}</style></head><body><main>
<h1>{title}</h1><p><b>{len(y):,} test consumers</b>: {int((y==0).sum()):,} normal, {int(y.sum()):,} theft-labeled ({y.mean():.2%}). Labels apply to consumers, not theft dates.</p>
<p>Selected before test evaluation: <b>{LABELS[selected]}</b>, by validation Average Precision. All models and thresholds were frozen before test inference. Class weighting can change calibration: these scores are not calibrated probabilities or proof of theft.</p>
<p class="muted">Run: <code>{html.escape(manifest['run_id'])}</code>. Device: {html.escape(manifest['hardware']['device'])}; logical CPUs: {manifest['hardware']['logical_cpus']}. Tree training time includes weighting comparisons and optional tuning; neural time includes all training/validation epochs.</p>
<section><h2>Measured results</h2>{metric_table}</section>'''
    page += ''.join('<section>' + plot + '</section>' for plot in plots)
    headings = ['Consumer ID', 'True label'] + [LABELS[n] + ' score / prediction' for n in names] + ['All agree']
    page += '<section><h2>Consumer predictions</h2>' + f'<p>Searchable first {len(cases):,} consumer IDs in lexical order, out of {len(combined):,}. Complete predictions remain in the CSV files. Binary predictions use frozen validation thresholds.</p>'
    page += '<input id="search" placeholder="Search consumer ID"><select id="label"><option value="">All labels</option><option value="0">Normal</option><option value="1">Theft</option></select><select id="agreement"><option value="">All consumers</option><option value="no">Models disagree</option></select><table><thead><tr>'
    page += ''.join('<th>' + h + '</th>' for h in headings) + '</tr></thead><tbody id="cases"></tbody></table><button id="prev">Previous</button><button id="next">Next</button><span id="count"></span></section>'
    page += f'''<script>const rows={case_json}, models={json.dumps(names)};let page=0;const el=id=>document.getElementById(id);function draw(){{const data=rows.filter(r=>(!el('search').value||r.CONS_NO.toLowerCase().includes(el('search').value.toLowerCase()))&&(!el('label').value||r.y_true===Number(el('label').value))&&(!el('agreement').value||r.agreement===false));page=Math.max(0,Math.min(page,Math.ceil(data.length/25)-1));el('cases').replaceChildren();for(const row of data.slice(page*25,page*25+25)){{const tr=document.createElement('tr');const values=[row.CONS_NO,row.y_true,...models.map(n=>row[n+'_score'].toFixed(4)+' / '+(row[n+'_prediction']===null?'Unavailable':row[n+'_prediction'])),row.agreement===null?'Unavailable':row.agreement?'Yes':'No'];for(const value of values){{const td=document.createElement('td');td.textContent=value;tr.appendChild(td)}}el('cases').appendChild(tr)}}el('count').textContent=`Page ${{page+1}}; ${{data.length}} matching consumers`}}for(const id of ['search','label','agreement'])el(id).addEventListener('input',()=>{{page=0;draw()}});el('prev').onclick=()=>{{page--;draw()}};el('next').onclick=()=>{{page++;draw()}};draw();</script>'''
    page += '<p class="muted">Single consumer split and seed; SGCC label noise, missingness differences and unknown theft timing limit interpretation. Comparison plots use the same held-out consumers. No test-based tuning or recalibration was performed. Development disclosure: the first implementation smoke run evaluated 128 final-test consumers before full training; its scores did not change the predefined settings. Future diagnostic inference uses validation consumers. The final test population was therefore not entirely blind during development.</p></main></body></html>'
    output = run_dir / 'dashboard/model_comparison.html'
    output.parent.mkdir(exist_ok=True)
    output.write_text(page, encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--run-dir')
    args = parser.parse_args()
    run = locate_run(load_training_config(args.config), args.run_dir)
    path = generate_dashboard(run)
    from shutil import copy2
    manifest = read_json(run / 'run.json')
    if not manifest['diagnostic']:
        target = Path(manifest['config']['experiment']['output_dir']) / 'dashboard'
        target.mkdir(exist_ok=True)
        copy2(path, target / path.name)
    print(path)


if __name__ == '__main__':
    main()
