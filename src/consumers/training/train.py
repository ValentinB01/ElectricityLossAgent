"""Train selected models; freeze validation selections before any test inference."""
from __future__ import annotations

import argparse
from copy import deepcopy
import logging
from pathlib import Path

import pandas as pd

from src.consumers.evaluation.metrics import classification_metrics
from src.consumers.evaluation.thresholds import select_threshold
from src.consumers.validation import ids_digest
from .config import MODELS, diagnostic_config, load_training_config
from .datasets import PreparedDataStore
from .experiment_tracker import choose_device, hardware_info, new_run, seed_everything, write_json
from .inference import predict_frozen_model
from .train_neural import train_neural
from .train_trees import train_tree

LOGGER = logging.getLogger(__name__)


def run_training(config: dict, models: list[str], evaluate=False, visualize=False, smoke=False) -> Path:
    if not models or any(name not in MODELS for name in models):
        raise ValueError('Select at least one supported enabled model')
    config = diagnostic_config(config) if smoke else deepcopy(config)
    store = PreparedDataStore(config['data']['processed_dir'])
    LOGGER.info('Checking existing data identities, schemas, masks and hybrid branch alignment')
    audit = store.alignment_audit()
    seed_everything(config['experiment']['seed'], config['experiment']['cpu_threads'])
    run_dir, manifest = new_run(config, models, audit)
    manifest['hardware'] = hardware_info(choose_device(config['experiment']['device']))
    write_json(run_dir / 'alignment_audit.json', audit)
    for name in models:
        train, val = store.load(name, 'train'), store.load(name, 'val')
        if smoke:
            train, val = train.sample(512, config['experiment']['seed']), val.sample(128, config['experiment']['seed'])
        base = Path(config['experiment']['model_dir']) / ('smoke' if smoke else '') / name / manifest['run_id']
        LOGGER.info('Training %s on %d consumers; validation %d', name, len(train.y), len(val.y))
        if name in ('lightgbm', 'xgboost'):
            result = train_tree(name, train, val, config, base)
        else:
            result = train_neural(name, train, val, config, base)
        probability, _ = predict_frozen_model(result, val, config)
        selection, grid = select_threshold(val.y, probability, config['evaluation'])
        result['threshold_selection'] = selection
        result['training_ids_digest'] = ids_digest(train.ids)
        result['validation_ids_digest'] = ids_digest(val.ids)
        result['seed'] = config['experiment']['seed']
        metrics = {'split': 'val', 'fixed_05': classification_metrics(val.y, probability, .5),
                   'optimized': classification_metrics(val.y, probability, selection['threshold']) if selection['threshold'] is not None else None}
        write_json(base / 'validation_metrics.json', metrics)
        write_json(base / 'threshold.json', selection)
        write_json(base / 'configuration.json', {'experiment': config['experiment'], 'model': config['models'][name],
                                                 'evaluation': config['evaluation'], 'dataset_metadata': store.manifest,
                                                 'hardware': manifest['hardware']})
        grid.to_csv(base / 'threshold_grid.csv', index=False)
        from src.consumers.evaluation.evaluator import prediction_table
        predictions = prediction_table(val, probability, selection)
        (run_dir / 'predictions').mkdir(exist_ok=True)
        predictions.to_csv(run_dir / 'predictions' / f'{name}_val.csv', index=False)
        manifest['models'][name] = result
        write_json(run_dir / 'run.json', manifest)
        LOGGER.info('%s selection frozen: val AP %.5f; threshold %s', name, result['validation_average_precision'], selection['threshold'])
    manifest['all_selections_frozen'] = True
    manifest['best_model_by_validation_ap'] = max(manifest['models'], key=lambda x: manifest['models'][x]['validation_average_precision'])
    write_json(run_dir / 'run.json', manifest)
    if not smoke:
        write_json(Path(config['experiment']['output_dir']) / 'latest_run.json', {'run_dir': str(run_dir), 'run_id': manifest['run_id']})
    if evaluate or visualize:
        from src.consumers.evaluation.evaluator import evaluate_run
        evaluate_run(run_dir, publish=not smoke)
    if visualize:
        from src.consumers.visualization.comparison import generate_comparison
        generate_comparison(run_dir, publish=not smoke)
    LOGGER.info('%s run complete: %s', 'DIAGNOSTIC ONLY' if smoke else 'Full-population', run_dir)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--all', action='store_true')
    selection.add_argument('--model', choices=MODELS)
    parser.add_argument('--config')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='Diagnostic subset, 8 tree rounds / 2 neural epochs; separate outputs')
    parser.add_argument('--seeds', help='Optional comma-separated reproducibility seeds')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    config = load_training_config(args.config)
    models = [m for m in MODELS if config['models'][m]['enabled']] if args.all else [args.model]
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else [config['experiment']['seed']]
    for seed in seeds:
        config['experiment']['seed'] = seed
        run_training(config, models, args.evaluate, args.visualize, args.smoke)


if __name__ == '__main__':
    main()
