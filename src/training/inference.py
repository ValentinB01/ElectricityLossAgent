from __future__ import annotations

import time

import joblib
import torch

from src.evaluation.metrics import validate_scores
from .experiment_tracker import choose_device
from .train_neural import load_network, make_loader, predict_network


def predict_frozen_model(result: dict, bundle, config: dict):
    if result['model'] in ('lightgbm', 'xgboost'):
        model = joblib.load(result['model_path'])
        start = time.perf_counter()
        probability = model.predict_proba(bundle.tabular())[:, 1]
        duration = time.perf_counter() - start
    else:
        device = choose_device(config['experiment']['device'])
        model, _ = load_network(result['model_path'], device)
        loader = make_loader(bundle, config['models'][result['model']], config)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()
        start = time.perf_counter()
        probability = predict_network(model, loader, device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()
        duration = time.perf_counter() - start
    validate_scores(bundle.y, probability)
    return probability, duration
