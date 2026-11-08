from __future__ import annotations

from contextlib import nullcontext
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import classification_metrics
from src.models.cnn1d_model import CNN1DClassifier
from src.models.hybrid_model import HybridClassifier
from .datasets import ConsumerDataset, SplitData
from .experiment_tracker import choose_device, seed_everything, write_json
from .train_trees import positive_class_weight

LOGGER = logging.getLogger(__name__)


def make_network(name, feature_count, settings):
    return CNN1DClassifier(settings) if name == 'cnn1d' else HybridClassifier(feature_count, settings)


def make_loader(data, settings, config, training=False):
    generator = torch.Generator().manual_seed(config['experiment']['seed'])
    return DataLoader(ConsumerDataset(data), batch_size=settings['batch_size'], shuffle=training,
                      num_workers=config['experiment']['num_workers'], generator=generator,
                      pin_memory=choose_device(config['experiment']['device']).type == 'cuda')


def predict_network(network, loader, device):
    network.eval()
    probabilities = []
    with torch.inference_mode():
        for sequence, features, _ in loader:
            logits = network(sequence.to(device), features.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def load_network(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    network = make_network(checkpoint['model'], checkpoint['feature_count'], checkpoint['settings'])
    network.load_state_dict(checkpoint['state_dict'])
    return network.to(device).eval(), checkpoint


def train_neural(name: str, train: SplitData, val: SplitData, config: dict, folder: Path) -> dict:
    if train.split != 'train' or val.split != 'val':
        raise ValueError('Neural fitting/early stopping requires train/val, never test')
    folder.mkdir(parents=True, exist_ok=False)
    settings = config['models'][name]
    seed_everything(config['experiment']['seed'], config['experiment']['cpu_threads'])
    device = choose_device(config['experiment']['device'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    feature_count = train.features.shape[1] if train.features is not None else 0
    network = make_network(name, feature_count, settings).to(device)
    weight = positive_class_weight(train.y) if settings['weighted_loss'] else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    optimizer = torch.optim.AdamW(network.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=settings['scheduler_patience'], factor=.5)
    use_amp = device.type == 'cuda' and settings['mixed_precision']
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    train_loader, val_loader = make_loader(train, settings, config, True), make_loader(val, settings, config)
    best_ap, stale, best_epoch = -np.inf, 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, settings['max_epochs'] + 1):
        epoch_start = time.perf_counter()
        network.train()
        total_loss = 0.
        for sequence, features, target in train_loader:
            sequence, features, target = sequence.to(device), features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
                loss = criterion(network(sequence, features), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; inspect the prepared inputs and learning rate')
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(network.parameters(), settings['gradient_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), settings['gradient_clip'])
                optimizer.step()
            total_loss += float(loss.detach()) * len(target)
        network.eval()
        val_loss, scores = 0., []
        with torch.inference_mode():
            for sequence, features, target in val_loader:
                sequence, features, target = sequence.to(device), features.to(device), target.to(device)
                logits = network(sequence, features)
                val_loss += float(criterion(logits, target)) * len(target)
                scores.append(torch.sigmoid(logits).cpu().numpy())
        probability = np.concatenate(scores)
        metrics = classification_metrics(val.y, probability)
        ap = metrics['average_precision']
        if ap is None or not np.isfinite(ap):
            raise ValueError('Validation Average Precision is undefined; cannot select a checkpoint')
        learning_rate = optimizer.param_groups[0]['lr']
        scheduler.step(ap)
        record = {'epoch': epoch, 'train_loss': total_loss / len(train.y), 'val_loss': val_loss / len(val.y),
                  'val_average_precision': ap, 'val_roc_auc': metrics['roc_auc'], 'val_f1_05': metrics['f1'],
                  'learning_rate': learning_rate, 'epoch_duration_seconds': time.perf_counter() - epoch_start}
        history.append(record)
        if ap > best_ap + settings['min_delta']:
            best_ap, stale, best_epoch = ap, 0, epoch
            torch.save({'state_dict': network.state_dict(), 'model': name, 'feature_count': feature_count,
                        'settings': settings, 'epoch': epoch, 'validation_average_precision': ap,
                        'seed': config['experiment']['seed'], 'positive_weight': weight}, folder / 'best.pt')
        else:
            stale += 1
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        LOGGER.info('%s epoch %d/%d: loss %.4f, val loss %.4f, val AP %.5f, ROC %.5f, %.1fs (best %d)',
                    name, epoch, settings['max_epochs'], record['train_loss'], record['val_loss'], ap,
                    metrics['roc_auc'], record['epoch_duration_seconds'], best_epoch)
        if stale >= settings['early_stopping_patience']:
            LOGGER.info('%s early stopping after %d epochs without sufficient AP improvement', name, stale)
            break
    write_json(folder / 'feature_names.json', train.feature_names or [])
    result = {'model': name, 'model_path': str((folder / 'best.pt').resolve()), 'experiment_dir': str(folder.resolve()),
              'validation_average_precision': float(best_ap), 'best_epoch': best_epoch,
              'epochs_completed': len(history), 'weighted': settings['weighted_loss'], 'positive_weight': weight,
              'parameter_count': sum(p.numel() for p in network.parameters()), 'feature_count': feature_count,
              'training_time_seconds': time.perf_counter() - start,
              'training_time_scope': 'all epochs including validation and checkpoint serialization',
              'model_size_bytes': (folder / 'best.pt').stat().st_size, 'device': str(device),
              'peak_gpu_memory_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else None,
              'mixed_precision': use_amp, 'early_stopping_split': 'val', 'fit_split': 'train',
              'early_stopping_metric': 'average_precision', 'settings': settings}
    write_json(folder / 'training_result.json', result)
    return result
