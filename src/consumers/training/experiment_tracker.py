from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import uuid

import numpy as np
import pandas as pd
import torch


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, (Path, np.str_)):
            return str(x)
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def seed_everything(seed: int, threads: int = 8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(request: str = 'auto') -> torch.device:
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if request == 'auto':
        request = 'cuda' if cuda else 'mps' if mps else 'cpu'
    if request == 'cuda' and not cuda or request == 'mps' and not mps:
        raise ValueError(f'Requested device {request} is unavailable; use auto or cpu')
    return torch.device(request)


def hardware_info(device) -> dict:
    return {'device': str(device), 'platform': platform.platform(), 'processor': platform.processor(),
            'processor_identifier': os.environ.get('PROCESSOR_IDENTIFIER'), 'logical_cpus': os.cpu_count(),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'versions': {name: version(name) for name in ('torch', 'lightgbm', 'xgboost', 'numpy', 'pandas',
                                                          'scikit-learn', 'matplotlib', 'plotly', 'optuna')},
            'python': platform.python_version()}


def new_run(config: dict, models: list[str], audit: dict) -> tuple[Path, dict]:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    run_id = f'{stamp}_{digest}_{uuid.uuid4().hex[:6]}'
    diagnostic = config['experiment'].get('diagnostic', False)
    root = Path(config['experiment']['output_dir'])
    run_dir = root / ('smoke' if diagnostic else 'runs') / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    source_dir = Path(__file__).resolve().parents[1]
    source_hashes = {str(p.relative_to(source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for subdir in ('models', 'training', 'evaluation', 'visualization') for p in sorted((source_dir / subdir).glob('*.py'))}
    manifest = {'run_id': run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                'diagnostic': diagnostic, 'requested_models': models, 'models': {}, 'alignment_audit': audit,
                'source_sha256': source_hashes, 'all_selections_frozen': False, 'evaluation_complete': False}
    write_json(run_dir / 'run.json', manifest)
    return run_dir, manifest


def append_registry(output_dir: Path, rows: list[dict]) -> None:
    path = output_dir / 'metrics/experiments.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
