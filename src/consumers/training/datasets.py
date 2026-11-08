"""ID-based alignment and memory-mapped input loading; never fits preprocessing."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from src.consumers.validation import FORBIDDEN_FEATURES, ids_digest


@dataclass
class SplitData:
    model: str
    split: str
    ids: np.ndarray
    y: np.ndarray
    order: np.ndarray
    features: np.ndarray | None = None
    sequence: np.ndarray | None = None
    feature_names: list[str] | None = None

    def tabular(self) -> pd.DataFrame:
        return pd.DataFrame(self.features[self.order], columns=self.feature_names)

    def sample(self, count: int, seed: int) -> 'SplitData':
        if len(self.y) <= count:
            return self
        chosen, _ = train_test_split(np.arange(len(self.y)), train_size=count, stratify=self.y, random_state=seed)
        return replace(self, ids=self.ids[chosen], y=self.y[chosen], order=self.order[chosen])


class PreparedDataStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        try:
            self.manifest = json.loads((self.root / 'shared/run_manifest.json').read_text())
            self.assignments = pd.read_csv(self.root / 'shared/splits.csv', dtype={'CONS_NO': str})
        except FileNotFoundError as exc:
            raise FileNotFoundError('Processed data missing. Run python -m src.consumers.prepare_data --config config/consumers/sgcc.yaml --all') from exc
        if not self.manifest.get('validation_passed'):
            raise ValueError('The preprocessing generation did not complete validation')
        if not self.assignments.CONS_NO.is_unique or set(self.assignments.split) != {'train', 'val', 'test'}:
            raise ValueError('Shared splits are not disjoint, exhaustive named assignments')
        if not self.assignments.FLAG.isin([0, 1]).all():
            raise ValueError('Shared labels must be binary')

    def load(self, model: str, split: str, verify_hybrid: bool = True) -> SplitData:
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown split')
        folder = self.root / model
        try:
            metadata = json.loads((folder / 'metadata.json').read_text())
            if metadata['run_fingerprint'] != self.manifest['run_fingerprint']:
                raise ValueError(f'{model} belongs to a different preprocessing generation')
            wanted = self.assignments.loc[self.assignments.split == split]
            features, sequence, names = None, None, None
            if model in ('lightgbm', 'xgboost'):
                frame = pd.read_parquet(folder / f'{split}.parquet')
                if not frame.FLAG.isin([0, 1]).all():
                    raise ValueError('Tree labels must be binary before numeric conversion')
                ids, y = frame.CONS_NO.astype(str).to_numpy(), frame.FLAG.to_numpy(dtype=np.int8)
                names = json.loads((folder / 'feature_names.json').read_text())
                if FORBIDDEN_FEATURES.intersection(names) or len(names) != len(set(names)):
                    raise ValueError('Identifiers, targets, duplicate names or row indices among features')
                features = frame[names].to_numpy(dtype=np.float64)
                if np.isinf(features).any():
                    raise ValueError('Infinite tree features')
            else:
                ids = pd.read_csv(folder / f'{split}_ids.csv', dtype=str).CONS_NO.to_numpy()
                y = np.load(folder / f'y_{split}.npy', allow_pickle=False)
                filename = f'X_{split}.npy' if model == 'cnn1d' else f'X_sequence_{split}.npy'
                sequence = np.load(folder / filename, mmap_mode='r', allow_pickle=False)
                if sequence.ndim != 3 or sequence.shape[0] != len(ids) or sequence.shape[1] != metadata['calendar_days'] or sequence.shape[2] != 2 or sequence.dtype != np.float32:
                    raise ValueError('Expected float32 neural sequence (consumers, days, 2)')
                if model == 'hybrid':
                    features = np.load(folder / f'X_features_{split}.npy', mmap_mode='r', allow_pickle=False)
                    names = json.loads((folder / 'feature_names.json').read_text())
                    if features.shape != (len(ids), len(names)) or features.dtype != np.float32:
                        raise ValueError('Hybrid feature shape or dtype mismatch')
                    if FORBIDDEN_FEATURES.intersection(names):
                        raise ValueError('Identifier or target among hybrid features')
                for start in range(0, len(ids), 512):
                    batch = sequence[start:start + 512]
                    if not np.isfinite(batch).all() or not np.isin(batch[:, :, 1], [0, 1]).all():
                        raise ValueError('Nonfinite neural sequence or invalid observation mask')
                    if features is not None and not np.isfinite(features[start:start + 512]).all():
                        raise ValueError('Nonfinite hybrid statistics')
        except FileNotFoundError as exc:
            raise FileNotFoundError(f'Missing prepared {model}/{split} file. Run python -m src.consumers.prepare_data --config config/consumers/sgcc.yaml --all') from exc
        if len(set(ids)) != len(ids) or set(ids) != set(wanted.CONS_NO) or y.shape != (len(ids),):
            raise ValueError(f'{model}/{split} consumer IDs differ from the shared assignment')
        order = pd.Series(np.arange(len(ids)), index=ids).loc[wanted.CONS_NO].to_numpy()
        labels = wanted.FLAG.to_numpy(dtype=np.int8)
        if not np.array_equal(y[order], labels):
            raise ValueError(f'{model}/{split} labels differ after consumer-ID alignment')
        result = SplitData(model, split, wanted.CONS_NO.to_numpy(), labels, order, features, sequence, names)
        if model == 'hybrid' and verify_hybrid:
            self.verify_hybrid(result)
        return result

    def verify_hybrid(self, bundle: SplitData) -> None:
        """Recompute both branches from shared consumer identities, not just their lengths."""
        shared = self.root / 'shared'
        consumers = pd.read_csv(shared / 'consumers.csv', dtype={'CONS_NO': str})
        lookup = pd.Series(np.arange(len(consumers)), index=consumers.CONS_NO)
        original = np.load(shared / 'consumption.npy', mmap_mode='r', allow_pickle=False)
        raw_features = pd.read_parquet(shared / 'engineered_features.parquet').set_index('CONS_NO')
        sequence_pre = joblib.load(self.root / 'hybrid/sequence_preprocessor.joblib')
        feature_pre = joblib.load(self.root / 'hybrid/feature_preprocessor.joblib')
        for start in range(0, len(bundle.ids), 512):
            ids, ix = bundle.ids[start:start + 512], bundle.order[start:start + 512]
            expected_features = feature_pre.transform(raw_features.loc[ids, feature_pre.input_names_])
            expected_sequence = sequence_pre.transform(original[lookup.loc[ids].to_numpy()])
            if not np.array_equal(expected_features, bundle.features[ix]) or not np.array_equal(expected_sequence, bundle.sequence[ix]):
                raise ValueError('Hybrid branches cannot be aligned to the same source consumers; refusing training')

    def alignment_audit(self) -> dict:
        result = {}
        for split in ('train', 'val', 'test'):
            reference = None
            for model in ('lightgbm', 'xgboost', 'cnn1d', 'hybrid'):
                bundle = self.load(model, split)
                if reference is not None and (not np.array_equal(reference.ids, bundle.ids) or not np.array_equal(reference.y, bundle.y)):
                    raise ValueError('Cross-model consumer alignment failed')
                reference = bundle
            result[split] = {'consumers': len(reference.y), 'positives': int(reference.y.sum()),
                             'ids_digest': ids_digest(reference.ids)}
        return {'passed': True, 'splits': result, 'preprocessing_fingerprint': self.manifest['run_fingerprint'],
                'note': 'Test data inspected only for static integrity; no fitted classifier or metric used.'}


class ConsumerDataset(Dataset):
    def __init__(self, data: SplitData):
        self.data = data

    def __len__(self):
        return len(self.data.y)

    def __getitem__(self, i):
        row = self.data.order[i]
        # Copy read-only memory maps before converting; Conv1d receives channels first.
        sequence = torch.from_numpy(np.array(self.data.sequence[row].T, dtype=np.float32, copy=True))
        features = torch.from_numpy(np.array(self.data.features[row], dtype=np.float32, copy=True)) if self.data.features is not None else torch.empty(0)
        return sequence, features, torch.tensor(float(self.data.y[i]), dtype=torch.float32)
