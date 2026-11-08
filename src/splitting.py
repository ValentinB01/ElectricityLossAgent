"""One reproducible stratified assignment, shared by every representation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import ConsumerData


def make_splits(data: ConsumerData, config: dict) -> dict[str, np.ndarray]:
    if len(np.unique(data.ids)) != len(data.ids):
        raise ValueError("Splitting requires unique consumer IDs")
    # Canonical ID ordering also makes assignments invariant to raw CSV row order.
    indices = np.argsort(data.ids, kind="stable")
    s = config["split"]
    try:
        train, heldout = train_test_split(indices, test_size=s["val_ratio"] + s["test_ratio"],
                                         random_state=s["random_state"], stratify=data.labels[indices])
        val, test = train_test_split(heldout, test_size=s["test_ratio"] / (s["val_ratio"] + s["test_ratio"]),
                                    random_state=s["random_state"], stratify=data.labels[heldout])
    except ValueError as exc:
        raise ValueError(f"Cannot create stratified train/val/test splits (need enough consumers per class): {exc}") from exc
    splits = {"train": train, "val": val, "test": test}
    joined = np.concatenate(list(splits.values()))
    if len(joined) != len(data.ids) or len(np.unique(joined)) != len(data.ids):
        raise ValueError("Split assignments must cover every consumer exactly once")
    return splits


def split_table(data: ConsumerData, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.concat([pd.DataFrame({"CONS_NO": data.ids[ix], "FLAG": data.labels[ix], "split": name})
                      for name, ix in splits.items()], ignore_index=True)
