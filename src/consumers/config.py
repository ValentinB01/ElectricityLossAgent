"""Configuration validation and paths relative to the configuration file."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


DEFAULTS: dict[str, Any] = {
    "data": {"input_path": "data.csv", "output_dir": "data/processed",
             "chunksize": 1024, "invalid_policy": "nan", "negative_policy": "nan",
             "suspicious_value_threshold": 100000.0},
    "split": {"train_ratio": .70, "val_ratio": .15, "test_ratio": .15,
              "stratify": True, "random_state": 42},
    "dates": {"start": "2014-01-01", "end": "2016-10-31", "frequency": "D"},
    "outliers": {"enabled": True, "upper_quantile": .999,
                 "method": "global_quantile_clip"},
    "cnn": {"dtype": "float32", "use_observation_mask": True, "log_transform": True,
            "scaling": "standard", "imputation": "placeholder",
            "max_interpolation_gap": 3, "missing_placeholder": 0.0},
    "hybrid": {"imputation": "median", "scaling": "robust", "add_missing_indicators": True},
    "features": {"sudden_drop_threshold": .5, "sudden_increase_threshold": .5,
                 "insufficient_history_threshold": 180, "include_quality_features": True,
                 "create_quality_ablation": False},
    "filtering": {"filter_insufficient_history": False},
    "runtime": {"random_seed": 42, "batch_size": 512},
}


def validate_config(config: dict) -> None:
    s = config["split"]
    ratios = [s[f"{name}_ratio"] for name in ("train", "val", "test")]
    if any(not isinstance(x, (int, float)) or not 0 < x < 1 for x in ratios):
        raise ValueError("Split ratios must be numbers strictly between 0 and 1")
    if not np.isclose(sum(ratios), 1, rtol=0, atol=1e-10):
        raise ValueError("Split ratios must sum to 1")
    if s["stratify"] is not True:
        raise ValueError("Consumer-level stratification must remain enabled")
    if config["dates"]["frequency"] != "D":
        raise ValueError("Only a complete daily calendar (frequency D) is supported")
    start = pd.to_datetime(config["dates"]["start"], format="%Y-%m-%d")
    end = pd.to_datetime(config["dates"]["end"], format="%Y-%m-%d")
    if pd.isna(start) or pd.isna(end) or end < start:
        raise ValueError("Calendar end must be on or after start")
    choices = [
        ("data", "invalid_policy", ("nan", "error")),
        ("data", "negative_policy", ("nan", "error", "clip_zero")),
        ("outliers", "method", ("global_quantile_clip",)),
        ("cnn", "dtype", ("float32",)),
        ("cnn", "scaling", ("standard", "robust", "none")),
        ("cnn", "imputation", ("placeholder", "short_linear")),
        ("hybrid", "imputation", ("median",)),
        ("hybrid", "scaling", ("standard", "robust", "none")),
    ]
    for section, key, allowed in choices:
        if config[section][key] not in allowed:
            raise ValueError(f"{section}.{key} must be one of {allowed}")
    if not 0 < config["outliers"]["upper_quantile"] <= 1:
        raise ValueError("outliers.upper_quantile must be in (0, 1]")
    for section, key in [("data", "chunksize"), ("runtime", "batch_size"),
                         ("features", "insufficient_history_threshold"),
                         ("cnn", "max_interpolation_gap")]:
        value = config[section][key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{section}.{key} must be a positive integer")
    for key in ("sudden_drop_threshold", "sudden_increase_threshold"):
        if not 0 < config["features"][key] <= 1:
            raise ValueError(f"features.{key} must be in (0, 1]")
    if config["cnn"]["use_observation_mask"] is not True:
        raise ValueError("The original observation mask is required")
    if config["cnn"]["missing_placeholder"] != 0:
        raise ValueError("Missing CNN values use the neutral normalized placeholder 0")
    if not np.isfinite(config["data"]["suspicious_value_threshold"]) or config["data"]["suspicious_value_threshold"] <= 0:
        raise ValueError("suspicious_value_threshold must be finite and positive")
    for section, key in [("outliers", "enabled"), ("cnn", "log_transform"),
                         ("hybrid", "add_missing_indicators"),
                         ("features", "include_quality_features"),
                         ("features", "create_quality_ablation"),
                         ("filtering", "filter_insufficient_history")]:
        if not isinstance(config[section][key], bool):
            raise ValueError(f"{section}.{key} must be a YAML boolean")
    for section, key in [("split", "random_state"), ("runtime", "random_seed")]:
        if not isinstance(config[section][key], int) or not 0 <= config[section][key] < 2**32:
            raise ValueError(f"{section}.{key} must be a 32-bit nonnegative integer")


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        supplied = yaml.safe_load(handle) or {}
    if not isinstance(supplied, dict):
        raise ValueError("Configuration must be a mapping")
    config = deepcopy(DEFAULTS)
    for section, values in supplied.items():
        if section not in config or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid configuration section: {section}")
        for key, value in values.items():
            if key not in config[section]:
                raise ValueError(f"Unknown configuration key: {section}.{key}")
            config[section][key] = value
    validate_config(config)
    for key in ("input_path", "output_dir"):
        value = Path(config["data"][key])
        config["data"][key] = str((path.parent / value).resolve())
    return config
