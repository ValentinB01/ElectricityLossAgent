"""Train-fitted, serializable transforms; transform never changes fitted parameters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler


@dataclass
class OutlierClipper:
    enabled: bool = True
    upper_quantile: float = .999

    def fit(self, values: np.ndarray) -> "OutlierClipper":
        observed = values[np.isfinite(values)]
        self.observed_count_ = len(observed)
        self.threshold_ = float(np.quantile(observed, self.upper_quantile)) if self.enabled and len(observed) else None
        return self


class SequencePreprocessor:
    def __init__(self, cnn_config: dict, outlier_config: dict):
        self.config = dict(cnn_config)
        self.clipper = OutlierClipper(outlier_config["enabled"], outlier_config["upper_quantile"])

    def fit(self, values: np.ndarray) -> "SequencePreprocessor":
        self.clipper.fit(values)
        observed = values[np.isfinite(values)].copy()
        if (observed < 0).any():
            raise ValueError("Negative values must be resolved by the loader before CNN preprocessing")
        if self.clipper.threshold_ is not None:
            np.minimum(observed, self.clipper.threshold_, out=observed)
        if self.config["log_transform"]:
            np.log1p(observed, out=observed)
        self.fitted_consumer_count_ = len(values)
        self.fitted_observation_count_ = len(observed)
        self.center_, self.scale_ = 0.0, 1.0
        if len(observed) and self.config["scaling"] == "standard":
            self.center_, self.scale_ = float(observed.mean()), float(observed.std())
        elif len(observed) and self.config["scaling"] == "robust":
            p25, self.center_, p75 = np.quantile(observed, [.25, .5, .75]).tolist()
            self.scale_ = p75 - p25
        if self.scale_ < 1e-12:
            self.scale_ = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not hasattr(self, "center_"):
            raise ValueError("SequencePreprocessor must be fitted on training consumers first")
        original_mask = np.isfinite(values)
        x = np.array(values, dtype=np.float64, copy=True)
        if (x[original_mask] < 0).any():
            raise ValueError("Negative input to the consumption transform")
        if self.config["imputation"] == "short_linear":
            for row in x:
                missing = ~np.isfinite(row)
                edges = np.diff(np.r_[False, missing, False].astype(np.int8))
                for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
                    if start > 0 and end < len(row) and end - start <= self.config["max_interpolation_gap"]:
                        row[start:end] = np.linspace(row[start - 1], row[end], end - start + 2)[1:-1]
        if self.clipper.threshold_ is not None:
            np.minimum(x, self.clipper.threshold_, out=x)
        if self.config["log_transform"]:
            np.log1p(x, out=x)
        x -= self.center_
        x /= self.scale_
        x[~np.isfinite(x)] = self.config["missing_placeholder"]
        result = np.empty((*values.shape, 2), dtype=np.float32)
        with np.errstate(over="ignore"):
            result[:, :, 0] = x
        result[:, :, 1] = original_mask
        if not np.isfinite(result).all():
            raise ValueError("CNN transform overflowed float32; enable clipping or log1p")
        return result


class FeaturePreprocessor:
    """Median imputation + scaling; train-empty columns stay present with zero fallback.

    Indicators cover every input feature so even newly missing validation values are
    represented. Indicators remain binary and are appended after scaling.
    """
    def __init__(self, config: dict):
        self.config = dict(config)

    @staticmethod
    def _numeric(frame: pd.DataFrame) -> np.ndarray:
        values = frame.to_numpy(dtype=np.float64, copy=True)
        values[~np.isfinite(values)] = np.nan
        return values

    def fit(self, frame: pd.DataFrame) -> "FeaturePreprocessor":
        if {"CONS_NO", "FLAG", "split", "index", "row_index"} & set(frame.columns):
            raise ValueError("Predictive features cannot contain consumer IDs, labels or row positions")
        self.input_names_ = list(frame.columns)
        x = self._numeric(frame)
        self.empty_columns_ = [name for name, empty in zip(self.input_names_, np.isnan(x).all(axis=0)) if empty]
        self.training_missing_columns_ = [name for name, missing in zip(self.input_names_, np.isnan(x).any(axis=0)) if missing]
        self.constant_columns_ = [name for i, name in enumerate(self.input_names_)
                                  if np.isfinite(x[:, i]).any() and np.nanmax(x[:, i]) == np.nanmin(x[:, i])]
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        imputed = self.imputer_.fit_transform(x)
        strategy = self.config["scaling"]
        self.scaler_ = RobustScaler() if strategy == "robust" else StandardScaler() if strategy == "standard" else None
        if self.scaler_ is not None:
            self.scaler_.fit(imputed)
        self.output_names_ = self.input_names_.copy()
        if self.config["add_missing_indicators"]:
            self.output_names_ += [f"{name}__missing" for name in self.input_names_]
        self.fitted_consumer_count_ = len(frame)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "input_names_"):
            raise ValueError("FeaturePreprocessor must be fitted on training consumers first")
        if list(frame.columns) != self.input_names_:
            raise ValueError("Feature names/order differ from the fitted training schema")
        x = self._numeric(frame)
        imputed = self.imputer_.transform(x)
        scaled = self.scaler_.transform(imputed) if self.scaler_ is not None else imputed
        if self.config["add_missing_indicators"]:
            scaled = np.column_stack([scaled, np.isnan(x).astype(float)])
        with np.errstate(over="ignore"):
            result = scaled.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Hybrid features overflowed float32; inspect raw values and scaling settings")
        return result
