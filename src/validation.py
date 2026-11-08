"""Full saved-artifact checks, also available as a standalone CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


FORBIDDEN_FEATURES = {"CONS_NO", "FLAG", "split", "index", "row_index"}


def ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(json.dumps(np.asarray(ids).tolist(), ensure_ascii=False).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Artifact validation failed: {message}")


def validate_artifacts(root: str | Path, models: list[str] | None = None,
                       batch_size: int = 512) -> dict:
    root = Path(root)
    manifest = json.loads((root / "shared/run_manifest.json").read_text(encoding="utf-8"))
    models = models or manifest["models"]
    assignments = pd.read_csv(root / "shared/splits.csv", dtype={"CONS_NO": str})
    require(assignments.CONS_NO.is_unique, "consumer IDs overlap between splits")
    require(set(assignments.split) == {"train", "val", "test"}, "split names")
    require(assignments.FLAG.isin([0, 1]).all(), "binary split labels")
    consumers = pd.read_csv(root / "shared/consumers.csv", dtype={"CONS_NO": str})
    require(set(assignments.CONS_NO) == set(consumers.CONS_NO), "complete consumer coverage")
    require(consumers.CONS_NO.is_unique, "unique standardized consumers")
    raw = np.load(root / "shared/consumption.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(root / "shared/observation_mask.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "shared/labels.npy", allow_pickle=False)
    calendar = pd.read_csv(root / "shared/calendar.csv")
    require(raw.shape == mask.shape == (len(consumers), len(calendar)), "standardized matrix dimensions")
    require(raw.dtype == np.float64 and mask.dtype == np.uint8, "standardized dtypes")
    require(np.array_equal(labels, consumers.FLAG), "standardized label order")
    require(np.array_equal(pd.to_datetime(calendar.date).to_numpy(),
                           pd.date_range(calendar.date.iloc[0], calendar.date.iloc[-1], freq="D").to_numpy()), "complete daily calendar")
    for start in range(0, len(raw), batch_size):
        require(np.array_equal(np.isfinite(raw[start:start + batch_size]), mask[start:start + batch_size]), "original observation mask")
        require(not np.isinf(raw[start:start + batch_size]).any(), "infinite standardized measurements")
    row_lookup = pd.Series(np.arange(len(consumers)), index=consumers.CONS_NO)
    features = pd.read_parquet(root / "shared/engineered_features.parquet")
    feature_names = json.loads((root / "shared/feature_names.json").read_text())
    require(not FORBIDDEN_FEATURES.intersection(feature_names), "identifier/target feature leakage")
    require(features.columns.tolist() == ["CONS_NO", *feature_names, "FLAG"], "shared feature schema")
    require(np.array_equal(features.CONS_NO, consumers.CONS_NO), "shared feature IDs")
    require(np.array_equal(features.FLAG, labels), "shared feature labels")
    train_ids = assignments.loc[assignments.split == "train", "CONS_NO"].to_numpy()
    sequence_preprocessor = joblib.load(root / "shared/sequence_preprocessor.joblib")
    require(sequence_preprocessor.fitted_consumer_count_ == len(train_ids), "CNN training fit count")
    require(sequence_preprocessor.fit_ids_digest_ == ids_digest(train_ids), "CNN training fit provenance")
    train_ix = row_lookup.loc[train_ids].to_numpy()
    # Refit from saved training rows to independently check the serialized parameters.
    from .preprocessing import FeaturePreprocessor, SequencePreprocessor
    expected_sequence = SequencePreprocessor(manifest["config"]["cnn"], manifest["config"]["outliers"]).fit(raw[train_ix])
    require((sequence_preprocessor.center_, sequence_preprocessor.scale_, sequence_preprocessor.clipper.threshold_) ==
            (expected_sequence.center_, expected_sequence.scale_, expected_sequence.clipper.threshold_),
            "CNN normalization/clipping must equal a fit on saved training rows alone")
    shared_clipper = joblib.load(root / "shared/outlier_preprocessor.joblib")
    require(shared_clipper.threshold_ == sequence_preprocessor.clipper.threshold_, "shared outlier threshold")
    hybrid_preprocessor = joblib.load(root / "hybrid/feature_preprocessor.joblib") if "hybrid" in models else None
    if hybrid_preprocessor is not None:
        require(hybrid_preprocessor.fit_ids_digest_ == ids_digest(train_ids), "hybrid training fit provenance")
        require(hybrid_preprocessor.fitted_consumer_count_ == len(train_ids), "hybrid training fit count")
        expected_hybrid = FeaturePreprocessor(manifest["config"]["hybrid"]).fit(features.iloc[train_ix][feature_names])
        require(np.array_equal(hybrid_preprocessor.imputer_.statistics_, expected_hybrid.imputer_.statistics_),
                "hybrid imputation medians must equal training-only medians")
        if expected_hybrid.scaler_ is not None:
            for attribute in ("center_", "mean_", "scale_"):
                if hasattr(expected_hybrid.scaler_, attribute):
                    require(np.array_equal(getattr(hybrid_preprocessor.scaler_, attribute), getattr(expected_hybrid.scaler_, attribute)),
                            f"hybrid {attribute} must equal training-only scaling parameters")
    shapes: dict = {}
    for model in models:
        folder = root / model
        metadata = json.loads((folder / "metadata.json").read_text())
        require(metadata["run_fingerprint"] == manifest["run_fingerprint"], f"{model} generation fingerprint")
        if model in ("cnn1d", "hybrid"):
            filename = "preprocessor.joblib" if model == "cnn1d" else "sequence_preprocessor.joblib"
            saved_preprocessor = joblib.load(folder / filename)
            require((saved_preprocessor.center_, saved_preprocessor.scale_, saved_preprocessor.clipper.threshold_) ==
                    (expected_sequence.center_, expected_sequence.scale_, expected_sequence.clipper.threshold_),
                    f"{model} saved sequence preprocessor")
        shapes[model] = {}
        for split in ("train", "val", "test"):
            assignment = assignments.loc[assignments.split == split]
            ids, y = assignment.CONS_NO.to_numpy(), assignment.FLAG.to_numpy(dtype=np.int8)
            ix = row_lookup.loc[ids].to_numpy()
            require(np.array_equal(labels[ix], y), "assignment labels match source")
            if model in ("lightgbm", "xgboost"):
                frame = pd.read_parquet(folder / f"{split}.parquet")
                names = json.loads((folder / "feature_names.json").read_text())
                require(names == feature_names, f"{model} feature names")
                require(frame.columns.tolist() == ["CONS_NO", *names, "FLAG"], f"{model} saved schema")
                require(np.array_equal(frame.CONS_NO, ids) and np.array_equal(frame.FLAG, y), f"{model} IDs and labels")
                require(all(pd.api.types.is_numeric_dtype(frame[n]) for n in names), "numeric predictive features")
                require(not np.isinf(frame[names].to_numpy()).any(), "infinite tree features")
                require(np.array_equal(frame[names].to_numpy(), features.iloc[ix][names].to_numpy(), equal_nan=True),
                        f"{model} saved feature values")
                shapes[model][split] = {"X": [len(ids), len(names)], "y": list(y.shape), "dtype": "float64"}
                continue
            saved_ids = pd.read_csv(folder / f"{split}_ids.csv", dtype={"CONS_NO": str}).CONS_NO.to_numpy()
            saved_y = np.load(folder / f"y_{split}.npy", allow_pickle=False)
            require(np.array_equal(saved_ids, ids) and np.array_equal(saved_y, y), f"{model} IDs and labels")
            require(saved_y.dtype == np.int8, f"{model} integer binary target dtype")
            sequence_name = f"X_{split}.npy" if model == "cnn1d" else f"X_sequence_{split}.npy"
            sequence = np.load(folder / sequence_name, mmap_mode="r", allow_pickle=False)
            require(sequence.shape == (len(ids), raw.shape[1], 2) and sequence.dtype == np.float32, f"{model} sequence shape and dtype")
            hybrid = np.load(folder / f"X_features_{split}.npy", mmap_mode="r", allow_pickle=False) if model == "hybrid" else None
            if hybrid is not None:
                names = json.loads((folder / "feature_names.json").read_text())
                require(names == hybrid_preprocessor.output_names_, "hybrid output feature order")
                require(hybrid.shape == (len(ids), len(names)) and hybrid.dtype == np.float32, "hybrid feature shape/dtype")
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                indices = ix[start:end]
                actual = sequence[start:end]
                require(np.isfinite(actual).all(), f"{model} finite sequence")
                require(np.array_equal(actual, sequence_preprocessor.transform(raw[indices])), f"{model} sequence matches consumer")
                if hybrid is not None:
                    require(np.isfinite(hybrid[start:end]).all(), "finite hybrid features")
                    require(np.array_equal(hybrid[start:end], hybrid_preprocessor.transform(features.iloc[indices][feature_names])),
                            "hybrid statistical branch matches the same consumer")
                if model == "hybrid" and "cnn1d" in models:
                    cnn = np.load(root / "cnn1d" / f"X_{split}.npy", mmap_mode="r", allow_pickle=False)
                    require(np.array_equal(actual, cnn[start:end]), "identical CNN/hybrid sequence branches")
            shapes[model][split] = {"X_sequence": list(sequence.shape), "y": list(y.shape), "dtype": "float32"}
            if hybrid is not None:
                shapes[model][split]["X_features"] = list(hybrid.shape)
    return {"passed": True, "all_consumer_rows_checked": len(consumers), "shapes": shapes,
            "checks": ["disjoint and exhaustive splits", "source labels and consumer order",
                       "numeric feature schemas", "original masks", "finite neural tensors",
                       "train-only fit provenance and refitted parameters", "every neural row recomputed from its source consumer",
                       "both hybrid branches aligned", "all saved arrays, tables and preprocessors loaded"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()
    result = validate_artifacts(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
