"""Build shared artifacts once and serialize consistent model-specific datasets."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
import logging
from pathlib import Path
import platform
import shutil

import joblib
import numpy as np
import pandas as pd

from .data_loader import ConsumerData, load_dataset
from .feature_engineering import QUALITY_FEATURES, engineer_features
from .preprocessing import FeaturePreprocessor, SequencePreprocessor
from .splitting import make_splits, split_table
from .validation import ids_digest, validate_artifacts


MODELS = ["lightgbm", "xgboost", "cnn1d", "hybrid"]
LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generation_manifest(config: dict, models: list[str]) -> dict:
    code = {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    raw_hash = file_digest(Path(config["data"]["input_path"]))
    # Paths do not change the generated numbers; content, configuration, code and versions do.
    numeric_config = json.loads(json.dumps(config))
    for key in ("input_path", "output_dir"):
        numeric_config["data"].pop(key)
    environment = {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "joblib", "PyYAML")}
    environment["python"] = platform.python_version()
    fingerprint = hashlib.sha256(json.dumps([raw_hash, numeric_config, code, environment], sort_keys=True).encode()).hexdigest()
    return {"run_fingerprint": fingerprint, "raw_csv_sha256": raw_hash, "source_sha256": code,
            "environment": environment, "config": config, "models": models, "validation_passed": False}


def _save_sequences(folder: Path, data: ConsumerData, indices: np.ndarray,
                    split: str, preprocessor: SequencePreprocessor, batch_size: int, name: str) -> None:
    output = np.lib.format.open_memmap(folder / name, mode="w+", dtype=np.float32,
                                       shape=(len(indices), len(data.dates), 2))
    for start in range(0, len(indices), batch_size):
        output[start:start + batch_size] = preprocessor.transform(data.values[indices[start:start + batch_size]])
    output.flush()
    del output


def _population_stats(data: ConsumerData, threshold_days: int, clip_threshold: float | None) -> dict:
    observed = np.isfinite(data.values).sum(axis=1)
    zeros = (data.values == 0).sum(axis=1)
    missing = data.values.size - int(observed.sum())
    classes = {}
    for label in (0, 1):
        selector = data.labels == label
        denominator = int(selector.sum()) * len(data.dates)
        classes[str(label)] = {
            "consumers": int(selector.sum()), "percentage": float(selector.mean() * 100),
            "missing_measurements": int((len(data.dates) - observed[selector]).sum()),
            "missing_percentage": float((len(data.dates) - observed[selector]).sum() / denominator * 100) if denominator else None,
            "mean_observed_days": float(observed[selector].mean()) if selector.any() else None,
        }
    return {"standardized_shape": list(data.values.shape), "standardized_dtype": str(data.values.dtype),
            "class_distribution_descriptive_only": classes,
            "missing_measurements_after_sanitization_and_calendar_insertion": missing,
            "missing_percentage_after_preprocessing": missing / data.values.size * 100,
            "observed_zero_measurements": int(zeros.sum()),
            "zero_percentage_of_observed": float(zeros.sum() / observed.sum() * 100) if observed.sum() else None,
            "consumers_with_insufficient_history": int((observed < threshold_days).sum()),
            "consumers_without_observations": int((observed == 0).sum()),
            "all_zero_consumers_with_observations": int(((observed > 0) & (zeros == observed)).sum()),
            "measurements_above_training_clip_threshold": int((data.values > clip_threshold).sum()) if clip_threshold is not None else 0}


def _write_report(root: Path, report: dict) -> None:
    write_json(root / "shared/dataset_report.json", report)
    lines = ["# SGCC preprocessing report", "", "Measured from the local input CSV; no models were trained.", "",
             "## Original data", "", f"- Source SHA256: `{report['raw_csv_sha256']}`",
             f"- Raw dimensions: {report['raw']['raw_dimensions']}",
             f"- Original daily columns: {report['raw']['original_date_columns']}",
             f"- Missing calendar dates: {', '.join(report['raw']['missing_calendar_dates']) or 'none'}",
             f"- Invalid numeric / infinite / negative values: {report['raw']['invalid_numeric_values']} / {report['raw']['infinite_values']} / {report['raw']['negative_measurements']}",
             f"- Raw finite minimum / maximum: {report['raw']['raw_finite_min']} / {report['raw']['raw_finite_max']}",
             f"- Suspiciously large values (reporting rule only): {report['raw']['suspiciously_large_values']}",
             "", "## After sanitization and calendar alignment", ""]
    lines += [f"- {key}: {json.dumps(value)}" for key, value in report["population"].items()]
    lines += ["", "## Shared splits", "", "| Split | Consumers | Normal | Theft |", "|---|---:|---:|---:|"]
    for name, summary in report["splits"].items():
        lines.append(f"| {name} | {summary['consumers']} | {summary['class_counts']['0']} | {summary['class_counts']['1']} |")
    lines += ["", "## Fitted transformations", "", "```json", json.dumps(report["preprocessing"], indent=2), "```",
              "", f"Engineered features: {report['features']['count']}; retained train-empty hybrid columns: {report['features']['training_all_missing_columns']}.",
              "Train-empty features use zero fallback; none are silently discarded. Monthly coverage is saved separately.",
              "", "## Artifacts and validation", "", "```json", json.dumps(report["validation"], indent=2), "```",
              "", "## Warnings", ""]
    lines += [f"- {message}" for message in report["warnings"]] or ["- None"]
    lines += ["", "Missingness by label is descriptive only. No anomaly establishes theft, and consumer labels do not identify theft dates.",
              "For exact feature names, undefined counts, environment versions and reproducibility settings, see dataset_report.json and run_manifest.json.", ""]
    (root / "shared/dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_datasets(data: ConsumerData, config: dict, models: list[str], manifest: dict,
                   root: Path, population_name: str = "complete") -> dict:
    root.mkdir(parents=True, exist_ok=True)
    shared = root / "shared"
    shared.mkdir(exist_ok=True)
    manifest["validation_passed"] = False
    write_json(shared / "run_manifest.json", manifest)
    splits = make_splits(data, config)
    assignments = split_table(data, splits)
    assignments.to_csv(shared / "splits.csv", index=False)
    pd.DataFrame({"CONS_NO": data.ids, "FLAG": data.labels}).to_csv(shared / "consumers.csv", index=False)
    pd.DataFrame({"date": data.dates.strftime("%Y-%m-%d")}).to_csv(shared / "calendar.csv", index=False)
    np.save(shared / "consumption.npy", data.values, allow_pickle=False)
    np.save(shared / "observation_mask.npy", data.mask.astype(np.uint8), allow_pickle=False)
    np.save(shared / "labels.npy", data.labels, allow_pickle=False)
    observed = data.mask.sum(axis=1)
    pd.DataFrame({"CONS_NO": data.ids, "FLAG": data.labels, "observed_days": observed}).loc[
        observed < config["features"]["insufficient_history_threshold"]].to_csv(shared / "insufficient_history.csv", index=False)
    months = data.dates.to_period("M")
    coverage = {f"observed_days_{m}": np.isfinite(data.values[:, months == m]).sum(axis=1) for m in months.unique()}
    pd.DataFrame({"CONS_NO": data.ids, **coverage}).to_parquet(shared / "monthly_coverage.parquet", index=False)
    LOGGER.info("Fitting clipping and sequence normalization on %s training consumers", len(splits["train"]))
    sequence_preprocessor = SequencePreprocessor(config["cnn"], config["outliers"]).fit(data.values[splits["train"]])
    sequence_preprocessor.fit_ids_digest_ = ids_digest(data.ids[splits["train"]])
    joblib.dump(sequence_preprocessor, shared / "sequence_preprocessor.joblib")
    joblib.dump(sequence_preprocessor.clipper, shared / "outlier_preprocessor.joblib")
    LOGGER.info("Engineering common consumer features once")
    features = engineer_features(data.values, data.dates, sequence_preprocessor.clipper.threshold_,
                                 config["features"], config["runtime"]["batch_size"])
    names = features.columns.tolist()
    common = features.copy()
    common.insert(0, "CONS_NO", data.ids)
    common["FLAG"] = data.labels
    common.to_parquet(shared / "engineered_features.parquet", index=False)
    write_json(shared / "feature_names.json", names)
    hybrid_preprocessor = None
    if "hybrid" in models:
        hybrid_preprocessor = FeaturePreprocessor(config["hybrid"]).fit(features.iloc[splits["train"]])
        hybrid_preprocessor.fit_ids_digest_ = sequence_preprocessor.fit_ids_digest_
    split_summary = {}
    for name, ix in splits.items():
        split_summary[name] = {"consumers": len(ix),
                               "class_counts": {str(v): int((data.labels[ix] == v).sum()) for v in (0, 1)},
                               "theft_percentage": float(data.labels[ix].mean() * 100),
                               "ids_digest": ids_digest(data.ids[ix])}
    for model in models:
        LOGGER.info("Writing %s datasets", model)
        folder = root / model
        folder.mkdir(exist_ok=True)
        if model != "cnn1d":
            write_json(folder / "feature_names.json", hybrid_preprocessor.output_names_ if model == "hybrid" else names)
        if model == "cnn1d":
            joblib.dump(sequence_preprocessor, folder / "preprocessor.joblib")
        elif model == "hybrid":
            joblib.dump(hybrid_preprocessor, folder / "feature_preprocessor.joblib")
            joblib.dump(sequence_preprocessor, folder / "sequence_preprocessor.joblib")
        for split, ix in splits.items():
            if model in ("lightgbm", "xgboost"):
                common.iloc[ix].to_parquet(folder / f"{split}.parquet", index=False)
            else:
                pd.DataFrame({"CONS_NO": data.ids[ix]}).to_csv(folder / f"{split}_ids.csv", index=False)
                np.save(folder / f"y_{split}.npy", data.labels[ix], allow_pickle=False)
                filename = f"X_{split}.npy" if model == "cnn1d" else f"X_sequence_{split}.npy"
                if model == "hybrid" and "cnn1d" in models:
                    shutil.copyfile(root / "cnn1d" / f"X_{split}.npy", folder / filename)
                else:
                    _save_sequences(folder, data, ix, split, sequence_preprocessor, config["runtime"]["batch_size"], filename)
                if model == "hybrid":
                    np.save(folder / f"X_features_{split}.npy", hybrid_preprocessor.transform(features.iloc[ix]), allow_pickle=False)
        metadata = {"model": model, "run_fingerprint": manifest["run_fingerprint"], "population": population_name,
                    "split_counts": split_summary, "calendar_start": str(data.dates[0].date()),
                    "calendar_end": str(data.dates[-1].date()), "calendar_days": len(data.dates),
                    "feature_count": len(hybrid_preprocessor.output_names_) if model == "hybrid" else len(names) if model in ("lightgbm", "xgboost") else None,
                    "target": {"name": "FLAG", "0": "normal", "1": "consumer labeled as theft", "unit": "consumer"},
                    "predictive_columns_exclude": ["CONS_NO", "FLAG", "split", "row_index"],
                    "sequence_channels": ["clipped, optionally log1p and scaled consumption", "original usable observation mask"],
                    "statistical_features_source": "sanitized raw values; no clipping/scaling for trees",
                    "fit_population": "train only", "training_ids_digest": sequence_preprocessor.fit_ids_digest_,
                    "config": config}
        write_json(folder / "metadata.json", metadata)
    if config["features"]["create_quality_ablation"]:
        ablation_names = [n for n in names if n not in QUALITY_FEATURES]
        for model in ("lightgbm", "xgboost"):
            if model not in models:
                continue
            folder = root / "ablations" / "without_quality" / model
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / "feature_names.json", ablation_names)
            for split, ix in splits.items():
                common.iloc[ix][["CONS_NO", *ablation_names, "FLAG"]].to_parquet(folder / f"{split}.parquet", index=False)
            write_json(folder / "metadata.json", {"run_fingerprint": manifest["run_fingerprint"],
                       "removed_features": sorted(set(names) - set(ablation_names)), "shared_splits": "../../../shared/splits.csv"})
    write_json(shared / "run_manifest.json", manifest)
    LOGGER.info("Validating every saved consumer and both hybrid branches")
    validation = validate_artifacts(root, manifest["models"], config["runtime"]["batch_size"])
    for model in models:
        path = root / model / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["output_shapes_and_dtypes"] = validation["shapes"][model]
        write_json(path, metadata)
    manifest["validation_passed"] = True
    write_json(shared / "run_manifest.json", manifest)
    undefined = features.isna().sum()
    train_features = features.iloc[splits["train"]]
    warnings = data.report["warnings"].copy()
    if (observed == 0).any():
        warnings.append("Consumers with fully missing histories were retained; undefined features stay NaN for trees")
    if (observed < config["features"]["insufficient_history_threshold"]).any():
        warnings.append("Consumers below the history threshold were retained; see insufficient_history.csv")
    if train_features.isna().all().any():
        warnings.append("Some features are entirely undefined in training; hybrid median-imputer fallback is zero")
    report = {"population_name": population_name, "raw_csv_sha256": manifest["raw_csv_sha256"],
              "raw": data.report, "population": _population_stats(data, config["features"]["insufficient_history_threshold"], sequence_preprocessor.clipper.threshold_),
              "splits": split_summary, "features": {"count": len(names), "names": names,
                  "undefined_counts": {n: int(undefined[n]) for n in names if undefined[n]},
                  "training_all_missing_columns": train_features.columns[train_features.isna().all()].tolist(),
                  "discarded_columns": [], "hybrid_output_count": len(hybrid_preprocessor.output_names_) if hybrid_preprocessor else None,
                  "ablation_created": config["features"]["create_quality_ablation"]},
              "preprocessing": {"fit_split": "train", "training_ids_digest": sequence_preprocessor.fit_ids_digest_,
                  "outliers": {**config["outliers"], "learned_upper_threshold": sequence_preprocessor.clipper.threshold_,
                               "fitted_observation_count": sequence_preprocessor.fitted_observation_count_},
                  "cnn": {**config["cnn"], "learned_center": sequence_preprocessor.center_, "learned_scale": sequence_preprocessor.scale_},
                  "hybrid": {**config["hybrid"], "all_missing_training_column_policy": "retain; impute zero; scale using training values",
                             "indicator_policy": "append a binary missing indicator for every input feature"}},
              "validation": validation, "environment": manifest["environment"], "warnings": warnings}
    _write_report(root, report)
    return report


def prepare(config: dict, models: list[str]) -> dict:
    models = [m for m in MODELS if m in models]
    if not models:
        raise ValueError("Select at least one supported dataset variant")
    np.random.seed(config["runtime"]["random_seed"])
    input_path = Path(config["data"]["input_path"])
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    root = Path(config["data"]["output_dir"])
    if root == input_path or input_path.is_relative_to(root):
        raise ValueError("The raw CSV must be outside the processed output directory")
    manifest = generation_manifest(config, models)
    existing = root / "shared/run_manifest.json"
    if existing.exists() and models != MODELS:
        old = json.loads(existing.read_text())
        if old["run_fingerprint"] != manifest["run_fingerprint"]:
            raise ValueError("Existing datasets use different source/config/code/environment. Run --all or choose a new output_dir to avoid mixed generations.")
        manifest["models"] = [m for m in MODELS if m in set(old["models"] + models)]
    write_json(existing, manifest)
    LOGGER.info("Loading and auditing %s", input_path)
    data = load_dataset(input_path, config, root / "shared/measurement_issues.csv")
    LOGGER.info("Aligned %s consumers to %s days", len(data.ids), len(data.dates))
    report = build_datasets(data, config, models, manifest, root)
    if config["filtering"]["filter_insufficient_history"]:
        keep = data.mask.sum(axis=1) >= config["features"]["insufficient_history_threshold"]
        if not keep.any():
            raise ValueError("Sensitivity population is empty at the configured history threshold")
        filtered = ConsumerData(data.ids[keep], data.labels[keep], data.values[keep], data.dates, data.report)
        filtered_root = root / "sensitivity" / f"min_{config['features']['insufficient_history_threshold']}_observed_days"
        filtered_manifest = {**manifest, "models": models, "validation_passed": False,
                             "population": "filtered sensitivity experiment"}
        build_datasets(filtered, config, models, filtered_manifest, filtered_root, "filtered sensitivity experiment")
    return report
