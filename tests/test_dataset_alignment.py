import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_config
from src.dataset_builders import MODELS, prepare
from src.validation import validate_artifacts


def test_all_variants_serialization_and_exact_consumer_alignment(synthetic_csv):
    _, config = synthetic_csv
    config["features"]["create_quality_ablation"] = True
    report = prepare(config, MODELS)
    root = Path(config["data"]["output_dir"])
    assert report["validation"]["passed"]
    assert report["features"]["count"] == 80
    assert report["features"]["hybrid_output_count"] == 160
    assignments = pd.read_csv(root / "shared/splits.csv", dtype={"CONS_NO": str})
    pre = joblib.load(root / "hybrid/feature_preprocessor.joblib")
    names = json.loads((root / "lightgbm/feature_names.json").read_text())
    for split, n in [("train", 28), ("val", 6), ("test", 6)]:
        lgb = pd.read_parquet(root / "lightgbm" / f"{split}.parquet")
        xgb = pd.read_parquet(root / "xgboost" / f"{split}.parquet")
        pd.testing.assert_frame_equal(lgb, xgb)
        expected_ids = assignments.loc[assignments.split == split, "CONS_NO"].to_numpy()
        np.testing.assert_array_equal(lgb.CONS_NO, expected_ids)
        cnn = np.load(root / "cnn1d" / f"X_{split}.npy")
        sequence = np.load(root / "hybrid" / f"X_sequence_{split}.npy")
        stats = np.load(root / "hybrid" / f"X_features_{split}.npy")
        assert cnn.shape == (n, 1035, 2)
        np.testing.assert_array_equal(cnn, sequence)
        # Explicitly prove each statistical and sequence row has the same consumer.
        np.testing.assert_array_equal(stats, pre.transform(lgb[names]))
        for model in ("cnn1d", "hybrid"):
            np.testing.assert_array_equal(np.load(root / model / f"y_{split}.npy"), lgb.FLAG)
            ids = pd.read_csv(root / model / f"{split}_ids.csv", dtype=str).CONS_NO
            np.testing.assert_array_equal(ids, expected_ids)
        ablation = pd.read_parquet(root / "ablations/without_quality/xgboost" / f"{split}.parquet")
        assert "observed_days" not in ablation and "missing_ratio" not in ablation
        np.testing.assert_array_equal(ablation.CONS_NO, expected_ids)
    assert validate_artifacts(root)["passed"]
    # Prove the validator actually rejects a branch shuffled away from its labels/IDs.
    path = root / "hybrid/X_features_train.npy"
    stats = np.load(path)
    np.save(path, stats[::-1])
    with pytest.raises(ValueError, match="statistical branch"):
        validate_artifacts(root)


@pytest.mark.parametrize("model", MODELS)
def test_individual_variants_and_cli_help(synthetic_csv, model):
    _, config = synthetic_csv
    report = prepare(config, [model])
    assert report["validation"]["passed"]
    assert set(report["validation"]["shapes"]) == {model}


def test_cli_and_separate_sensitivity_experiment(synthetic_csv, tmp_path):
    _, config = synthetic_csv
    config["filtering"]["filter_insufficient_history"] = True
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run([sys.executable, "-m", "src.prepare_data", "--config", str(config_path), "--all"],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    root = Path(config["data"]["output_dir"])
    filtered = root / "sensitivity/min_180_observed_days"
    main_splits = pd.read_csv(root / "shared/splits.csv")
    filtered_splits = pd.read_csv(filtered / "shared/splits.csv")
    assert len(main_splits) == 40 and len(filtered_splits) == 39
    assert validate_artifacts(filtered)["passed"]
    assert json.loads((filtered / "shared/dataset_report.json").read_text())["population_name"] == "filtered sensitivity experiment"


def test_different_config_cannot_silently_mix_individual_outputs(synthetic_csv):
    _, config = synthetic_csv
    prepare(config, ["cnn1d"])
    config["outliers"]["upper_quantile"] = .99
    with pytest.raises(ValueError, match="mixed generations"):
        prepare(config, ["hybrid"])


def test_config_errors_and_relative_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("data:\n  input_path: data.csv\n")
    config = load_config(config_path)
    assert Path(config["data"]["input_path"]) == tmp_path / "data.csv"
    for text, match in [
        ("split:\n  train_ratio: 0.8\n", "sum to 1"),
        ("split:\n  stratify: false\n", "stratification"),
        ("cnn:\n  use_observation_mask: false\n", "mask"),
        ("features:\n  misspelled: 1\n", "Unknown configuration"),
        ("outliers:\n  upper_quantile: 1.5\n", "upper_quantile"),
    ]:
        config_path.write_text(text)
        with pytest.raises(ValueError, match=match):
            load_config(config_path)
