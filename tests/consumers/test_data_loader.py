import numpy as np
import pandas as pd
import pytest

from src.consumers.data_loader import load_dataset


def test_full_calendar_and_chronology(synthetic_csv):
    path, config = synthetic_csv
    data = load_dataset(path, config)
    assert data.values.shape == (40, 1035)
    assert data.dates.is_monotonic_increasing
    assert data.report["original_date_columns"] == 1034
    assert data.report["missing_calendar_dates"] == ["2016-09-18"]
    assert np.isnan(data.values[:, data.dates.get_loc("2016-09-18")]).all()
    original = pd.read_csv(path)
    assert data.values[2, data.dates.get_loc("2014-01-10")] == original.loc[2, "2014/1/10"]
    assert data.values[1, 0] == 0 and data.mask[1, 0]
    assert np.isnan(data.values[0, 0]) and not data.mask[0, 0]
    assert data.ids[0] == "000000"
    assert data.labels.dtype == np.int8


@pytest.mark.parametrize("text,match", [
    ("FLAG,2014/1/1\n0,1\n", "CONS_NO"),
    ("CONS_NO,2014/1/1\na,1\n", "FLAG"),
    ("CONS_NO,FLAG,2014/1/1\na,0,1\na,1,2\n", "unique"),
    ("CONS_NO,FLAG,2014/1/1\na,2,1\n", "binary"),
    ("CONS_NO,FLAG,2014/1/1\n,0,1\n", "empty"),
    ("CONS_NO,FLAG,2014/1/1,2014/01/01\na,0,1,2\n", "Duplicate dates"),
    ("CONS_NO,FLAG,2014/1/1,2014/1/1\na,0,1,2\n", "Duplicate CSV"),
    ("CONS_NO,FLAG,2014/2/30\na,0,1\n", "Invalid calendar date"),
    ("CONS_NO,FLAG,other\na,0,1\n", "Unexpected column"),
    ("CONS_NO,FLAG,2017/1/1\na,0,1\n", "outside"),
])
def test_validation_errors(tmp_path, config, text, match):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_dataset(path, config)


def test_measurement_audit_and_configurable_policies(tmp_path, config):
    path = tmp_path / "bad.csv"
    path.write_text("CONS_NO,FLAG,2014/1/1,2014/1/2,2014/1/3,2014/1/4,2014/1/5,2014/1/6\na,0,oops,-2,inf,800000,0,\n")
    audit = tmp_path / "audit.csv"
    data = load_dataset(path, config, audit)
    assert np.isnan(data.values[0, :3]).all()
    assert data.values[0, 3] == 800000
    assert data.mask[0, 4] and data.values[0, 4] == 0
    assert not data.mask[0, 5]
    for key in ["invalid_numeric_values", "negative_measurements", "infinite_values", "suspiciously_large_values"]:
        assert data.report[key] == 1
    assert len(pd.read_csv(audit)) == 4
    config["data"]["negative_policy"] = "clip_zero"
    assert load_dataset(path, config).values[0, 1] == 0
    config["data"]["negative_policy"] = "error"
    with pytest.raises(ValueError, match="Negative"):
        load_dataset(path, config)
    config["data"]["invalid_policy"] = "error"
    with pytest.raises(ValueError, match="Invalid/infinite"):
        load_dataset(path, config)
