from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.consumers.config import DEFAULTS


@pytest.fixture
def config():
    return deepcopy(DEFAULTS)


@pytest.fixture
def synthetic_csv(tmp_path, config):
    """Label-grouped CSV, unsorted dates, zeros, gaps, empty/constant/extreme histories."""
    dates = pd.date_range("2014-01-01", "2016-10-31")
    original = dates[dates != pd.Timestamp("2016-09-18")]
    rng = np.random.default_rng(123)
    x = rng.gamma(2, 5, (40, len(original)))
    x[rng.random(x.shape) < .15] = np.nan
    x[0] = np.nan
    x[1] = 0
    x[2] = 7
    x[3, :10] = 800000
    x[4, :100] = np.nan
    columns = [f"{d.year}/{d.month}/{d.day}" for d in original]
    frame = pd.DataFrame(x, columns=columns)
    frame.insert(0, "FLAG", np.repeat([1, 0], 20))
    frame.insert(0, "CONS_NO", [f"000{i:03d}" for i in range(40)])
    frame = frame[["CONS_NO", "FLAG", *rng.permutation(columns)]]
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    config["data"]["input_path"] = str(path)
    config["data"]["output_dir"] = str(tmp_path / "processed")
    config["data"]["chunksize"] = 7
    config["runtime"]["batch_size"] = 9
    return path, config
