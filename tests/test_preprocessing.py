import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import FeaturePreprocessor, SequencePreprocessor


def test_sequence_train_only_fit_extreme_heldout_and_mask(config, tmp_path):
    train = np.array([[0, 1, np.nan, 2], [3, 4, 5, 6]], dtype=float)
    pre = SequencePreprocessor(config["cnn"], config["outliers"]).fit(train)
    expected_threshold = np.quantile(train[np.isfinite(train)], .999)
    assert pre.clipper.threshold_ == expected_threshold
    expected = np.log1p(np.minimum(train[np.isfinite(train)], expected_threshold))
    assert pre.center_ == pytest.approx(expected.mean())
    assert pre.scale_ == pytest.approx(expected.std())
    fitted = (pre.center_, pre.scale_, pre.clipper.threshold_, pre.fitted_observation_count_)
    heldout = np.array([[1e12, np.nan, 0, 1e9], [np.nan] * 4])
    result = pre.transform(heldout)
    assert fitted == (pre.center_, pre.scale_, pre.clipper.threshold_, pre.fitted_observation_count_)
    assert result.dtype == np.float32 and result.shape == (2, 4, 2)
    assert np.isfinite(result).all()
    assert result[0, 1, 0] == 0 and result[0, 1, 1] == 0
    assert result[0, 2, 1] == 1  # real zero is observed
    assert result[0, 2, 0] != 0  # scaled log(1+0), unlike the neutral placeholder
    assert (result[1] == 0).all()
    path = tmp_path / "pre.joblib"
    joblib.dump(pre, path)
    np.testing.assert_array_equal(joblib.load(path).transform(heldout), result)


def test_hybrid_imputer_scaler_train_only_and_empty_columns(config):
    train = pd.DataFrame({"a": [1., np.nan, 3.], "empty": [np.nan] * 3, "constant": [7.] * 3})
    pre = FeaturePreprocessor(config["hybrid"]).fit(train)
    median, center, scale = pre.imputer_.statistics_.copy(), pre.scaler_.center_.copy(), pre.scaler_.scale_.copy()
    heldout = pd.DataFrame({"a": [1e9, np.nan], "empty": [np.inf, np.nan], "constant": [7., 7.]})
    result = pre.transform(heldout)
    np.testing.assert_array_equal(pre.imputer_.statistics_, median)
    np.testing.assert_array_equal(pre.scaler_.center_, center)
    np.testing.assert_array_equal(pre.scaler_.scale_, scale)
    assert median.tolist() == [2., 0., 7.]
    assert result.shape == (2, 6) and result.dtype == np.float32
    assert pre.empty_columns_ == ["empty"] and pre.constant_columns_ == ["constant"]
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[:, 3:], [[0, 1, 0], [1, 1, 0]])
    assert (result[:, 1:3] == 0).all()
    with pytest.raises(ValueError, match="order"):
        pre.transform(heldout[["empty", "a", "constant"]])


@pytest.mark.parametrize("scaling", ["standard", "robust", "none"])
def test_sequence_empty_and_constant_fits(config, scaling):
    config["cnn"]["scaling"] = scaling
    for train in [np.full((3, 40), np.nan), np.zeros((3, 40)), np.full((3, 40), 7.)]:
        pre = SequencePreprocessor(config["cnn"], config["outliers"]).fit(train)
        assert pre.scale_ > 0
        assert np.isfinite(pre.transform(train)).all()


def test_optional_interpolation_is_short_internal_and_preserves_mask(config):
    config["cnn"].update({"imputation": "short_linear", "max_interpolation_gap": 1,
                          "scaling": "none", "log_transform": False})
    config["outliers"]["enabled"] = False
    x = np.array([[np.nan, 2, np.nan, 4, np.nan, np.nan, 8, np.nan]])
    pre = SequencePreprocessor(config["cnn"], config["outliers"]).fit(x)
    result = pre.transform(x)
    np.testing.assert_array_equal(result[0, :, 0], [0, 2, 3, 4, 0, 0, 8, 0])
    np.testing.assert_array_equal(result[0, :, 1], np.isfinite(x)[0])
