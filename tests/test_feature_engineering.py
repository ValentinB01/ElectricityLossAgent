import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import engineer_features, pair_correlation


def extract(rows, config, start="2014-01-01", threshold=100):
    x = np.asarray(rows, dtype=float)
    return engineer_features(x, pd.date_range(start, periods=x.shape[1]), threshold, config["features"])


def test_general_features_and_raw_outliers(config):
    f = extract([[0, 1, 2, 3, np.nan, 800000]], config).iloc[0]
    assert len(f) == 80
    assert f.observed_days == 5
    assert f.total_consumption == 800006
    assert f.mean_consumption == pytest.approx(800006 / 5)
    assert f.median_consumption == 2
    assert f.p25_consumption == 1
    assert f.max_consumption == 800000  # raw, not clipped
    assert f.outlier_count == 1 and f.outlier_ratio == .2
    assert f.zero_ratio == .2 and f.missing_ratio == 1 / 6
    assert f.std_consumption == pytest.approx(np.std([0, 1, 2, 3, 800000]))


def test_missing_gaps_and_temporal_streaks(config):
    f = extract([[10, np.nan, 0, 0, np.nan, 20, 10, 0, 0]], config).iloc[0]
    assert f.mean_daily_absolute_change == 5
    assert f.max_daily_decrease == 10
    assert f.max_daily_increase == 0
    assert f.sudden_drop_count == 2 and f.sudden_drop_ratio == 1
    assert f.mean_daily_percentage_change == -.75
    assert f.max_consecutive_zero_days == 2 and f.mean_consecutive_zero_days == 2
    assert f.longest_missing_streak == 1 and f.number_of_missing_streaks == 2
    assert f.first_valid_day_index == 0 and f.last_valid_day_index == 8
    assert f.valid_observation_span == 9
    assert f.autocorrelation_lag_1 == pytest.approx(np.corrcoef([0, 20, 10, 0], [0, 10, 0, 0])[0, 1])
    g = extract([[1, np.nan, 3, np.nan, 5]], config).iloc[0]
    assert np.isnan(g.mean_daily_absolute_change)
    assert np.isnan(g.autocorrelation_lag_1)
    assert np.isnan(g.sudden_drop_ratio)


def test_lag_endpoints_only():
    x = np.array([[1, 2, np.nan, 4, 5, 6, 7, 8, 9, np.nan]], dtype=float)
    assert pair_correlation(x, 7)[0] == pytest.approx(1)


def test_single_day_calendar_has_undefined_temporal_features(config):
    f = extract([[1], [np.nan]], config)
    assert f.loc[0, "observed_days"] == 1
    assert np.isnan(f.loc[0, "mean_daily_absolute_change"])
    assert np.isnan(f.loc[0, "monthly_trend_slope"])


def test_empty_zero_constant_and_zero_denominators(config):
    f = extract([[np.nan] * 40, [0] * 40, [7] * 40], config)
    assert f.loc[0, "all_observations_missing"] == 1
    assert np.isnan(f.loc[0, "total_consumption"])
    assert np.isnan(f.loc[0, "first_valid_day_index"])
    assert f.loc[0, "longest_missing_streak"] == 40
    assert f.loc[0, "all_observations_zero"] == 0
    assert f.loc[1, "all_observations_zero"] == 1
    assert f.loc[1, "zero_ratio"] == 1
    assert f.loc[1, "max_consecutive_zero_days"] == 40
    assert np.isnan(f.loc[1, "mean_to_median_ratio"])
    assert np.isnan(f.loc[1, "mean_daily_percentage_change"])
    assert np.isnan(f.loc[2, "autocorrelation_lag_1"])
    assert np.isnan(f.loc[2, "consumption_skewness"])
    assert f.loc[2, "std_consumption"] == 0
    assert not np.isinf(f.to_numpy()).any()


def test_calendar_weekends_months_trend_and_recent_windows(config):
    dates = pd.date_range("2014-01-01", "2014-12-31")
    row = np.where(dates.dayofweek < 5, 10., 20.)
    f = engineer_features(row[None, :], dates, None, config["features"]).iloc[0]
    assert f.weekday_mean == 10 and f.weekend_mean == 20
    assert f.weekday_weekend_ratio == .5
    assert f.monthly_mean_01 == pytest.approx(row[dates.month == 1].mean())
    row[-90:] = np.nan
    g = engineer_features(row[None, :], dates, None, config["features"]).iloc[0]
    assert np.isnan(g.recent_30d_mean) and np.isnan(g.recent_90d_mean)
    assert g.recent_90d_observed_ratio == 0
    monthly_values = dates.month.to_numpy(dtype=float)
    h = engineer_features(monthly_values[None, :], dates, None, config["features"]).iloc[0]
    assert h.monthly_trend_slope == pytest.approx(1)


def test_monthly_pooling_across_years_and_no_target_identifier_features(config):
    dates = pd.date_range("2014-01-01", "2015-01-31")
    row = np.full(len(dates), np.nan)
    row[(dates.year == 2014) & (dates.month == 1)] = 1
    row[-1] = 32
    f = engineer_features(row[None, :], dates, None, config["features"])
    assert f.loc[0, "monthly_mean_01"] == pytest.approx((31 + 32) / 32)
    assert not {"FLAG", "CONS_NO", "index", "split", "row_index"} & set(f.columns)
