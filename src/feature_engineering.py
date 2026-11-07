"""Eighty consumer features. All consumption statistics use sanitized RAW values.

Only outlier_count/ratio depend on a population parameter: a training-only threshold.
Undefined statistics are NaN; finite zeros remain genuine measurements.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


QUALITY_FEATURES = {
    "observed_days", "missing_days", "missing_ratio", "zero_days", "zero_ratio", "nonzero_days",
    "longest_missing_streak", "number_of_missing_streaks", "first_valid_day_index",
    "last_valid_day_index", "valid_observation_span", "observed_days_ratio",
    "has_insufficient_history", "all_observations_zero", "all_observations_missing",
    "recent_30d_observed_ratio", "recent_90d_observed_ratio",
    "winter_observed_ratio", "spring_observed_ratio", "summer_observed_ratio", "autumn_observed_ratio",
}


def safe_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    result = np.full(a.shape, np.nan)
    np.divide(a, b, out=result, where=np.isfinite(b) & (b != 0))
    result[~np.isfinite(result)] = np.nan
    return result


def streaks(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum length, mean length and count of True runs per consumer."""
    current = np.zeros(len(mask), dtype=int)
    maximum = current.copy()
    count = current.copy()
    previous = np.zeros(len(mask), dtype=bool)
    for col in mask.T:
        count += col & ~previous
        current = np.where(col, current + 1, 0)
        maximum = np.maximum(maximum, current)
        previous = col
    mean = np.divide(mask.sum(axis=1), count, out=np.zeros(len(mask), dtype=float), where=count > 0)
    return maximum, mean, count


def pair_correlation(x: np.ndarray, lag: int) -> np.ndarray:
    if lag >= x.shape[1]:
        return np.full(len(x), np.nan)
    left, right = x[:, :-lag], x[:, lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    left = np.where(valid, left, np.nan)
    right = np.where(valid, right, np.nan)
    a = left - np.nanmean(left, axis=1)[:, None]
    b = right - np.nanmean(right, axis=1)[:, None]
    result = safe_ratio(np.nansum(a * b, axis=1),
                        np.sqrt(np.nansum(a * a, axis=1) * np.nansum(b * b, axis=1)))
    result[valid.sum(axis=1) < 2] = np.nan
    return np.clip(result, -1, 1)


def _batch_features(x: np.ndarray, dates: pd.DatetimeIndex, threshold: float | None, config: dict) -> dict:
    valid = np.isfinite(x)
    observed = valid.sum(axis=1)
    n = x.shape[1]
    mean = np.nanmean(x, axis=1)
    std = np.nanstd(x, axis=1)
    p10, p25, median, p75, p90, p95 = np.nanpercentile(x, [10, 25, 50, 75, 90, 95], axis=1)
    maximum = np.nanmax(x, axis=1)
    centered = x - mean[:, None]
    skew = safe_ratio(np.nanmean(centered**3, axis=1), std**3)
    skew[observed < 3] = np.nan
    total = np.nansum(x, axis=1)
    total[observed == 0] = np.nan
    outlier_count = ((x > threshold) & valid).sum(axis=1) if threshold is not None else np.zeros(len(x))
    f = {
        "observed_days": observed, "total_consumption": total, "mean_consumption": mean,
        "median_consumption": median, "std_consumption": std, "min_consumption": np.nanmin(x, axis=1),
        "max_consumption": maximum, "p10_consumption": p10, "p25_consumption": p25,
        "p75_consumption": p75, "p90_consumption": p90, "p95_consumption": p95,
        "iqr_consumption": p75 - p25, "coefficient_of_variation": safe_ratio(std, mean),
        "mean_to_median_ratio": safe_ratio(mean, median), "max_to_median_ratio": safe_ratio(maximum, median),
        "consumption_skewness": skew, "outlier_count": outlier_count,
        "outlier_ratio": safe_ratio(outlier_count, observed),
    }
    # Adjacent positions are consecutive days because the calendar is complete.
    previous = x[:, :-1]
    changes = x[:, 1:] - previous
    if changes.shape[1] == 0:
        changes = np.full((len(x), 1), np.nan)
        previous = np.full_like(changes, np.nan)
    eligible = np.isfinite(changes) & (previous > 0)
    percentage = np.full(changes.shape, np.nan)
    np.divide(changes, previous, out=percentage, where=eligible)
    zero = valid & (x == 0)
    max_zero, mean_zero, _ = streaks(zero)
    drops = (percentage <= -config["sudden_drop_threshold"]).sum(axis=1)
    f.update({
        "mean_daily_absolute_change": np.nanmean(np.abs(changes), axis=1),
        "median_daily_absolute_change": np.nanmedian(np.abs(changes), axis=1),
        "std_daily_change": np.nanstd(changes, axis=1),
        "max_daily_increase": np.maximum(np.nanmax(changes, axis=1), 0),
        "max_daily_decrease": np.maximum(-np.nanmin(changes, axis=1), 0),
        "mean_daily_percentage_change": np.nanmean(percentage, axis=1),
        "sudden_drop_count": drops, "sudden_drop_ratio": safe_ratio(drops, eligible.sum(axis=1)),
        "sudden_increase_count": (percentage >= config["sudden_increase_threshold"]).sum(axis=1),
        "max_consecutive_zero_days": max_zero, "mean_consecutive_zero_days": mean_zero,
        "recent_30d_mean": np.nanmean(x[:, -30:], axis=1),
        "recent_90d_mean": np.nanmean(x[:, -90:], axis=1),
        "recent_vs_historical_ratio": safe_ratio(np.nanmean(x[:, -90:], axis=1),
                                                 np.nanmean(x[:, :-90], axis=1)),
        "first_half_vs_second_half_ratio": safe_ratio(np.nanmean(x[:, :n // 2], axis=1),
                                                      np.nanmean(x[:, n // 2:], axis=1)),
    })
    # Equal weight per available year-month; elapsed calendar months retain missing gaps.
    months = dates.to_period("M")
    unique_months = months.unique()
    monthly = np.column_stack([np.nanmean(x[:, months == m], axis=1) for m in unique_months])
    time = np.arange(len(unique_months), dtype=float)[None, :]
    available = np.isfinite(monthly)
    tx = np.where(available, time, np.nan)
    centered_t = tx - np.nanmean(tx, axis=1)[:, None]
    centered_y = monthly - np.nanmean(monthly, axis=1)[:, None]
    f["monthly_trend_slope"] = safe_ratio(np.nansum(centered_t * centered_y, axis=1),
                                         np.nansum(centered_t**2, axis=1))
    for lag in (1, 7, 30):
        f[f"autocorrelation_lag_{lag}"] = pair_correlation(x, lag)
    weekday = dates.dayofweek < 5
    f["weekday_mean"] = np.nanmean(x[:, weekday], axis=1)
    f["weekend_mean"] = np.nanmean(x[:, ~weekday], axis=1)
    f["weekday_weekend_ratio"] = safe_ratio(f["weekday_mean"], f["weekend_mean"])
    seasons = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
    for name, month_numbers in seasons.items():
        selector = np.isin(dates.month, month_numbers)
        f[f"{name}_mean"] = np.nanmean(x[:, selector], axis=1)
        f[f"{name}_observed_ratio"] = safe_ratio(valid[:, selector].sum(axis=1), selector.sum())
    month_of_year = np.column_stack([np.nanmean(x[:, dates.month == m], axis=1) for m in range(1, 13)])
    for m in range(1, 13):
        f[f"monthly_mean_{m:02d}"] = month_of_year[:, m - 1]
    f["monthly_mean_std"] = np.nanstd(month_of_year, axis=1)
    f["monthly_mean_range"] = np.nanmax(month_of_year, axis=1) - np.nanmin(month_of_year, axis=1)
    weekday_means = np.column_stack([np.nanmean(x[:, dates.dayofweek == d], axis=1) for d in range(7)])
    weekly_variance = np.nanvar(weekday_means, axis=1)
    # Descriptive weekday between-group variance / total variance; no day-level labels implied.
    f["weekly_pattern_strength"] = np.clip(safe_ratio(weekly_variance, std**2), 0, 1)
    missing_max, _, missing_count = streaks(~valid)
    first = np.argmax(valid, axis=1).astype(float)
    last = (n - 1 - np.argmax(valid[:, ::-1], axis=1)).astype(float)
    first[observed == 0] = np.nan
    last[observed == 0] = np.nan
    f.update({
        "missing_days": n - observed, "missing_ratio": (n - observed) / n,
        "zero_days": zero.sum(axis=1), "zero_ratio": safe_ratio(zero.sum(axis=1), observed),
        "nonzero_days": (valid & (x != 0)).sum(axis=1), "longest_missing_streak": missing_max,
        "number_of_missing_streaks": missing_count, "first_valid_day_index": first,
        "last_valid_day_index": last, "valid_observation_span": last - first + 1,
        "observed_days_ratio": observed / n,
        "has_insufficient_history": (observed < config["insufficient_history_threshold"]).astype(int),
        "all_observations_zero": ((observed > 0) & (zero.sum(axis=1) == observed)).astype(int),
        "all_observations_missing": (observed == 0).astype(int),
        "recent_30d_observed_ratio": valid[:, -30:].mean(axis=1),
        "recent_90d_observed_ratio": valid[:, -90:].mean(axis=1),
    })
    return f


def engineer_features(values: np.ndarray, dates: pd.DatetimeIndex, threshold: float | None,
                      config: dict, batch_size: int = 512) -> pd.DataFrame:
    if values.ndim != 2 or values.shape[1] != len(dates) or not len(dates):
        raise ValueError("Feature matrix must match a nonempty daily calendar")
    if not dates.equals(pd.date_range(dates[0], dates[-1], freq="D")):
        raise ValueError("Temporal features require a complete chronological daily calendar")
    frames = []
    # Empty/constant histories intentionally produce undefined statistics.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start in range(0, len(values), batch_size):
            frames.append(pd.DataFrame(_batch_features(values[start:start + batch_size], dates, threshold, config)))
    result = pd.concat(frames, ignore_index=True).astype(np.float64)
    if not config["include_quality_features"]:
        result = result.drop(columns=sorted(QUALITY_FEATURES & set(result.columns)))
    if not np.isfinite(result.to_numpy()[~result.isna().to_numpy()]).all():
        raise ValueError("Engineered features contain infinity; inspect the raw numeric range")
    if {"CONS_NO", "FLAG", "split", "index", "row_index"} & set(result.columns):
        raise ValueError("Forbidden predictive identifier/target feature")
    return result
