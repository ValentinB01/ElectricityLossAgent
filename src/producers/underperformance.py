"""Underperformance (M2): is a turbine producing less than it should, since when, why, and how many MWh a day is that?

    python -m src.producers.underperformance --train   # Alarm threshold and cause classifier, from Jan-Oct 2019 (about 10 minutes)
    python -m src.producers.underperformance --eval    # Score on faults injected into Nov-Dec 2019, against the fixed-threshold baseline
    python -m src.producers.underperformance --eval --final  # The 2020 test year. Once only, after everything is frozen.

Three pieces (CLAUDE.md 9.3). The 72 h performance index says how much is lost. A two-sided
CUSUM says the drop is persistent, not a bad hour. A classifier on the shape of the deviation
says why, and an alarm it confidently calls 'no fault' is dropped. Each turbine is compared with the rest of the fleet as well as with M1, because most
of a healthy turbine's drift against M1 (up to 1.2% over 72 h) is shared by all six.

Everything M2 learns from is out of sample: Jan-Oct expected values come from M1 refitted without
the two months being predicted, so injected faults look the way they will on new data.
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.common import CONF_MIN, COLORS, MODELS, QUANTILES, SEED, STEP_H, chart_style, plot_confusion, save_figure, write_result
from src.producers import expected_power
from src.simulate import inject_faults as faults

VAL_START, VAL_END = expected_power.TRAIN_END, expected_power.VAL_END
# Detection. Assumptions, fixed before looking at validation.
PI_WINDOW_H = 72
MIN_EXPECTED_FRAC = 0.10            # Only intervals where M1 expects over 10% of rated (CLAUDE.md 9.3).
CUSUM_K = 0.5                       # Allowance, in standard deviations of the hourly deviation.
CUSUM_H_GRID = [4, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 260]
FALSE_ALARMS_PER_TURBINE_MONTH = 0.1  # About one a turbine a year (24 a month at Fantanele's 240). Stricter than CLAUDE.md 9.3's 1, agreed with Valentin.
REFRACTORY_H = 72                   # After an alarm, that turbine stays quiet this long.
BASELINE_PI = 0.95                  # Baseline: alarm when the 72 h index falls below this.
# Cause classifier.
FEATURE_DAYS = 7                    # It reads the week before the moment it classifies...
CLASSIFY_AFTER_H = 48               # ...which is this long after the alarm, to see the shape. Trades delay for accuracy.
TRAIN_PASSES, TRAIN_FAULTS, WINDOWS_PER_FAULT = 6, 30, 2
EVAL_PASSES, EVAL_FAULTS = 8, 8     # Independent fault sets on Nov-Dec, so scores rest on about 48 faults (icing finds no frost), not 6.
CLASSIFIER = {"objective": "multiclass", "num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 5, "seed": SEED, "deterministic": True, "force_row_wise": True, "verbose": -1}
CLASSIFIER_TREES = 300
NO_LOSS = {"anemometer_bias", "none"}  # A wrong wind reading loses no energy: it needs a check, not a repair.
LOSS_ONLY = {"pitch_fault", "yaw_misalignment", "icing"}  # Faults that can only lower output.


# ---------- Expected values that never saw the row ----------
def fold_models(d: pd.DataFrame) -> tuple[list, dict]:
    """M1 refitted five times, each without two of the ten training months, with the tree counts M1 chose."""
    _, meta = expected_power.load_models()
    keep = expected_power.clean_mask(d) & (d.ts < expected_power.TRAIN_END).to_numpy()
    X, y, month = expected_power.build_features(d)[meta["features"]], d.frac.to_numpy(float), d.ts.dt.month.to_numpy()
    folds = []
    for k in range(5):
        tr = keep & ~np.isin(month, [2 * k + 1, 2 * k + 2])
        folds.append({n: expected_power.fit_fixed(X[tr], y[tr], q, meta["quantiles"][n]["best_iteration"]) for n, q in QUANTILES.items()})
        print(f"  M1 without months {2 * k + 1}-{2 * k + 2}: fitted", flush=True)
    return folds, meta


def predict_oof(d: pd.DataFrame, folds: list, meta: dict) -> pd.DataFrame:
    fold = (d.ts.dt.month.to_numpy() - 1) // 2
    return pd.concat([expected_power.predict_expected(d[fold == k], folds[k], meta) for k in range(5) if (fold == k).any()]).sort_index()


# ---------- Detection ----------
def usable(d: pd.DataFrame) -> np.ndarray:
    """Normal operation with enough expected output to compare. Stops, service and curtailment are left out, not called faults."""
    return d.healthy.to_numpy(bool) & d.power_kw.notna().to_numpy() & (d.exp_p50_kw.to_numpy(float) > MIN_EXPECTED_FRAC * d.rated_kw.to_numpy(float))


def hourly(d: pd.DataFrame) -> dict:
    """Hour x turbine tables. z: (real - expected) / rated. z_rel: z minus the median of the other five turbines."""
    use = usable(d)
    f = pd.DataFrame({"h": d.ts.dt.floor("h"), "t": d.turbine, "dev": np.where(use, (d.power_kw - d.exp_p50_kw) / d.rated_kw, np.nan),
                      "p": np.where(use, d.power_kw, np.nan), "e": np.where(use, d.exp_p50_kw, np.nan), "w": d.wind_ms, "pa": d.power_kw.where(d.healthy)})
    g = f.groupby(["h", "t"])
    z = g.dev.mean().unstack("t")
    a, others = z.to_numpy(float), np.full(z.shape, np.nan)
    for j in range(a.shape[1]):
        rest = np.delete(a, j, axis=1)
        ok = ~np.isnan(rest).all(axis=1)
        others[ok, j] = np.nanmedian(rest[ok], axis=1)
    return {"z": z, "z_rel": z - others, "power": g.p.sum(min_count=1).unstack("t"), "expected": g.e.sum(min_count=1).unstack("t"),
            "wind": g.w.mean().unstack("t"), "power_all": g.pa.mean().unstack("t")}


def cusum(x: np.ndarray, h: float) -> list[tuple[int, str]]:
    """Two-sided CUSUM on standardised deviations: (index, 'low' or 'high') per alarm.

    Two-sided because an anemometer reading low makes a healthy turbine look like it
    overproduces. A one-sided test would never see that fault.
    """
    lo = hi = 0.0
    quiet, alarms = 0, []
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        if quiet:
            quiet -= 1
            continue
        lo, hi = max(0.0, lo - v - CUSUM_K), max(0.0, hi + v - CUSUM_K)
        if lo > h or hi > h:
            alarms.append((i, "low" if lo >= hi else "high"))
            lo = hi = 0.0
            quiet = REFRACTORY_H
    return alarms


def scan(hr: dict, sigma: float, h: float, start=None) -> pd.DataFrame:
    """Every alarm from start on. An alarm is stamped at the end of the hour that raised it."""
    z = hr["z_rel"] if start is None else hr["z_rel"][hr["z_rel"].index >= start]
    rows = [(z.index[i] + pd.Timedelta(hours=1), t, side) for t in z.columns for i, side in cusum(z[t].to_numpy() / sigma, h)]
    return pd.DataFrame(rows, columns=["ts", "turbine", "side"])


def pi_table(hr: dict) -> pd.DataFrame:
    """Performance index: real / expected, summed over the trailing 72 h of usable intervals."""
    return hr["power"].rolling(PI_WINDOW_H, min_periods=12).sum() / hr["expected"].rolling(PI_WINDOW_H, min_periods=12).sum()


def baseline_alarms(hr: dict, start=None) -> pd.DataFrame:
    """Baseline (CLAUDE.md 9.3): the 72 h index below 0.95, with no CUSUM and no classifier."""
    pi = pi_table(hr)
    pi = pi if start is None else pi[pi.index >= start]
    rows = []
    for t in pi.columns:
        last = -10**9
        for i in np.flatnonzero((pi[t] < BASELINE_PI).to_numpy()):
            if i - last > REFRACTORY_H:
                rows.append((pi.index[i] + pd.Timedelta(hours=1), t, "low"))
                last = i
    return pd.DataFrame(rows, columns=["ts", "turbine", "side"])


def calibrate(hr: dict) -> dict:
    """The CUSUM threshold: the most sensitive one with at most one alarm per turbine-month on fault-free Jan-Oct.

    Real faults nobody labelled count as false alarms here, so the threshold errs on the quiet side.
    """
    sigma = float(np.nanstd(hr["z_rel"].to_numpy()))
    turbine_months = len(hr["z_rel"]) / (24 * 30.44) * hr["z_rel"].shape[1]
    curve, chosen = [], None
    for h in CUSUM_H_GRID:
        rate = len(scan(hr, sigma, h)) / turbine_months
        curve.append({"h": h, "alarms_per_turbine_month": round(rate, 3)})
        if chosen is None and rate <= FALSE_ALARMS_PER_TURBINE_MONTH:
            chosen = h
    return {"sigma": sigma, "h": chosen or CUSUM_H_GRID[-1], "curve": curve}


# ---------- Why: the shape of the deviation ----------
BANDS = [(0, 6), (6, 9), (9, 12), (12, 99)]
FEATURES = [*(f"loss_wind_{a}_{b}" for a, b in BANDS), "pi_window", "below_p10", "above_p90", "cold_humid_share", "cold_humid_loss", "other_loss",
            "pi_change", "pi_daily_std", "high_wind_p95", "high_wind_std", "z_mean", "z_rel_mean", "power_vs_fleet_change", "wind_vs_fleet_change"]


def _ratio(a, b) -> float:
    s = float(np.nansum(b))
    return float(np.nansum(a)) / s if s > 0 else np.nan


def _rel_loss(p, e) -> float:
    s = float(np.nansum(e))
    return (s - float(np.nansum(p))) / s if s > 0 else np.nan


def _vs_fleet(table: pd.DataFrame, turbine: str, t0, t1, prior_days: int = 30) -> float:
    """This turbine against the median of the others in the window, relative to the month before. Fixed offsets, like hub height, cancel."""
    others = table.drop(columns=turbine).median(axis=1).to_numpy()
    mine = table[turbine].to_numpy()

    def ratio(a, b):
        m = (table.index >= a) & (table.index < b)
        return _ratio(mine[m], others[m])
    now, before = ratio(t0, t1), ratio(t0 - pd.Timedelta(days=prior_days), t0)
    return now / before - 1 if np.isfinite(now) and np.isfinite(before) and before else 0.0


def shape_features(rows: pd.DataFrame, hr: dict, turbine: str, t: pd.Timestamp) -> dict:
    """Where the loss sits (wind band, cold and humid hours), how it started (abrupt or gradual), and how the turbine compares with the fleet."""
    t0 = t - pd.Timedelta(days=FEATURE_DAYS)
    P, E, W = rows.power_kw.to_numpy(float), rows.exp_p50_kw.to_numpy(float), rows.wind_ms.to_numpy(float)
    rated = float(rows.rated_kw.iloc[0])
    f = {f"loss_wind_{a}_{b}": _rel_loss(P[(W >= a) & (W < b)], E[(W >= a) & (W < b)]) for a, b in BANDS}
    f["pi_window"] = _ratio(P, E)
    f["below_p10"], f["above_p90"] = float((P < rows.exp_p10_kw.to_numpy(float)).mean()), float((P > rows.exp_p90_kw.to_numpy(float)).mean())
    cold = faults.icing_conditions(rows)
    f["cold_humid_share"], f["cold_humid_loss"], f["other_loss"] = float(cold.mean()), _rel_loss(P[cold], E[cold]), _rel_loss(P[~cold], E[~cold])
    daily = pd.DataFrame({"d": rows.ts.dt.floor("D").to_numpy(), "p": P, "e": E}).groupby("d").sum()
    pi_d = (daily.p / daily.e.replace(0, np.nan)).dropna()
    half = len(pi_d) // 2
    f["pi_change"] = float(pi_d.iloc[half:].mean() - pi_d.iloc[:half].mean()) if half else 0.0
    f["pi_daily_std"] = float(pi_d.std()) if len(pi_d) > 1 else 0.0
    windy = W > faults.PITCH_WIND_MS
    f["high_wind_p95"] = float(np.nanpercentile(P[windy] / rated, 95)) if windy.sum() >= 5 else np.nan
    f["high_wind_std"] = float(np.nanstd(P[windy] / rated)) if windy.sum() >= 5 else np.nan
    m = (hr["z"].index >= t0) & (hr["z"].index < t)
    for k in ("z", "z_rel"):
        v = hr[k][turbine].to_numpy()[m]
        f[f"{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    # A wrong wind reading: power keeps pace with the neighbours, the wind reading does not.
    f["power_vs_fleet_change"], f["wind_vs_fleet_change"] = _vs_fleet(hr["power_all"], turbine, t0, t), _vs_fleet(hr["wind"], turbine, t0, t)
    return f


class Series:
    """One turbine's rows between two times, fast."""

    def __init__(self, d: pd.DataFrame):
        self.rows = {t: g.sort_values("ts").reset_index(drop=True) for t, g in d.groupby("turbine")}
        self.ts = {t: g.ts.to_numpy("datetime64[ns]") for t, g in self.rows.items()}

    def window(self, turbine: str, t0, t1) -> pd.DataFrame:
        ts = self.ts[turbine]
        return self.rows[turbine].iloc[np.searchsorted(ts, np.datetime64(t0), "left"): np.searchsorted(ts, np.datetime64(t1), "left")]


def training_windows(pred: pd.DataFrame, labels: pd.DataFrame, hr: dict, rng) -> tuple[list, list]:
    """Weeks inside faults, labelled with the cause, plus weeks with no fault ('none')."""
    series, X, y = Series(pred), [], []
    for f in labels[labels.detectable].itertuples():
        earliest = f.start + pd.Timedelta(hours=CLASSIFY_AFTER_H)
        for _ in range(WINDOWS_PER_FAULT):
            t = (f.end if earliest >= f.end else earliest + (f.end - earliest) * float(rng.uniform())).floor("h")
            rows = series.window(f.turbine, t - pd.Timedelta(days=FEATURE_DAYS), t)
            rows = rows[usable(rows)]
            if len(rows) >= 12:
                X.append(shape_features(rows, hr, f.turbine, t))
                y.append(f.cause)
    want, lo, hi, tries = max(len(y) // max(labels.cause.nunique(), 1), 4), pred.ts.min(), pred.ts.max(), 0
    while y.count("none") < want and tries < want * 30:
        tries += 1
        turbine = sorted(series.rows)[int(rng.integers(len(series.rows)))]
        t = (lo + (hi - lo) * float(rng.uniform(0.15, 1))).floor("h")
        if ((labels.turbine == turbine) & (labels.start < t) & (labels.end > t - pd.Timedelta(days=FEATURE_DAYS))).any():
            continue
        rows = series.window(turbine, t - pd.Timedelta(days=FEATURE_DAYS), t)
        rows = rows[usable(rows)]
        if len(rows) >= 12:
            X.append(shape_features(rows, hr, turbine, t))
            y.append("none")
    return X, y


def loss_per_day(rows72: pd.DataFrame, hr: dict, turbine: str, truth: bool = False) -> tuple[float, float]:
    """MWh lost per day over the window, and the share of intervals usable for it.

    Measured against M1 shifted by the other turbines' median deviation, which removes what the whole
    fleet shares, such as M1 reading low in winter. Divided by all normal-operation hours, calm ones
    included: dividing by just the windy hours would overstate the daily rate. truth=True sums the injected loss instead.
    """
    if not len(rows72):
        return 0.0, 0.0
    use = usable(rows72)
    normal_days = float(rows72.healthy.sum()) * STEP_H / 24
    if truth:
        kwh = float(np.nansum(rows72.true_loss_kw.to_numpy(float))) * STEP_H
    else:
        fleet = (hr["z"][turbine] - hr["z_rel"][turbine]).reindex(rows72.ts.dt.floor("h")).fillna(0).to_numpy()
        kwh = float(np.nansum(((rows72.exp_p50_kw - rows72.power_kw).to_numpy(float) + fleet * rows72.rated_kw.to_numpy(float))[use])) * STEP_H
    return (max(0.0, kwh / 1000 / normal_days) if normal_days > 0 else 0.0), float(use.mean())


def assess(series: Series, hr: dict, turbine: str, t: pd.Timestamp, side: str, booster, meta: dict) -> dict:
    """M2's answer for one alarm (CLAUDE.md 9.3, step 4): index, MWh lost per day, cause, confidence, and the top 3 reasons."""
    w72 = series.window(turbine, t - pd.Timedelta(hours=PI_WINDOW_H), t)
    u = usable(w72)
    pi = _ratio(w72.power_kw.to_numpy(float)[u], w72.exp_p50_kw.to_numpy(float)[u])
    loss, coverage = loss_per_day(w72, hr, turbine)
    rows = series.window(turbine, t - pd.Timedelta(days=FEATURE_DAYS), t)
    rows = rows[usable(rows)]
    out = {"turbine": turbine, "ts": str(t), "side": side, "pi_72h": None if not np.isfinite(pi) else round(pi, 4), "coverage_72h": round(coverage, 3)}
    if len(rows) < 12:
        return {**out, "loss_mwh_per_day": round(loss, 3), "cause": "unknown", "cause_proba": 0.0, "confidence": 0.0, "reasons": []}
    ruled_out = []
    if "icing" in meta["classes"] and not faults.icing_conditions(rows).any():
        ruled_out.append({"icing"})  # No cold, humid hour in the week: icing is physically impossible.
    if side == "high":
        ruled_out.append(LOSS_ONLY)  # A turbine producing more than expected cannot have a fault that only loses power.
    cause, proba, reasons = classify(booster, meta["classes"], meta["features"], shape_features(rows, hr, turbine, t), ruled_out)
    return {**out, "loss_mwh_per_day": 0.0 if cause in NO_LOSS else round(loss, 3), "cause": cause, "cause_proba": round(proba, 3),
            "confidence": round(proba * min(1.0, coverage / 0.25), 3), "reasons": reasons}


def classify(booster, classes: list, features: list, f: dict, ruled_out: list) -> tuple[str, float, list]:
    """The most likely cause, its probability, and the top 3 reasons (TreeSHAP contributions). Shared with the solar branch.

    ruled_out: sets of causes that are physically impossible here; each set is zeroed and the rest renormalised, in order.
    """
    X = pd.DataFrame([f], columns=features).to_numpy(float)
    proba = booster.predict(X)[0]
    for causes in ruled_out:
        for c in causes & set(classes):
            proba[classes.index(c)] = 0.0
        proba = proba / proba.sum()
    best = int(np.argmax(proba))
    contrib = booster.predict(X, pred_contrib=True).reshape(len(classes), len(features) + 1)[best, :-1]
    reasons = [{"feature": features[j], "value": None if np.isnan(X[0, j]) else round(float(X[0, j]), 3), "contribution": round(float(contrib[j]), 3)} for j in np.argsort(-np.abs(contrib))[:3]]
    return classes[best], float(proba[best]), reasons


def detect_as_of(pred: pd.DataFrame, as_of: pd.Timestamp, active_days: int = 14) -> list[dict]:
    """Open findings at a moment, from data before it only: one per turbine whose latest alarm is recent and survives the filter.

    pred: rows with M1's expected values, from some weeks before as_of (the alarm detector starts at the first row).
    """
    booster, meta = load_models()
    hist = pred[pred.ts < as_of]
    hr, series = hourly(hist), Series(hist)
    th = meta["thresholds"]
    out = []
    for turbine, g in scan(hr, th["sigma"], th["h"]).groupby("turbine"):
        last = g.sort_values("ts").iloc[-1]
        if last.ts < as_of - pd.Timedelta(days=active_days):
            continue
        a = assess(series, hr, turbine, as_of, last.side, booster, meta)
        if not dismissed(a):
            out.append({**a, "alarm_ts": str(last.ts)})
    return sorted(out, key=lambda a: -a["loss_mwh_per_day"])


# ---------- Train and evaluate ----------
def _sorted(d: pd.DataFrame) -> pd.DataFrame:
    return d.sort_values(["turbine", "ts"]).reset_index(drop=True)


def train() -> dict:
    d = expected_power.load()
    print("Refitting M1 five times for out-of-sample expected values...", flush=True)
    folds, meta1 = fold_models(d)
    period = _sorted(d[d.ts < expected_power.TRAIN_END])
    hr = hourly(predict_oof(period, folds, meta1))
    th = calibrate(hr)
    print(f"CUSUM threshold {th['h']}: {th['curve']}", flush=True)
    X, y = [], []
    for k in range(TRAIN_PASSES):
        rng = np.random.default_rng([SEED, 1, k])
        faulted, labels = faults.apply(period, faults.sample(period, TRAIN_FAULTS, period.ts.min(), expected_power.TRAIN_END, rng))
        pred = predict_oof(faulted, folds, meta1)
        Xk, yk = training_windows(pred, labels, hourly(pred), rng)
        X, y = X + Xk, y + yk
        print(f"  fault set {k + 1}/{TRAIN_PASSES}: {len(labels)} faults, {len(y)} windows so far", flush=True)
    classes = sorted(set(y))
    booster = lgb.train({**CLASSIFIER, "num_class": len(classes)}, lgb.Dataset(pd.DataFrame(X, columns=FEATURES).to_numpy(float), np.array([classes.index(c) for c in y]), feature_name=FEATURES),
                        num_boost_round=CLASSIFIER_TREES)
    MODELS.mkdir(exist_ok=True)
    booster.save_model(str(MODELS / "wind_cause.txt"))
    meta = {"classes": classes, "features": FEATURES, "thresholds": th, "windows": len(y), "class_counts": {c: y.count(c) for c in classes}, "lightgbm": lgb.__version__, "seed": SEED}
    (MODELS / "wind_cause_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_models() -> tuple:
    return lgb.Booster(model_file=str(MODELS / "wind_cause.txt")), json.loads((MODELS / "wind_cause_meta.json").read_text())


def _bucket(f) -> str:
    if f.cause == "anemometer_bias":
        return "wrong wind reading (no loss)"
    return "loss under 2%" if f.loss_share < 0.02 else "loss 2-5%" if f.loss_share < 0.05 else "loss over 5%"


def _match(alarms: pd.DataFrame, labels: pd.DataFrame) -> tuple[dict, int]:
    """First alarm inside each fault (up to a day after its end) is a detection; an alarm outside every fault is false."""
    hits = {}
    for f in labels[labels.detectable].itertuples():
        a = alarms[(alarms.turbine == f.turbine) & (alarms.ts > f.start) & (alarms.ts <= f.end + pd.Timedelta(hours=24))]
        if len(a):
            hits[f.fault_id] = a.sort_values("ts").iloc[0]
    false = sum(not ((labels.turbine == r.turbine) & (labels.start < r.ts) & (labels.end + pd.Timedelta(hours=24) >= r.ts)).any() for r in alarms.itertuples())
    return hits, false


def _summary(rows: list, false: int, turbine_months: float) -> dict:
    r = pd.DataFrame(rows)
    det = r[r.detected]
    by = lambda col: {k: f"{int(g.detected.sum())}/{len(g)}" for k, g in r.groupby(col)}
    return {"faults": len(r), "detected": int(r.detected.sum()), "detection_rate": round(float(r.detected.mean()), 3), "mean_delay_days": round(float(det.delay_days.mean()), 2) if len(det) else None,
            "false_alarms": int(false), "false_alarms_per_turbine_month": round(false / turbine_months, 3), "by_cause": by("cause"), "by_size": by("size")}


def _assess_alarms(alarms: pd.DataFrame, series: Series, hr: dict, labels, booster, meta: dict, end: pd.Timestamp) -> list[dict]:
    """M2's answer for each alarm, given CLASSIFY_AFTER_H later. An injected fault ends by itself, a real one waits for repair, so the look is cut at the fault's end."""
    out = []
    for r in alarms.itertuples():
        t = r.ts + pd.Timedelta(hours=CLASSIFY_AFTER_H)
        if labels is not None:
            f = labels[(labels.turbine == r.turbine) & (labels.start < r.ts) & (labels.end + pd.Timedelta(hours=24) >= r.ts)]
            if len(f):
                t = max(r.ts, min(t, f.end.iloc[0]))
        out.append(assess(series, hr, r.turbine, min(t, end).ceil("h"), r.side, booster, meta))
    return out


def dismissed(a: dict) -> bool:
    """The filter: an alarm the classifier confidently calls 'no fault' raises no job."""
    return a["cause"] == "none" and a["confidence"] >= CONF_MIN


def evaluate(final: bool = False) -> dict:
    """Validation (Nov-Dec 2019) or, with final, the 2020 test year: run once, after everything is frozen."""
    lo, hi, period_name = (VAL_END, pd.Timestamp("2021-01-01"), "test_2020") if final else (VAL_START, VAL_END, "val_2019_nov_dec")
    n_faults = max(EVAL_FAULTS, round(EVAL_FAULTS * (hi - lo).days / (VAL_END - VAL_START).days))  # Same density of faults as on validation.
    booster, meta = load_models()
    th = meta["thresholds"]
    boosters1, meta1 = expected_power.load_models()
    d = expected_power.load((2019, 2020) if final else (2019,))
    # Five weeks of lead-in so the first alarms in November have their feature window and month-before comparison.
    period = _sorted(d[(d.ts >= lo - pd.Timedelta(days=37)) & (d.ts < hi)])
    clean = expected_power.predict_expected(period, boosters1, meta1)
    clean_hr = hourly(clean)
    real_alarms = scan(clean_hr, th["sigma"], th["h"], lo)
    real = [{**a, "dismissed_by_filter": dismissed(a)} for a in _assess_alarms(real_alarms, Series(clean), clean_hr, None, booster, meta, hi)]
    turbine_months = (hi - lo).days / 30.44 * clean_hr["z"].shape[1] * EVAL_PASSES
    systems = ("model", "model_without_filter", "baseline")
    rows, false = {s: [] for s in systems}, {s: 0 for s in systems}
    y_true, y_pred, loss_err, loss_true, example = [], [], [], [], None
    for k in range(EVAL_PASSES):
        rng = np.random.default_rng([SEED, 2, k])
        faulted, labels = faults.apply(period, faults.sample(period, n_faults, lo, hi, rng))
        pred = expected_power.predict_expected(faulted, boosters1, meta1)
        hr, series = hourly(pred), Series(pred)
        raw = scan(hr, th["sigma"], th["h"], lo)
        assessed = _assess_alarms(raw, series, hr, labels, booster, meta, hi)
        kept = raw[[not dismissed(a) for a in assessed]]
        for name, alarms in (("model", kept), ("model_without_filter", raw), ("baseline", baseline_alarms(hr, lo))):
            hits, n_false = _match(alarms, labels)
            false[name] += n_false
            for f in labels[labels.detectable].itertuples():
                hit = hits.get(f.fault_id)
                rows[name].append({"cause": f.cause, "size": _bucket(f), "detected": hit is not None, "delay_days": None if hit is None else (hit.ts - f.start).total_seconds() / 86400})
                if name != "model" or hit is None:
                    continue
                out = assessed[hit.name]
                truth, _ = loss_per_day(series.window(f.turbine, pd.Timestamp(out["ts"]) - pd.Timedelta(hours=PI_WINDOW_H), pd.Timestamp(out["ts"])), hr, f.turbine, truth=True)
                y_true.append(f.cause)
                y_pred.append(out["cause"])
                loss_err.append(abs(out["loss_mwh_per_day"] - truth))
                loss_true.append(truth)
                if example is None and f.cause == "pitch_fault":
                    example = {"pred": pred, "fault": f, "alarm": hit.ts, "assessment": out}
    labels_seen = sorted(set(y_true) | set(y_pred))
    res = {"period": period_name, "note": "Faults are synthetic, injected into real data. Real faults nobody labelled count as false alarms. " + ("" if final else "No icing: Nov-Dec 2019 had no frost."),
           "cusum_threshold": th["h"], "false_alarm_target_per_turbine_month": FALSE_ALARMS_PER_TURBINE_MONTH, "fault_sets": EVAL_PASSES,
           "model": _summary(rows["model"], false["model"], turbine_months), "model_without_filter": _summary(rows["model_without_filter"], false["model_without_filter"], turbine_months),
           "baseline_pi_below_0.95": _summary(rows["baseline"], false["baseline"], turbine_months),
           "cause_macro_f1": round(float(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)), 3),
           "cause_accuracy": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 3),
           "confusion_labels": labels_seen, "confusion_rows_true_cols_predicted": confusion_matrix(y_true, y_pred, labels=labels_seen).tolist(),
           "loss_mae_mwh_per_day": round(float(np.mean(loss_err)), 3), "mean_true_loss_mwh_per_day": round(float(np.mean(loss_true)), 3),
           "real_data_alarms_no_injection": real, "example": None if example is None else example["assessment"]}
    res["detects_more_than_baseline"] = res["model"]["detection_rate"] > res["baseline_pi_below_0.95"]["detection_rate"]
    write_result(f"wind_underperformance_{period_name}", res)
    _plot(res, example, period_name, lo)
    return res


def _plot(res: dict, example, period_name: str, lo: pd.Timestamp) -> None:
    import matplotlib.dates as mdates
    plt = chart_style()
    C = COLORS
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.1, 0.9, 1.4]})
    ax = axes[0]
    sizes = ["loss under 2%", "loss 2-5%", "loss over 5%", "wrong wind reading (no loss)"]
    rate = lambda key, s: detected_pct(res[key]["by_size"].get(s, "0/0"))
    pos = np.arange(len(sizes))
    b, m = res["baseline_pi_below_0.95"], res["model"]
    ax.bar(pos - 0.2, [rate("baseline_pi_below_0.95", s) for s in sizes], 0.38, color=C["orange"], label=f"Baseline: index < 0.95 ({b['false_alarms_per_turbine_month']} false alarms/turbine-month)")
    ax.bar(pos + 0.2, [rate("model", s) for s in sizes], 0.38, color=C["blue"], label=f"M2 ({m['false_alarms_per_turbine_month']} false alarms/turbine-month)")
    ax.set_xticks(pos, ["loss\nunder 2%", "loss\n2-5%", "loss\nover 5%", "wrong wind\nreading (no loss)"], fontsize=9)
    ax.set_ylim(0, 125); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="x", visible=False)
    ax.set_title("Faults detected, by size: higher is better"); ax.set_ylabel("Detected (%)"); ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    lab = res["confusion_labels"]
    plot_confusion(ax, lab, res["confusion_rows_true_cols_predicted"], [s.replace("_", " ").replace("misalignment", "error").replace("anemometer bias", "wind reading").replace("none", "no fault") for s in lab])
    ax.grid(False); ax.set_title("Cause of detected faults"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax = axes[2]
    if example is not None:
        p, f, a = example["pred"], example["fault"], example["assessment"]
        u = usable(p)
        g = pd.DataFrame({"d": p.ts.dt.floor("D"), "t": p.turbine, "real": np.where(u, p.power_kw, 0.0), "exp": np.where(u, p.exp_p50_kw, 0.0)})
        g = g[g.d >= lo]
        ratio = lambda x: (x.real / x.exp * 100).where(x.exp * STEP_H / 1000 >= 2)  # Calm days (under 2 MWh expected) say little.
        mine, rest = g[g.t == f.turbine].groupby("d")[["real", "exp"]].sum(), g[g.t != f.turbine].groupby("d")[["real", "exp"]].sum() / 5
        ax.axvspan(f.start, f.end, color=C["orange"], alpha=0.14, linewidth=0, label=f"Injected pitch fault: capped at {f.severity['k']:.0%} of rated")
        ax.axhline(100, color=C["muted"], linewidth=1)
        ax.plot(rest.index, ratio(rest), color=C["muted"], linewidth=1.2, linestyle=":", label="Other five turbines")
        ax.plot(mine.index, ratio(mine), color=C["blue"], linewidth=2, marker="o", markersize=3, label=f"{f.turbine}")
        ax.axvline(example["alarm"], color=C["ink"], linestyle="--", linewidth=1)
        ax.annotate(f"alarm: {a['cause'].replace('_', ' ')} ({a['cause_proba']:.0%})\n{a['loss_mwh_per_day']} MWh/day lost", (example["alarm"], 0.04), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=C["ink"])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b")); ax.tick_params(axis="x", labelsize=8)
        ax.set_title("Example: real / expected each day, one fault set"); ax.set_ylabel("Real / M1 expected (%)"); ax.legend(fontsize=8, loc="upper left", ncol=2)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (hi - lo) * 0.25)
    fig.tight_layout()
    save_figure(fig, f"wind_underperformance_{period_name}")


def detected_pct(hits: str) -> float:
    """'detected/total', as written in the summaries, to a percentage."""
    n, total = (int(x) for x in hits.split("/"))
    return n / max(total, 1) * 100


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--final", action="store_true", help="With --eval: score the 2020 test year. Once only.")
    a = ap.parse_args()
    if a.train:
        m = train()
        print(json.dumps({k: m[k] for k in ("classes", "windows", "class_counts")} | {"cusum_h": m["thresholds"]["h"]}, indent=1))
    if a.eval:
        r = evaluate(a.final)
        print(json.dumps({k: r[k] for k in ("model", "model_without_filter", "baseline_pi_below_0.95", "detects_more_than_baseline", "cause_macro_f1", "cause_accuracy", "loss_mae_mwh_per_day", "mean_true_loss_mwh_per_day")}, indent=1))
