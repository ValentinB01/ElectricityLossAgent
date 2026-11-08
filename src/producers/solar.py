"""Solar (PV) branch: expected production, underperformance and downtime cost (CLAUDE.md 9.8).

    python -m src.producers.solar --train            # M1-PV, its forecast-only sibling, and M2-PV, from the first 22 days (about a minute)
    python -m src.producers.solar --eval             # Validation: the next 5 days
    python -m src.producers.solar --eval --final     # Test: the last 7 days. Once only, after everything is frozen.

Kaggle Solar Power Generation Data: two plants in India, 22 inverters each, 34 days at 15 minutes. The set
is short, from one season, with no location and no rain record, so results are indicative (README).

Same structure as wind. M1-PV predicts AC power as a share of rated from irradiance, module temperature,
time of day and inverter. Its forecast-only sibling uses irradiance and air temperature, which a forecast
gives, so it can cost a stop at Calugareni. M2-PV reuses M2's machinery: each inverter is compared with
M1-PV and with the other inverters of its plant, a two-sided CUSUM raises alarms, and a classifier names
the cause from the shape of the loss (by irradiance, by hour, and over the days).
"""
import argparse
import datetime as dt
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.common import COLORS, MODELS, PV_DIR, QUANTILES, SEED, chart_style, plot_confusion, save_figure, write_result
from src.producers import downtime_cost, expected_power, underperformance
from src.simulate import inject_faults as faults

DATA = PV_DIR / "pv_inverters.parquet"
TRAIN_END, VAL_END, TEST_END = pd.Timestamp("2020-06-06"), pd.Timestamp("2020-06-11"), pd.Timestamp("2020-06-18")  # 22 / 5 / 7 days (CLAUDE.md 9.8)
GAMMA = -0.004             # Module power temperature coefficient per degC, for the baseline. Assumption: real value from the datasheet.
MIN_IRR_WM2 = 50           # Daylight: below this, expected output is set to 0 and the row is not used.
FEATURES = ["irr_wm2", "t_mod_c", "irr_corr", "hour_sin", "hour_cos", "asset_code"]
FORECAST_FEATURES = ["irr_wm2", "t_amb_c"]  # What a weather forecast gives.
# M2-PV. Faults here last 2-7 days, so the windows are shorter than for wind.
FEATURE_DAYS, CLASSIFY_AFTER_H, FAULT_DAYS = 3, 24, (2, 7)
TRAIN_PASSES, TRAIN_FAULTS, WINDOWS_PER_FAULT = 8, 24, 2
EVAL_PASSES, EVAL_FAULTS = 8, 16
FOLDS = 4                  # M1-PV refitted without each block of training days, for out-of-sample residuals.
IRR_BANDS, HOUR_BANDS = [(50, 300), (300, 700), (700, 2000)], [(6, 9), (9, 12), (12, 15), (15, 19)]
PV_FEATURES = [f"loss_irr_{a}_{b}" for a, b in IRR_BANDS] + ["pi_window", "hour_loss_max", "hour_loss_spread", "high_irr_p95", "high_irr_std",
                                                             "pi_slope", "pi_change", "pi_daily_std", "below_p10", "z_rel_mean", "rain_in_window"]
PV_FORECAST_VARS = "shortwave_radiation,temperature_2m,wind_gusts_10m,precipitation"
CLEANING_RAIN_MM = 5.0     # A day with this much rain washes the panels. Assumption.


# ---------- Data and M1-PV ----------
def load() -> pd.DataFrame:
    """The prepared rows, named so M2's and the fault injector's functions work on them unchanged."""
    if not DATA.exists():
        raise SystemExit(f"{DATA} is missing. Run: python -m src.data.prepare --pv")
    d = pd.read_parquet(DATA)
    return d.assign(turbine=d.inverter, power_kw=d.ac_kw.astype(float), wind_ms=d.irr_wm2.astype(float), frac=d.ac_kw / d.rated_kw)


def features(d: pd.DataFrame, codes: dict) -> pd.DataFrame:
    hour = 2 * np.pi * (d.ts.dt.hour + d.ts.dt.minute / 60).to_numpy(float) / 24
    irr, t = d.irr_wm2.to_numpy(float), d.t_mod_c.to_numpy(float)
    return pd.DataFrame({"irr_wm2": irr, "t_mod_c": t, "irr_corr": irr * (1 + GAMMA * (t - 25)), "hour_sin": np.sin(hour), "hour_cos": np.cos(hour),
                         "asset_code": d.inverter.map(codes).to_numpy(float), "t_amb_c": d.t_amb_c.to_numpy(float)}, index=d.index)


def clean(d: pd.DataFrame) -> np.ndarray:
    return (d.healthy & d.ac_kw.notna() & (d.irr_wm2 > MIN_IRR_WM2)).to_numpy()


def baseline_frac(irr, t_mod) -> np.ndarray:
    """CLAUDE.md 9.1: rated x irradiance/1000 x (1 + gamma (T_mod - 25)), as a share of rated."""
    return np.asarray(irr, float) / 1000 * (1 + GAMMA * (np.asarray(t_mod, float) - 25))


def predict(d: pd.DataFrame, boosters: dict, meta: dict, cols=FEATURES) -> pd.DataFrame:
    """Adds exp_p10_kw, exp_p50_kw, exp_p90_kw. Night (under MIN_IRR_WM2) is 0, so M2 never looks at it."""
    X = features(d, meta["codes"])[cols].to_numpy(float)
    q = np.column_stack([boosters[n].predict(X) for n in QUANTILES])
    q.sort(axis=1)
    q = q * d.rated_kw.to_numpy(float)[:, None]
    q[~(d.irr_wm2.to_numpy(float) > MIN_IRR_WM2)] = 0.0
    return d.assign(exp_p10_kw=q[:, 0], exp_p50_kw=q[:, 1], exp_p90_kw=q[:, 2])


def load_models(prefix: str = "pv_expected") -> tuple[dict, dict]:
    """pv_expected: M1-PV, from the inverter's own sensors. pv_weather: its forecast-only sibling, for the cost of a stop."""
    meta = json.loads((MODELS / "pv_meta.json").read_text())
    return {n: lgb.Booster(model_file=str(MODELS / f"{prefix}_{n}.txt")) for n in QUANTILES}, meta


# ---------- M2-PV ----------
def hourly_by_plant(pred: pd.DataFrame) -> dict:
    """M2's hour x inverter tables, one set per plant: each inverter is compared with its own plant's others."""
    return {p: underperformance.hourly(g) for p, g in pred.groupby("plant")}


def scan(hrs: dict, th: dict, start=None) -> pd.DataFrame:
    parts = [x for x in (underperformance.scan(hr, th["sigma"][str(p)], th["h"], start) for p, hr in hrs.items()) if len(x)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["ts", "turbine", "side"])


def calibrate(hrs: dict) -> dict:
    """Most sensitive threshold with at most m2's target of false alarms per inverter-month, on out-of-sample training days.

    The scale of normal deviation is robust (1.4826 x median absolute deviation): a few inverters with real
    faults, as in Plant 2, inflate a standard deviation up to seven times and would make the alarm deaf.
    """
    def robust(z):
        z = z[np.isfinite(z)]
        return float(1.4826 * np.median(np.abs(z - np.median(z))))
    sigma = {str(p): robust(hr["z_rel"].to_numpy().ravel()) for p, hr in hrs.items()}
    # The false-alarm rate is set on inverters that look healthy. Those swinging over 3 x the plant's robust scale
    # have real faults (outages, weak strings): their alarms are real, and would push the threshold to the top.
    quiet = {p: [c for c in hr["z_rel"].columns if hr["z_rel"][c].std() <= 3 * sigma[str(p)]] for p, hr in hrs.items()}
    calm = {p: {**hr, "z_rel": hr["z_rel"][quiet[p]]} for p, hr in hrs.items()}
    months = sum(len(hr["z_rel"]) / (24 * 30.44) * hr["z_rel"].shape[1] for hr in calm.values())
    curve, chosen = [], None
    for h in underperformance.CUSUM_H_GRID:
        rate = len(scan(calm, {"sigma": sigma, "h": h})) / months
        curve.append({"h": h, "alarms_per_inverter_month": round(rate, 3)})
        if chosen is None and rate <= underperformance.FALSE_ALARMS_PER_TURBINE_MONTH:
            chosen = h
    return {"sigma": sigma, "h": chosen or underperformance.CUSUM_H_GRID[-1], "curve": curve, "left_out_of_calibration": {str(p): sorted(set(hr["z_rel"].columns) - set(quiet[p])) for p, hr in hrs.items()}}


def shape_features(rows: pd.DataFrame, hr: dict, inverter: str, t: pd.Timestamp) -> dict:
    """Where the loss sits (irradiance band: a derate shows only at high irradiance; hour of day: shading) and how it moves over the days (soiling grows, a string outage is a step)."""
    P, E, irr = rows.power_kw.to_numpy(float), rows.exp_p50_kw.to_numpy(float), rows.irr_wm2.to_numpy(float)
    rated = float(rows.rated_kw.iloc[0])
    f = {f"loss_irr_{a}_{b}": underperformance._rel_loss(P[(irr >= a) & (irr < b)], E[(irr >= a) & (irr < b)]) for a, b in IRR_BANDS}
    f["pi_window"] = underperformance._ratio(P, E)
    h = rows.ts.dt.hour.to_numpy()
    band = np.array([underperformance._rel_loss(P[(h >= a) & (h < b)], E[(h >= a) & (h < b)]) for a, b in HOUR_BANDS], float)
    f["hour_loss_max"] = float(np.nanmax(band)) if np.isfinite(band).any() else np.nan
    f["hour_loss_spread"] = float(np.nanmax(band) - np.nanmedian(band)) if np.isfinite(band).any() else np.nan
    hi = irr > 700
    f["high_irr_p95"] = float(np.nanpercentile(P[hi] / rated, 95)) if hi.sum() >= 5 else np.nan
    f["high_irr_std"] = float(np.nanstd(P[hi] / rated)) if hi.sum() >= 5 else np.nan
    daily = pd.DataFrame({"d": rows.ts.dt.floor("D").to_numpy(), "p": P, "e": E}).groupby("d").sum()
    pi_d = (daily.p / daily.e.replace(0, np.nan)).dropna()
    f["pi_slope"] = float(np.polyfit(np.arange(len(pi_d)), pi_d.to_numpy(), 1)[0]) if len(pi_d) >= 2 else 0.0
    half = len(pi_d) // 2
    f["pi_change"] = float(pi_d.iloc[half:].mean() - pi_d.iloc[:half].mean()) if half else 0.0
    f["pi_daily_std"] = float(pi_d.std()) if len(pi_d) > 1 else 0.0
    f["below_p10"] = float((P < rows.exp_p10_kw.to_numpy(float)).mean())
    m = (hr["z_rel"].index >= t - pd.Timedelta(days=FEATURE_DAYS)) & (hr["z_rel"].index < t)
    v = hr["z_rel"][inverter].to_numpy()[m]
    f["z_rel_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    f["rain_in_window"] = float(bool(faults.rain_days(rows.ts.dt.date) & set(rows.ts.dt.date)))
    return f


def assess(series: underperformance.Series, hr: dict, inverter: str, t: pd.Timestamp, side: str, booster, meta: dict) -> dict:
    """M2-PV's answer for one alarm: index, MWh lost per day (against the plant's other inverters), cause, confidence, top 3 reasons."""
    w72 = series.window(inverter, t - pd.Timedelta(hours=underperformance.PI_WINDOW_H), t)
    u = underperformance.usable(w72)
    pi = underperformance._ratio(w72.power_kw.to_numpy(float)[u], w72.exp_p50_kw.to_numpy(float)[u])
    loss, coverage = underperformance.loss_per_day(w72, hr, inverter)
    rows = series.window(inverter, t - pd.Timedelta(days=FEATURE_DAYS), t)
    rows = rows[underperformance.usable(rows)]
    out = {"turbine": inverter, "ts": str(t), "side": side, "pi_72h": None if not np.isfinite(pi) else round(pi, 4), "coverage_72h": round(coverage, 3)}
    if len(rows) < 8:
        return {**out, "loss_mwh_per_day": round(loss, 3), "cause": "unknown", "cause_proba": 0.0, "confidence": 0.0, "reasons": []}
    ruled_out = [set(faults.PV_CAUSES)] if side == "high" else []  # Every PV fault here only loses power.
    cause, proba, reasons = underperformance.classify(booster, meta["classes"], meta["pv_features"], shape_features(rows, hr, inverter, t), ruled_out)
    return {**out, "loss_mwh_per_day": 0.0 if cause == "none" else round(loss, 3), "cause": cause, "cause_proba": round(proba, 3),
            "confidence": round(proba * min(1.0, coverage / 0.15), 3), "reasons": reasons}


def _assess_all(alarms, pred, hrs, labels, booster, meta, end) -> list[dict]:
    out, plant_of = [], pred.groupby("turbine").plant.first().to_dict()
    series = {p: underperformance.Series(g) for p, g in pred.groupby("plant")}
    for r in alarms.itertuples():
        plant = plant_of[r.turbine]
        t = r.ts + pd.Timedelta(hours=CLASSIFY_AFTER_H)
        if labels is not None:
            f = labels[(labels.turbine == r.turbine) & (labels.start < r.ts) & (labels.end + pd.Timedelta(hours=24) >= r.ts)]
            if len(f):
                t = max(r.ts, min(t, f.end.iloc[0]))
        out.append(assess(series[plant], hrs[plant], r.turbine, min(t, end).ceil("h"), r.side, booster, meta))
    return out


def _windows(pred, labels, hrs, rng) -> tuple[list, list]:
    """Training windows inside faults (labelled by cause) and with no fault ('none')."""
    X, y = [], []
    series = {p: underperformance.Series(g) for p, g in pred.groupby("plant")}
    plant_of = pred.groupby("turbine").plant.first().to_dict()

    def take(inv, t, label):
        p = plant_of[inv]
        rows = series[p].window(inv, t - pd.Timedelta(days=FEATURE_DAYS), t)
        rows = rows[underperformance.usable(rows)]
        if len(rows) >= 8:
            X.append(shape_features(rows, hrs[p], inv, t))
            y.append(label)
    for f in labels[labels.detectable].itertuples():
        earliest = f.start + pd.Timedelta(hours=CLASSIFY_AFTER_H)
        for _ in range(WINDOWS_PER_FAULT):
            take(f.turbine, (f.end if earliest >= f.end else earliest + (f.end - earliest) * float(rng.uniform())).floor("h"), f.cause)
    want, lo, hi, inv, tries = max(len(y) // 4, 4), pred.ts.min(), pred.ts.max(), sorted(plant_of), 0
    while y.count("none") < want and tries < want * 30:
        tries += 1
        i, t = inv[int(rng.integers(len(inv)))], (lo + (hi - lo) * float(rng.uniform(0.2, 1))).floor("h")
        if not ((labels.turbine == i) & (labels.start < t) & (labels.end > t - pd.Timedelta(days=FEATURE_DAYS))).any():
            take(i, t, "none")
    return X, y


# ---------- For the calendar: cost of a stop, rain, and replayed findings ----------
def cost_map(site: str, w: pd.DataFrame) -> pd.DataFrame:
    """Same columns as downtime_cost.cost_map. A stopped hour loses that hour's forecast PV output; night costs nothing.

    Radiation, gusts and rain in Open-Meteo cover the preceding hour, so hour [H, H+1) takes the value at H+1;
    temperature is an instant, so the two ends are averaged. Workable: gusts and rain only (no climbing).
    """
    w = w.sort_values("ts_utc").reset_index(drop=True)
    nxt = w.shift(-1)
    wh = pd.DataFrame({"hour_utc": w.ts_utc.dt.tz_convert("UTC").dt.tz_localize(None), "irr_wm2": nxt.shortwave_radiation, "t_amb_c": (w.temperature_2m + nxt.temperature_2m) / 2,
                       "gust_ms": nxt.wind_gusts_10m, "rain_mm": nxt.precipitation}).iloc[:-1].dropna().reset_index(drop=True)
    b, _ = load_models("pv_weather")
    q = np.column_stack([b[n].predict(wh[FORECAST_FEATURES].to_numpy(float)) for n in QUANTILES])
    q.sort(axis=1)
    q = np.clip(q, 0, 1)
    q[wh.irr_wm2.to_numpy() < MIN_IRR_WM2] = 0.0
    workable = (wh.gust_ms < downtime_cost.G_SAFE_MS) & (wh.rain_mm < downtime_cost.P_MAX_MM)
    return pd.concat([pd.DataFrame({"turbine": f.turbine, "hour_utc": wh.hour_utc, "cost_mwh": q[:, 1] * f.rated_kw / 1000, "cost_mwh_p10": q[:, 0] * f.rated_kw / 1000, "cost_mwh_p90": q[:, 2] * f.rated_kw / 1000,
                                    "workable": workable, "v_hub": np.nan, "gust_ms": wh.gust_ms, "rain_mm": wh.rain_mm, "irr_wm2": wh.irr_wm2}) for f in downtime_cost.fleet(site).itertuples()], ignore_index=True)


def days_to_rain(w: pd.DataFrame, tz: str, today) -> int:
    """Days until the forecast gives a washing rain (CLEANING_RAIN_MM in a local day); the horizon if none."""
    daily = w.assign(d=w.ts_utc.dt.tz_convert(tz).dt.date).groupby("d").precipitation.sum()
    wet = [(d - today).days for d, v in daily.items() if v >= CLEANING_RAIN_MM and d >= today]
    return min(wet) if wet else downtime_cost.HORIZON_DAYS


REPLAY_AS_OF = VAL_END     # Kaggle data replayed as "today": the end of validation, never the test days.
DEMO_FAULTS = [            # Injected so the demo has known problems to find; labelled as such in every output.
    {"slot": 2, "cause": "soiling", "start": "2020-05-29", "severity": {"r": 0.005}},
    {"slot": 3, "cause": "string_outage", "start": "2020-06-06", "severity": {"n": 8}},
    {"slot": 5, "cause": "inverter_derate", "start": "2020-06-07", "severity": {"k": 0.6}},
]


def replayed_findings(site: str, today) -> list[dict]:
    """M2-PV on Plant 1 up to REPLAY_AS_OF with DEMO_FAULTS injected. The site's six blocks replay six Plant 1 inverters that look healthy."""
    booster, meta = lgb.Booster(model_file=str(MODELS / "pv_cause.txt")), json.loads((MODELS / "pv_meta.json").read_text())
    boosters, _ = load_models()
    d = load()
    d = d[(d.plant == 1) & (d.ts >= REPLAY_AS_OF - pd.Timedelta(days=21)) & (d.ts < REPLAY_AS_OF)].reset_index(drop=True)
    quiet = [i for i in sorted(d.inverter.unique()) if i not in meta["thresholds"]["left_out_of_calibration"]["1"]][:6]
    f = pd.DataFrame([{"fault_id": i, "turbine": quiet[x["slot"] - 1], "cause": x["cause"], "start": pd.Timestamp(x["start"]), "end": REPLAY_AS_OF, "severity": x["severity"]} for i, x in enumerate(DEMO_FAULTS)])
    faulted, _ = faults.apply(d, f)
    pred = predict(faulted, boosters, meta)
    hrs = hourly_by_plant(pred)
    alarms = scan(hrs, meta["thresholds"])
    names = dict(zip(quiet, downtime_cost.fleet(site).turbine))
    out = []
    for inv, g in alarms[alarms.turbine.isin(quiet)].groupby("turbine"):
        last = g.sort_values("ts").iloc[-1]
        a = assess(underperformance.Series(pred), hrs[1], inv, REPLAY_AS_OF, last.side, booster, meta)
        if not underperformance.dismissed(a):
            out.append({**a, "turbine": names[inv], "alarm_ts": str(last.ts), "replayed_from": f"Kaggle plant 1 inverter {inv} on {REPLAY_AS_OF:%Y-%m-%d}, with injected faults",
                        "first_detected": today - dt.timedelta(days=(REPLAY_AS_OF - last.ts).days)})
    return sorted(out, key=lambda a: -a["loss_mwh_per_day"])


# ---------- Train and evaluate ----------
def train() -> dict:
    d = load()
    codes = {k: i for i, k in enumerate(sorted(d.inverter.unique()))}
    X, y, ok = features(d, codes), d.frac.to_numpy(float), clean(d)
    tr, va = ok & (d.ts < TRAIN_END).to_numpy(), ok & ((d.ts >= TRAIN_END) & (d.ts < VAL_END)).to_numpy()
    meta = {"features": FEATURES, "forecast_features": FORECAST_FEATURES, "pv_features": PV_FEATURES, "codes": codes, "train_rows": int(tr.sum()), "val_rows": int(va.sum()),
            "trees": {}, "lightgbm": lgb.__version__, "seed": SEED}
    MODELS.mkdir(exist_ok=True)
    for prefix, cols in (("pv_expected", FEATURES), ("pv_weather", FORECAST_FEATURES)):
        for n, q in QUANTILES.items():
            b = expected_power.fit(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va], q)
            b.save_model(str(MODELS / f"{prefix}_{n}.txt"), num_iteration=b.best_iteration)
            meta["trees"][f"{prefix}_{n}"] = int(b.best_iteration)
    # Out-of-sample expected values on the training days, for the alarm threshold and the classifier.
    days = sorted(d.ts[d.ts < TRAIN_END].dt.normalize().unique())
    block = {day: i * FOLDS // len(days) for i, day in enumerate(days)}
    fold = d.ts.dt.normalize().map(block)
    train_rows = d[d.ts < TRAIN_END]
    oof = []
    for k in range(FOLDS):
        fit_on = tr & (fold != k).to_numpy()
        bk = {n: expected_power.fit_fixed(X.loc[fit_on, FEATURES], y[fit_on], q, meta["trees"][f"pv_expected_{n}"]) for n, q in QUANTILES.items()}
        oof.append((k, bk))
    def predict_oof(rows):
        f = rows.ts.dt.normalize().map(block)
        return pd.concat([predict(rows[f == k], bk, meta) for k, bk in oof if (f == k).any()]).sort_index()
    th = calibrate(hourly_by_plant(predict_oof(train_rows)))
    print(f"M2-PV threshold {th['h']}, sigma {th['sigma']}", flush=True)
    Xw, yw = [], []
    for k in range(TRAIN_PASSES):
        rng = np.random.default_rng([SEED, 11, k])
        faulted, labels = faults.apply(train_rows, faults.sample(train_rows, TRAIN_FAULTS, train_rows.ts.min(), TRAIN_END, rng, faults.PV_CAUSES, FAULT_DAYS))
        pred = predict_oof(faulted)
        a, b = _windows(pred, labels, hourly_by_plant(pred), rng)
        Xw, yw = Xw + a, yw + b
    classes = sorted(set(yw))
    booster = lgb.train({**underperformance.CLASSIFIER, "num_class": len(classes)}, lgb.Dataset(pd.DataFrame(Xw, columns=PV_FEATURES).to_numpy(float), np.array([classes.index(c) for c in yw]), feature_name=PV_FEATURES),
                        num_boost_round=underperformance.CLASSIFIER_TREES)
    booster.save_model(str(MODELS / "pv_cause.txt"))
    meta.update(classes=classes, thresholds=th, windows=len(yw), class_counts={c: yw.count(c) for c in classes})
    (MODELS / "pv_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def _score(y, p, rated) -> dict:
    return expected_power.score(np.asarray(y) * rated, np.asarray(p) * rated, rated)


def evaluate(final: bool = False) -> dict:
    lo, hi, name = (VAL_END, TEST_END, "test_2020_06_11_17") if final else (TRAIN_END, VAL_END, "val_2020_06_06_10")
    d = load()
    boosters, meta = load_models()
    fboosters, _ = load_models("pv_weather")
    booster = lgb.Booster(model_file=str(MODELS / "pv_cause.txt"))
    th = meta["thresholds"]
    span = d[(d.ts >= lo - pd.Timedelta(days=FEATURE_DAYS)) & (d.ts < hi)].reset_index(drop=True)
    period = span[(span.ts >= lo).to_numpy()]
    ok = clean(period)
    pr, fr = predict(period, boosters, meta)[ok], predict(period, fboosters, meta, FORECAST_FEATURES)[ok]
    y, rated = pr.frac.to_numpy(float), pr.rated_kw.to_numpy(float)
    base = baseline_frac(pr.irr_wm2, pr.t_mod_c)
    covered = (pr.ac_kw >= pr.exp_p10_kw) & (pr.ac_kw <= pr.exp_p90_kw)
    bins = pd.cut(pr.irr_wm2, [MIN_IRR_WM2, 300, 700, 2000])
    res = {"period": name, "rows": int(ok.sum()), "note": "34 days of data in all: indicative only. Faults are synthetic; soiling resets on seeded rain days.",
           "m1_pv": _score(y, pr.exp_p50_kw / pr.rated_kw, rated), "forecast_only_model": _score(y, fr.exp_p50_kw / fr.rated_kw, rated), "baseline_physical": _score(y, base, rated),
           "coverage_10_90": round(float(covered.mean()), 3),
           "by_irradiance": [{"irr_wm2": str(k), "rows": len(g), "m1_mae_pct": round(float((g.exp_p50_kw - g.ac_kw).abs().mean() / g.rated_kw.mean() * 100), 3),
                              "baseline_mae_pct": round(float((baseline_frac(g.irr_wm2, g.t_mod_c) * g.rated_kw - g.ac_kw).abs().mean() / g.rated_kw.mean() * 100), 3)} for k, g in pr.groupby(bins, observed=True)]}
    res["m1_beats_baseline"] = res["m1_pv"]["mae_kw"] < res["baseline_physical"]["mae_kw"]
    # M2-PV on injected faults, and on the untouched data (Plant 2 has real outages and weak inverters).
    clean_pred = predict(span, boosters, meta)
    clean_hrs = hourly_by_plant(clean_pred)
    real = scan(clean_hrs, th, lo)
    res["real_data_alarms_no_injection"] = [{**a, "dismissed_by_filter": underperformance.dismissed(a)} for a in _assess_all(real, clean_pred, clean_hrs, None, booster, meta, hi)]
    inv_months = (hi - lo).days / 30.44 * span.inverter.nunique() * EVAL_PASSES
    rows, false = {"model": [], "baseline": []}, {"model": 0, "baseline": 0}
    y_true, y_pred = [], []
    for k in range(EVAL_PASSES):
        rng = np.random.default_rng([SEED, 12, k])
        faulted, labels = faults.apply(span, faults.sample(span, EVAL_FAULTS, lo, hi, rng, faults.PV_CAUSES, (FAULT_DAYS[0], min(FAULT_DAYS[1], (hi - lo).days))))
        pred = predict(faulted, boosters, meta)
        hrs = hourly_by_plant(pred)
        raw = scan(hrs, th, lo)
        assessed = _assess_all(raw, pred, hrs, labels, booster, meta, hi)
        kept = raw[[not underperformance.dismissed(a) for a in assessed]]
        base_alarms = pd.concat([x for x in (underperformance.baseline_alarms(hr, lo) for hr in hrs.values()) if len(x)] or [pd.DataFrame(columns=["ts", "turbine", "side"])], ignore_index=True)
        for sys_name, alarms in (("model", kept), ("baseline", base_alarms)):
            hits, n_false = underperformance._match(alarms, labels)
            false[sys_name] += n_false
            for f in labels[labels.detectable].itertuples():
                hit = hits.get(f.fault_id)
                rows[sys_name].append({"cause": f.cause, "size": underperformance._bucket(f),
                                       "detected": hit is not None, "delay_days": None if hit is None else (hit.ts - f.start).total_seconds() / 86400})
                if sys_name == "model" and hit is not None:
                    y_true.append(f.cause)
                    y_pred.append(assessed[hit.name]["cause"])
    lab = sorted(set(y_true) | set(y_pred))
    res.update({"m2_pv": underperformance._summary(rows["model"], false["model"], inv_months), "m2_baseline_pi_below_0.95": underperformance._summary(rows["baseline"], false["baseline"], inv_months),
                "cause_macro_f1": round(float(f1_score(y_true, y_pred, labels=sorted(set(y_true)), average="macro", zero_division=0)), 3) if y_true else None,
                "confusion_labels": lab, "confusion_rows_true_cols_predicted": confusion_matrix(y_true, y_pred, labels=lab).tolist() if y_true else []})
    for k_ in ("m2_pv", "m2_baseline_pi_below_0.95"):
        res[k_]["false_alarms_per_inverter_month"] = res[k_].pop("false_alarms_per_turbine_month")
    write_result(f"pv_{name}", res)
    _plot(pr, res, name)
    return res


def _plot(pr: pd.DataFrame, res: dict, name: str) -> None:
    plt = chart_style()
    C = COLORS
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.2, 1.1, 0.9]})
    ax = axes[0]
    g = pr.assign(b=(pr.irr_wm2 // 50) * 50 + 25).groupby("b")
    g = {k: v for k, v in g if len(v) >= 20}
    x = np.array(list(g))
    mean = lambda f: np.array([f(v) for v in g.values()]) * 100
    ax.fill_between(x, mean(lambda v: (v.exp_p10_kw / v.rated_kw).mean()), mean(lambda v: (v.exp_p90_kw / v.rated_kw).mean()), color=C["blue"], alpha=0.18, linewidth=0, label="M1-PV 10-90% band")
    ax.plot(x, mean(lambda v: (v.exp_p50_kw / v.rated_kw).mean()), color=C["blue"], linewidth=2, label="M1-PV expected")
    ax.plot(x, mean(lambda v: baseline_frac(v.irr_wm2, v.t_mod_c).mean()), color=C["orange"], linewidth=1.5, linestyle="--", label="Baseline: physical formula")
    ax.scatter(x, mean(lambda v: v.frac.mean()), s=14, color=C["ink"], zorder=3, label="Actual")
    ax.set_title(f"Real vs. expected: M1 {res['m1_pv']['mae_pct_rated']}% vs. baseline {res['baseline_physical']['mae_pct_rated']}% error")
    ax.set_xlabel("Irradiance (W/m2)"); ax.set_ylabel("AC power (% of rated, mean per bin)"); ax.legend(fontsize=8)
    ax = axes[1]
    causes = faults.PV_CAUSES
    rate = lambda key, c: underperformance.detected_pct(res[key]["by_cause"].get(c, "0/0"))
    pos = np.arange(len(causes))
    b, m = res["m2_baseline_pi_below_0.95"], res["m2_pv"]
    ax.bar(pos - 0.2, [rate("m2_baseline_pi_below_0.95", c) for c in causes], 0.38, color=C["orange"], label=f"Baseline: index < 0.95 ({b['false_alarms_per_inverter_month']} false/inverter-month)")
    ax.bar(pos + 0.2, [rate("m2_pv", c) for c in causes], 0.38, color=C["blue"], label=f"M2-PV ({m['false_alarms_per_inverter_month']} false/inverter-month)")
    ax.set_xticks(pos, [c.replace("_", "\n") for c in causes], fontsize=9); ax.set_ylim(0, 125); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="x", visible=False)
    ax.set_title("Faults detected, by cause: higher is better"); ax.set_ylabel("Detected (%)"); ax.legend(fontsize=8, loc="upper left")
    ax = axes[2]
    lab = res["confusion_labels"]
    if len(lab):
        plot_confusion(ax, lab, res["confusion_rows_true_cols_predicted"], [s.replace("_", " ").replace("inverter ", "").replace("none", "no fault") for s in lab])
    ax.grid(False); ax.set_title("Cause of detected faults"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.text(0.01, 0.01, "Kaggle Solar Power Generation Data, 2 plants x 22 inverters, 34 days: indicative. Faults synthetic; soiling resets on seeded rain days.", fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, f"pv_{name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--final", action="store_true", help="With --eval: score the last 7 days. Once only.")
    a = ap.parse_args()
    if a.train:
        m = train()
        print(json.dumps({k: m[k] for k in ("train_rows", "val_rows", "trees", "classes", "class_counts")} | {"threshold": m["thresholds"]["h"]}, indent=1))
    if a.eval:
        r = evaluate(a.final)
        print(json.dumps({k: r[k] for k in ("rows", "m1_pv", "forecast_only_model", "baseline_physical", "coverage_10_90", "m1_beats_baseline", "m2_pv", "m2_baseline_pi_below_0.95", "cause_macro_f1")}, indent=1))
        print("real-data alarms:", [(a_["turbine"], a_["ts"], a_["cause"], a_["loss_mwh_per_day"], a_["dismissed_by_filter"]) for a_ in r["real_data_alarms_no_injection"]])
