"""Labelled synthetic wind faults, deterministic with a seed (CLAUDE.md 9.2).

The real data has almost no visible faults (0.18% of healthy rows sit far below normal),
so M2 is trained and scored on controlled faults applied to real production. Each fault
records the power it really removed, so detections and loss estimates are scored against
the truth. Used by M2, the backtest and the solar branch. The same functions serve PV rows once the
inverter is in `turbine`, AC power in `power_kw` and irradiance in `wind_ms` (see src/producers/solar.py).
"""
import numpy as np
import pandas as pd

from src.common import SEED, STEP_H

CAUSES = ["yaw_misalignment", "pitch_fault", "anemometer_bias", "icing"]
PV_CAUSES = ["soiling", "string_outage", "shading", "inverter_derate"]
# Severity ranges from CLAUDE.md 9.2. All are assumptions about how bad real faults get.
DURATION_DAYS = (3, 30)
YAW_DEG = (4.0, 15.0)        # Power x cos(angle)^2 below rated: a documented approximation.
PITCH_K = (0.80, 0.95)       # Power capped at k x rated...
PITCH_WIND_MS = 9.0          # ...once the wind passes this.
ANEMOMETER_B = (0.04, 0.10)  # Wind reading off by 4-10%, either way. Under 4% is within sensor tolerance (assumption).
ICING_S = (0.2, 0.6)         # Share of power lost while icing conditions hold.
ICING_TEMP_C, ICING_RH_MIN = (-3.0, 1.0), 85.0
SOILING_R = (0.001, 0.005)   # PV: loss grows by r per day since the fault began, reset by rain.
RAIN_DAYS_PER_30 = 3         # PV: the Kaggle plants have no location, so rain days are drawn with a fixed seed. Assumption.
STRING_N = (8, 16)           # PV: strings per inverter; one fails, so 1/N of output is lost. Real value: plant layout.
SHADING_S, SHADING_WINDOW_H, SHADING_WEEKLY_GROWTH = (0.15, 0.5), 3, 0.25  # PV: loss s in a fixed 3-hour window, growing 25% a week.
DERATE_K = (0.5, 0.85)       # PV: inverter capped at k x rated.
MIN_EFFECT_HOURS = 12        # A fault that changes less than this has nothing to detect, so it is not scored.


def severity(cause: str, rng: np.random.Generator) -> dict:
    if cause == "yaw_misalignment":
        return {"angle_deg": float(rng.uniform(*YAW_DEG))}
    if cause == "pitch_fault":
        return {"k": float(rng.uniform(*PITCH_K))}
    if cause == "anemometer_bias":
        return {"b": float(rng.uniform(*ANEMOMETER_B) * rng.choice([-1, 1]))}
    if cause == "icing":
        return {"s": float(rng.uniform(*ICING_S))}
    if cause == "soiling":
        return {"r": float(rng.uniform(*SOILING_R))}
    if cause == "string_outage":
        return {"n": int(rng.integers(STRING_N[0], STRING_N[1] + 1))}
    if cause == "shading":
        return {"s": float(rng.uniform(*SHADING_S)), "start_h": int(rng.integers(7, 15))}  # Local hour, inside daylight.
    if cause == "inverter_derate":
        return {"k": float(rng.uniform(*DERATE_K))}
    raise ValueError(f"Unknown cause {cause!r}")


def icing_conditions(rows: pd.DataFrame) -> np.ndarray:
    t = rows.temp_c.fillna(rows.temperature_2m).to_numpy(float)
    return (t >= ICING_TEMP_C[0]) & (t <= ICING_TEMP_C[1]) & (rows.relative_humidity_2m.to_numpy(float) >= ICING_RH_MIN)


def rain_days(dates) -> set:
    """Seeded rain days for PV soiling. The same date always gets the same answer."""
    return {d for d in pd.to_datetime(pd.Series(list(dates))).dt.date.unique()
            if np.random.default_rng([SEED, 99, pd.Timestamp(d).toordinal()]).uniform() < RAIN_DAYS_PER_30 / 30}


def effect(rows: pd.DataFrame, cause: str, sev: dict) -> tuple[np.ndarray, np.ndarray]:
    """Faulted power and wind reading (irradiance for PV) for one fault's rows, which start when the fault starts. The input is not changed."""
    power, wind, rated = rows.power_kw.to_numpy(float).copy(), rows.wind_ms.to_numpy(float).copy(), rows.rated_kw.to_numpy(float)
    if cause in PV_CAUSES:
        return _pv_effect(rows, cause, sev, power), wind
    if cause == "yaw_misalignment":
        on = power < 0.98 * rated  # No loss once there is enough wind to hold rated power.
        power[on] *= np.cos(np.deg2rad(sev["angle_deg"])) ** 2
    elif cause == "pitch_fault":
        on = wind > PITCH_WIND_MS
        power[on] = np.minimum(power[on], sev["k"] * rated[on])
    elif cause == "anemometer_bias":
        wind *= 1 + sev["b"]  # Power is unchanged: this fault loses no energy, it only misleads.
    elif cause == "icing":
        power[icing_conditions(rows)] *= 1 - sev["s"]
    else:
        raise ValueError(f"Unknown cause {cause!r}")
    return power, wind


def _pv_effect(rows: pd.DataFrame, cause: str, sev: dict, power: np.ndarray) -> np.ndarray:
    ts = rows.ts
    if cause == "soiling":
        if not len(ts):
            return power
        t = ts.to_numpy("datetime64[ns]")
        day, last_clean = ts.dt.normalize().to_numpy("datetime64[ns]"), np.full(len(ts), t.min())
        for r in sorted(rain_days(ts.dt.date)):
            r = np.datetime64(pd.Timestamp(r), "ns")
            last_clean = np.where(day >= r, np.maximum(last_clean, r), last_clean)  # Rain washes the panels.
        days = (t - last_clean) / np.timedelta64(1, "D")
        return power * np.clip(1 - sev["r"] * days, 0.5, 1)
    if cause == "string_outage":
        return power * (1 - 1 / sev["n"])
    if cause == "shading":
        h = ts.dt.hour.to_numpy()
        on = (h >= sev["start_h"]) & (h < sev["start_h"] + SHADING_WINDOW_H)
        weeks = ((ts - ts.min()).dt.total_seconds() / (7 * 86400)).to_numpy()
        out = power.copy()
        out[on] *= 1 - np.clip(sev["s"] * (1 + SHADING_WEEKLY_GROWTH * weeks[on]), 0, 0.9)
        return out
    if cause == "inverter_derate":
        return np.minimum(power, sev["k"] * rows.rated_kw.to_numpy(float))
    raise ValueError(f"Unknown cause {cause!r}")


def _changed(rows: pd.DataFrame, power: np.ndarray, wind: np.ndarray) -> np.ndarray:
    return (np.abs(power - rows.power_kw.to_numpy(float)) > 1e-6) | (np.abs(wind - rows.wind_ms.to_numpy(float)) > 1e-6)


def sample(d: pd.DataFrame, n: int, lo: pd.Timestamp, hi: pd.Timestamp, rng: np.random.Generator, causes=CAUSES, duration_days=DURATION_DAYS) -> pd.DataFrame:
    """n faults inside [lo, hi), causes in turn, never two at once on a turbine.

    A placement with too little real effect (icing in a warm week) is retried elsewhere,
    and dropped after 60 tries, so a period without frost gets no icing faults.
    """
    groups = {t: g for t, g in d.groupby("turbine")}
    turbines, taken, out = sorted(groups), {t: [] for t in groups}, []
    for i in range(n):
        cause = causes[i % len(causes)]
        for _ in range(60):
            t = turbines[int(rng.integers(len(turbines)))]
            dur = pd.Timedelta(days=float(rng.uniform(*duration_days)))
            start = (lo + (hi - lo - dur) * float(rng.uniform())).floor("h")
            end = min(start + dur, hi)
            if any(start < e and s < end for s, e in taken[t]):
                continue
            g = groups[t]
            rows = g[(g.ts >= start) & (g.ts < end)]
            sev = severity(cause, rng)
            if _changed(rows, *effect(rows, cause, sev)).sum() * STEP_H >= MIN_EFFECT_HOURS:
                taken[t].append((start, end))
                out.append({"fault_id": len(out), "turbine": t, "start": start, "end": end, "cause": cause, "severity": sev})
                break
    return pd.DataFrame(out, columns=["fault_id", "turbine", "start", "end", "cause", "severity"])


def apply(d: pd.DataFrame, faults: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Faulted copy of d (adds fault_id, -1 if none, and true_loss_kw), and the labels with what each fault really did."""
    out = d.copy()
    out["fault_id"], out["true_loss_kw"] = -1, 0.0
    labels = faults.copy()
    hours, loss_mwh, share = [], [], []
    for f in faults.itertuples():
        m = ((out.turbine == f.turbine) & (out.ts >= f.start) & (out.ts < f.end)).to_numpy()
        rows = out[m]
        power, wind = effect(rows, f.cause, f.severity)
        loss = np.nan_to_num(rows.power_kw.to_numpy(float) - power)
        hours.append(float(_changed(rows, power, wind).sum() * STEP_H))
        out.loc[m, "power_kw"], out.loc[m, "wind_ms"] = power.astype("float32"), wind.astype("float32")
        out.loc[m, "fault_id"], out.loc[m, "true_loss_kw"] = f.fault_id, loss
        clean_kwh = float(np.nansum(np.clip(rows.power_kw.to_numpy(float), 0, None)))
        loss_mwh.append(float(loss.sum() * STEP_H / 1000))
        share.append(float(loss.sum() / clean_kwh) if clean_kwh > 0 else 0.0)
    labels["effect_hours"], labels["true_loss_mwh"], labels["loss_share"] = hours, loss_mwh, share
    labels["detectable"] = labels.effect_hours >= MIN_EFFECT_HOURS
    return out, labels
