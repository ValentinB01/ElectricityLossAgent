"""Downtime cost (M4): what would stopping each turbine cost, hour by hour, over the next 10 days? (CLAUDE.md 9.5)

    python -m src.producers.downtime_cost --train            # M1-W: expected power from weather alone, trained on Jan-Oct 2019 (about a minute)
    python -m src.producers.downtime_cost --eval             # M1-W against its baseline on Nov-Dec 2019
    python -m src.producers.downtime_cost --eval --final     # The 2020 test year. Once only, after everything is frozen.
    python -m src.producers.downtime_cost --site kelmarsh    # Cost map from today's Open-Meteo forecast (or --site fantanele)

A stopped hour loses that hour's production, so the cost of a stop is the expected production on the
forecast. M1 cannot give it: it reads the nacelle anemometer, which no forecast has. M1-W maps what a
forecast does give (hub-height wind, air density, gustiness, temperature) to hourly power as a share of
rated. It has no turbine, month or hour input, so it carries over to other sites. At Fantanele that is an
approximation: it learned a Senvion MM92's power curve, not a GE 2.5xl's.

Weather data by Open-Meteo.com (CC BY 4.0).
"""
import argparse
import datetime as dt
import functools
import json
import urllib.parse
import urllib.request

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import COLORS, FORECASTS, KELMARSH, MODELS, QUANTILES, SEED, chart_style, save_figure, write_result
from src.producers import expected_power

FEATURES = ["v_hub", "rho", "gust_factor", "temp_c"]
LEVELS = {80: "wind_speed_80m", 100: "wind_speed_100m", 120: "wind_speed_120m"}  # Archive has 100 m; the forecast has 80 and 120 m.
SHEAR_ALPHA = 0.143   # Power-law shear from the nearest level to hub height (CLAUDE.md 9.5). Assumption: real shear varies with stability.
MIN_INTERVALS = 5     # An hour counts if at least 5 of its six 10-minute rows are healthy.
# Working limits. Assumptions: the real values come from the manufacturer's manual and PPC field operations.
V_SAFE_MS = 12.0      # Top hub-height wind for climbing and working in the nacelle.
G_SAFE_MS = 18.0      # Top gust at 10 m.
P_MAX_MM = 2.0        # Rain in an hour that stops work.
SHIFT_LOCAL = (8, 18)  # Crew day, local time.
JOB_HOURS = 8         # For the summary: the cheapest 8-hour safe window in each day's shift.
HORIZON_DAYS = 10
FORECAST_VARS = "wind_speed_80m,wind_speed_120m,wind_speed_10m,wind_gusts_10m,temperature_2m,surface_pressure,precipitation"
SITES = {
    "kelmarsh": {"lat": 52.4006, "lon": -0.9471, "tz": "Europe/London"},  # Turbines, ratings and hub heights from Kelmarsh_WT_static.csv.
    # PPC, 240 GE 2.5xl turbines. The demo shows six, the number replayed from Kelmarsh (CLAUDE.md 1.2).
    "fantanele": {"lat": 44.59028, "lon": 28.56528, "tz": "Europe/Bucharest", "rated_kw": 2500.0, "turbines": 6, "prefix": "FAN-T",
                  "hub_m": 100.0},  # Assumption: a common GE 2.5xl tower. Real value: PPC asset register.
    # PPC Lumina PV park, 63 MW. The demo shows six inverter blocks the size of a Kaggle inverter, so replayed findings fit.
    "calugareni": {"lat": 44.0951, "lon": 25.9761, "tz": "Europe/Bucharest", "type": "pv", "rated_kw": 1400.0, "turbines": 6, "prefix": "CAL-INV", "hub_m": None},
}


# ---------- Weather to model inputs ----------
def hour_means(w: pd.DataFrame) -> pd.DataFrame:
    """Weather for each hour [H, H+1), to match hourly turbine means (column 'hour', naive UTC).

    Wind, temperature and pressure are instants, so the two ends of the hour are averaged.
    Gusts and rain already cover the preceding hour, so the value at the end is taken.
    """
    w = w.sort_values("ts_utc").reset_index(drop=True)
    nxt = w.shift(-1)
    out = pd.DataFrame({"hour": w.ts_utc.dt.tz_convert("UTC").dt.tz_localize(None)})
    for c in [*LEVELS.values(), "wind_speed_10m", "temperature_2m", "surface_pressure"]:
        if c in w:
            out[c] = (w[c] + nxt[c]) / 2
    for c in ("wind_gusts_10m", "precipitation"):
        out[c] = nxt[c]
    return out.iloc[:-1].reset_index(drop=True)


def hub_wind(wh: pd.DataFrame, hub_m: float) -> np.ndarray:
    """Wind at hub height: interpolated in log height between two levels, or moved by power-law shear from the nearest one."""
    have = sorted(h for h, c in LEVELS.items() if c in wh and wh[c].notna().any())
    if not have:
        raise ValueError(f"No wind level in the weather table: {list(wh.columns)}")
    below, above = [h for h in have if h <= hub_m], [h for h in have if h >= hub_m]
    if below and above:
        lo, hi = below[-1], above[0]
        v_lo, v_hi = wh[LEVELS[lo]].to_numpy(float), wh[LEVELS[hi]].to_numpy(float)
        return v_lo if lo == hi else v_lo + (v_hi - v_lo) * (np.log(hub_m / lo) / np.log(hi / lo))
    ref = below[-1] if below else above[0]
    return wh[LEVELS[ref]].to_numpy(float) * (hub_m / ref) ** SHEAR_ALPHA


def features(wh: pd.DataFrame, hub_m: float) -> pd.DataFrame:
    missing = {"wind_speed_10m", "wind_gusts_10m", "temperature_2m", "surface_pressure"} - set(wh.columns)
    if missing:
        raise SystemExit(f"Weather is missing {sorted(missing)}. Run: python -m src.data.prepare (it fetches the weather again).")
    t = wh.temperature_2m.to_numpy(float)
    return pd.DataFrame({"v_hub": hub_wind(wh, hub_m), "rho": wh.surface_pressure.to_numpy(float) * 100 / (287.05 * (t + 273.15)),
                         "gust_factor": wh.wind_gusts_10m.to_numpy(float) / np.maximum(wh.wind_speed_10m.to_numpy(float), 1.0), "temp_c": t}, index=wh.index)


def fleet(site: str) -> pd.DataFrame:
    if site == "kelmarsh":
        f = pd.read_csv(KELMARSH / "Kelmarsh_WT_static.csv", encoding="utf-8-sig")
        return pd.DataFrame({"turbine": f["Alternative Title"], "rated_kw": f["Rated power (kW)"].astype(float), "hub_m": f["Hub Height (m)"].astype(float)})
    s = SITES[site]
    return pd.DataFrame({"turbine": [f"{s['prefix']}{i + 1:02d}" for i in range(s["turbines"])], "rated_kw": s["rated_kw"], "hub_m": s["hub_m"]})


# ---------- M1-W ----------
def hourly_table(year: int = 2019) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hourly mean power share per turbine on healthy hours, with that hour's weather, and the model inputs."""
    d = expected_power.load((year,))
    keep = expected_power.clean_mask(d)
    g = (pd.DataFrame({"hour": d.ts.dt.floor("h"), "turbine": d.turbine, "frac": d.frac.where(keep), "rated_kw": d.rated_kw})
         .groupby(["hour", "turbine"]).agg(frac=("frac", "mean"), n=("frac", "count"), rated_kw=("rated_kw", "first")).reset_index())
    t = g[g.n >= MIN_INTERVALS].merge(hour_means(pd.read_parquet(KELMARSH / f"weather_{year}.parquet")), on="hour").reset_index(drop=True)
    t["hub_m"] = t.turbine.map(fleet("kelmarsh").set_index("turbine").hub_m)
    X = pd.concat([features(g, hub) for hub, g in t.groupby("hub_m")]).loc[t.index, FEATURES]
    return t, X


def train() -> dict:
    t, X = hourly_table()
    tr, va = (t.hour < expected_power.TRAIN_END).to_numpy(), ((t.hour >= expected_power.TRAIN_END) & (t.hour < expected_power.VAL_END)).to_numpy()
    y = t.frac.to_numpy(float)
    meta = {"features": FEATURES, "train_hours": int(tr.sum()), "val_hours": int(va.sum()), "quantiles": {}, "lightgbm": lgb.__version__, "seed": SEED}
    MODELS.mkdir(exist_ok=True)
    for name, q in QUANTILES.items():
        b = expected_power.fit(X[tr], y[tr], X[va], y[va], q)
        b.save_model(str(MODELS / f"wind_weather_{name}.txt"), num_iteration=b.best_iteration)
        _boosters.cache_clear()
        meta["quantiles"][name] = int(b.best_iteration)
    (MODELS / "wind_weather_baseline.json").write_text(json.dumps(expected_power.binned_curve(X.v_hub[tr], y[tr])))
    (MODELS / "wind_weather_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


@functools.lru_cache(maxsize=1)
def _boosters() -> dict:
    return {n: lgb.Booster(model_file=str(MODELS / f"wind_weather_{n}.txt")) for n in QUANTILES}


def predict_frac(X: pd.DataFrame) -> np.ndarray:
    """Power as a share of rated, columns p10, p50, p90."""
    b = _boosters()
    q = np.column_stack([b[n].predict(X[FEATURES].to_numpy(float)) for n in QUANTILES])
    q.sort(axis=1)
    return q


def evaluate(final: bool = False) -> dict:
    """Validation (Nov-Dec 2019) or, with final, the 2020 test year: run once, after everything is frozen."""
    t, X = hourly_table(2020 if final else 2019)
    period_name = "test_2020" if final else "val_2019_nov_dec"
    va = np.ones(len(t), bool) if final else ((t.hour >= expected_power.TRAIN_END) & (t.hour < expected_power.VAL_END)).to_numpy()
    t, X = t[va].reset_index(drop=True), X[va].reset_index(drop=True)
    q = predict_frac(X) * t.rated_kw.to_numpy(float)[:, None]
    curve = json.loads((MODELS / "wind_weather_baseline.json").read_text())
    d = pd.DataFrame({"turbine": t.turbine, "day": t.hour.dt.floor("D"), "wind_ms": X.v_hub, "power_kw": t.frac * t.rated_kw, "rated_kw": t.rated_kw,
                      "exp_p10_kw": q[:, 0], "exp_p50_kw": q[:, 1], "exp_p90_kw": q[:, 2], "base_kw": np.interp(X.v_hub, curve["centres"], curve["values"]) * t.rated_kw})
    covered = (d.power_kw >= d.exp_p10_kw) & (d.power_kw <= d.exp_p90_kw)
    bins = pd.cut(d.wind_ms, [0, 4, 6, 8, 10, 12, 15, 40], include_lowest=True)
    by_wind = [{"wind_ms": str(k), "hours": len(g), "model_mae_pct": expected_power.score(g.power_kw, g.exp_p50_kw, g.rated_kw)["mae_pct_rated"],
                "baseline_mae_pct": expected_power.score(g.power_kw, g.base_kw, g.rated_kw)["mae_pct_rated"], "coverage_10_90": round(float(covered[g.index].mean()), 3)} for k, g in d.groupby(bins, observed=True)]
    # What the optimizer adds up: energy over a working day. Hourly errors partly cancel.
    day = d.groupby(["turbine", "day"]).agg(real=("power_kw", "sum"), model=("exp_p50_kw", "sum"), base=("base_kw", "sum"), hours=("power_kw", "size"))
    day = day[day.hours >= 20]
    res = {"period": period_name, "hours": len(d), "note": "Hourly power from reanalysis weather (Open-Meteo archive). A forecast adds its own error on top.",
           "model": expected_power.score(d.power_kw, d.exp_p50_kw, d.rated_kw), "baseline_binned_curve_on_hub_wind": expected_power.score(d.power_kw, d.base_kw, d.rated_kw),
           "coverage_10_90": round(float(covered.mean()), 3), "by_wind": by_wind,
           "daily_energy_mae_mwh": {"model": round(float((day.model - day.real).abs().mean() / 1000), 2), "baseline": round(float((day.base - day.real).abs().mean() / 1000), 2),
                                    "mean_daily_energy_mwh": round(float(day.real.mean() / 1000), 2)}}
    res["model_beats_baseline"] = res["model"]["mae_kw"] < res["baseline_binned_curve_on_hub_wind"]["mae_kw"]
    write_result(f"wind_weather_{period_name}", res)
    expected_power.plot_expected(d, res, f"wind_weather_{period_name}", model="M1-W", xlabel="Hub-height wind from weather data (m/s)")
    return res


# ---------- Cost map ----------
def get_forecast(site: str, variables: str = FORECAST_VARS) -> tuple[pd.DataFrame, dict]:
    """Today's hourly forecast. If Open-Meteo cannot be reached, the last saved one is used and marked stale (CLAUDE.md section 12)."""
    s, path = SITES[site], FORECASTS / f"{site}.parquet"
    try:
        query = urllib.parse.urlencode({"latitude": s["lat"], "longitude": s["lon"], "hourly": variables, "forecast_days": HORIZON_DAYS + 1, "wind_speed_unit": "ms", "timezone": "UTC"})
        with urllib.request.urlopen(f"https://api.open-meteo.com/v1/forecast?{query}", timeout=60) as r:
            w = pd.DataFrame(json.load(r)["hourly"])
        w["ts_utc"] = pd.to_datetime(w.pop("time"), utc=True)
        meta = {"site": site, "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "stale": False, "source": "Open-Meteo forecast API"}
        FORECASTS.mkdir(parents=True, exist_ok=True)
        w.to_parquet(path)
        path.with_suffix(".json").write_text(json.dumps(meta))
    except (OSError, ValueError, KeyError) as e:
        if not path.exists():
            raise SystemExit(f"No forecast for {site}: Open-Meteo failed ({e}) and nothing is saved at {path}.")
        w, meta = pd.read_parquet(path), {**json.loads(path.with_suffix(".json").read_text()), "stale": True, "stale_reason": str(e)[:200]}
    return w, meta


def cost_map(site: str, w: pd.DataFrame) -> pd.DataFrame:
    """One row per turbine and hour: cost_mwh (production lost if stopped, p50), cost_mwh_p90, workable, and the weather behind them."""
    wh = hour_means(w)
    parts = []
    for f in fleet(site).itertuples():
        X = features(wh, f.hub_m)
        q = np.clip(predict_frac(X), 0, 1) * f.rated_kw / 1000  # Idle turbines draw a little power; a stop never costs less than nothing.
        ok = X[["v_hub", "gust_factor"]].notna().all(axis=1).to_numpy() & wh.precipitation.notna().to_numpy()
        parts.append(pd.DataFrame({"turbine": f.turbine, "hour_utc": wh.hour, "cost_mwh": q[:, 1], "cost_mwh_p90": q[:, 2],
                                   "workable": ok & (X.v_hub.to_numpy() < V_SAFE_MS) & (wh.wind_gusts_10m.to_numpy(float) < G_SAFE_MS) & (wh.precipitation.to_numpy(float) < P_MAX_MM),
                                   "v_hub": X.v_hub.round(2), "gust_ms": wh.wind_gusts_10m, "rain_mm": wh.precipitation})[ok])
    return pd.concat(parts, ignore_index=True)


def _fleet_hours(cm: pd.DataFrame, site: str) -> pd.DataFrame:
    """Per hour: mean cost per turbine, whether every turbine is workable, and the site's local time."""
    f = cm.groupby("hour_utc").agg(cost=("cost_mwh", "mean"), workable=("workable", "all")).reset_index()
    f["local"] = f.hour_utc.dt.tz_localize("UTC").dt.tz_convert(SITES[site]["tz"])
    return f


def daily_windows(cm: pd.DataFrame, site: str) -> pd.DataFrame:
    """For each local day: the cheapest JOB_HOURS-long window inside the shift where every turbine can be worked on, and the cost of stopping all day (full days only)."""
    f = _fleet_hours(cm, site)
    f["date"], f["h"] = f.local.dt.date, f.local.dt.hour
    rows = []
    for date, g in f.groupby("date"):
        g = g.set_index("h")
        if not ((g.index >= SHIFT_LOCAL[0]) & (g.index < SHIFT_LOCAL[1])).any():
            continue  # Today's shift is over, or the forecast ends before the last day's shift.
        best = None
        for start in range(SHIFT_LOCAL[0], SHIFT_LOCAL[1] - JOB_HOURS + 1):
            hrs = range(start, start + JOB_HOURS)
            if all(h in g.index and g.workable[h] for h in hrs):
                c = float(g.cost[list(hrs)].sum())
                if best is None or c < best[0]:
                    best = (c, start)
        shift = g.loc[SHIFT_LOCAL[0]:SHIFT_LOCAL[1] - 1]
        rows.append({"date": str(date), "cheapest_window_mwh": None if best is None else round(best[0], 2), "window_start_local": None if best is None else f"{best[1]:02d}:00",
                     "workable_shift_hours": int(shift.workable.sum()), "whole_day_mwh": round(float(g.cost.sum()), 2) if len(g) == 24 else None})
    return pd.DataFrame(rows)


def run_site(site: str) -> dict:
    w, meta = get_forecast(site)
    cm = cost_map(site, w)
    cm = cm[cm.hour_utc >= pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)]  # The forecast starts at midnight UTC: past hours cannot be scheduled.
    days = daily_windows(cm, site)
    ok = days.dropna(subset=["cheapest_window_mwh"])
    res = {**meta, "turbines": int(cm.turbine.nunique()), "hours": int(cm.hour_utc.nunique()),
           "assumptions": {"v_safe_ms": V_SAFE_MS, "g_safe_ms": G_SAFE_MS, "p_max_mm": P_MAX_MM, "shift_local": SHIFT_LOCAL, "job_hours": JOB_HOURS, "shear_alpha": SHEAR_ALPHA},
           "per_turbine_daily": days.to_dict("records")}
    if len(ok):
        res["cheapest_vs_dearest_safe_window_mwh"] = [float(ok.cheapest_window_mwh.min()), float(ok.cheapest_window_mwh.max())]
    write_result(f"downtime_cost_{site}", res)
    _plot_cost(cm, days, site, meta)
    return res


def _plot_cost(cm: pd.DataFrame, days: pd.DataFrame, site: str, meta: dict) -> None:
    plt = chart_style()
    C = COLORS
    f = _fleet_hours(cm, site)
    f = f[f.local.dt.date.astype(str).isin(days.date)]
    f["day"], f["h"] = f.local.dt.strftime("%a %d %b"), f.local.dt.hour
    order = list(dict.fromkeys(f.day))
    cost = f.pivot_table(index="day", columns="h", values="cost").reindex(order)
    work = f.pivot_table(index="day", columns="h", values="workable", aggfunc="first").reindex(order).astype(float)
    rated = float(fleet(site).rated_kw.mean()) / 1000
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.42 * len(order) + 2.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    im = ax.imshow(cost.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=rated)
    yy, xx = np.nonzero(work.to_numpy() == 0)
    ax.scatter(xx, yy, marker="x", s=14, color=C["orange"], linewidths=1, label="Not safe to work (wind, gusts or rain)")
    ax.axvline(SHIFT_LOCAL[0] - 0.5, color=C["ink"], linewidth=0.8, linestyle=":"); ax.axvline(SHIFT_LOCAL[1] - 0.5, color=C["ink"], linewidth=0.8, linestyle=":")
    ax.set_yticks(range(len(order)), order, fontsize=9); ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)], fontsize=9)
    ax.grid(False); ax.set_xlabel(f"Local hour ({SITES[site]['tz']}); dotted lines: crew shift")
    ax.set_title(f"Production lost per stopped hour, per turbine (MWh): {site.title()}")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    ax = axes[1]
    y = np.arange(len(days))
    vals = days.cheapest_window_mwh.astype(float)
    ax.barh(y, vals.fillna(0), color=C["blue"], height=0.6)
    for i, (v, st) in enumerate(zip(vals, days.window_start_local)):
        ax.text((0 if np.isnan(v) else v) + rated * 0.05, i, "no safe window" if np.isnan(v) else f"{v:.1f} MWh from {st}", va="center", fontsize=8.5, color=C["muted"] if np.isnan(v) else C["ink"])
    ax.set_yticks(y, [pd.Timestamp(x).strftime("%a %d %b") for x in days.date], fontsize=9); ax.invert_yaxis()
    ax.set_xlim(0, max(float(np.nan_to_num(vals).max()), rated) * 1.6); ax.grid(axis="y", visible=False)
    ax.set_title(f"Cheapest safe {JOB_HOURS}-hour stop each day, per turbine"); ax.set_xlabel("MWh lost")
    fig.text(0.01, 0.005, f"Forecast fetched {meta['fetched_utc']}{' (STALE)' if meta.get('stale') else ''}. Weather data by Open-Meteo.com. Safety limits are assumptions.", fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_figure(fig, f"downtime_cost_{site}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--final", action="store_true", help="With --eval: score the 2020 test year. Once only.")
    ap.add_argument("--site", choices=sorted(k for k, v in SITES.items() if v.get("type", "wind") == "wind"), help="Wind sites; solar costs come from src/producers/solar.py")
    a = ap.parse_args()
    if a.train:
        print(json.dumps(train(), indent=1))
    if a.eval:
        r = evaluate(a.final)
        print(json.dumps({k: r[k] for k in ("hours", "model", "baseline_binned_curve_on_hub_wind", "coverage_10_90", "daily_energy_mae_mwh", "model_beats_baseline")}, indent=1))
    if a.site:
        r = run_site(a.site)
        print(json.dumps({k: v for k, v in r.items() if k != "per_turbine_daily"}, indent=1, default=str))
        print(pd.DataFrame(r["per_turbine_daily"]).to_string(index=False))
