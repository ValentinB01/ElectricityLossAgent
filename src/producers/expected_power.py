"""Expected production (M1): how much would this turbine produce right now if nothing were wrong?

    python -m src.producers.expected_power --compare      # Which inputs help, on validation (every third row)
    python -m src.producers.expected_power --train        # Train the p10, p50 and p90 models
    python -m src.producers.expected_power --eval         # Score on Nov-Dec 2019 against the baseline
    python -m src.producers.expected_power --eval --final # Score on 2020. Once only, after everything is frozen.

Trained on healthy rows only: training on faulty periods would teach the fault as normal.
Predicts power as a share of rated, so outputs convert to any turbine size. Inputs describe
the conditions, never the turbine's own response (blade angle, rotor speed) or vendor
calculations: the first would hide faults from M2, the second would leak a vendor's answer.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common import COLORS, KELMARSH, MODELS, QUANTILES, SEED, chart_style, save_figure, write_result

TRAIN_END, VAL_END = pd.Timestamp("2019-11-01"), pd.Timestamp("2020-01-01")  # Fixed before seeing any result.

# Starting hyperparameters from the guide, CLAUDE.md 9.1.
LGB = {"num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 50, "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.9,
       "seed": SEED, "deterministic": True, "force_row_wise": True, "verbose": -1}
MAX_TREES, PATIENCE = 800, 50
BIN_WIDTH_MS = 0.5  # Baseline: method of bins, as in IEC 61400-12-1.

ALL_FEATURES = ["wind_ms", "rho", "ti", "dir_sin", "dir_cos", "hour_sin", "hour_cos", "month_sin", "month_cos", "asset_code"]
# Chosen with --compare on validation: direction, hour and month did not lower the error, so the simpler set is used.
FEATURES = ["wind_ms", "rho", "ti", "asset_code"]
ASSET_CODES = {f"KWF{i}": i - 1 for i in range(1, 7)}


def load(years=(2019,)) -> pd.DataFrame:
    """Turbine rows with the hour's weather attached, and power as a share of rated."""
    frames = []
    for y in years:
        p = KELMARSH / f"turbines_{y}.parquet"
        if not p.exists():
            raise SystemExit(f"{p} is missing. Run: python -m src.data.prepare --year {y}")
        d = pd.read_parquet(p)
        w = pd.read_parquet(KELMARSH / f"weather_{y}.parquet")
        w = w.assign(hour=w.ts_utc.dt.tz_localize(None))[["hour", "surface_pressure", "temperature_2m", "relative_humidity_2m"]]  # Humidity: icing, for M2.
        frames.append(d.assign(hour=d.ts.dt.floor("h")).merge(w, on="hour", how="left").drop(columns="hour"))
    d = pd.concat(frames, ignore_index=True)
    facts = pd.read_csv(KELMARSH / "Kelmarsh_WT_static.csv", encoding="utf-8-sig")
    d["rated_kw"] = d.turbine.map(dict(zip(facts["Alternative Title"], facts["Rated power (kW)"].astype(float))))
    d["frac"] = d.power_kw / d.rated_kw
    return d


def clean_mask(d: pd.DataFrame) -> np.ndarray:
    """Healthy rows with possible values, no stuck wind sensor and no flat cap below rated in high wind."""
    ok = d.healthy.to_numpy(bool) & d[["power_kw", "wind_ms"]].notna().all(axis=1).to_numpy()
    ok &= ((d.frac >= -0.05) & d.wind_ms.between(0, 40)).to_numpy()
    for _, idx in d.groupby("turbine", sort=False).indices.items():
        g = d.iloc[idx]
        same = g.wind_ms.diff().eq(0)
        run = same.groupby((~same).cumsum()).cumsum()
        stuck = run.groupby((run == 0).cumsum()).transform("max").to_numpy() >= 11  # The same value for about 2 hours.
        roll = g.frac.rolling(6, min_periods=6)
        capped = ((roll.std() < 0.005) & (roll.mean() < 0.95) & (g.wind_ms > 12)).to_numpy()
        ok[idx] &= ~stuck & ~capped
    return ok


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    wind = d.wind_ms.to_numpy(float)
    temp = d.temp_c.fillna(d.temperature_2m).to_numpy(float)
    direction = np.deg2rad(d.wind_dir_deg.to_numpy(float))
    hour = 2 * np.pi * (d.ts.dt.hour + d.ts.dt.minute / 60).to_numpy(float) / 24
    year = 2 * np.pi * d.ts.dt.dayofyear.to_numpy(float) / 365.25
    return pd.DataFrame({"wind_ms": wind, "rho": d.surface_pressure.to_numpy(float) * 100 / (287.05 * (temp + 273.15)),
                         "ti": np.clip(d.wind_std_ms.to_numpy(float) / np.maximum(wind, 1.0), 0, 2),
                         "dir_sin": np.sin(direction), "dir_cos": np.cos(direction), "hour_sin": np.sin(hour), "hour_cos": np.cos(hour),
                         "month_sin": np.sin(year), "month_cos": np.cos(year), "asset_code": d.turbine.map(ASSET_CODES).to_numpy(float)}, index=d.index)


def _categorical(X: pd.DataFrame):
    return ["asset_code"] if "asset_code" in X.columns else "auto"


def fit(X_tr, y_tr, X_va, y_va, alpha: float, **overrides) -> lgb.Booster:
    """Quantile LightGBM with early stopping on validation. Shared by every expected-power model."""
    ds = lgb.Dataset(X_tr, y_tr, categorical_feature=_categorical(X_tr))
    trees = overrides.pop("trees", MAX_TREES)
    params = {**LGB, "objective": "quantile", "alpha": alpha, "metric": "quantile", **overrides}
    return lgb.train(params, ds, num_boost_round=trees, valid_sets=[lgb.Dataset(X_va, y_va, reference=ds)], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])


def fit_fixed(X, y, alpha: float, trees: int) -> lgb.Booster:
    """The same model with a tree count already chosen: the refits that give M2 out-of-sample expected values."""
    return lgb.train({**LGB, "objective": "quantile", "alpha": alpha}, lgb.Dataset(X, y, categorical_feature=_categorical(X)), num_boost_round=trees)


def binned_curve(wind, frac) -> dict:
    """Mean output per wind bin, from training rows only. The simple baseline M1 has to beat."""
    g = pd.Series(frac).groupby(np.floor(np.asarray(wind) / BIN_WIDTH_MS).astype(int)).agg(["mean", "size"])
    g = g[g["size"] >= 5]
    return {"centres": ((g.index.to_numpy() + 0.5) * BIN_WIDTH_MS).tolist(), "values": g["mean"].tolist()}


def score(y_kw, pred_kw, rated) -> dict:
    e = np.asarray(pred_kw) - np.asarray(y_kw)
    r = float(np.mean(rated))
    return {"mae_kw": round(float(np.abs(e).mean()), 2), "rmse_kw": round(float(np.sqrt((e**2).mean())), 2), "mae_pct_rated": round(float(np.abs(e).mean() / r * 100), 3),
            "rmse_pct_rated": round(float(np.sqrt((e**2).mean()) / r * 100), 3), "bias_kw": round(float(e.mean()), 2)}


def compare() -> pd.DataFrame:
    """Which inputs earn their place? Uses every third row and a faster learning rate, so it takes minutes."""
    d = load()
    keep = clean_mask(d)
    tr, va = keep & (d.ts < TRAIN_END).to_numpy(), keep & ((d.ts >= TRAIN_END) & (d.ts < VAL_END)).to_numpy()
    tr_idx = np.flatnonzero(tr)[::3]
    X, y = build_features(d), d.frac.to_numpy(float)
    wind = d.wind_ms.to_numpy(float)
    curve = binned_curve(wind[tr], y[tr])
    base = np.interp(wind[va], curve["centres"], curve["values"])
    base_mae = np.abs(base - y[va]).mean() * 100
    sets = {"wind, turbine": ["wind_ms", "asset_code"],
            "+ air density": ["wind_ms", "asset_code", "rho"],
            "+ turbulence": ["wind_ms", "asset_code", "rho", "ti"],
            "+ direction": ["wind_ms", "asset_code", "rho", "ti", "dir_sin", "dir_cos"],
            "+ hour": ["wind_ms", "asset_code", "rho", "ti", "dir_sin", "dir_cos", "hour_sin", "hour_cos"],
            "+ month (all)": ALL_FEATURES,
            "all, no turbine": [f for f in ALL_FEATURES if f != "asset_code"]}
    rows = [{"inputs": "baseline: binned curve", "val_mae_pct_rated": round(float(base_mae), 3), "val_bias_pct_rated": round(float((base - y[va]).mean() * 100), 3), "trees": 0, "seconds": 0}]
    for name, cols in sets.items():
        t = time.time()
        b = fit(X.iloc[tr_idx][cols], y[tr_idx], X.loc[va, cols], y[va], 0.5, learning_rate=0.1, trees=400)
        p = b.predict(X.loc[va, cols].to_numpy(float))
        rows.append({"inputs": name, "val_mae_pct_rated": round(float(np.abs(p - y[va]).mean() * 100), 3), "val_bias_pct_rated": round(float((p - y[va]).mean() * 100), 3), "trees": b.best_iteration, "seconds": round(time.time() - t)})
        print(rows[-1], flush=True)
    write_result("wind_expected_feature_comparison", rows)
    return pd.DataFrame(rows)


def train() -> dict:
    np.random.seed(SEED)
    d = load()
    keep = clean_mask(d)
    tr, va = keep & (d.ts < TRAIN_END).to_numpy(), keep & ((d.ts >= TRAIN_END) & (d.ts < VAL_END)).to_numpy()
    if keep.mean() < 0.6 or tr.sum() < 1000 or va.sum() < 100:
        raise SystemExit(f"Only {keep.mean():.1%} of rows pass the filter, so the model would learn a skewed sample.")
    X, y = build_features(d)[FEATURES], d.frac.to_numpy(float)
    MODELS.mkdir(exist_ok=True)
    meta = {"features": FEATURES, "asset_codes": ASSET_CODES, "train_rows": int(tr.sum()), "val_rows": int(va.sum()), "clean_share": round(float(keep.mean()), 4), "quantiles": {}, "lightgbm": lgb.__version__, "seed": SEED}
    for name, q in QUANTILES.items():
        t = time.time()
        b = fit(X[tr], y[tr], X[va], y[va], q)
        b.save_model(str(MODELS / f"wind_expected_{name}.txt"), num_iteration=b.best_iteration)
        meta["quantiles"][name] = {"best_iteration": int(b.best_iteration), "minutes": round((time.time() - t) / 60, 1)}
        print(name, meta["quantiles"][name], flush=True)
    (MODELS / "wind_expected_baseline.json").write_text(json.dumps(binned_curve(d.wind_ms[tr], y[tr])))
    (MODELS / "wind_expected_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def load_models() -> tuple[dict, dict]:
    meta = json.loads((MODELS / "wind_expected_meta.json").read_text())
    return {n: lgb.Booster(model_file=str(MODELS / f"wind_expected_{n}.txt")) for n in QUANTILES}, meta


def predict_expected(df: pd.DataFrame, boosters=None, meta=None) -> pd.DataFrame:
    """Returns df with exp_p10_kw, exp_p50_kw and exp_p90_kw added (CLAUDE.md 9.1)."""
    if boosters is None:
        boosters, meta = load_models()
    unknown = set(df.turbine.unique()) - set(meta["asset_codes"])
    if unknown:
        raise ValueError(f"M1 does not know these turbines: {sorted(unknown)}")
    X = build_features(df)[meta["features"]].to_numpy(float)
    q = np.column_stack([boosters[n].predict(X) for n in QUANTILES])
    q.sort(axis=1)  # Separate quantile models can cross.
    rated = df.rated_kw.to_numpy(float)[:, None]
    q = q * rated  # Not clipped at 0: an idle turbine draws a little power, so real output can be slightly negative.
    return df.assign(exp_p10_kw=q[:, 0], exp_p50_kw=q[:, 1], exp_p90_kw=q[:, 2])


def _errors(g: pd.DataFrame) -> dict:
    """Model and baseline error, and model bias, for one slice of the evaluation rows."""
    return {"model_mae_pct": score(g.power_kw, g.exp_p50_kw, g.rated_kw)["mae_pct_rated"], "baseline_mae_pct": score(g.power_kw, g.base_kw, g.rated_kw)["mae_pct_rated"],
            "bias_pct": round(float((g.exp_p50_kw - g.power_kw).mean() / g.rated_kw.mean() * 100), 3)}


def evaluate(final: bool = False) -> dict:
    d = load((2019, 2020) if final else (2019,))
    keep = clean_mask(d)
    lo, hi, name = (VAL_END, pd.Timestamp("2021-01-01"), "test_2020") if final else (TRAIN_END, VAL_END, "val_2019_nov_dec")
    d = d[keep & ((d.ts >= lo) & (d.ts < hi)).to_numpy()].reset_index(drop=True)
    d = predict_expected(d)
    curve = json.loads((MODELS / "wind_expected_baseline.json").read_text())
    d["base_kw"] = np.interp(d.wind_ms, curve["centres"], curve["values"]) * d.rated_kw
    y, r = d.power_kw.to_numpy(float), d.rated_kw.to_numpy(float)
    d["covered"] = (y >= d.exp_p10_kw) & (y <= d.exp_p90_kw)
    bins = pd.cut(d.wind_ms, [0, 4, 6, 8, 10, 12, 15, 40], include_lowest=True)
    by_bin = [{"wind_ms": str(k), "rows": len(g), "model_mae_pct": score(g.power_kw, g.exp_p50_kw, g.rated_kw)["mae_pct_rated"], "baseline_mae_pct": score(g.power_kw, g.base_kw, g.rated_kw)["mae_pct_rated"],
               "model_bias_pct": round(float((g.exp_p50_kw - g.power_kw).mean() / g.rated_kw.mean() * 100), 3), "coverage_10_90": round(float(g.covered.mean()), 3)} for k, g in d.groupby(bins, observed=True)]
    by_turbine = {t: _errors(g) for t, g in d.groupby("turbine")}
    by_month = {str(m): _errors(g) for m, g in d.groupby(d.ts.dt.to_period("M"))}
    # The noise floor for M2: on healthy data, how far does real / expected wander over 72 hours?
    h = d.assign(hour=d.ts.dt.floor("h"), use=d.exp_p50_kw > 0.1 * d.rated_kw)
    h = h[h.use].groupby(["hour", "turbine"])[["power_kw", "exp_p50_kw"]].sum().unstack("turbine").asfreq("h")
    pi = h["power_kw"].rolling(72, min_periods=12).sum() / h["exp_p50_kw"].rolling(72, min_periods=12).sum()
    pi_all = pi.stack()
    res = {"period": name, "rows": len(d), "model": score(y, d.exp_p50_kw, r), "baseline_binned_curve": score(y, d.base_kw, r), "coverage_10_90": round(float(d.covered.mean()), 3),
           "below_p10": round(float((y < d.exp_p10_kw).mean()), 3), "above_p90": round(float((y > d.exp_p90_kw).mean()), 3), "model_beats_baseline": bool(np.abs(y - d.exp_p50_kw).mean() < np.abs(y - d.base_kw).mean()),
           "by_wind": by_bin, "by_turbine": by_turbine, "by_month": by_month,
           "pi_72h_on_healthy_data": {"p05": round(float(pi_all.quantile(.05)), 4), "median": round(float(pi_all.median()), 4), "p95": round(float(pi_all.quantile(.95)), 4), "std": round(float(pi_all.std()), 4)}}
    res["top_factors_by_mean_abs_kw"] = _importance(d)
    write_result(f"wind_expected_{name}", res)
    plot_expected(d, res, f"wind_expected_{name}")
    return res


def _importance(d: pd.DataFrame) -> dict:
    """Mean absolute contribution of each input in kW, from LightGBM's built-in TreeSHAP (the algorithm behind shap.TreeExplainer)."""
    boosters, meta = load_models()
    s = d.sample(min(5000, len(d)), random_state=SEED)
    c = np.abs(boosters["p50"].predict(build_features(s)[meta["features"]].to_numpy(float), pred_contrib=True)[:, :-1]) * s.rated_kw.to_numpy(float)[:, None]
    return {f: round(float(v), 1) for f, v in sorted(zip(meta["features"], c.mean(axis=0)), key=lambda x: -x[1])}


def plot_expected(d: pd.DataFrame, res: dict, file: str, model: str = "M1", xlabel: str = "Nacelle wind speed (m/s)") -> None:
    """Real vs. expected along the power curve, and error by wind speed against the baseline. Shared with M1-W."""
    plt = chart_style()
    BLUE, ORANGE, INK = COLORS["blue"], COLORS["orange"], COLORS["ink"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    g = d.groupby((d.wind_ms * 2).round() / 2)
    g = {k: v for k, v in g if len(v) >= 20}
    x = np.array(list(g))
    m = lambda c: np.array([(v[c] / v.rated_kw).mean() * 100 for v in g.values()])
    axes[0].fill_between(x, m("exp_p10_kw"), m("exp_p90_kw"), color=BLUE, alpha=0.18, linewidth=0, label=f"{model} 10-90% band")
    axes[0].plot(x, m("exp_p50_kw"), color=BLUE, linewidth=2, label=f"{model} expected")
    axes[0].scatter(x, m("power_kw"), s=18, color=INK, zorder=3, label="Actual")
    axes[0].set_title("Real vs. expected, healthy rows"); axes[0].set_xlabel(xlabel); axes[0].set_ylabel("Power (% of rated, mean per bin)"); axes[0].legend(fontsize=9)
    labels = [b["wind_ms"].strip("(]").replace(", ", "-").replace("-0.001", "0") for b in res["by_wind"]]
    pos = np.arange(len(labels))
    axes[1].bar(pos - 0.2, [b["baseline_mae_pct"] for b in res["by_wind"]], width=0.38, color=ORANGE, label="Baseline: binned curve")
    axes[1].bar(pos + 0.2, [b["model_mae_pct"] for b in res["by_wind"]], width=0.38, color=BLUE, label=model)
    axes[1].set_xticks(pos, labels); axes[1].grid(axis="x", visible=False)
    axes[1].set_title("Error by wind speed: lower is better"); axes[1].set_xlabel(xlabel); axes[1].set_ylabel("Mean absolute error (% of rated)"); axes[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, file)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--final", action="store_true", help="With --eval: score the 2020 test year. Once only.")
    a = ap.parse_args()
    if a.compare:
        print(compare().to_string(index=False))
    if a.train:
        print(json.dumps(train(), indent=1))
    if a.eval:
        r = evaluate(a.final)
        print(json.dumps({k: r[k] for k in ("period", "rows", "model", "baseline_binned_curve", "coverage_10_90", "model_beats_baseline", "pi_72h_on_healthy_data", "top_factors_by_mean_abs_kw")}, indent=1))
