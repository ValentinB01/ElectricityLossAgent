"""Backtest: a year replayed day by day under three maintenance policies (CLAUDE.md 9.7).

    python backtest.py --dev      # Nov-Dec 2019: builds and checks the replay. Not a result.
    python backtest.py --final    # 2020, the test year. Run once, after every setting is frozen.

Every policy gets the same faults, detections, crews, safety limits and revisions. Only the decision differs:
    immediate     each repair at the earliest safe slot after detection, whatever the stop costs
    at_revision   repairs wait for the turbine's next revision
    agent         the optimizer each morning on the forecast; only today's jobs are carried out
Each runs with a perfect forecast (the real weather of the next 10 days) and a noisy one. Decisions use
the forecast; energy is charged on what really happened: the turbine's real production while stopped
(SCADA) and each fault's real loss until it is fixed. A job whose hours turn out unsafe is called off.

Detection is shared: the injected faults are applied to the real data as if never repaired, and M2 runs
once over it. M2 uses only data up to each moment, so these are the alarms a daily run would raise.
A policy acts on a finding only while its fault is still unrepaired.
"""
import argparse
import copy
import datetime as dt
import json

import numpy as np
import pandas as pd

import inject_faults as faults
import m1_expected as m1
import m2_underperf as m2
import m4_cost as m4
import optimizer as opt

SITE, TZ, STEP_H, SEED = "kelmarsh", m4.SITES["kelmarsh"]["tz"], 1 / 6, m1.SEED
FAULTS_PER_YEAR = 30         # CLAUDE.md 9.7 asks for 20-40.
MIN_GAP_DAYS = 60            # Faults on one turbine start at least this far apart.
NOISE_PER_LEAD_DAY = 0.03    # Noisy forecast: wind and gusts x (1 + e), e ~ N(0, 0.03 x (days ahead + 1)), one draw per day ahead. Assumption.
SEEDS = 3                    # Independent fault sets; results are the mean and range.
LEAD_IN_DAYS = 45            # Data before the period, for M2's feature windows.
IMMEDIATE_WAIT_MWH = 1000.0  # How "immediate" is told that any delay is unacceptable.
POLICIES = ("immediate", "at_revision", "agent")
PERIODS = {"dev": ("2019-11-01", "2020-01-01", (2019,)), "final": ("2020-01-01", "2021-01-01", (2019, 2020))}


# ---------- What happened: faults, detections, real production and weather ----------
def sample_faults(d: pd.DataFrame, lo, hi, n: int, rng) -> pd.DataFrame:
    """n faults starting at random times, causes in turn. They last until repaired, so 'end' is the end of the period."""
    groups = {t: g for t, g in d.groupby("turbine")}
    turbines, starts, out = sorted(groups), {t: [] for t in groups}, []
    for i in range(n):
        cause = faults.CAUSES[i % len(faults.CAUSES)]
        for _ in range(60):
            t = turbines[int(rng.integers(len(turbines)))]
            start = (lo + (hi - lo - pd.Timedelta(days=3)) * float(rng.uniform())).floor("h")
            if any(abs((start - s).days) < MIN_GAP_DAYS for s in starts[t]):
                continue
            g = groups[t]
            rows, sev = g[(g.ts >= start) & (g.ts < start + pd.Timedelta(days=30))], faults.severity(cause, rng)
            if faults._changed(rows, *faults.effect(rows, cause, sev)).sum() * STEP_H >= faults.MIN_EFFECT_HOURS:
                starts[t].append(start)
                out.append({"fault_id": len(out), "turbine": t, "start": start, "end": hi, "cause": cause, "severity": sev})
                break
    return pd.DataFrame(out)


def fault_losses(d: pd.DataFrame, fl: pd.DataFrame) -> dict:
    """Per fault: row times and cumulative energy lost (MWh) if it is never fixed, so any fix time can be looked up."""
    out = {}
    for f in fl.itertuples():
        rows = d[(d.turbine == f.turbine) & (d.ts >= f.start) & (d.ts < f.end)]
        power, _ = faults.effect(rows, f.cause, f.severity)
        out[f.fault_id] = (rows.ts.to_numpy("datetime64[ns]"), np.cumsum(np.nan_to_num(rows.power_kw.to_numpy(float) - power)) * STEP_H / 1000)
    return out


def lost_until(losses: dict, fid: int, t) -> float:
    ts, cum = losses[fid]
    i = np.searchsorted(ts, np.datetime64(t), "left")
    return float(cum[i - 1]) if i > 0 else 0.0


def detections(d: pd.DataFrame, fl: pd.DataFrame, lo, hi) -> list[dict]:
    """M2 once over the period with every fault left in place. Each finding is known 48 h after its alarm, when it is classified."""
    booster, meta = m2._load()
    th = meta["thresholds"]
    faulted, _ = faults.apply(d, fl)
    b1, meta1 = m1._models()
    pred = m1.predict_expected(faulted, b1, meta1)
    hr, series = m2.hourly(pred), m2.Series(pred)
    alarms = m2.scan(hr, th["sigma"], th["h"], lo)
    out = []
    for r, a in zip(alarms.itertuples(), m2._assess_alarms(alarms, series, hr, None, booster, meta, hi)):
        if m2.dismissed(a):
            continue
        before = fl[(fl.turbine == r.turbine) & (fl.start < r.ts)]
        out.append({**a, "alarm_ts": r.ts, "known_at": pd.Timestamp(a["ts"]), "fault_id": int(before.sort_values("start").fault_id.iloc[-1]) if len(before) else -1})
    return sorted(out, key=lambda x: x["known_at"])


class World:
    """Real production and safety per turbine-hour, and the forecast as seen from each morning."""

    def __init__(self, d: pd.DataFrame, years: tuple, lo, hi):
        w = pd.concat([pd.read_parquet(m1.DATA / f"weather_{y}.parquet") for y in years], ignore_index=True)
        self.weather = w[(w.ts_utc >= pd.Timestamp(lo, tz="UTC") - pd.Timedelta(days=1)) & (w.ts_utc <= pd.Timestamp(hi, tz="UTC") + pd.Timedelta(days=11))].reset_index(drop=True)
        self.actual = m4.cost_map(SITE, self.weather)  # M1-W on the real weather: the perfect forecast, and the real safety flags.
        self.safe = set(map(tuple, self.actual.loc[self.actual.workable, ["turbine", "hour_utc"]].to_numpy()))
        g = d.assign(hour=d.ts.dt.floor("h"), mwh=d.power_kw.clip(lower=0).fillna(0) * STEP_H / 1000).groupby(["turbine", "hour"]).mwh.sum()
        self.production = g.to_dict()
        self.cache = {}

    def forecast(self, day: dt.date, start_utc: pd.Timestamp, mode: str, seed: int) -> pd.DataFrame:
        key = (day, mode)
        if key not in self.cache:
            end = start_utc + pd.Timedelta(days=opt.HORIZON_DAYS)
            if mode == "perfect":
                cm = self.actual[(self.actual.hour_utc >= start_utc) & (self.actual.hour_utc < end)]
            else:
                w = self.weather[(self.weather.ts_utc >= start_utc.tz_localize("UTC")) & (self.weather.ts_utc <= end.tz_localize("UTC"))].copy()
                lead = ((w.ts_utc.dt.tz_localize(None) - start_utc) // pd.Timedelta(days=1)).clip(0, opt.HORIZON_DAYS - 1).to_numpy()
                eps = np.random.default_rng([SEED, 6, seed, day.toordinal()]).normal(0, NOISE_PER_LEAD_DAY * (np.arange(opt.HORIZON_DAYS) + 1))
                for c in ("wind_speed_100m", "wind_speed_10m", "wind_gusts_10m"):
                    w[c] = (w[c] * (1 + eps[lead])).clip(lower=0)
                cm = m4.cost_map(SITE, w)
                cm = cm[cm.hour_utc < end]
            self.cache[key] = cm
        return self.cache[key]


def utc_hour(date: str, hour: int) -> pd.Timestamp:
    return pd.Timestamp(f"{date} {hour:02d}:00").tz_localize(TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC").tz_localize(None)


# ---------- One policy through the period ----------
def revisions(lo, hi, policy: str) -> list:
    """The service calendar. Only revisions whose whole window lies in the period, so no policy can push one past the end.
    The two simple policies hold each revision on its due date or the first safe day after; the agent may use the full window."""
    tasks = opt.generate_revisions(list(m4.fleet(SITE).turbine), SITE, lo.date(), (hi - pd.Timedelta(days=1)).date(), epoch=dt.date(2019, 1, 5))
    tasks = [t for t in tasks if t.due_start >= lo.date() and t.due_end < hi.date()]
    for t in tasks:
        if policy != "agent":
            t.due_start = dt.date.fromisoformat(t.trace["due_date"])
    return tasks


def simulate(policy: str, mode: str, seed: int, found: list, fl: pd.DataFrame, losses: dict, world: World, lo, hi) -> dict:
    open_tasks = {t.task_id: t for t in revisions(lo, hi, policy)}
    fixed, busy, trips, log, by_kind = {}, [], set(), [], {}
    downtime = drone = cancelled = deferred = false_jobs = 0.0
    cause_of = dict(zip(fl.fault_id, fl.cause))
    k = 0

    def add_repair(turbine, cause, loss, fid, day, source):
        t = opt.Task(f"{source}-confirmed", turbine, SITE, "repair", float(opt.REPAIR_HOURS.get(cause, 8)), loss_mwh_per_day=IMMEDIATE_WAIT_MWH if policy == "immediate" else loss,
                     confidence=1.0, first_detected=day, job_type=f"{cause}_repair", trace={"fault_id": fid, "confirmed_by_inspection": True, "m2_cause": cause})
        open_tasks[t.task_id] = t

    def fix(fid, at):
        if fid >= 0 and fid not in fixed:
            fixed[fid] = at
        for tid in [i for i, t in open_tasks.items() if t.kind != "revision" and t.trace.get("fault_id") == fid and fid >= 0]:
            del open_tasks[tid]

    for day in pd.date_range(lo, hi - pd.Timedelta(days=1), freq="D").date:
        start = utc_hour(day.isoformat(), 0)
        # 1. Findings known by this morning.
        while k < len(found) and found[k]["known_at"] <= start:
            f, k = found[k], k + 1
            fid = f["fault_id"]
            if (fid >= 0 and fid in fixed and fixed[fid] <= f["known_at"]) or any(t.turbine == f["turbine"] and t.kind != "revision" for t in open_tasks.values()):
                continue  # Already repaired, or this turbine already has an open job.
            task = opt.tasks_from_findings([{**f, "first_detected": day}], SITE, day)[0]
            task.task_id, task.trace["fault_id"] = f"F{k}-{task.task_id}", fid
            false_jobs += fid < 0
            if task.kind == "repair" and policy == "at_revision":
                deferred += 1
                continue
            if task.kind == "repair" and policy == "immediate":
                task.loss_mwh_per_day = IMMEDIATE_WAIT_MWH
            open_tasks[task.task_id] = task
        if not open_tasks:
            continue
        # 2. Plan on the forecast. The simple policies do not look at what a stop costs.
        cm = world.forecast(day, start, mode, seed)
        if policy != "agent":
            cm = cm.assign(cost_mwh=0.0, cost_mwh_p90=0.0)
        s = opt.optimize(copy.deepcopy(list(open_tasks.values())), cm, opt.Constraints(day, unavailable=[b for b in busy if b["date"] >= day.isoformat()]), TZ)
        # 3. Carry out today's jobs, charged on what really happened.
        jobs, done = {j["task_id"]: j for j in s["jobs"]}, set()
        for j in sorted(s["jobs"], key=lambda j: bool(j.get("bundled_with"))):  # Revisions before the repairs bundled into them.
            if j["date"] != day.isoformat() or j["task_id"] not in open_tasks:
                continue
            task = open_tasks[j["task_id"]]
            if j.get("bundled_with"):
                if j["bundled_with"] not in done:
                    continue  # Its revision did not happen today.
                hours = [utc_hour(dd, h) for dd, h in jobs[j["bundled_with"]]["hours"]]
            else:
                hours = [utc_hour(dd, h) for dd, h in j["hours"]]
            fid = task.trace.get("fault_id", -1)
            if j["kind"] == "inspection":
                drone += 1
                end = hours[-1] + pd.Timedelta(hours=1)
                if task.kind == "repair":  # The optimizer chose to check first: the check confirms the job.
                    task.trace["confirmed_by_inspection"] = True
                    continue
                del open_tasks[task.task_id]
                if fid >= 0 and fid not in fixed:
                    if cause_of[fid] in ("anemometer_bias", "icing"):
                        fix(fid, end)  # Recalibrated or de-iced on the spot.
                    elif policy != "at_revision":
                        add_repair(task.turbine, cause_of[fid], task.loss_mwh_per_day, fid, day, task.task_id)
                    else:
                        deferred += 1
                continue
            if not j.get("bundled_with"):
                if not all((task.turbine, h) in world.safe for h in hours):
                    cancelled += 1  # The forecast said safe; the real weather did not.
                    continue
                lost = sum(world.production.get((task.turbine, h), 0.0) for h in hours)
                downtime += lost
                by_kind[task.kind] = by_kind.get(task.kind, 0.0) + lost
                for dd, _ in j["hours"]:
                    trips.add((j["crew"], dd))
                    if dd > day.isoformat():
                        busy.append({"crew": j["crew"], "date": dd})
            end = hours[-1] + pd.Timedelta(hours=1)
            del open_tasks[task.task_id]
            done.add(task.task_id)
            log.append({"date": day.isoformat(), "turbine": task.turbine, "task": task.task_id, "kind": task.kind, "hours": len(j["hours"]), "fault_id": fid})
            if task.kind == "revision":
                for f in fl[(fl.turbine == task.turbine) & (fl.start < end)].fault_id:
                    fix(int(f), end)  # A full service finds and fixes whatever is wrong.
            else:
                fix(fid, end)
    loss_faults = fl[fl.cause != "anemometer_bias"]
    days_to_fix = [((fixed.get(f, hi) - s) / pd.Timedelta(days=1)) for f, s in zip(loss_faults.fault_id, loss_faults.start)]
    under = sum(lost_until(losses, f, min(fixed.get(f, hi), hi)) for f in fl.fault_id)
    return {"policy": policy, "forecast": mode, "seed": seed, "downtime_mwh": round(downtime, 2), "downtime_revisions_mwh": round(by_kind.get("revision", 0.0), 2), "downtime_repairs_mwh": round(by_kind.get("repair", 0.0), 2), "underperformance_mwh": round(under, 2), "total_mwh": round(downtime + under, 2),
            "crew_trips": len(trips), "drone_checks": int(drone), "faults_fixed": f"{sum(f in fixed for f in loss_faults.fault_id)}/{len(loss_faults)}",
            "mean_days_to_fix": round(float(np.mean(days_to_fix)), 1) if days_to_fix else None, "weather_cancellations": int(cancelled),
            "repairs_deferred_to_revision": int(deferred), "jobs_from_false_alarms": int(false_jobs), "jobs": log}


def run(period: str) -> dict:
    lo, hi, years = PERIODS[period]
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    d = m1.load(years)
    d = d[(d.ts >= lo - pd.Timedelta(days=LEAD_IN_DAYS)) & (d.ts < hi)].sort_values(["turbine", "ts"]).reset_index(drop=True)
    world = World(d, years, lo, hi)
    n = max(4, round(FAULTS_PER_YEAR * (hi - lo).days / 365))
    runs, detail, placed = [], {}, []
    for seed in range(SEEDS):
        fl = sample_faults(d, lo, hi, n, np.random.default_rng([SEED, 7, seed]))
        losses, found = fault_losses(d, fl), detections(d, fl, lo, hi)
        placed.append(len(fl))
        real = [x for x in found if x["fault_id"] >= 0]
        print(f"seed {seed}: {len(fl)} faults, {len(found)} findings ({len(found) - len(real)} with no injected fault), {len({x['fault_id'] for x in real})} faults found", flush=True)
        for mode in ("perfect", "noisy"):
            for policy in POLICIES:
                r = simulate(policy, mode, seed, found, fl, losses, world, lo, hi)
                detail[(seed, mode, policy)] = r.pop("jobs")
                runs.append(r)
                print(f"  {mode:<8}{policy:<12} total {r['total_mwh']:8.1f} MWh = downtime {r['downtime_mwh']:7.1f} + underperformance {r['underperformance_mwh']:7.1f}; "
                      f"trips {r['crew_trips']}, fixed {r['faults_fixed']}, {r['mean_days_to_fix']} days to fix, {r['weather_cancellations']} called off", flush=True)
        world.cache.clear()
    df = pd.DataFrame(runs)
    summary = df.groupby(["forecast", "policy"]).agg(total_mwh=("total_mwh", "mean"), total_min=("total_mwh", "min"), total_max=("total_mwh", "max"), downtime_mwh=("downtime_mwh", "mean"), downtime_revisions_mwh=("downtime_revisions_mwh", "mean"), downtime_repairs_mwh=("downtime_repairs_mwh", "mean"),
                                                     underperformance_mwh=("underperformance_mwh", "mean"), crew_trips=("crew_trips", "mean"), mean_days_to_fix=("mean_days_to_fix", "mean")).round(2).reset_index()
    res = {"period": period, "from": str(lo.date()), "to": str((hi - pd.Timedelta(days=1)).date()), "seeds": SEEDS, "faults_requested_per_seed": n, "faults_placed_per_seed": placed,
           "note": "Faults are injected into real SCADA; detections come from M2; downtime is charged on real production. Failures (M3) are out of scope, so their cost is 0.",
           "assumptions": {"noise_per_lead_day": NOISE_PER_LEAD_DAY, "crews_per_day": opt.CREWS_PER_DAY, "repair_hours": opt.REPAIR_HOURS, "revision_hours": [opt.SEMIANNUAL_HOURS, opt.ANNUAL_HOURS],
                           "v_safe_ms": m4.V_SAFE_MS, "g_safe_ms": m4.G_SAFE_MS},
           "summary": summary.to_dict("records"), "runs": runs, "agent_jobs_perfect_seed0": detail[(0, "perfect", "agent")]}
    m1.RESULTS.mkdir(exist_ok=True)
    (m1.RESULTS / f"backtest_{period}.json").write_text(json.dumps(res, indent=1, default=str))
    _plot(summary, period, res)
    return res


def _plot(summary: pd.DataFrame, period: str, res: dict) -> None:
    plt = m1.chart_style()
    C = m1.COLORS
    names = {"immediate": "Repair as soon\nas detected", "at_revision": "Repair at the\nnext revision", "agent": "Agent"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, mode in zip(axes, ("perfect", "noisy")):
        s = summary[summary.forecast == mode].set_index("policy").loc[list(POLICIES)]
        x = np.arange(len(s))
        ax.bar(x, s.downtime_mwh, 0.55, color=C["orange"], label="Lost while stopped (downtime)")
        ax.bar(x, s.underperformance_mwh, 0.55, bottom=s.downtime_mwh, color=C["blue"], label="Lost to faults while waiting")
        ax.errorbar(x, s.total_mwh, yerr=[s.total_mwh - s.total_min, s.total_max - s.total_mwh], fmt="none", ecolor=C["ink"], capsize=4, linewidth=1)
        for i, v in enumerate(s.total_mwh):
            ax.text(i, s.total_max.iloc[i] * 1.03 + 1, f"{v:.0f} MWh", ha="center", fontsize=9, color=C["ink"])
        ax.set_xticks(x, [names[p] for p in s.index], fontsize=9); ax.grid(axis="x", visible=False)
        ax.set_title(f"{'Perfect' if mode == 'perfect' else 'Noisy'} forecast")
    axes[0].set_ylabel(f"Energy lost, {res['from']} to {res['to']} (MWh)")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.text(0.01, 0.01, f"Kelmarsh, 6 turbines; {np.mean(res['faults_placed_per_seed']):.0f} injected faults per set (icing needs frost), mean of {res['seeds']} sets (bars: range). Failures (M3) out of scope.", fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    (m1.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(m1.RESULTS / "figures" / f"backtest_{period}.png", dpi=150, facecolor=C["surface"])
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dev", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    r = run("final" if a.final else "dev")
    print(pd.DataFrame(r["summary"]).to_string(index=False))
