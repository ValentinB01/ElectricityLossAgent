"""The decision: which day, hours and crew for each job, so the total energy lost is lowest (CLAUDE.md 9.6).

    python optimizer.py --site fantanele   # A real calendar: today's forecast, the service calendar, replayed M2 findings
    python optimizer.py --site kelmarsh
    python optimizer.py --site fantanele calugareni   # Wind and solar in one calendar (two crews per site)
    python -m pytest tests                 # Hand-built cases, including the 7.6 vs. 19.4 MWh example

For one job, the cost of doing it on day d (counted from today) is

    cost = production lost while stopped (M4, the hours worked)
         + daily loss of the fault x d (M2, waiting)
         + failure probability by day d x failure cost (M3; out of scope, so 0)

Deterministic and greedy, with no LLM involved. Revisions go first, since their windows are hard
limits, tightest first; then repairs, largest daily loss first; then inspections. Each job takes its
cheapest feasible slot and books the crew. Every job keeps its cost breakdown, the options it beat
and why, and a trace back to the model outputs that produced it.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

import m4_cost as m4

RESULTS = m4.RESULTS
# Capacity and rules. Assumptions: the real values come from PPC field operations.
CREWS_PER_DAY = 2
HOURS_PER_CREW = 8          # Hours of work in a crew's day, inside the shift (m4_cost.SHIFT_LOCAL, 08-18 local).
INSPECTIONS_PER_DAY = 4     # Drone or remote checks, by a separate team.
HORIZON_DAYS = m4.HORIZON_DAYS
TRIP_BONUS_MWH = 0.5        # Credit when a crew is already at the site that day (shared travel). From travel cost / energy price.
FORECAST_SPREAD_MAX_MWH = 3.0  # If p90 - p50 downtime on the best slot exceeds this, inspect first instead of committing a crew.
MAX_DEFERRAL_DAYS = 14      # A fault found this long ago takes the earliest slot: no endless wait for "the next calm day".
N_ALTERNATIVES = 3
CONF_MIN = 0.55             # Same as m2_underperf.CONF_MIN: below it, a finding gets a check, not a repair crew.
MIN_LOSS_FOR_REPAIR = 0.05  # MWh/day. A fault with no measurable loss gets a check: a crew would recover nothing.
# Job lengths in hours. Assumptions: real values come from the manufacturer's service manual.
REPAIR_HOURS = {"pitch_fault": 8, "yaw_misalignment": 6}
INSPECTION_HOURS = 1
REVISION_MONTHS, REVISION_WINDOW_DAYS = 6, 14  # A public O&M plan: revisions every 6 months (docs/design.md), +-14 days tolerance.
SEMIANNUAL_HOURS, ANNUAL_HOURS = 16, 24        # Two and three crew days; the public plan gives 2-4 days for the annual one.
# PV jobs (CLAUDE.md 9.8): job, hours, allowed local hours. Assumptions: real values from PPC's PV operations.
PV_JOBS = {"soiling": ("cleaning", 4, (5, 9)),        # At dawn: cheap, and the panels are cool.
           "string_outage": ("string_repair", 4, None), "shading": ("shading_removal", 4, None), "inverter_derate": ("inverter_check", 4, None)}
CLEANING_COST_MWH = 0.5     # Cleaning a block, in MWh equivalent: clean only if the loss until rain exceeds it. From contractor price / energy price.


@dataclass
class Task:
    task_id: str
    turbine: str
    site: str
    kind: str                                  # revision, repair, inspection
    duration_h: float
    due_start: dt.date | None = None           # Tolerance window, for revisions
    due_end: dt.date | None = None
    loss_mwh_per_day: float = 0.0              # From M2: what each day of waiting costs
    p_fail_by_day: list | None = None          # From M3; None until M3 exists, so the risk term is 0
    fail_cost_mwh_eq: float = 0.0
    confidence: float = 1.0
    first_detected: dt.date | None = None
    job_type: str = ""
    trace: dict = field(default_factory=dict)  # Where the job came from: model outputs or the service calendar
    hours: tuple | None = None                 # Allowed local hours, if the job has its own (PV cleaning at dawn); else the crew shift


@dataclass
class Constraints:
    today: dt.date
    crews_per_day: int = CREWS_PER_DAY
    hours_per_crew: int = HOURS_PER_CREW
    inspections_per_day: int = INSPECTIONS_PER_DAY
    horizon_days: int = HORIZON_DAYS
    shift: tuple = m4.SHIFT_LOCAL
    unavailable: list = field(default_factory=list)  # [{"crew": "crew-2", "date": "2026-11-12"}]

    def __post_init__(self):
        for u in self.unavailable:
            if u.get("crew") not in self.crews:
                raise ValueError(f"Unknown crew in unavailable: {u.get('crew')!r}. Known crews: {self.crews}")

    @property
    def crews(self) -> list[str]:
        return [f"crew-{i + 1}" for i in range(self.crews_per_day)]

    def crew_off(self, crew: str, day: dt.date) -> bool:
        return any(u["crew"] == crew and u["date"] == day.isoformat() for u in self.unavailable)


# ---------- Jobs ----------
def generate_revisions(turbines: list[str], site: str, start: dt.date, end: dt.date, epoch: dt.date = dt.date(2026, 1, 5)) -> list[Task]:
    """Six-monthly revisions, staggered across the fleet; every second one is the longer annual revision."""
    tasks, win = [], dt.timedelta(days=REVISION_WINDOW_DAYS)
    for i, t in enumerate(turbines):
        due, k = pd.Timestamp(epoch) + pd.Timedelta(days=int(i * 30.4 * REVISION_MONTHS / max(len(turbines), 1))), 0
        while due.date() - win <= end:
            if due.date() + win >= start:
                annual = k % 2 == 1
                tasks.append(Task(f"{t}-rev-{due:%Y%m}", t, site, "revision", float(ANNUAL_HOURS if annual else SEMIANNUAL_HOURS), due.date() - win, due.date() + win,
                                  job_type="revision_annual" if annual else "revision_semiannual", trace={"due_date": due.date().isoformat(), "source": "service calendar (assumed)"}))
            k, due = k + 1, due + pd.DateOffset(months=REVISION_MONTHS)
    return tasks


def tasks_from_findings(findings: list[dict], site: str, today: dt.date, days_to_rain: int | None = None, skipped: list | None = None) -> list[Task]:
    """M2 findings to jobs. A wrong wind reading, icing, a confidence under CONF_MIN or no measurable loss gets a check rather than a repair crew.
    Soiling is cleaned only if its loss until the next washing rain exceeds the cost of cleaning (CLAUDE.md 9.8)."""
    tasks = []
    for i, f in enumerate(findings):
        base = dict(turbine=f["turbine"], site=site, loss_mwh_per_day=float(f["loss_mwh_per_day"]), confidence=float(f["confidence"]), first_detected=f.get("first_detected", today),
                    trace={"m2_cause": f["cause"], "m2_proba": f["cause_proba"], "m2_loss_mwh_day": f["loss_mwh_per_day"], "m2_pi_72h": f["pi_72h"], "m2_reasons": f.get("reasons", []), "m2_alarm_ts": f.get("alarm_ts")})
        tid = f"{f['turbine']}-{f['cause']}-{i + 1:02d}"
        if f["cause"] in PV_JOBS and f["confidence"] >= CONF_MIN and f["loss_mwh_per_day"] >= MIN_LOSS_FOR_REPAIR:
            job, hours, allowed = PV_JOBS[f["cause"]]
            if job == "cleaning":
                wait = m4.HORIZON_DAYS if days_to_rain is None else days_to_rain
                base["trace"]["days_to_rain"] = wait
                if f["loss_mwh_per_day"] * wait <= CLEANING_COST_MWH:
                    if skipped is not None:
                        skipped.append({"task_id": tid, "reason": f"cleaning not worth it: {f['loss_mwh_per_day']:.2f} MWh/day x {wait} days to rain <= {CLEANING_COST_MWH} MWh cleaning cost"})
                    continue
            tasks.append(Task(tid, kind="repair", duration_h=float(hours), job_type=job, hours=allowed, **base))
        elif f["cause"] in REPAIR_HOURS and f["confidence"] >= CONF_MIN and f["loss_mwh_per_day"] >= MIN_LOSS_FOR_REPAIR:
            tasks.append(Task(tid, kind="repair", duration_h=float(REPAIR_HOURS[f["cause"]]), job_type=f"{f['cause']}_repair", **base))
        else:
            why = {"anemometer_bias": "sensor_check", "icing": "icing_check"}.get(f["cause"], "low_confidence_check" if f["confidence"] < CONF_MIN else "no_loss_check")
            tasks.append(Task(tid, kind="inspection", duration_h=float(INSPECTION_HOURS), job_type=why, **base))
    return tasks


# ---------- Planning ----------
class CostGrid:
    """M4's cost map as (day, local hour) arrays per turbine. Hours missing from the map (the past, beyond the forecast) are not workable."""

    def __init__(self, cost_map: pd.DataFrame, today: dt.date, days: int, tz: str):
        if not len(cost_map):
            raise ValueError("The cost map is empty")
        self.today, self.days = today, days
        local = cost_map.hour_utc.dt.tz_localize("UTC").dt.tz_convert(tz)
        d = np.array([(x - today).days for x in local.dt.date])
        h = local.dt.hour.to_numpy()
        self.cost, self.p90, self.ok, self.gust_ok, self.v = {}, {}, {}, {}, {}
        for t, idx in cost_map.groupby("turbine").indices.items():
            keep = idx[(d[idx] >= 0) & (d[idx] < days)]
            arr = lambda col, fill, dtype=float: self._fill(cost_map[col].to_numpy(dtype)[keep], d[keep], h[keep], fill)
            self.cost[t], self.p90[t], self.v[t] = arr("cost_mwh", np.nan), arr("cost_mwh_p90", np.nan), arr("v_hub", np.nan)
            self.ok[t] = arr("workable", False, bool).astype(bool)
            self.gust_ok[t] = self._fill(cost_map.gust_ms.to_numpy(float)[keep] < m4.G_SAFE_MS, d[keep], h[keep], False).astype(bool)

    def _fill(self, values, d, h, fill):
        a = np.full((self.days, 24), fill, dtype=float)
        a[d, h] = values
        return a

    def date(self, d: int) -> dt.date:
        return self.today + dt.timedelta(days=d)


class Bookings:
    def __init__(self):
        self.crew, self.turbine, self.trips = set(), set(), set()
        self.crew_hours: dict = {}   # (crew, day) -> hours booked
        self.crew_site: dict = {}    # (crew, day) -> site
        self.inspections: dict = {}  # day -> checks booked

    def reserve(self, crew: str, turbine: str, site: str, hours: list) -> None:
        for d, h in hours:
            self.crew.add((crew, d, h))
            self.turbine.add((turbine, d, h))
            self.crew_hours[(crew, d)] = self.crew_hours.get((crew, d), 0) + 1
            self.crew_site[(crew, d)] = site
            self.trips.add((crew, site, d))


def _slot_hours(task: Task, c: Constraints, grid: CostGrid, day: int, start: int) -> list | None:
    """The hours worked from a start. A job that fits in a crew day is done in one go; a longer one continues from the next morning, with no gap day."""
    ok, (s0, s1) = grid.ok.get(task.turbine), task.hours or c.shift
    if ok is None:
        return None
    need, hours, d, h, n_today = math.ceil(task.duration_h), [], day, start, 0
    one_day = need <= c.hours_per_crew
    while need > 0:
        if d >= grid.days:
            return None
        if h >= s1 or n_today >= c.hours_per_crew:
            if one_day:
                return None  # A short job is not split overnight.
            if n_today == 0 and d != day:
                return None  # A whole continuing day without a safe hour: the job would be split by a gap.
            d, h, n_today = d + 1, s0, 0
            continue
        if not ok[d, h]:
            if (d == day and not hours) or one_day:
                return None  # A job starts on a safe hour, and a short one runs without a break.
            if n_today == 0:
                h += 1  # On a continuing day, start at the first safe hour.
                continue
            d, h, n_today = d + 1, s0, 0  # Stopped by the weather: carry on next morning.
            continue
        hours.append((d, h))
        need, n_today, h = need - 1, n_today + 1, h + 1
    return hours


def _crew_for(hours: list, site: str, c: Constraints, b: Bookings, grid: CostGrid) -> str | None:
    per_day = pd.Series([d for d, _ in hours]).value_counts().to_dict()
    for crew in c.crews:
        if any((crew, d, h) in b.crew or c.crew_off(crew, grid.date(d)) or b.crew_site.get((crew, d), site) != site for d, h in hours):
            continue
        if all(b.crew_hours.get((crew, d), 0) + n <= c.hours_per_crew for d, n in per_day.items()):
            return crew
    return None


def _breakdown(task: Task, grid: CostGrid, hours: list, day: int, bonus: float = 0.0, stops: bool = True) -> dict:
    down = float(sum(grid.cost[task.turbine][d, h] for d, h in hours)) if stops else 0.0
    wait = task.loss_mwh_per_day * day
    risk = task.p_fail_by_day[min(day, len(task.p_fail_by_day) - 1)] * task.fail_cost_mwh_eq if task.p_fail_by_day and task.fail_cost_mwh_eq > 0 else 0.0
    return {"downtime": round(down, 3), "waiting": round(wait, 3), "risk": round(risk, 3), "travel_credit": round(-bonus, 3), "total": round(down + wait + risk - bonus, 3)}


def _best_on_day(task: Task, d: int, c: Constraints, grid: CostGrid, b: Bookings) -> tuple:
    """The cheapest slot on one day, or the reason there is none."""
    day = grid.date(d)
    if task.kind == "revision" and task.due_end and task.due_end >= c.today and not (task.due_start <= day <= task.due_end):
        return None, "outside the revision's window"
    if grid.ok.get(task.turbine) is None:
        return None, "no cost data for this turbine"
    s0, s1 = task.hours or c.shift
    reason = "no safe hours in the shift"
    if not grid.ok[task.turbine][d, s0:s1].any():
        v = grid.v[task.turbine][d, s0:s1]
        reason = f"unsafe all shift: hub wind up to {np.nanmax(v):.1f} m/s, or gusts or rain" if np.isfinite(v).any() else "outside the forecast, or already past"
    best, no_crew = None, False
    for start in range(s0, s1):
        hours = _slot_hours(task, c, grid, d, start)
        if hours is None or any((task.turbine, x, y) in b.turbine for x, y in hours):
            continue
        crew = _crew_for(hours, task.site, c, b, grid)
        if crew is None:
            no_crew = True
            continue
        bonus = TRIP_BONUS_MWH if b.crew_site.get((crew, d)) == task.site else 0.0
        cost = _breakdown(task, grid, hours, d, bonus)
        if best is None or cost["total"] < best["cost"]["total"] - 1e-9:
            best = {"day": d, "start_hour": hours[0][1], "hours": hours, "crew": crew, "cost": cost}
    if best is None and no_crew:
        reason = "no crew free"
    return best, reason


def _plan_inspection(task: Task, c: Constraints, grid: CostGrid, b: Bookings) -> tuple:
    """A drone or remote check: stops nothing, has its own team, and is limited by gusts, not the climbing limit. Earliest day wins."""
    s0, s1 = task.hours or c.shift
    rejected = []
    for d in range(grid.days):
        g = grid.gust_ok.get(task.turbine)
        if b.inspections.get(d, 0) >= c.inspections_per_day:
            rejected.append({"day": d, "reason": "inspection team fully booked"})
        elif g is None or not g[d, s0:s1].any():
            rejected.append({"day": d, "reason": "gusts too strong to fly, or outside the forecast"})
        else:
            hour = s0 + int(np.argmax(g[d, s0:s1]))
            return {"day": d, "start_hour": hour, "hours": [(d, hour)], "crew": "drone-team", "cost": _breakdown(task, grid, [], d, stops=False)}, rejected
    return None, rejected


def _why_not(alt: dict, chosen: dict) -> str:
    a, c = alt["cost"], chosen["cost"]
    parts = [f"more {k}: {a[k]:.1f} vs. {c[k]:.1f} MWh" for k in ("downtime", "waiting", "risk") if a[k] > c[k] + 0.01]
    return "; ".join(parts) or "similar cost; lost on a tie or to the travel credit"


def _urgency(t: Task):
    """Revisions first (hard windows, tightest first), then repairs by daily loss, then checks. Ties break on task_id, so runs repeat exactly."""
    if t.kind == "revision":
        return (0, t.due_end or dt.date.max, t.task_id)
    if t.kind == "inspection":
        return (2, -t.loss_mwh_per_day, t.task_id)
    return (1, -max(t.loss_mwh_per_day, max(t.p_fail_by_day or [0]) * t.fail_cost_mwh_eq), t.task_id)


def optimize(tasks: list[Task], cost_map: pd.DataFrame, c: Constraints, tz: str = "UTC", forecast: dict | None = None) -> dict:
    grid, b = CostGrid(cost_map, c.today, c.horizon_days, tz), Bookings()
    horizon_end = grid.date(c.horizon_days - 1)
    jobs, unscheduled, warnings, revisions = [], [], [], {}
    if forecast and forecast.get("stale"):
        warnings.append(f"Stale forecast: planned on the copy saved {forecast.get('fetched_utc')}.")
    for task in sorted(tasks, key=_urgency):
        if task.kind == "revision":
            if task.due_start and task.due_start > horizon_end:
                unscheduled.append({"task_id": task.task_id, "reason": f"window opens {task.due_start}, after the horizon"})
                continue
            if task.due_end and task.due_end < c.today:
                warnings.append(f"{task.task_id} is overdue: its window ended {task.due_end}.")
        if task.kind == "inspection":
            chosen, rejected = _plan_inspection(task, c, grid, b)
            if chosen is None:
                unscheduled.append({"task_id": task.task_id, "reason": "no inspection slot in the horizon"})
                continue
            b.inspections[chosen["day"]] = b.inspections.get(chosen["day"], 0) + 1
            jobs.append(_job(task, chosen, [{"date": grid.date(r["day"]).isoformat(), "total": None, "why_not": r["reason"]} for r in rejected][:N_ALTERNATIVES], [], grid, forecast))
            continue
        options, reasons = [], []
        for d in range(c.horizon_days):
            best, why = _best_on_day(task, d, c, grid, b)
            (options.append(best) if best else reasons.append({"day": d, "reason": why}))
        earliest = min(options, key=lambda x: x["day"]) if options else None
        forced = bool(task.first_detected and (c.today - task.first_detected).days >= MAX_DEFERRAL_DAYS)
        if forced and options:
            options = [earliest]
        # Grouping (CLAUDE.md 9.6): a repair done inside the revision's stop costs no extra downtime.
        rev = revisions.get(task.turbine)
        if task.kind == "repair" and rev and rev["duration_h"] >= task.duration_h and not forced:
            options.append({"day": rev["day"], "start_hour": rev["start_hour"], "hours": [], "crew": rev["crew"], "cost": _breakdown(task, grid, [], rev["day"]), "bundled_with": rev["task_id"]})
        if not options:
            unscheduled.append({"task_id": task.task_id, "reason": reasons[0]["reason"] if reasons else "no feasible slot",
                                "per_day": [{"date": grid.date(r["day"]).isoformat(), "reason": r["reason"]} for r in reasons]})
            continue
        options.sort(key=lambda x: (x["cost"]["total"], x["day"]))
        chosen = options[0]
        # An uncertain forecast on the best slot: check first rather than commit a crew (CLAUDE.md 9.6, step 3).
        if task.kind == "repair" and chosen["hours"] and not task.trace.get("confirmed_by_inspection"):
            spread = sum(float(grid.p90[task.turbine][x, y] - grid.cost[task.turbine][x, y]) for x, y in chosen["hours"])
            if spread > FORECAST_SPREAD_MAX_MWH:
                task.trace["converted_to_inspection"] = f"forecast spread {spread:.1f} MWh on the best slot"
                task.kind, task.job_type, task.duration_h = "inspection", "uncertain_forecast_check", float(INSPECTION_HOURS)
                alt, _ = _plan_inspection(task, c, grid, b)
                if alt:
                    b.inspections[alt["day"]] = b.inspections.get(alt["day"], 0) + 1
                    jobs.append(_job(task, alt, [], [], grid, forecast))
                else:
                    unscheduled.append({"task_id": task.task_id, "reason": "uncertain forecast and no inspection slot"})
                continue
        if chosen["hours"]:
            b.reserve(chosen["crew"], task.turbine, task.site, chosen["hours"])
        alts = [{"date": grid.date(x["day"]).isoformat(), "total": x["cost"]["total"], "why_not": _why_not(x, chosen)} for x in options[1:N_ALTERNATIVES + 1]]
        alts += [{"date": grid.date(r["day"]).isoformat(), "total": None, "why_not": r["reason"]} for r in reasons if r["reason"] != "outside the revision's window"][:N_ALTERNATIVES]
        jobs.append(_job(task, chosen, alts, [earliest] if earliest else [], grid, forecast, forced))
        if task.kind == "revision":
            revisions[task.turbine] = {"task_id": task.task_id, "crew": chosen["crew"], "start_hour": chosen["start_hour"], "day": chosen["day"], "duration_h": task.duration_h}
    jobs.sort(key=lambda j: (j["date"], j["start_hour"], j["task_id"]))
    totals = {k: round(sum(j["cost_breakdown_mwh"][k] for j in jobs), 3) for k in ("downtime", "waiting", "risk", "total")}
    key = json.dumps([[t.task_id, t.loss_mwh_per_day, t.duration_h] for t in tasks] + [c.today.isoformat(), c.unavailable, c.crews_per_day], default=str)
    return {"schedule_id": hashlib.sha1(key.encode()).hexdigest()[:10], "as_of": c.today.isoformat(), "horizon_days": c.horizon_days, "timezone": tz,
            "constraints": {"crews_per_day": c.crews_per_day, "hours_per_crew": c.hours_per_crew, "shift_local": list(c.shift), "unavailable": c.unavailable},
            "forecast": forecast or {}, "totals_mwh": totals, "trips": len(b.trips), "warnings": warnings, "jobs": jobs, "unscheduled": unscheduled}


def _job(task: Task, chosen: dict, alternatives: list, earliest: list, grid: CostGrid, forecast, forced: bool = False) -> dict:
    day = grid.date(chosen["day"])
    trace = {**task.trace, "m4_forecast_fetched": (forecast or {}).get("fetched_utc")}
    if forced:
        trace["forced_by_max_deferral"] = True
    cost = {k: chosen["cost"][k] for k in ("downtime", "waiting", "risk", "total")}
    job = {"task_id": task.task_id, "turbine": task.turbine, "site": task.site, "kind": task.kind, "job_type": task.job_type, "date": day.isoformat(), "start_hour": chosen["start_hour"],
           "duration_h": task.duration_h, "crew": chosen["crew"], "hours": [[grid.date(d).isoformat(), h] for d, h in chosen["hours"]], "confidence": task.confidence,
           "cost_breakdown_mwh": cost, "alternatives": alternatives, "trace": trace}
    if "bundled_with" in chosen:
        job["bundled_with"] = chosen["bundled_with"]
    if earliest and earliest[0]["day"] != chosen["day"]:
        job["earliest_option"] = {"date": grid.date(earliest[0]["day"]).isoformat(), "total": earliest[0]["cost"]["total"]}
    job["why"] = _why(job, day)
    return job


def _why(job: dict, day: dt.date) -> str:
    c = job["cost_breakdown_mwh"]
    when = f"{day:%a %d %b} from {job['start_hour']:02d}:00"
    if job["kind"] == "inspection":
        return f"{job['job_type'].replace('_', ' ').capitalize()} on {when}: a drone or remote check stops nothing, so the earliest flyable day is best."
    text = f"{when}: {c['downtime']:.1f} MWh lost while stopped + {c['waiting']:.1f} MWh lost while waiting = {c['total']:.1f} MWh."
    if job.get("bundled_with"):
        text += f" Done inside {job['bundled_with']}, whose stop already pays the downtime."
    priced = [a for a in job["alternatives"] if a["total"] is not None]
    if job.get("earliest_option"):
        e = job["earliest_option"]
        text += f" The earliest possible day, {pd.Timestamp(e['date']):%a %d %b}, would cost {e['total']:.1f} MWh."
    elif priced:
        text += f" Next best: {pd.Timestamp(priced[0]['date']):%a %d %b} at {priced[0]['total']:.1f} MWh ({priced[0]['why_not']})."
    return text


# ---------- A real calendar ----------
REPLAY_AS_OF = pd.Timestamp("2019-12-15")  # Kelmarsh data replayed as "today's" SCADA: a date M1 and M2 never trained on.
DEMO_FAULTS = [  # Injected so the demo has known problems to find. Not PPC data, and labelled so in every output.
    {"turbine": "KWF2", "cause": "pitch_fault", "start": "2019-12-03", "severity": {"k": 0.85}},
    {"turbine": "KWF5", "cause": "yaw_misalignment", "start": "2019-12-05", "severity": {"angle_deg": 12.0}},
    {"turbine": "KWF3", "cause": "anemometer_bias", "start": "2019-12-01", "severity": {"b": 0.08}},
]


def replayed_findings(site: str, today: dt.date) -> list[dict]:
    """M2 on Kelmarsh data up to REPLAY_AS_OF, with DEMO_FAULTS injected. Kelmarsh turbine n stands in for the site's turbine n."""
    import inject_faults as faults
    import m1_expected as m1
    import m2_underperf as m2
    d = m1.load()
    d = d[(d.ts >= REPLAY_AS_OF - pd.Timedelta(days=60)) & (d.ts < REPLAY_AS_OF)].sort_values(["turbine", "ts"]).reset_index(drop=True)
    f = pd.DataFrame([{"fault_id": i, **x, "start": pd.Timestamp(x["start"]), "end": REPLAY_AS_OF} for i, x in enumerate(DEMO_FAULTS)])
    faulted, _ = faults.apply(d, f)
    found = m2.detect_as_of(m1.predict_expected(faulted), REPLAY_AS_OF)
    names = dict(zip(m4.fleet("kelmarsh").turbine, m4.fleet(site).turbine))
    for x in found:
        x["replayed_from"] = f"{x['turbine']} on {REPLAY_AS_OF:%Y-%m-%d}, with injected faults"
        x["turbine"] = names[x["turbine"]]
        x["first_detected"] = today - dt.timedelta(days=(REPLAY_AS_OF - pd.Timestamp(x["alarm_ts"])).days)
    return found


def plan(sites: list[str], unavailable: list | None = None) -> dict:
    """One calendar for one or more sites: wind costs from m4_cost, solar costs from pv, each site's replayed findings and revisions."""
    import pv
    tzs = {m4.SITES[x]["tz"] for x in sites}
    if len(tzs) > 1:
        raise SystemExit(f"Sites in one calendar must share a time zone, got {sorted(tzs)}")
    tz, now = tzs.pop(), pd.Timestamp.now(tz="UTC").floor("h").tz_localize(None)
    today = pd.Timestamp.now(tz=tz).date()
    c = Constraints(today, crews_per_day=CREWS_PER_DAY * len(sites), unavailable=unavailable or [])  # Each site keeps its own crews.
    maps, tasks, findings, skipped, metas = [], [], [], [], {}
    for site in sites:
        solar = m4.SITES[site].get("type", "wind") == "pv"
        w, metas[site] = m4.get_forecast(site, pv.PV_FORECAST_VARS if solar else m4.FORECAST_VARS)
        cm = pv.cost_map(site, w) if solar else m4.cost_map(site, w)
        maps.append(cm[cm.hour_utc >= now])
        found = pv.replayed_findings(site, today) if solar else replayed_findings(site, today)
        findings += found
        if solar:
            rain = pv.days_to_rain(w, tz, today)
            metas[site]["days_to_rain"] = rain
            tasks += tasks_from_findings(found, site, today, rain, skipped)
        else:
            tasks += generate_revisions(list(m4.fleet(site).turbine), site, today, today + dt.timedelta(days=c.horizon_days - 1)) + tasks_from_findings(found, site, today)
    cm = pd.concat(maps, ignore_index=True)
    meta = {"fetched_utc": min(m["fetched_utc"] for m in metas.values()), "stale": any(m.get("stale") for m in metas.values()), "sites": metas}
    s = optimize(tasks, cm, c, tz, meta)
    s["unscheduled"] += skipped
    s["inputs"] = {"m2_findings": [{k: v for k, v in x.items() if k != "reasons"} for x in findings],
                   "note": "Findings are replayed, not live PPC data: wind from Kelmarsh, solar from Kaggle plant 1, both with injected faults."}
    name = "_".join(sites)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"schedule_{name}.json").write_text(json.dumps(s, indent=1, default=str))
    _plot(s, cm, name)
    return s


def _plot(s: dict, cm: pd.DataFrame, name: str) -> None:
    import matplotlib.patches as mpatches
    plt = m4.m1.chart_style()
    C = m4.m1.COLORS
    tz, days = s["timezone"], s["horizon_days"]
    turbines = list(dict.fromkeys(cm.turbine))
    grid = CostGrid(cm, dt.date.fromisoformat(s["as_of"]), days, tz)
    kind_color = {"revision": C["ink"], "repair": C["blue"], "inspection": C["orange"]}
    fig, axes = plt.subplots(1, 2, figsize=(17, 0.55 * len(turbines) + 3), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    rated = float(cm.groupby("turbine").cost_mwh.max().median())
    img = np.vstack([grid.cost[t].ravel() for t in turbines])
    ax.imshow(np.nan_to_num(img, nan=0), aspect="auto", cmap="Blues", vmin=0, vmax=rated * 1.6, extent=(0, days * 24, len(turbines) - 0.5, -0.5))
    unsafe = np.vstack([~grid.ok[t].ravel() & np.isfinite(grid.cost[t].ravel()) for t in turbines])
    yy, xx = np.nonzero(unsafe)
    ax.scatter(xx + 0.5, yy, marker="|", s=40, color=C["orange"], alpha=0.5, linewidths=1)
    for j in s["jobs"]:
        y = turbines.index(j["turbine"])
        hrs = [(dt.date.fromisoformat(d) - grid.today).days * 24 + h for d, h in j["hours"]] or [(dt.date.fromisoformat(j["date"]) - grid.today).days * 24 + j["start_hour"]]
        for h in hrs:
            ax.add_patch(mpatches.Rectangle((h, y - 0.32), 1, 0.64, color=kind_color[j["kind"]], linewidth=0))
        label = j["job_type"].replace("_", " ").replace("revision ", "").replace(" repair", "")
        bundled = bool(j.get("bundled_with"))
        ax.text(hrs[0], y + (0.42 if bundled else -0.4), f"{label}{' (bundled into the revision)' if bundled else ''}", fontsize=7.5, color=C["ink"], va="top" if bundled else "bottom")
    ax.set_yticks(range(len(turbines)), turbines, fontsize=9)
    ax.set_xticks([d * 24 + 12 for d in range(days)], [f"{grid.date(d):%a %d}" for d in range(days)], fontsize=8.5)
    for d in range(1, days):
        ax.axvline(d * 24, color=C["grid"], linewidth=1)
    ax.grid(False); ax.set_xlim(0, days * 24)
    handles = [mpatches.Patch(color=v, label=k) for k, v in kind_color.items()] + [plt.Line2D([], [], marker="|", linestyle="", color=C["orange"], label="unsafe hour")]
    ax.legend(handles=handles, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    ax.set_title(f"Maintenance calendar, {name.replace('_', ' + ').title()}: background is production lost per stopped hour")
    ax = axes[1]
    priced = [j for j in s["jobs"] if j["kind"] != "inspection"]
    y = np.arange(len(priced))
    chosen = [j["cost_breakdown_mwh"]["total"] for j in priced]
    early = [j.get("earliest_option", {}).get("total", j["cost_breakdown_mwh"]["total"]) for j in priced]
    ax.barh(y - 0.2, early, 0.38, color=C["orange"], label="Earliest possible day")
    ax.barh(y + 0.2, chosen, 0.38, color=C["blue"], label="Chosen day")
    for i, j in enumerate(priced):
        ax.text(max(chosen[i], early[i]) + 0.1, i, f"{pd.Timestamp(j['date']):%a %d}", va="center", fontsize=8, color=C["muted"])
    ax.set_yticks(y, [j["task_id"] for j in priced], fontsize=8); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("MWh lost (downtime + waiting)"); ax.set_title("Cost of each crew job: chosen vs. earliest day"); ax.legend(fontsize=8, loc="lower right")
    fig.text(0.01, 0.005, f"Forecast fetched {s['forecast'].get('fetched_utc')}{' (STALE)' if s['forecast'].get('stale') else ''}; Open-Meteo.com. Findings replayed with injected faults (wind: Kelmarsh; solar: Kaggle). "
             f"Crews, limits and job lengths are assumptions.", fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / f"schedule_{name}.png", dpi=150, facecolor=C["surface"])
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", nargs="+", choices=sorted(m4.SITES), required=True, help="One or more sites in one calendar, e.g. --site fantanele calugareni")
    ap.add_argument("--crew-off", nargs=2, action="append", metavar=("CREW", "DATE"), help="e.g. --crew-off crew-2 2026-11-12")
    a = ap.parse_args()
    s = plan(a.site, [{"crew": c, "date": d} for c, d in a.crew_off or []])
    print(f"Schedule {s['schedule_id']} for {a.site}, as of {s['as_of']}: totals {s['totals_mwh']}, {s['trips']} crew trips")
    for w_ in s["warnings"]:
        print("WARNING:", w_)
    for j in s["jobs"]:
        print(f"  {j['task_id']:<32} {j['crew']:<10} {j['why']}")
    for u in s["unscheduled"]:
        print(f"  UNSCHEDULED {u['task_id']}: {u['reason']}")
