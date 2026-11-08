"""Live backend: the agent's tools answered by the real producer pipeline.

Every number comes from src.decision.optimizer: plan_inputs() (today's Open-Meteo forecast -> M4 cost map;
replayed M2 findings and the service calendar -> jobs) and optimize() (the calendar, its cost breakdowns,
the options it beat and why, and the trace back to the model outputs).

plan_inputs() takes about 10 s per site (M2 replays weeks of SCADA), so it is computed once per site and hour
and reused: a what-if ("crew 2 is off on Thursday") only re-runs optimize(), which takes milliseconds.

A tool with no model behind it yet is not answered here: it returns an error and stays on "fixture" in
config/agent.yaml, so the provenance shown to the user is always true.
"""

import copy
import datetime as dt
import json
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.agent.backends.fixture import DEFAULT_FIXTURE_BACKEND, FixtureBackend
from src.agent.schemas.common import ErrorCode, StructuredError

ROOT = Path(__file__).resolve().parents[3]
PARAMS = ROOT / "config" / "params.yaml"
SITES = ("fantanele", "calugareni")    # The PPC parks the agent plans for (config/sites in src/producers/downtime_cost.py)
PREFIX_SITE = {"FAN-": "fantanele", "CAL-": "calugareni"}
DEFAULT_SITE = "fantanele"             # Used when a what-if names no site and no live plan exists yet
INPUTS_TTL_S = 3600                    # Refresh the forecast and findings at most once an hour
PERIOD_DAYS = {"week": 7, "month": 30}
KNOWN_CONSTRAINTS = {"crews_per_day", "unavailable", "crew", "crew_id", "date", "day", "defer", "task_id", "defer_days"}

WEEKDAYS = {"luni": 0, "marti": 1, "miercuri": 2, "joi": 3, "vineri": 4, "sambata": 5, "duminica": 6,
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
RELATIVE_DAYS = {"azi": 0, "astazi": 0, "today": 0, "maine": 1, "tomorrow": 1, "poimaine": 2}

KIND_TO_JOB_TYPE = {"revision": "periodic_revision", "repair": "repair", "inspection": "inspection"}
TITLES = {  # Romanian labels for the optimizer's job types (the agent answers in Romanian)
    "revision_semiannual": "Revizie semestrială", "revision_annual": "Revizie anuală",
    "pitch_fault_repair": "Reparație sistem pitch", "yaw_misalignment_repair": "Corecție aliniere yaw",
    "sensor_check": "Verificare anemometru (dronă / la distanță)", "icing_check": "Verificare givraj",
    "low_confidence_check": "Verificare (încredere scăzută)", "no_loss_check": "Verificare (fără pierdere măsurabilă)",
    "uncertain_forecast_check": "Verificare (prognoză incertă)", "cleaning": "Curățare panouri",
    "string_repair": "Reparație string", "shading_removal": "Eliminare umbrire", "inverter_check": "Verificare invertor",
}


class _Fail(Exception):
    """A clear, structured refusal (unknown asset, bad date, unsupported constraint) raised inside a handler."""

    def __init__(self, code: ErrorCode, message: str, **extra: Any):
        super().__init__(message)
        self.error = StructuredError(code=code, message=message, **extra)


def energy_price() -> float:
    """RON per MWh, from config/params.yaml (an assumption, documented there)."""
    return float(yaml.safe_load(PARAMS.read_text(encoding="utf-8"))["economics"]["energy_price_ron_per_mwh"])


def _plain(s: Any) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower().strip()


def parse_day(value: Any, today: dt.date) -> dt.date:
    """YYYY-MM-DD, a weekday (Romanian or English, with or without diacritics) or azi/mâine: the next such day from today."""
    s = _plain(value)
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        pass
    if s in RELATIVE_DAYS:
        return today + dt.timedelta(days=RELATIVE_DAYS[s])
    if s in WEEKDAYS:
        return today + dt.timedelta(days=(WEEKDAYS[s] - today.weekday()) % 7)
    raise _Fail(ErrorCode.INVALID_DATE, f"Unrecognised date {value!r}: use YYYY-MM-DD or a weekday name", date_str=str(value))


def parse_crew(value: Any) -> str:
    m = re.search(r"\d+", str(value or ""))
    if not m:
        raise _Fail(ErrorCode.UNSUPPORTED_SCENARIO, f"Unrecognised crew {value!r}: use crew-1, crew-2, ...")
    return f"crew-{int(m.group())}"


def _jsonable(o: Any) -> Any:
    """Dates, numpy numbers and NaN into plain JSON values, recursively."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if o != o else float(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


def _num(v: Any, nd: int = 4) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, nd)


def _provenance(tool: str, x: Optional[dict] = None) -> Dict[str, Any]:
    return {"result_id": f"res-{tool[:8]}-{uuid.uuid4().hex[:6]}", "tool": tool, "backend": "live",
            "as_of": x["today"].isoformat() if x else dt.date.today().isoformat(),
            "forecast_run": x["forecast"]["fetched_utc"] if x else None,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}


class LiveBackend:
    """Adapters from the agent's tool contracts (src/agent/schemas) to the producer pipeline."""

    def __init__(self, fixture_backend: Optional[FixtureBackend] = None):
        self._fixture_backend = fixture_backend or DEFAULT_FIXTURE_BACKEND
        self._lock = threading.Lock()
        self._inputs: Dict[str, tuple] = {}            # site -> (monotonic time, plan_inputs result)
        self._plans: Dict[str, Dict[str, Any]] = {}    # schedule_id -> optimize_maintenance output, newest last
        self.task_overrides: Dict[str, Dict[str, Any]] = {}  # task_id -> confirmed field report

    # ---------- Dispatch ----------
    def call(self, tool_name: str, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        handler = getattr(self, f"handle_{tool_name}", None)
        if handler is None:
            return StructuredError(code=ErrorCode.UNSUPPORTED_SCENARIO,
                                   message=f"No live backend for '{tool_name}' yet: set it to \"fixture\" in config/agent.yaml.", tool_name=tool_name)
        try:
            return handler(args or {}, store=store)
        except _Fail as f:
            return f.error
        except SystemExit as e:  # get_forecast exits when Open-Meteo fails and nothing is cached; never let it stop the server
            return StructuredError(code=ErrorCode.BACKEND_CONTRACT_VIOLATION, message=str(e), tool_name=tool_name)
        except FileNotFoundError as e:
            return StructuredError(code=ErrorCode.BACKEND_CONTRACT_VIOLATION, tool_name=tool_name,
                                   message=f"Live data or models missing ({e}). Run `make data` and `make train`, or set the tool to \"fixture\".")

    # ---------- Shared inputs ----------
    def _inputs_for(self, site: str) -> dict:
        """plan_inputs for one site, cached for an hour (and never across midnight)."""
        from src.decision.optimizer import plan_inputs
        with self._lock:
            hit = self._inputs.get(site)
            if hit and time.monotonic() - hit[0] < INPUTS_TTL_S and hit[1]["today"] == dt.date.today():
                return hit[1]
            x = plan_inputs([site])
            self._inputs[site] = (time.monotonic(), x)
            return x

    @staticmethod
    def _fleet(site: str) -> List[str]:
        from src.producers import downtime_cost
        return list(downtime_cost.fleet(site).turbine)

    @staticmethod
    def _check_site(site_id: str) -> str:
        if site_id not in SITES:
            raise _Fail(ErrorCode.UNKNOWN_ITEM, f"Unknown site '{site_id}'", valid_ids=list(SITES), item_id=site_id)
        return site_id

    def _sites_for(self, site_id: Optional[str], asset_ids: List[str]) -> List[str]:
        if site_id:
            return [self._check_site(site_id)]
        if asset_ids:
            sites = []
            for a in asset_ids:
                site = next((s for p, s in PREFIX_SITE.items() if str(a).startswith(p)), None)
                if site is None:
                    raise _Fail(ErrorCode.UNKNOWN_ASSET, f"Asset '{a}' belongs to no known site", valid_ids=sorted(sum((self._fleet(s) for s in SITES), [])))
                sites.append(site)
            return sorted(set(sites), key=SITES.index)
        return list(SITES)

    def _check_assets(self, asset_ids: List[str], sites: List[str]) -> None:
        fleet = sorted(sum((self._fleet(s) for s in sites), []))
        unknown = [a for a in asset_ids if a not in fleet]
        if unknown:
            raise _Fail(ErrorCode.UNKNOWN_ASSET, f"Asset '{unknown[0]}' is not in the live fleet", valid_ids=fleet)

    def known_task_ids(self) -> set:
        """Task ids a field report may refer to: every job in today's live inputs."""
        ids = set()
        for site in SITES:
            ids |= {t.task_id for t in self._inputs_for(site)["tasks"]}
        return ids

    # ---------- Optimizer ----------
    def _constraints(self, raw: Dict[str, Any], today: dt.date) -> tuple:
        unknown = set(raw) - KNOWN_CONSTRAINTS
        if unknown:
            raise _Fail(ErrorCode.UNSUPPORTED_SCENARIO, f"Unsupported constraints {sorted(unknown)}; supported: {sorted(KNOWN_CONSTRAINTS)}",
                        details={"constraints": raw})
        from src.decision.optimizer import CREWS_PER_DAY
        items = list(raw.get("unavailable") or [])
        if raw.get("crew") or raw.get("crew_id"):
            items.append({"crew": raw.get("crew") or raw.get("crew_id"), "date": raw.get("date") or raw.get("day")})
        unavailable = []
        for u in items:
            if not isinstance(u, dict) or not (u.get("date") or u.get("day")):
                raise _Fail(ErrorCode.UNSUPPORTED_SCENARIO, f"Each unavailable entry needs a crew and a date, got {u!r}")
            unavailable.append({"crew": parse_crew(u.get("crew") or u.get("crew_id")), "date": parse_day(u.get("date") or u.get("day"), today).isoformat()})
        defer = {d["task_id"]: int(d["days"]) for d in raw.get("defer") or [] if isinstance(d, dict) and "task_id" in d and "days" in d}
        if raw.get("task_id") and raw.get("defer_days") is not None:
            defer[raw["task_id"]] = int(raw["defer_days"])
        return int(raw.get("crews_per_day") or CREWS_PER_DAY), unavailable, defer

    def _jobs_in(self, x: dict, defer: Dict[str, int]) -> list:
        """The site's jobs after confirmed field reports (done: dropped; deferred: not before today + N days) and what-if deferrals."""
        ids = {t.task_id for t in x["tasks"]}
        missing = [t for t in defer if t not in ids]
        if missing:
            raise _Fail(ErrorCode.UNKNOWN_ITEM, f"Task '{missing[0]}' is not open at this site", item_id=missing[0], valid_ids=sorted(ids))
        out = []
        for t in x["tasks"]:
            o = self.task_overrides.get(t.task_id, {})
            if o.get("status") in ("done", "not_found"):
                continue
            days = defer.get(t.task_id, o.get("defer_days"))
            t = copy.deepcopy(t)
            if days:
                start = x["today"] + dt.timedelta(days=int(days))
                t.due_start = max(t.due_start, start) if t.due_start else start
                if t.due_end and t.due_end < t.due_start:
                    t.due_end = t.due_start
                t.trace = {**t.trace, "deferred_days": int(days), "deferred_by": "what-if" if t.task_id in defer else "field report"}
            if o:
                t.trace = {**t.trace, "field_report": o}
            out.append(t)
        return out

    def _run(self, site: str, constraints: Dict[str, Any]) -> tuple:
        from src.decision.optimizer import Constraints, optimize
        x = self._inputs_for(site)
        crews, unavailable, defer = self._constraints(constraints, x["today"])
        try:
            c = Constraints(x["today"], crews_per_day=crews, unavailable=unavailable)
        except ValueError as e:
            raise _Fail(ErrorCode.UNSUPPORTED_SCENARIO, str(e))
        s = optimize(self._jobs_in(x, defer), x["cost_map"], c, x["tz"], x["forecast"])
        s["unscheduled"] += x["skipped"]
        return x, s

    @staticmethod
    def _job_out(j: dict, price: float) -> Dict[str, Any]:
        from src.decision.optimizer import HOURS_PER_CREW
        cost = {k: j["cost_breakdown_mwh"][k] for k in ("downtime", "waiting", "risk", "total")}
        n = len([h for d, h in j["hours"] if d == j["date"]]) or (1 if j["kind"] == "inspection" else int(min(j["duration_h"], HOURS_PER_CREW)))
        trace = {**j["trace"], "kind": j["kind"], "job_type": j["job_type"], "confidence": j["confidence"]}
        for k in ("bundled_with", "earliest_option"):
            if k in j:
                trace[k] = j[k]
        return _jsonable({
            "task_id": j["task_id"], "asset_id": j["turbine"], "site": j["site"], "kind": j["kind"], "job_type": j["job_type"],
            "title": TITLES.get(j["job_type"], j["job_type"]), "date": j["date"], "start_hour": j["start_hour"],
            "duration_hours": float(j["duration_h"]), "crew": j["crew"], "hours": j["hours"],
            "window_start": f"{j['start_hour']:02d}:00", "window_end": f"{j['start_hour'] + n:02d}:00",
            "cost_mwh": cost["total"], "cost_ron": round(cost["total"] * price, 2),
            "cost_breakdown_mwh": cost, "cost_breakdown_ron": {k: round(v * price, 2) for k, v in cost.items()},
            "alternatives": [{"date": a["date"], "total_mwh": a["total"], "total_ron": None if a["total"] is None else round(a["total"] * price, 2),
                              "why_not": a["why_not"], "feasible": a["total"] is not None} for a in j["alternatives"]],
            "why": j["why"], "trace": trace,
        })

    def _schedule_out(self, site: str, x: dict, s: dict) -> Dict[str, Any]:
        from src.decision.optimizer import REPLAY_NOTE
        price = energy_price()
        total = s["totals_mwh"]["total"]
        return _jsonable({
            "schedule_id": f"sched-{site[:3]}-{s['schedule_id']}", "status": "draft", "site_id": site, "live_sites": [site], "as_of": s["as_of"],
            "jobs": [self._job_out(j, price) for j in s["jobs"]],
            "total_mwh": round(total, 3), "total_ron": round(total * price, 2), "totals_breakdown_mwh": s["totals_mwh"],
            "trips": s["trips"], "warnings": s["warnings"], "unscheduled": s["unscheduled"], "constraints": s["constraints"],
            "forecast": {"fetched_utc": x["forecast"]["fetched_utc"], "stale": x["forecast"]["stale"]},
            "energy_price_ron_per_mwh": price, "data_note": REPLAY_NOTE,
        })

    def _reference(self, site: str, out: dict, store: Optional[Any]) -> tuple:
        """The plan a what-if is compared with: the active plan if it is today's live plan for this site, else today's plan without constraints."""
        active = store.get_active_schedule() if store else None
        p = (active or {}).get("payload") or {}
        if p.get("live_sites") == [site] and p.get("as_of") == out["as_of"]:
            return active["id"], p
        x, s = self._run(site, {})
        return None, self._schedule_out(site, x, s)

    def handle_optimize_maintenance(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        site = args.get("site_id")
        if site is None:  # A what-if without a site ("crew 2 is off on Thursday") re-plans the site planned last
            site = next((p["site_id"] for p in reversed(list(self._plans.values()))), DEFAULT_SITE)
        site = self._check_site(site)
        x, s = self._run(site, args.get("constraints") or {})
        out = self._schedule_out(site, x, s)
        ref_id, ref = self._reference(site, out, store)
        before = {j["task_id"]: (j["date"], j["start_hour"], j["crew"]) for j in ref["jobs"]}
        now = {j["task_id"]: (j["date"], j["start_hour"], j["crew"]) for j in out["jobs"]}
        out.update({
            "parent_schedule_id": ref_id, "active_total_mwh": ref["total_mwh"], "active_total_ron": ref["total_ron"],
            "delta_mwh": round(out["total_mwh"] - ref["total_mwh"], 3), "delta_ron": round(out["total_ron"] - ref["total_ron"], 2),
            "moved_jobs": sorted([t for t in now if before.get(t) != now[t]] + [t for t in before if t not in now]),
            "provenance": _provenance("optimize_maintenance", x),
        })
        self._plans.pop(out["schedule_id"], None)
        self._plans[out["schedule_id"]] = out
        return out

    # ---------- Read tools ----------
    def handle_list_tasks(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        price = energy_price()
        tasks, x = [], None
        for site in self._sites_for(args.get("site_id"), []):
            x = self._inputs_for(site)
            for t in x["tasks"]:
                o = self.task_overrides.get(t.task_id, {})
                status = {"done": "completed", "not_found": "completed", "deferred": "deferred", "needs_part": "deferred"}.get(o.get("status"), "open")
                if t.kind == "revision":
                    left = (t.due_end - x["today"]).days if t.due_end else 99
                    urgency = "critical" if left < 0 else "high" if left <= 3 else "normal"
                    desc = f"Revizie din calendarul de service, scadentă {t.trace.get('due_date')}, fereastră {t.due_start} – {t.due_end}."
                else:
                    tr = t.trace
                    urgency = "high" if t.kind == "repair" and t.loss_mwh_per_day >= 1.0 else "normal" if t.kind == "repair" else "low"
                    desc = (f"M2: {tr.get('m2_cause')} (probabilitate {tr.get('m2_proba')}), pierdere {tr.get('m2_loss_mwh_day')} MWh/zi, "
                            f"PI 72h {tr.get('m2_pi_72h')}, încredere {t.confidence}. Date reluate, nu SCADA PPC.")
                job_type = KIND_TO_JOB_TYPE[t.kind]
                tasks.append(_jsonable({
                    "task_id": t.task_id, "asset_id": t.turbine, "site": site, "job_type": job_type, "type": job_type, "job_detail": t.job_type,
                    "title": TITLES.get(t.job_type, t.job_type), "description": desc,
                    "loss_mwh_per_day": round(t.loss_mwh_per_day, 3), "loss_ron_per_day": round(t.loss_mwh_per_day * price, 2),
                    "due_date": t.trace.get("due_date"), "due_start": t.due_start, "due_end": t.due_end,
                    "estimated_hours": float(t.duration_h), "requires_outage": t.kind != "inspection",
                    "urgency": urgency, "priority": urgency, "status": status, "confidence": t.confidence, "trace": t.trace,
                }))
        return {"tasks": tasks, "provenance": _provenance("list_tasks", x)}

    def handle_detect_underperformance(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        from src.decision.optimizer import REPLAY_NOTE
        price, asset_ids = energy_price(), list(args.get("asset_ids") or [])
        sites = self._sites_for(args.get("site_id"), asset_ids)
        self._check_assets(asset_ids, sites)
        dets, x = [], None
        for site in sites:
            x = self._inputs_for(site)
            jobs = {t.task_id: t for t in x["tasks"]}
            skipped = {u["task_id"]: u["reason"] for u in x["skipped"]}
            for i, f in enumerate([f for f in x["findings"] if f["site"] == site]):
                if asset_ids and f["turbine"] not in asset_ids:
                    continue
                tid = f"{f['turbine']}-{f['cause']}-{i + 1:02d}"  # The id tasks_from_findings gives this finding
                job = jobs.get(tid)
                planned = job.kind if job else "none"
                dets.append(_jsonable({
                    "asset_id": f["turbine"], "site": site, "task_id": tid if job else None,
                    "pi_72h": _num(f.get("pi_72h")), "loss_mwh_per_day": round(float(f["loss_mwh_per_day"]), 3),
                    "loss_ron_per_day": round(float(f["loss_mwh_per_day"]) * price, 2),
                    "cause": f["cause"], "cause_probability": float(f["cause_proba"]), "confidence": float(f["confidence"]),
                    "planned_action": planned, "planned_action_reason": skipped.get(tid),
                    "top_reasons": f.get("reasons", []), "alarm_ts": f.get("alarm_ts"), "first_detected": f.get("first_detected"),
                    "side": f.get("side"), "replayed_from": f.get("replayed_from"),
                }))
        return {"detections": dets, "checked_sites": sites, "data_note": REPLAY_NOTE, "provenance": _provenance("detect_underperformance", x)}

    def handle_compute_cost_map(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        import pandas as pd
        site = self._check_site(args.get("site_id"))
        x, price = self._inputs_for(site), energy_price()
        cm = x["cost_map"]
        end = pd.Timestamp(x["today"]) + pd.Timedelta(days=int(args.get("horizon_days") or 7))
        cm = cm[cm.hour_utc < end]
        local = cm.hour_utc.dt.tz_localize("UTC").dt.tz_convert(x["tz"])
        entries = [{"asset_id": r.turbine, "timestamp": f"{r.hour_utc:%Y-%m-%dT%H:00Z}", "date": f"{lt:%Y-%m-%d}", "hour": int(lt.hour),
                    "cost_mwh": round(float(r.cost_mwh), 4), "cost_mwh_p90": round(float(r.cost_mwh_p90), 4), "cost_ron": round(float(r.cost_mwh) * price, 2),
                    "workable": bool(r.workable), "wind_speed_ms": _num(r.v_hub, 2), "gust_ms": _num(r.gust_ms, 2), "rain_mm": _num(r.rain_mm, 2)}
                   for r, lt in zip(cm.itertuples(), local)]
        return {"site_id": site, "entries": entries, **self._staleness(x), "provenance": _provenance("compute_cost_map", x)}

    @staticmethod
    def _staleness(x: dict) -> Dict[str, Any]:
        fetched = dt.datetime.strptime(x["forecast"]["fetched_utc"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)
        age = (dt.datetime.now(dt.timezone.utc) - fetched).total_seconds() / 3600
        return {"stale": bool(x["forecast"]["stale"]), "age_hours": round(age, 1) if x["forecast"]["stale"] else None}

    def handle_get_weather_forecast(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        import pandas as pd
        site = self._check_site(args.get("site_id"))
        x = self._inputs_for(site)
        w = x["weather"][site]
        now = pd.Timestamp.now(tz="UTC").floor("h")
        w = w[(w.ts_utc >= now) & (w.ts_utc < now + pd.Timedelta(days=int(args.get("days") or 7)))]
        g = lambda r, c: _num(getattr(r, c, None), 2)  # noqa: E731  The wind and PV forecasts fetch different variables
        hourly = [{"timestamp": f"{r.ts_utc:%Y-%m-%dT%H:00Z}", "temperature_c": g(r, "temperature_2m"), "wind_speed_ms": g(r, "wind_speed_10m"),
                   "wind_speed_80m_ms": g(r, "wind_speed_80m"), "wind_speed_120m_ms": g(r, "wind_speed_120m"), "wind_gust_ms": g(r, "wind_gusts_10m"),
                   "wind_direction_deg": None, "surface_solar_radiation_wm2": g(r, "shortwave_radiation"), "precipitation_mm": g(r, "precipitation")}
                  for r in w.itertuples()]
        return {"site_id": site, "hourly": hourly, **self._staleness(x), "source": "Weather data by Open-Meteo.com",
                "provenance": _provenance("get_weather_forecast", x)}

    def handle_predict_expected_production(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        import pandas as pd
        asset_ids = list(args.get("asset_ids") or [])
        if not asset_ids:
            raise _Fail(ErrorCode.UNKNOWN_ASSET, "Name at least one asset", valid_ids=sorted(sum((self._fleet(s) for s in SITES), [])))
        sites = self._sites_for(None, asset_ids)
        self._check_assets(asset_ids, sites)

        def ts(v: str, end: bool = False) -> pd.Timestamp:
            t = pd.Timestamp(v)
            t = t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t
            return t + pd.Timedelta(days=1) if end and len(str(v)) == 10 else t  # A bare end date includes that day
        start, end = ts(args["start"]), ts(args["end"], end=True)
        preds, x, span = [], None, []
        for site in sites:
            x = self._inputs_for(site)
            cm = x["cost_map"]
            span += [cm.hour_utc.min(), cm.hour_utc.max()]
            cm = cm[cm.turbine.isin(asset_ids) & (cm.hour_utc >= start) & (cm.hour_utc < end)]
            preds += [{"asset_id": r.turbine, "timestamp": f"{r.hour_utc:%Y-%m-%dT%H:00Z}", "expected_kw_p10": round(r.cost_mwh_p10 * 1000, 1),
                       "expected_kw_p50": round(r.cost_mwh * 1000, 1), "expected_kw_p90": round(r.cost_mwh_p90 * 1000, 1)} for r in cm.itertuples()]
        if not preds:
            raise _Fail(ErrorCode.INVALID_DATE, f"Live expected production covers the forecast only: {min(span):%Y-%m-%d %H:00} to {max(span):%Y-%m-%d %H:00} UTC",
                        date_str=f"{args['start']}..{args['end']}")
        return {"predictions": preds, "model": "M1-W: weather-only expected production on the Open-Meteo forecast (M1-PV-W for solar)",
                "provenance": _provenance("predict_expected_production", x)}

    def handle_loss_overview(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        period = args.get("period", "week")
        if period not in PERIOD_DAYS:
            raise _Fail(ErrorCode.UNSUPPORTED_SCENARIO, f"Unknown period '{period}': use week or month")
        days, price, rows, x = PERIOD_DAYS[period], energy_price(), [], None
        under, conf, down = 0.0, [], 0.0
        for site in SITES:
            x, s = self._run(site, {})
            under += sum(float(f["loss_mwh_per_day"]) for f in x["findings"])
            conf += [float(f["confidence"]) for f in x["findings"]]
            horizon_end = x["today"] + dt.timedelta(days=days)
            down += sum(j["cost_breakdown_mwh"]["downtime"] for j in s["jobs"] if dt.date.fromisoformat(j["date"]) < horizon_end)
        rows.append({"category": "producers_underperformance", "mwh": round(under * days, 2), "ron": round(under * days * price, 2),
                     "description": f"Subperformanță detectată de M2 ({under:.2f} MWh/zi), dacă nu se repară, pe {days} zile. Date reluate, nu SCADA PPC.",
                     "source_ref": "detect_underperformance", "confidence": round(sum(conf) / len(conf), 3) if conf else None})
        rows.append({"category": "producers_downtime", "mwh": round(down, 2), "ron": round(down * price, 2),
                     "description": f"Producție pierdută în opririle planificate (planul de azi, orizont {x['today']} + {min(days, 10)} zile).",
                     "source_ref": "optimize_maintenance", "confidence": 1.0})
        demo = self._fixture_backend.call("loss_overview", {"period": period})
        for r in demo.get("rows", []):
            if r.get("category") == "consumers_suspect":
                rows.append({**r, "description": r["description"] + " (date demo: modulul consumatori nu este încă conectat)", "data_source": "fixture"})
        total = sum(r["mwh"] for r in rows)
        return {"period": period, "rows": rows, "total_mwh": round(total, 2), "total_ron": round(sum(r["ron"] for r in rows), 2),
                "energy_price_ron_per_mwh": price, "provenance": _provenance("loss_overview", x)}

    def handle_explain(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        item_id = str(args.get("item_id", ""))
        if item_id.upper().startswith("HH-"):  # Households: the consumers module is not connected; the answer says it is demo data
            return self._fixture_backend.call("explain", args)
        found = self._find_job(item_id)
        if found is None:
            site = next((s for p, s in PREFIX_SITE.items() if item_id.startswith(p)), None)
            if site:
                self.handle_optimize_maintenance({"site_id": site}, store=store)
                found = self._find_job(item_id)
        if found is None:
            latest = next(reversed(list(self._plans.values())), None)
            unscheduled = {u["task_id"]: u["reason"] for u in (latest or {}).get("unscheduled", [])}
            if item_id in unscheduled:
                raise _Fail(ErrorCode.UNSUPPORTED_SCENARIO, f"'{item_id}' is not in the calendar: {unscheduled[item_id]}", item_id=item_id)
            raise _Fail(ErrorCode.UNKNOWN_ITEM, f"Item '{item_id}' is not in the live calendar", item_id=item_id,
                        valid_ids=sorted(j["task_id"] for j in (latest or {}).get("jobs", [])))
        plan, job = found
        keys = ("task_id", "asset_id", "date", "crew", "cost_breakdown_mwh", "cost_breakdown_ron", "alternatives", "trace",
                "why", "title", "kind", "job_type", "start_hour", "window_start", "window_end", "duration_hours")
        return {"item_id": item_id, "item_type": "scheduled_job", "schedule_id": plan["schedule_id"],
                "scheduled_job_explanation": {k: job[k] for k in keys}, "provenance": {**_provenance("explain"), "as_of": plan["as_of"]}}

    def _find_job(self, task_id: str) -> Optional[tuple]:
        for plan in reversed(list(self._plans.values())):
            for j in plan["jobs"]:
                if j["task_id"] == task_id:
                    return plan, j
        return None

    # ---------- Write tools (run only after the user confirms, via the action gate) ----------
    def handle_update_from_field_report(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        from src.agent.actions import process_field_report
        res = process_field_report(args)
        if not res["needs_input"]:  # Confirmed and valid: the next plan drops a finished job or holds a deferred one
            self.task_overrides[args["task_id"]] = {k: args[k] for k in ("status", "defer_days", "new_finding", "confidence") if args.get(k) is not None}
        return res

    def handle_create_work_orders(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        from src.agent.store import DEFAULT_STORE
        sid = str(args.get("schedule_id", ""))
        plan = self._plans.get(sid)
        if plan is None:
            rec = (store or DEFAULT_STORE).get_schedule(sid)
            plan = (rec or {}).get("payload")
        if not plan or not plan.get("live_sites"):
            raise _Fail(ErrorCode.UNKNOWN_ITEM, f"Schedule '{sid}' is not a live plan", item_id=sid, valid_ids=list(self._plans))
        safe = re.sub(r"[^A-Za-z0-9_-]", "", sid) or "schedule"
        out_dir = Path("results/work_orders")
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc)
        orders = [{k: j[k] for k in ("task_id", "asset_id", "title", "date", "window_start", "window_end", "crew", "why")} for j in plan["jobs"]]
        json_file = out_dir / f"{safe}.json"
        json_file.write_text(json.dumps({"schedule_id": sid, "generated_utc": stamp.isoformat(), "site_id": plan["site_id"],
                                         "data_note": plan.get("data_note"), "work_orders": orders}, indent=2, ensure_ascii=False), encoding="utf-8")
        events = []
        for j in plan["jobs"]:
            day = j["date"].replace("-", "")
            events += ["BEGIN:VEVENT", f"UID:{sid}-{j['task_id']}@ppc-maintenance-agent", f"DTSTAMP:{stamp:%Y%m%dT%H%M%SZ}",
                       f"DTSTART;TZID=Europe/Bucharest:{day}T{j['window_start'].replace(':', '')}00",
                       f"DTEND;TZID=Europe/Bucharest:{day}T{j['window_end'].replace(':', '')}00",
                       f"SUMMARY:{j['title']} {j['asset_id']} ({j['crew']})", f"DESCRIPTION:{j['why']}", "END:VEVENT"]
        ics_file = out_dir / f"{safe}.ics"
        ics_file.write_text("\r\n".join(["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//PPC maintenance agent//RO", *events, "END:VCALENDAR"]) + "\r\n", encoding="utf-8")
        return {"schedule_id": sid, "created_files": [json_file.as_posix(), ics_file.as_posix()], "work_orders_count": len(orders),
                "provenance": _provenance("create_work_orders")}

    def handle_send_alert(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        from src.agent.actions import dispatch_alert
        return dispatch_alert(level=args.get("level", "warning"), message=args.get("message", ""), asset_id=args.get("asset_id", ""))

    def handle_promote_schedule(self, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        from src.agent.actions import promote_schedule_to_active
        return promote_schedule_to_active(draft_id=args.get("draft_id", ""), store=store)


DEFAULT_LIVE_BACKEND = LiveBackend()
