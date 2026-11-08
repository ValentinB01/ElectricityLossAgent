"""The live backend: real producer models and today's forecast behind the agent's tool contracts.

Checks the contracts (every output validates against its schema), that the numbers are the optimizer's own
(nothing staged), that what-ifs, deferrals and field reports change the plan, and that bad inputs get clear errors.
Needs `make data` and `make train`, plus a forecast (Open-Meteo, or the copy saved by an earlier run); skipped otherwise.
"""
import datetime as dt

import pytest

from src.agent.backends.live import LiveBackend, _Fail, parse_crew, parse_day
from src.agent.schemas.common import ErrorCode, StructuredError
from src.agent.schemas.producers import (
    ComputeCostMapOutput, DetectUnderperformanceOutput, GetWeatherForecastOutput, ListTasksOutput,
    OptimizeMaintenanceOutput, PredictExpectedProductionOutput,
)
from src.agent.schemas.shared import CreateWorkOrdersOutput, ExplainOutput, LossOverviewOutput
from src.common import KELMARSH, ROOT

SUNDAY = dt.date(2026, 11, 8)


# ---------- Parsing (no data needed) ----------
@pytest.mark.parametrize("value, expected", [
    ("joi", dt.date(2026, 11, 12)), ("Joi", dt.date(2026, 11, 12)), ("thursday", dt.date(2026, 11, 12)),
    ("duminică", SUNDAY), ("mâine", dt.date(2026, 11, 9)), ("azi", SUNDAY), ("2026-11-13", dt.date(2026, 11, 13)),
])
def test_parse_day(value, expected):
    assert parse_day(value, SUNDAY) == expected


def test_parse_day_rejects_garbage():
    with pytest.raises(_Fail) as e:
        parse_day("next blue moon", SUNDAY)
    assert e.value.error.code == ErrorCode.INVALID_DATE


@pytest.mark.parametrize("value", ["2", "crew-2", "crew_2", "echipa 2", 2])
def test_parse_crew(value):
    assert parse_crew(value) == "crew-2"


# ---------- Live pipeline ----------
@pytest.fixture(scope="module")
def live():
    if not (ROOT / "models" / "wind_expected_p50.txt").exists() or not (ROOT / "models" / "pv_cause.txt").exists() or not KELMARSH.exists():
        pytest.skip("Live data or models missing: run `make data` and `make train`")
    b = LiveBackend()
    base = b.call("optimize_maintenance", {"site_id": "fantanele"})
    if isinstance(base, StructuredError):
        pytest.skip(f"Live pipeline unavailable: {base.message}")
    return b


@pytest.fixture(scope="module")
def base(live):
    return live.call("optimize_maintenance", {"site_id": "fantanele"})


def test_plan_is_the_optimizers_own(live, base):
    """Totals and per-job costs equal a direct optimize() run on the same inputs: nothing is overwritten or staged."""
    from src.decision.optimizer import Constraints, optimize
    out = OptimizeMaintenanceOutput.model_validate(base)
    x = live._inputs_for("fantanele")
    direct = optimize(live._jobs_in(x, {}), x["cost_map"], Constraints(x["today"]), x["tz"], x["forecast"])
    assert out.total_mwh == pytest.approx(direct["totals_mwh"]["total"], abs=1e-3)
    assert {j.task_id: j.cost_breakdown_mwh.total for j in out.jobs} == pytest.approx({j["task_id"]: j["cost_breakdown_mwh"]["total"] for j in direct["jobs"]})
    assert out.provenance.backend == "live" and out.delta_mwh == 0 and out.moved_jobs == []
    for j in out.jobs:
        c = j.cost_breakdown_mwh
        assert c.total <= c.downtime + c.waiting + c.risk + 1e-6  # Equal, or lower by the shared-travel credit
        assert j.cost_breakdown_ron.total == pytest.approx(c.total * base["energy_price_ron_per_mwh"], abs=0.01)


def test_crew_off_replans_and_reports_the_difference(live, base):
    today = dt.date.fromisoformat(base["as_of"])
    out = live.call("optimize_maintenance", {"site_id": "fantanele", "constraints": {"unavailable": [{"crew": "2", "date": today.isoformat()}]}})
    OptimizeMaintenanceOutput.model_validate(out)
    assert not any(j["crew"] == "crew-2" and j["date"] == today.isoformat() for j in out["jobs"])
    assert out["delta_mwh"] == pytest.approx(out["total_mwh"] - base["total_mwh"], abs=1e-3)
    assert out["delta_mwh"] >= -1e-6  # Fewer crews can never make the greedy plan cheaper here
    moved = {j["task_id"] for j in base["jobs"] if j["crew"] == "crew-2" and j["date"] == today.isoformat()}
    assert moved <= set(out["moved_jobs"])


def test_deferral_holds_the_job(live, base):
    job = next(j for j in base["jobs"] if j["kind"] == "repair")
    out = live.call("optimize_maintenance", {"site_id": "fantanele", "constraints": {"defer": [{"task_id": job["task_id"], "days": 3}]}})
    moved = next((j for j in out["jobs"] if j["task_id"] == job["task_id"]), None)
    if moved is not None:
        assert dt.date.fromisoformat(moved["date"]) >= dt.date.fromisoformat(base["as_of"]) + dt.timedelta(days=3)
        assert moved["trace"]["deferred_days"] == 3
    else:
        assert job["task_id"] in {u["task_id"] for u in out["unscheduled"]}


def test_explain_matches_the_plan(live, base):
    live.call("optimize_maintenance", {"site_id": "fantanele"})  # The latest plan is the base one
    job = next(j for j in base["jobs"] if j["kind"] == "repair")
    out = ExplainOutput.model_validate(live.call("explain", {"item_id": job["task_id"]}))
    e = out.scheduled_job_explanation
    assert e.cost_breakdown_mwh.total == pytest.approx(job["cost_breakdown_mwh"]["total"])
    assert e.trace["m2_cause"] and e.trace["m2_reasons"], "a repair explains which M2 cause and which factors led to it"
    assert e.model_extra["why"]


def test_read_tools_validate(live):
    ListTasksOutput.model_validate(live.call("list_tasks", {"site_id": "fantanele"}))
    det = DetectUnderperformanceOutput.model_validate(live.call("detect_underperformance", {"site_id": "fantanele"}))
    assert all(d.model_extra.get("replayed_from") for d in det.detections), "replayed findings say so"
    cm = ComputeCostMapOutput.model_validate(live.call("compute_cost_map", {"site_id": "fantanele", "horizon_days": 2}))
    assert cm.entries and all(e.cost_mwh >= 0 for e in cm.entries)
    GetWeatherForecastOutput.model_validate(live.call("get_weather_forecast", {"site_id": "fantanele", "days": 1}))
    first = cm.entries[0]
    pred = PredictExpectedProductionOutput.model_validate(
        live.call("predict_expected_production", {"asset_ids": [first.asset_id], "start": first.timestamp, "end": cm.entries[-1].timestamp}))
    assert all(p.expected_kw_p10 <= p.expected_kw_p50 <= p.expected_kw_p90 for p in pred.predictions)
    losses = LossOverviewOutput.model_validate(live.call("loss_overview", {"period": "week"}))
    assert losses.total_mwh == pytest.approx(sum(r.mwh for r in losses.rows), abs=0.01)
    assert any(r.model_extra.get("data_source") == "fixture" for r in losses.rows if r.category == "consumers_suspect")


@pytest.mark.parametrize("tool, args, code", [
    ("detect_underperformance", {"asset_ids": ["FAN-T17"]}, ErrorCode.UNKNOWN_ASSET),
    ("optimize_maintenance", {"site_id": "fantanele", "constraints": {"weather": "nice"}}, ErrorCode.UNSUPPORTED_SCENARIO),
    ("optimize_maintenance", {"site_id": "fantanele", "constraints": {"unavailable": [{"crew": "crew-9", "date": "joi"}]}}, ErrorCode.UNSUPPORTED_SCENARIO),
    ("optimize_maintenance", {"site_id": "fantanele", "constraints": {"unavailable": [{"crew": "crew-2", "date": "someday"}]}}, ErrorCode.INVALID_DATE),
    ("optimize_maintenance", {"site_id": "mars"}, ErrorCode.UNKNOWN_ITEM),
    ("explain", {"item_id": "FAN-T01-nothing-99"}, ErrorCode.UNKNOWN_ITEM),
    ("predict_expected_production", {"asset_ids": ["FAN-T01"], "start": "2020-01-01", "end": "2020-01-02"}, ErrorCode.INVALID_DATE),
    ("score_households", {}, ErrorCode.UNSUPPORTED_SCENARIO),  # No live consumer model yet: refused, not relabelled
])
def test_bad_inputs_get_clear_errors(live, tool, args, code):
    out = live.call(tool, args)
    assert isinstance(out, StructuredError) and out.code == code


def test_field_report_done_removes_the_job(live, base, monkeypatch):
    monkeypatch.setenv("AGENT_BACKENDS", "live")  # The action gate checks task ids against the live jobs
    job = next(j for j in base["jobs"] if j["kind"] == "inspection")
    try:
        res = live.call("update_from_field_report", {"task_id": job["task_id"], "status": "done", "confidence": 0.95, "raw_text": "verificat, senzor ok"})
        assert not res["needs_input"]
        out = live.call("optimize_maintenance", {"site_id": "fantanele"})
        assert job["task_id"] not in {j["task_id"] for j in out["jobs"]}
    finally:
        live.task_overrides.pop(job["task_id"], None)


def test_work_orders_come_from_the_plan(live, base, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = CreateWorkOrdersOutput.model_validate(live.call("create_work_orders", {"schedule_id": base["schedule_id"]}))
    assert out.work_orders_count == len(base["jobs"])
    ics = (tmp_path / out.created_files[1]).read_text(encoding="utf-8")
    assert ics.count("BEGIN:VEVENT") == len(base["jobs"]) and all(j["asset_id"] in ics for j in base["jobs"])
