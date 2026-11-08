"""The optimizer on small hand-built cases (CLAUDE.md section 12). Run: make test"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.decision.optimizer import Constraints, Task, generate_revisions, optimize, tasks_from_findings

TODAY = dt.date(2026, 11, 8)


def cmap(turbines=("A",), days=10, hourly=None, workable=None, start=TODAY):
    """A cost map like downtime_cost.cost_map. hourly(day) -> MWh per stopped hour; default 1.0 everywhere, all safe, no forecast spread."""
    rows = []
    for t in turbines:
        for d in range(days):
            for h in range(24):
                c = hourly(d) if hourly else 1.0
                rows.append({"turbine": t, "hour_utc": pd.Timestamp(start) + pd.Timedelta(days=d, hours=h), "cost_mwh": c, "cost_mwh_p90": c,
                             "workable": workable(d, h) if workable else True, "v_hub": 5.0, "gust_ms": 5.0})
    return pd.DataFrame(rows)


def repair(tid="T", turbine="A", hours=8, loss=1.4, **kw):
    return Task(tid, turbine, "s", "repair", hours, loss_mwh_per_day=loss, **kw)


def finding(cause, confidence=0.9, loss=1.0):
    return {"turbine": "A", "cause": cause, "cause_proba": confidence, "confidence": confidence, "loss_mwh_per_day": loss, "pi_72h": 0.95, "reasons": []}


def test_reproduces_the_design_doc_example():
    """Tomorrow: 18 MWh downtime + 1 day x 1.4 = 19.4. In four days: 2 MWh + 4 x 1.4 = 7.6. The calm day wins."""
    costs = {1: 18 / 8, 4: 2 / 8}
    job = optimize([repair()], cmap(hourly=lambda d: costs.get(d, 50.0)), Constraints(TODAY))["jobs"][0]
    assert job["date"] == "2026-11-12"
    assert job["cost_breakdown_mwh"] == {"downtime": 2.0, "waiting": 5.6, "risk": 0.0, "total": 7.6}
    tomorrow = next(a for a in job["alternatives"] if a["date"] == "2026-11-09")
    assert tomorrow["total"] == 19.4 and "more downtime" in tomorrow["why_not"]
    assert "7.6 MWh" in job["why"]


def test_risk_can_overturn_the_choice():
    costs = {1: 18 / 8, 4: 2 / 8}
    t = repair(p_fail_by_day=[0, 0, 0.1, 0.5, 0.9, 1, 1, 1, 1, 1], fail_cost_mwh_eq=100.0)
    assert optimize([t], cmap(hourly=lambda d: costs.get(d, 50.0)), Constraints(TODAY))["jobs"][0]["date"] == "2026-11-09"


def test_never_uses_unsafe_hours():
    job = optimize([repair(loss=0.0)], cmap(workable=lambda d, h: d >= 3, hourly=lambda d: 5.0 - d * 0.1), Constraints(TODAY))["jobs"][0]
    assert all(pd.Timestamp(h[0]) >= pd.Timestamp("2026-11-11") for h in job["hours"])


def test_capacity_is_respected():
    tasks = [repair(f"T{a}", a, hours=8, loss=10.0) for a in "ABCD"]
    s = optimize(tasks, cmap(turbines=tuple("ABCD"), days=3), Constraints(TODAY, crews_per_day=2, hours_per_crew=8, horizon_days=3))
    per_day = {}
    for j in s["jobs"]:
        per_day.setdefault(j["date"], []).append(j["crew"])
    assert len(s["jobs"]) == 4 and all(len(v) <= 2 and len(set(v)) == len(v) for v in per_day.values())


def test_unavailable_crew_is_not_used_and_costs_more():
    tasks = [repair("TA", "A", loss=5.0), repair("TB", "B", loss=5.0)]
    cm = cmap(turbines=("A", "B"), days=3)
    base = optimize(tasks, cm, Constraints(TODAY, horizon_days=3))
    off = optimize(tasks, cm, Constraints(TODAY, horizon_days=3, unavailable=[{"crew": "crew-2", "date": TODAY.isoformat()}]))
    assert not any(j["crew"] == "crew-2" and j["date"] == TODAY.isoformat() for j in off["jobs"])
    assert off["totals_mwh"]["total"] > base["totals_mwh"]["total"]


def test_unknown_crew_fails_clearly():
    with pytest.raises(ValueError, match="Unknown crew"):
        Constraints(TODAY, unavailable=[{"crew": "crew-9", "date": "2026-11-12"}])


def test_revision_stays_in_its_window():
    t = Task("R", "A", "s", "revision", 8, due_start=TODAY + dt.timedelta(days=3), due_end=TODAY + dt.timedelta(days=5))
    job = optimize([t], cmap(hourly=lambda d: 0.1 if d == 0 else 5.0 - d * 0.1), Constraints(TODAY))["jobs"][0]
    assert job["date"] == (TODAY + dt.timedelta(days=5)).isoformat()


def test_multi_day_revision_takes_consecutive_shifts():
    t = Task("R", "A", "s", "revision", 24, due_start=TODAY, due_end=TODAY + dt.timedelta(days=9))
    job = optimize([t], cmap(), Constraints(TODAY))["jobs"][0]
    days = sorted({h[0] for h in job["hours"]})
    assert len(job["hours"]) == 24 and len(days) == 3
    assert (np.diff([pd.Timestamp(d).toordinal() for d in days]) == 1).all()


def test_repair_groups_with_a_revision_when_cheaper():
    rev = Task("R", "A", "s", "revision", 16, due_start=TODAY + dt.timedelta(days=1), due_end=TODAY + dt.timedelta(days=1))
    s = optimize([rev, repair(loss=0.1, hours=8)], cmap(hourly=lambda d: 3.0), Constraints(TODAY))
    r = next(j for j in s["jobs"] if j["task_id"] == "T")
    assert r.get("bundled_with") == "R" and r["cost_breakdown_mwh"]["downtime"] == 0


def test_deferred_repair_waits_even_when_a_revision_could_carry_it():
    """A deferral means "not before": a revision's stop today does not pull the repair back."""
    rev = Task("R", "A", "s", "revision", 16, TODAY, TODAY + dt.timedelta(days=9))
    jobs = {j["task_id"]: j for j in optimize([rev, repair(hours=6, due_start=TODAY + dt.timedelta(days=3))], cmap(), Constraints(TODAY))["jobs"]}
    assert jobs["T"]["date"] >= (TODAY + dt.timedelta(days=3)).isoformat()
    assert "2026-11-08" in {a["date"] for a in jobs["T"]["alternatives"]} and "deferred until" in next(a["why_not"] for a in jobs["T"]["alternatives"] if a["date"] == "2026-11-08")


def test_deferred_inspection_waits():
    check = Task("C", "A", "s", "inspection", 1, due_start=TODAY + dt.timedelta(days=2))
    assert optimize([check], cmap(), Constraints(TODAY))["jobs"][0]["date"] == "2026-11-10"


def test_a_deferred_plan_gets_its_own_id():
    """A what-if must not overwrite the plan it is compared with."""
    base = optimize([repair()], cmap(), Constraints(TODAY))["schedule_id"]
    assert optimize([repair(due_start=TODAY + dt.timedelta(days=3))], cmap(), Constraints(TODAY))["schedule_id"] != base


def test_low_confidence_becomes_a_check_that_stops_nothing():
    tasks = tasks_from_findings([finding("yaw_misalignment", confidence=0.4)], "s", TODAY)
    assert tasks[0].kind == "inspection" and tasks[0].job_type == "low_confidence_check"
    job = optimize(tasks, cmap(), Constraints(TODAY))["jobs"][0]
    assert job["crew"] == "drone-team" and job["cost_breakdown_mwh"]["downtime"] == 0


@pytest.mark.parametrize("cause,job_type", [("anemometer_bias", "sensor_check"), ("icing", "icing_check")])
def test_no_repair_crew_for_a_wrong_reading_or_icing(cause, job_type):
    t = tasks_from_findings([finding(cause, confidence=0.95, loss=0.0)], "s", TODAY)[0]
    assert t.kind == "inspection" and t.job_type == job_type


def test_no_repair_crew_when_nothing_is_lost():
    t = tasks_from_findings([finding("yaw_misalignment", confidence=0.9, loss=0.0)], "s", TODAY)[0]
    assert t.kind == "inspection" and t.job_type == "no_loss_check"


def test_repairs_take_their_assumed_length():
    t = tasks_from_findings([finding("pitch_fault"), finding("yaw_misalignment")], "s", TODAY)
    assert [x.kind for x in t] == ["repair", "repair"] and [x.duration_h for x in t] == [8.0, 6.0]


def test_shift_is_in_local_time():
    """The cost map is in UTC; the crew shift is 08-18 in the site's own time (UTC+2 in Bucharest in November)."""
    job = optimize([repair(loss=0.0)], cmap(), Constraints(TODAY), tz="Europe/Bucharest")["jobs"][0]
    assert all(8 <= h < 18 for _, h in job["hours"])


def test_uncertain_forecast_becomes_a_check():
    cm = cmap(hourly=lambda d: 2.0)
    cm["cost_mwh_p90"] = cm.cost_mwh + 1.0  # 8 hours x 1 MWh of spread > the 3 MWh limit
    job = optimize([repair()], cm, Constraints(TODAY))["jobs"][0]
    assert job["kind"] == "inspection" and "converted_to_inspection" in job["trace"]


def test_stale_forecast_is_flagged():
    s = optimize([repair()], cmap(), Constraints(TODAY), forecast={"stale": True, "fetched_utc": "x"})
    assert any("Stale forecast" in w for w in s["warnings"])


def test_max_deferral_forces_the_earliest_slot():
    t = repair(loss=0.0, first_detected=TODAY - dt.timedelta(days=30))
    job = optimize([t], cmap(hourly=lambda d: 10.0 if d == 0 else 0.1), Constraints(TODAY))["jobs"][0]
    assert job["date"] == TODAY.isoformat() and job["trace"]["forced_by_max_deferral"]


def test_past_hours_are_never_used():
    cm = cmap(start=TODAY)
    cm = cm[cm.hour_utc >= pd.Timestamp(TODAY) + pd.Timedelta(hours=14)]  # As if planned at 14:00
    job = optimize([repair(loss=0.0)], cm, Constraints(TODAY))["jobs"][0]
    assert all(h >= 14 for d, h in job["hours"] if d == TODAY.isoformat())


def test_short_job_is_not_split_overnight():
    """Late-afternoon hours are cheap, but a 4-hour job started at 16:00 would need the next morning too."""
    cm = cmap(hourly=lambda d: 1.0)
    cm.loc[cm.hour_utc.dt.hour >= 16, "cost_mwh"] = 0.0
    cm["cost_mwh_p90"] = cm.cost_mwh
    job = optimize([repair(hours=4, loss=0.0)], cm, Constraints(TODAY))["jobs"][0]
    assert len({d for d, _ in job["hours"]}) == 1


def test_is_deterministic():
    cm = cmap(turbines=tuple("ABC"), hourly=lambda d: 1.0 + d % 3)
    tasks = lambda: [repair(f"T{a}", a, loss=2.0) for a in "ABC"]
    assert optimize(tasks(), cm, Constraints(TODAY)) == optimize(tasks(), cm, Constraints(TODAY))


def test_revisions_cover_each_turbine_in_a_year():
    revs = generate_revisions(["A", "B"], "s", dt.date(2026, 1, 1), dt.date(2026, 12, 31))
    assert all(sum(t.turbine == a for t in revs) >= 2 for a in "AB")
    assert {t.job_type for t in revs} == {"revision_annual", "revision_semiannual"} and len({t.task_id for t in revs}) == len(revs)
