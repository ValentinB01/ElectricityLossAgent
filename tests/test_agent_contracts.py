"""Contract tests for agent tool schemas, fixtures, dispatcher error handling, and llm_view reducers."""

import json
from pathlib import Path
from typing import Any, Set
import pytest

from src.agent.registry import registry
import src.agent.tools  # Register all 15 tools
from src.agent.schemas import (
    TOOL_SCHEMAS,
    EffectClass,
    ErrorCode,
    OptimizeMaintenanceOutput,
    StructuredError,
)


def _collect_numeric_leaves(obj: Any) -> Set[float]:
    """Recursively extract all numeric (float, int) values from a dict, list, or primitive."""
    numbers = set()
    # In Python, bool is a subclass of int, so exclude bool
    if isinstance(obj, bool):
        return numbers
    if isinstance(obj, (int, float)):
        numbers.add(round(float(obj), 6))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers.update(_collect_numeric_leaves(v))
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            numbers.update(_collect_numeric_leaves(item))
    return numbers


def test_schemas_export_json_schema():
    """Verify that every tool's input and output schema exports a valid JSON schema."""
    assert len(TOOL_SCHEMAS) == 15, f"Expected 15 tools, found {len(TOOL_SCHEMAS)}"

    for tool_name, (input_cls, output_cls) in TOOL_SCHEMAS.items():
        in_schema = input_cls.model_json_schema()
        assert in_schema is not None
        assert "properties" in in_schema or "type" in in_schema, f"{tool_name} input schema invalid"

        out_schema = output_cls.model_json_schema()
        assert out_schema is not None
        assert "properties" in out_schema or "type" in out_schema, f"{tool_name} output schema invalid"


def test_agent_contract_documentation():
    """Verify that docs/agent-contract.md documents all 15 tools and is linked from docs/design.md."""
    contract_path = Path("docs/agent-contract.md")
    assert contract_path.exists(), "docs/agent-contract.md must exist"
    content = contract_path.read_text(encoding="utf-8")

    for tool_name in TOOL_SCHEMAS:
        assert f"`{tool_name}`" in content, f"Tool {tool_name} must be documented in docs/agent-contract.md"

    design_path = Path("docs/design.md")
    assert design_path.exists(), "docs/design.md must exist"
    design_content = design_path.read_text(encoding="utf-8")
    assert "agent-contract.md" in design_content, "docs/design.md must link to agent-contract.md"


def test_fixtures_validate():
    """Verify that all JSON files in src/agent/fixtures/demo_week/ validate against their output schemas."""
    fixture_dir = Path("src/agent/fixtures/demo_week")
    assert fixture_dir.exists(), "Fixture directory must exist"

    fixture_files = list(fixture_dir.glob("*.json"))
    assert len(fixture_files) == 15, f"Expected 15 fixture files, found {len(fixture_files)}"

    for file_path in fixture_files:
        tool_name = file_path.stem
        assert tool_name in TOOL_SCHEMAS, f"Unknown fixture tool: {tool_name}"
        _, output_cls = TOOL_SCHEMAS[tool_name]

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = output_cls.model_validate(data)
        assert instance is not None
        assert instance.provenance is not None
        assert instance.provenance.tool == tool_name
        assert instance.provenance.backend == "fixture"


def test_optimize_maintenance_variants_deltas():
    """Verify that each optimize_maintenance variant has delta = draft_total - active_total."""
    file_path = Path("src/agent/fixtures/demo_week/optimize_maintenance.json")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    variants = data.get("variants", {})
    assert "baseline" in variants
    assert "crew_2_off_thursday" in variants
    assert "t05_deferred_5_days" in variants

    for name, v_data in variants.items():
        variant = OptimizeMaintenanceOutput.model_validate(v_data)
        expected_delta_mwh = round(variant.total_mwh - variant.active_total_mwh, 4)
        expected_delta_ron = round(variant.total_ron - variant.active_total_ron, 2)
        assert round(variant.delta_mwh, 4) == expected_delta_mwh, f"MWh delta mismatch in variant {name}"
        assert round(variant.delta_ron, 2) == expected_delta_ron, f"RON delta mismatch in variant {name}"


def test_catalogue_listing_and_effects():
    """Verify that catalogue lists all 15 tools with exact effect classifications."""
    tools = registry.list_tools()
    assert len(tools) == 15, f"Expected 15 registered tools, got {len(tools)}"

    tool_by_name = {t["name"]: t for t in tools}

    # Write tools
    expected_writes = {"update_from_field_report", "create_work_orders", "send_alert", "promote_schedule"}
    for t_name in expected_writes:
        assert tool_by_name[t_name]["effect"] == EffectClass.WRITE.value, f"{t_name} should be write"

    # What-if tools
    assert tool_by_name["optimize_maintenance"]["effect"] == EffectClass.WHAT_IF.value

    # Read tools
    expected_reads = {
        "get_weather_forecast",
        "predict_expected_production",
        "detect_underperformance",
        "compute_cost_map",
        "list_tasks",
        "predict_expected_consumption",
        "score_households",
        "build_inspection_list",
        "loss_overview",
        "explain",
    }
    for t_name in expected_reads:
        assert tool_by_name[t_name]["effect"] == EffectClass.READ.value, f"{t_name} should be read"


def test_bad_asset_id_rejected():
    """Verify that an unknown asset ID returns structured error code unknown_asset with valid_ids."""
    res = registry.execute("detect_underperformance", {"asset_ids": ["FAN-T999"]})
    assert res.is_error
    assert res.error is not None
    assert res.error.code == ErrorCode.UNKNOWN_ASSET
    assert res.error.valid_ids is not None
    assert "FAN-T17" in res.error.valid_ids


def test_past_date_rejected():
    """Verify that an unavailability date in the past returns structured error code invalid_date."""
    res = registry.execute(
        "optimize_maintenance",
        {"constraints": {"unavailable": [{"crew": "crew-2", "date": "2026-11-01"}]}},
    )
    assert res.is_error
    assert res.error is not None
    assert res.error.code == ErrorCode.INVALID_DATE
    assert res.error.date_str == "2026-11-01"


def test_malformed_date_rejected():
    """Verify that a malformed date string returns structured error code invalid_date."""
    res = registry.execute(
        "optimize_maintenance",
        {"constraints": {"unavailable": [{"crew": "crew-2", "date": "not-a-real-date"}]}},
    )
    assert res.is_error
    assert res.error is not None
    assert res.error.code == ErrorCode.INVALID_DATE


def test_backend_contract_violation():
    """Verify that malformed backend output returns structured error code backend_contract_violation."""
    defn = registry.get_tool("get_weather_forecast")
    original_handler = defn.handler

    # Temporarily force handler to return malformed output
    try:
        defn.handler = lambda args, backend: {"malformed_key": 123}
        res = registry.execute("get_weather_forecast", {"site_id": "fantanele"})
        assert res.is_error
        assert res.error is not None
        assert res.error.code == ErrorCode.BACKEND_CONTRACT_VIOLATION
    finally:
        defn.handler = original_handler


def test_unknown_tool_rejected():
    """Verify that an unknown tool returns structured error code unknown_tool."""
    res = registry.execute("nonexistent_tool", {})
    assert res.is_error
    assert res.error is not None
    assert res.error.code == ErrorCode.UNKNOWN_TOOL


def test_unknown_item_in_explain():
    """Verify that an unknown item in explain returns structured error code unknown_item."""
    res = registry.execute("explain", {"item_id": "UNKNOWN-999"})
    assert res.is_error
    assert res.error is not None
    assert res.error.code == ErrorCode.UNKNOWN_ITEM
    assert res.error.item_id == "UNKNOWN-999"


def test_unsupported_scenario_in_optimize():
    """Verify that unknown constraints in fixture mode return structured error code unsupported_scenario."""
    res = registry.execute(
        "optimize_maintenance",
        {"constraints": {"exotic_constraint_xyz": 999}},
    )
    assert res.is_error
    assert res.error is not None
    assert res.error.code == ErrorCode.UNSUPPORTED_SCENARIO


def test_side_effects_confined_to_results_dir():
    """Verify that write tools only create or touch files under results/."""
    wo_res = registry.execute("create_work_orders", {"schedule_id": "sched-test-01"})
    assert not wo_res.is_error
    for path_str in wo_res.full_result.created_files:
        p = Path(path_str)
        assert str(p).replace("\\", "/").startswith("results/"), f"File outside results/: {path_str}"
        assert p.exists()

    alert_res = registry.execute("send_alert", {"level": "info", "message": "Test alert", "asset_id": "FAN-T17"})
    assert not alert_res.is_error
    assert alert_res.full_result.logged_to.replace("\\", "/").startswith("results/")
    assert Path(alert_res.full_result.logged_to).exists()


def test_every_numeric_leaf_in_llm_view_exists_in_full_result():
    """Verify that every numeric leaf in each tool's llm_view exists in the full result (grounding prerequisite)."""
    default_args = {
        "get_weather_forecast": {"site_id": "fantanele"},
        "predict_expected_production": {"asset_ids": ["FAN-T17"], "start": "2026-11-08T00:00:00Z", "end": "2026-11-09T00:00:00Z"},
        "detect_underperformance": {"site_id": "fantanele"},
        "compute_cost_map": {"site_id": "fantanele"},
        "list_tasks": {"site_id": "fantanele"},
        "optimize_maintenance": {},
        "predict_expected_consumption": {"household_ids": ["HH-001"], "month": "2026-10"},
        "score_households": {"top_n": 10},
        "build_inspection_list": {},
        "loss_overview": {"period": "week"},
        "explain": {"item_id": "T17-pitch-001"},
        "update_from_field_report": {"task_id": "T05-bearing-001", "status": "deferred", "defer_days": 5},
        "create_work_orders": {"schedule_id": "sched-active-001"},
        "send_alert": {"level": "warning", "message": "Alert test", "asset_id": "FAN-T17"},
        "promote_schedule": {"draft_id": "sched-draft-base"},
    }

    for tool_name, args in default_args.items():
        res = registry.execute(tool_name, args)
        assert not res.is_error, f"Tool {tool_name} failed execution: {res.error}"

        full_dump = res.full_result.model_dump()
        full_numbers = _collect_numeric_leaves(full_dump)
        llm_numbers = _collect_numeric_leaves(res.llm_view)

        # Every number in llm_view must exist in full_result
        for num in llm_numbers:
            assert num in full_numbers, (
                f"Tool '{tool_name}' llm_view contains number {num} not found in full result: {full_numbers}"
            )


def test_direct_tool_call_enforces_validation():
    """Verify that calling tool function directly still routes through registry pipeline and validates."""
    from src.agent.tools import detect_underperformance
    res = detect_underperformance({"asset_ids": ["FAN-T999"]})
    assert isinstance(res, StructuredError)
    assert res.code == ErrorCode.UNKNOWN_ASSET

    valid_res = detect_underperformance({"asset_ids": ["FAN-T17"]})
    assert not isinstance(valid_res, StructuredError)
    assert len(valid_res.detections) == 1
    assert valid_res.detections[0].asset_id == "FAN-T17"


def test_stale_forecast_surfacing():
    """Verify that stale weather and costmap queries return stale: true with age_hours."""
    res_wx = registry.execute("get_weather_forecast", {"site_id": "fantanele", "stale": True, "age_hours": 14.5})
    assert not res_wx.is_error
    assert res_wx.full_result.stale is True
    assert res_wx.full_result.age_hours == 14.5

    res_cm = registry.execute("compute_cost_map", {"site_id": "fantanele", "stale": True, "age_hours": 14.5})
    assert not res_cm.is_error
    assert res_cm.full_result.stale is True
    assert res_cm.full_result.age_hours == 14.5


def test_low_confidence_invariant():
    """Verify that underperformance detections with confidence < 0.7 enforce low_confidence and inspection."""
    from src.agent.schemas.producers import UnderperformanceDetection
    det = UnderperformanceDetection(
        asset_id="FAN-T09",
        pi_72h=0.9,
        loss_mwh_per_day=0.5,
        loss_ron_per_day=225.0,
        cause="icing",
        cause_probability=0.6,
        confidence=0.55,
        low_confidence=False,  # deliberately passed false
        recommended_action="repair",  # deliberately passed repair
    )
    assert det.low_confidence is True
    assert det.recommended_action == "inspection"


def test_explain_field_aliases():
    """Verify that explain results provide cost_breakdown and household factor aliases."""
    res_job = registry.execute("explain", {"item_id": "T17-pitch-001"})
    assert not res_job.is_error
    job_exp = res_job.full_result.scheduled_job_explanation
    assert job_exp is not None
    assert job_exp.cost_breakdown is not None
    assert job_exp.cost_breakdown.downtime == 2.0
    assert job_exp.cost_breakdown.total == 7.6

    res_hh = registry.execute("explain", {"item_id": "HH-FAULTY-01"})
    assert not res_hh.is_error
    hh_exp = res_hh.full_result.household_explanation
    assert hh_exp is not None
    assert hh_exp.actual_series is not None
    assert hh_exp.expected_series is not None
    assert hh_exp.top_3_factors is not None


def test_romanian_day_name_unavailability():
    """Verify that Romanian day name 'joi' in unavailable constraint returns crew_2_off_thursday variant."""
    res = registry.execute(
        "optimize_maintenance",
        {"constraints": {"unavailable": [{"crew": "crew-2", "date": "joi"}]}},
    )
    assert not res.is_error
    assert res.full_result.schedule_id == "sched-draft-crew2-off"
    assert res.full_result.delta_mwh == 1.2


def test_baseline_crews_per_day_constraint():
    """Verify that passing standard baseline crews_per_day: 2 returns baseline schedule."""
    res = registry.execute(
        "optimize_maintenance",
        {"constraints": {"crews_per_day": 2}},
    )
    assert not res.is_error
    assert res.full_result.schedule_id == "sched-draft-base"
    assert res.full_result.delta_mwh == 0.0


def test_monthly_loss_overview():
    """Verify that loss_overview with period='month' scales totals accordingly."""
    res_week = registry.execute("loss_overview", {"period": "week"})
    res_month = registry.execute("loss_overview", {"period": "month"})
    assert not res_week.is_error
    assert not res_month.is_error
    assert res_month.full_result.period == "month"
    assert round(res_month.full_result.total_mwh, 1) == round(res_week.full_result.total_mwh * 4.0, 1)


def test_create_work_orders_sanitizes_path_traversal():
    """Verify that path traversal in schedule_id is sanitized and confined to results/work_orders/."""
    res = registry.execute("create_work_orders", {"schedule_id": "../../malicious_name"})
    assert not res.is_error
    for path_str in res.full_result.created_files:
        p = Path(path_str)
        assert str(p).replace("\\", "/").startswith("results/work_orders/")
        assert ".." not in str(p)

