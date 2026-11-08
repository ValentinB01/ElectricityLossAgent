"""Producer tool wrappers and llm_view reducers."""

from collections import defaultdict
from typing import Any, Dict, List
from src.agent.backends.fixture import DEFAULT_FIXTURE_BACKEND
from src.agent.registry import tool
from src.agent.schemas.common import EffectClass
from src.agent.store import DEFAULT_STORE
from src.agent.schemas.producers import (
    ComputeCostMapInput,
    ComputeCostMapOutput,
    DetectUnderperformanceInput,
    DetectUnderperformanceOutput,
    GetWeatherForecastInput,
    GetWeatherForecastOutput,
    ListTasksInput,
    ListTasksOutput,
    OptimizeMaintenanceInput,
    OptimizeMaintenanceOutput,
    PredictExpectedProductionInput,
    PredictExpectedProductionOutput,
)

_fixture_backend = DEFAULT_FIXTURE_BACKEND


# 1. get_weather_forecast
def weather_forecast_llm_view(out: GetWeatherForecastOutput) -> Dict[str, Any]:
    # Select sample entries (up to 3) so context stays compact
    samples = []
    for rec in out.hourly[:3]:
        samples.append({
            "timestamp": rec.timestamp,
            "temperature_c": rec.temperature_c,
            "wind_speed_ms": rec.wind_speed_ms,
            "surface_solar_radiation_wm2": rec.surface_solar_radiation_wm2,
        })
    res: Dict[str, Any] = {
        "site_id": out.site_id,
        "sample_forecast": samples,
        "stale": out.stale,
    }
    if out.age_hours is not None:
        res["age_hours"] = out.age_hours
    return res


@tool(
    name="get_weather_forecast",
    description="Fetch hourly weather forecast for wind or solar site",
    effect=EffectClass.READ,
    input_schema=GetWeatherForecastInput,
    output_schema=GetWeatherForecastOutput,
    llm_view=weather_forecast_llm_view,
)
def get_weather_forecast(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("get_weather_forecast", args)


# 2. predict_expected_production
def expected_production_llm_view(out: PredictExpectedProductionOutput) -> Dict[str, Any]:
    return {
        "predictions": [
            {
                "asset_id": p.asset_id,
                "timestamp": p.timestamp,
                "expected_kw_p10": p.expected_kw_p10,
                "expected_kw_p50": p.expected_kw_p50,
                "expected_kw_p90": p.expected_kw_p90,
            }
            for p in out.predictions
        ]
    }


@tool(
    name="predict_expected_production",
    description="Predict expected power generation quantiles (10%, 50%, 90%) under healthy conditions",
    effect=EffectClass.READ,
    input_schema=PredictExpectedProductionInput,
    output_schema=PredictExpectedProductionOutput,
    llm_view=expected_production_llm_view,
)
def predict_expected_production(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("predict_expected_production", args)


# 3. detect_underperformance
def underperformance_llm_view(out: DetectUnderperformanceOutput) -> Dict[str, Any]:
    return {
        "detections": [
            {
                "asset_id": d.asset_id,
                "pi_72h": d.pi_72h,
                "loss_mwh_per_day": d.loss_mwh_per_day,
                "loss_ron_per_day": d.loss_ron_per_day,
                "cause": d.cause,
                "confidence": d.confidence,
                "low_confidence": d.low_confidence,
                "recommended_action": d.recommended_action,
            }
            for d in out.detections
        ]
    }


@tool(
    name="detect_underperformance",
    description="Detect persistent underperformance, probable cause, and loss per day",
    effect=EffectClass.READ,
    input_schema=DetectUnderperformanceInput,
    output_schema=DetectUnderperformanceOutput,
    llm_view=underperformance_llm_view,
)
def detect_underperformance(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("detect_underperformance", args)


# 4. compute_cost_map
def cost_map_llm_view(out: ComputeCostMapOutput) -> Dict[str, Any]:
    """Reduce cost map to cheapest workable window per asset per day."""
    grouped = defaultdict(list)
    for entry in out.entries:
        if entry.workable:
            grouped[(entry.asset_id, entry.date)].append(entry)

    cheapest_windows = []
    for (asset_id, d_str), entries in grouped.items():
        min_entry = min(entries, key=lambda e: e.cost_mwh)
        item: Dict[str, Any] = {
            "asset_id": min_entry.asset_id,
            "date": min_entry.date,
            "hour": min_entry.hour,
            "cost_mwh": min_entry.cost_mwh,
            "cost_ron": min_entry.cost_ron,
        }
        if min_entry.wind_speed_ms is not None:
            item["wind_speed_ms"] = min_entry.wind_speed_ms
        cheapest_windows.append(item)

    return {
        "site_id": out.site_id,
        "cheapest_workable_windows": cheapest_windows,
        "stale": out.stale,
    }


@tool(
    name="compute_cost_map",
    description="Compute hourly downtime opportunity cost map over forecast horizon",
    effect=EffectClass.READ,
    input_schema=ComputeCostMapInput,
    output_schema=ComputeCostMapOutput,
    llm_view=cost_map_llm_view,
)
def compute_cost_map(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("compute_cost_map", args)


# 5. list_tasks
def list_tasks_llm_view(out: ListTasksOutput) -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": t.task_id,
                "asset_id": t.asset_id,
                "job_type": t.job_type,
                "loss_mwh_per_day": t.loss_mwh_per_day,
                "loss_ron_per_day": t.loss_ron_per_day,
                "estimated_hours": t.estimated_hours,
                "urgency": t.urgency,
            }
            for t in out.tasks
        ]
    }


@tool(
    name="list_tasks",
    description="List open maintenance tasks, scheduled revisions, and detected repairs",
    effect=EffectClass.READ,
    input_schema=ListTasksInput,
    output_schema=ListTasksOutput,
    llm_view=list_tasks_llm_view,
)
def list_tasks(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("list_tasks", args)


# 6. optimize_maintenance
def optimize_maintenance_llm_view(out: OptimizeMaintenanceOutput) -> Dict[str, Any]:
    return {
        "schedule_id": out.schedule_id,
        "total_mwh": out.total_mwh,
        "total_ron": out.total_ron,
        "active_total_mwh": out.active_total_mwh,
        "active_total_ron": out.active_total_ron,
        "delta_mwh": out.delta_mwh,
        "delta_ron": out.delta_ron,
        "moved_jobs": list(out.moved_jobs),
        "scheduled_jobs": [
            {
                "task_id": j.task_id,
                "asset_id": j.asset_id,
                "date": j.date,
                "start_hour": j.start_hour,
                "duration_hours": j.duration_hours,
                "crew": j.crew,
                "total_cost_mwh": j.cost_breakdown_mwh.total,
                "total_cost_ron": j.cost_breakdown_ron.total,
            }
            for j in out.jobs
        ],
    }


@tool(
    name="optimize_maintenance",
    description="Optimize maintenance schedule given constraints; computes what-if draft with deltas",
    effect=EffectClass.WHAT_IF,
    input_schema=OptimizeMaintenanceInput,
    output_schema=OptimizeMaintenanceOutput,
    llm_view=optimize_maintenance_llm_view,
)
def optimize_maintenance(args: Dict[str, Any], backend: str = "fixture") -> Any:
    res = _fixture_backend.call("optimize_maintenance", args)
    if isinstance(res, dict) and "schedule_id" in res:
        # Seed baseline active schedule if no active schedule exists in store
        if not DEFAULT_STORE.get_active_schedule():
            DEFAULT_STORE.store_schedule(
                schedule_id="sched-active-001",
                status="active",
                payload=res,
                total_mwh=float(res.get("active_total_mwh", res.get("total_mwh", 16.1))),
                total_ron=float(res.get("active_total_ron", res.get("total_ron", 7245.0))),
                parent_id=None,
            )
        # Store draft schedule linked to parent active schedule
        parent = DEFAULT_STORE.get_active_schedule()
        parent_id = parent["id"] if parent else "sched-active-001"
        DEFAULT_STORE.store_schedule(
            schedule_id=res["schedule_id"],
            status="draft",
            payload=res,
            total_mwh=float(res.get("total_mwh", 0.0)),
            total_ron=float(res.get("total_ron", 0.0)),
            parent_id=parent_id,
        )
    return res

