"""Shared tool wrappers and llm_view reducers."""

from typing import Any, Dict
from src.agent.backends.fixture import DEFAULT_FIXTURE_BACKEND
from src.agent.backends.live import DEFAULT_LIVE_BACKEND
from src.agent.registry import tool
from src.agent.schemas.common import EffectClass
from src.agent.schemas.shared import (
    CreateWorkOrdersInput,
    CreateWorkOrdersOutput,
    ExplainInput,
    ExplainOutput,
    LossOverviewInput,
    LossOverviewOutput,
    PromoteScheduleInput,
    PromoteScheduleOutput,
    SendAlertInput,
    SendAlertOutput,
    UpdateFromFieldReportInput,
    UpdateFromFieldReportOutput,
)

_fixture_backend = DEFAULT_FIXTURE_BACKEND
_live_backend = DEFAULT_LIVE_BACKEND


# 10. loss_overview
def loss_overview_llm_view(out: LossOverviewOutput) -> Dict[str, Any]:
    return {
        "period": out.period,
        "total_mwh": out.total_mwh,
        "total_ron": out.total_ron,
        "loss_categories": [
            {
                "category": r.category,
                "description": r.description,
                "mwh": r.mwh,
                "ron": r.ron,
                "source_ref": r.source_ref,
            }
            for r in out.rows
        ],
    }


@tool(
    name="loss_overview",
    description="Overview of technical and non-technical energy losses in physical (MWh) and monetary (RON) units",
    effect=EffectClass.READ,
    input_schema=LossOverviewInput,
    output_schema=LossOverviewOutput,
    llm_view=loss_overview_llm_view,
)
def loss_overview(args: Dict[str, Any], backend: str = "fixture") -> Any:
    if backend == "live":
        return _live_backend.call("loss_overview", args)
    return _fixture_backend.call("loss_overview", args)


# 11. explain
def explain_llm_view(out: ExplainOutput) -> Dict[str, Any]:
    view: Dict[str, Any] = {
        "item_id": out.item_id,
        "item_type": out.item_type,
    }
    if out.scheduled_job_explanation:
        exp = out.scheduled_job_explanation
        view["scheduled_job"] = {
            "task_id": exp.task_id,
            "asset_id": exp.asset_id,
            "date": exp.date,
            "crew": exp.crew,
            "downtime_mwh": exp.cost_breakdown_mwh.downtime,
            "waiting_mwh": exp.cost_breakdown_mwh.waiting,
            "total_mwh": exp.cost_breakdown_mwh.total,
            "total_ron": exp.cost_breakdown_ron.total,
            "alternatives": [
                {
                    "date": a.date,
                    "total_mwh": a.total_mwh,
                    "total_ron": a.total_ron,
                    "why_not": a.why_not,
                }
                for a in exp.alternatives
            ],
        }
        # Live explanations carry the optimizer's reason and the model outputs behind the job (M2 cause, SHAP reasons).
        extra = exp.model_extra or {}
        if extra.get("why"):
            view["scheduled_job"]["why"] = extra["why"]
        trace_keys = ("m2_cause", "m2_proba", "m2_loss_mwh_day", "m2_pi_72h", "m2_reasons", "bundled_with", "earliest_option",
                      "deferred_days", "deferred_by", "converted_to_inspection", "forced_by_max_deferral", "due_date", "source")
        trace = {k: exp.trace[k] for k in trace_keys if k in exp.trace}
        if trace:
            view["scheduled_job"]["trace"] = trace
    elif out.household_explanation:
        hh = out.household_explanation
        view["household"] = {
            "household_id": hh.household_id,
            "area": hh.area,
            "theft_probability": hh.theft_probability,
            "signal_type": hh.signal_type,
            "top_factors": [
                {
                    "factor": f.factor,
                    "importance": f.importance,
                    "description": f.description,
                }
                for f in hh.top_factors
            ],
        }
    return view


@tool(
    name="explain",
    description="Provide explainability breakdown for a scheduled job or suspect consumer household",
    effect=EffectClass.READ,
    input_schema=ExplainInput,
    output_schema=ExplainOutput,
    llm_view=explain_llm_view,
)
def explain(args: Dict[str, Any], backend: str = "fixture") -> Any:
    if backend == "live":
        return _live_backend.call("explain", args)
    return _fixture_backend.call("explain", args)


# 12. update_from_field_report
def field_report_llm_view(out: UpdateFromFieldReportOutput) -> Dict[str, Any]:
    res: Dict[str, Any] = {
        "task_id": out.task_id,
        "status": out.status,
        "confidence": out.confidence,
        "effective_confidence": out.effective_confidence,
        "needs_input": out.needs_input,
    }
    if out.defer_days is not None:
        res["defer_days"] = out.defer_days
    return res


@tool(
    name="update_from_field_report",
    description="Turn field technician inspection/repair report into structured update (write action)",
    effect=EffectClass.WRITE,
    input_schema=UpdateFromFieldReportInput,
    output_schema=UpdateFromFieldReportOutput,
    llm_view=field_report_llm_view,
)
def update_from_field_report(args: Dict[str, Any], backend: str = "fixture") -> Any:
    if backend == "live":
        return _live_backend.call("update_from_field_report", args)
    return _fixture_backend.call("update_from_field_report", args)


# 13. create_work_orders
def create_work_orders_llm_view(out: CreateWorkOrdersOutput) -> Dict[str, Any]:
    return {
        "schedule_id": out.schedule_id,
        "work_orders_count": out.work_orders_count,
        "created_files": out.created_files,
    }


@tool(
    name="create_work_orders",
    description="Generate calendar (ICS) and JSON work orders for scheduled maintenance (write action)",
    effect=EffectClass.WRITE,
    input_schema=CreateWorkOrdersInput,
    output_schema=CreateWorkOrdersOutput,
    llm_view=create_work_orders_llm_view,
)
def create_work_orders(args: Dict[str, Any], backend: str = "fixture", store: Any = None) -> Any:
    if backend == "live":
        return _live_backend.call("create_work_orders", args, store=store)
    return _fixture_backend.call("create_work_orders", args)


# 14. send_alert
def send_alert_llm_view(out: SendAlertOutput) -> Dict[str, Any]:
    return {
        "alert_id": out.alert_id,
        "level": out.level,
        "message": out.message,
        "asset_id": out.asset_id,
    }


@tool(
    name="send_alert",
    description="Dispatch maintenance or anomaly alert record (write action)",
    effect=EffectClass.WRITE,
    input_schema=SendAlertInput,
    output_schema=SendAlertOutput,
    llm_view=send_alert_llm_view,
)
def send_alert(args: Dict[str, Any], backend: str = "fixture") -> Any:
    if backend == "live":
        return _live_backend.call("send_alert", args)
    return _fixture_backend.call("send_alert", args)


# 15. promote_schedule
def promote_schedule_llm_view(out: PromoteScheduleOutput) -> Dict[str, Any]:
    return {
        "promoted_id": out.promoted_id,
        "previous_active_id": out.previous_active_id,
        "status": out.status,
        "total_mwh": out.total_mwh,
        "total_ron": out.total_ron,
        "delta_mwh": out.delta_mwh,
        "delta_ron": out.delta_ron,
    }


@tool(
    name="promote_schedule",
    description="Promote a draft maintenance plan to active schedule, archiving previous active plan (write action)",
    effect=EffectClass.WRITE,
    input_schema=PromoteScheduleInput,
    output_schema=PromoteScheduleOutput,
    llm_view=promote_schedule_llm_view,
)
def promote_schedule(args: Dict[str, Any], backend: str = "fixture") -> Any:
    if backend == "live":
        return _live_backend.call("promote_schedule", args)
    return _fixture_backend.call("promote_schedule", args)

