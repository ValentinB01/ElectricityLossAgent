"""Consumer tool wrappers and llm_view reducers."""

from typing import Any, Dict
from src.agent.backends.fixture import DEFAULT_FIXTURE_BACKEND
from src.agent.registry import tool
from src.agent.schemas.common import EffectClass
from src.agent.schemas.consumers import (
    BuildInspectionListInput,
    BuildInspectionListOutput,
    PredictExpectedConsumptionInput,
    PredictExpectedConsumptionOutput,
    ScoreHouseholdsInput,
    ScoreHouseholdsOutput,
)

_fixture_backend = DEFAULT_FIXTURE_BACKEND


# 7. predict_expected_consumption
def expected_consumption_llm_view(out: PredictExpectedConsumptionOutput) -> Dict[str, Any]:
    return {
        "predictions": [
            {
                "household_id": p.household_id,
                "month": p.month,
                "expected_kwh_p10": p.expected_kwh_p10,
                "expected_kwh_p50": p.expected_kwh_p50,
                "expected_kwh_p90": p.expected_kwh_p90,
            }
            for p in out.predictions
        ]
    }


@tool(
    name="predict_expected_consumption",
    description="Predict expected household consumption quantiles under normal conditions",
    effect=EffectClass.READ,
    input_schema=PredictExpectedConsumptionInput,
    output_schema=PredictExpectedConsumptionOutput,
    llm_view=expected_consumption_llm_view,
)
def predict_expected_consumption(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("predict_expected_consumption", args)


# 8. score_households
def score_households_llm_view(out: ScoreHouseholdsOutput) -> Dict[str, Any]:
    """Reduce to top scored households."""
    return {
        "suspect_households": [
            {
                "household_id": h.household_id,
                "area": h.area,
                "theft_probability": h.theft_probability,
                "monthly_loss_kwh": h.monthly_loss_kwh,
                "monthly_loss_ron": h.monthly_loss_ron,
                "signal_type": h.signal_type,
                "confidence": h.confidence,
                "recommended_action": h.recommended_action,
            }
            for h in out.households[:5]
        ]
    }


@tool(
    name="score_households",
    description="Score households for non-technical losses (theft, faulty meter)",
    effect=EffectClass.READ,
    input_schema=ScoreHouseholdsInput,
    output_schema=ScoreHouseholdsOutput,
    llm_view=score_households_llm_view,
)
def score_households(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("score_households", args)


# 9. build_inspection_list
def build_inspection_list_llm_view(out: BuildInspectionListOutput) -> Dict[str, Any]:
    return {
        "total_expected_recovery_kwh": out.total_expected_recovery_kwh,
        "total_expected_recovery_ron": out.total_expected_recovery_ron,
        "top_inspections": [
            {
                "rank": i.rank,
                "household_id": i.household_id,
                "area": i.area,
                "assigned_crew": i.assigned_crew,
                "theft_probability": i.theft_probability,
                "monthly_loss_kwh": i.monthly_loss_kwh,
                "monthly_loss_ron": i.monthly_loss_ron,
                "expected_recovery_ron": i.expected_recovery_ron,
            }
            for i in out.inspections
        ],
    }


@tool(
    name="build_inspection_list",
    description="Build daily prioritized inspection queue for field and metering crews",
    effect=EffectClass.READ,
    input_schema=BuildInspectionListInput,
    output_schema=BuildInspectionListOutput,
    llm_view=build_inspection_list_llm_view,
)
def build_inspection_list(args: Dict[str, Any], backend: str = "fixture") -> Any:
    return _fixture_backend.call("build_inspection_list", args)

