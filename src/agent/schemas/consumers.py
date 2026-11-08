"""Pydantic schemas for consumer tools (C1, C2 and inspection queue)."""

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.agent.schemas.common import Provenance


# 7. predict_expected_consumption
class ConsumptionPrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    household_id: str
    month: str
    expected_kwh_p10: float
    expected_kwh_p50: float
    expected_kwh_p90: float


class PredictExpectedConsumptionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    household_ids: List[str] = Field(description="List of household identifiers")
    month: str = Field(description="Month in YYYY-MM format")


class PredictExpectedConsumptionOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    predictions: List[ConsumptionPrediction]
    provenance: Provenance


# 8. score_households
class HouseholdScore(BaseModel):
    model_config = ConfigDict(extra="allow")

    household_id: str
    area: str
    theft_probability: float
    monthly_loss_kwh: float
    monthly_loss_ron: float
    signal_type: str = Field(description="theft | faulty_meter | uncertain")
    confidence: float
    recommended_action: str = Field(description="inspection | remote_check | meter_replacement")


class ScoreHouseholdsInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    as_of: Optional[str] = None
    top_n: int = Field(default=50, ge=1)


class ScoreHouseholdsOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    households: List[HouseholdScore]
    provenance: Provenance


# 9. build_inspection_list
class InspectionQueueItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    household_id: str
    area: str
    assigned_crew: str
    theft_probability: float
    monthly_loss_kwh: float
    monthly_loss_ron: float
    signal_type: str
    expected_recovery_ron: float
    priority: str = Field(default="high")


class BuildInspectionListInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    as_of: Optional[str] = None
    crews_count: int = Field(default=2, ge=1)
    capacity_per_crew: int = Field(default=5, ge=1)


class BuildInspectionListOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    inspections: List[InspectionQueueItem]
    total_expected_recovery_kwh: float
    total_expected_recovery_ron: float
    provenance: Provenance

