"""Pydantic schemas for shared tools (overview, explain, writes, actions)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.agent.schemas.common import Provenance
from src.agent.schemas.producers import CostBreakdown, JobAlternative


# 10. loss_overview
class LossOverviewRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str = Field(description="producers_underperformance | producers_downtime | consumers_suspect")
    description: str
    mwh: float
    ron: float
    source_ref: str = Field(description="Traceable reference to result_id, task_id or household_id")
    confidence: Optional[float] = None


class LossOverviewInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    period: str = Field(default="week", description="week | month")
    as_of: Optional[str] = None


class LossOverviewOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    period: str
    rows: List[LossOverviewRow]
    total_mwh: float
    total_ron: float
    provenance: Provenance


# 11. explain
class MonthlyDataPoint(BaseModel):
    model_config = ConfigDict(extra="allow")

    month: str
    kwh: float


class PeerComparison(BaseModel):
    model_config = ConfigDict(extra="allow")

    peer_group_name: str
    avg_kwh: float
    household_kwh: float
    deviation_pct: float


class FactorContribution(BaseModel):
    model_config = ConfigDict(extra="allow")

    factor: str
    importance: float
    description: str


class ScheduledJobExplanation(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    asset_id: str
    date: str
    crew: str
    cost_breakdown_mwh: CostBreakdown
    cost_breakdown_ron: CostBreakdown
    cost_breakdown: Optional[CostBreakdown] = None
    alternatives: List[JobAlternative] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def populate_cost_breakdown_alias(self) -> "ScheduledJobExplanation":
        if self.cost_breakdown is None:
            self.cost_breakdown = self.cost_breakdown_mwh
        return self


class HouseholdExplanation(BaseModel):
    model_config = ConfigDict(extra="allow")

    household_id: str
    area: str
    theft_probability: float
    signal_type: str
    monthly_actual_kwh: List[MonthlyDataPoint]
    monthly_expected_kwh: List[MonthlyDataPoint]
    peer_comparison: PeerComparison
    top_factors: List[FactorContribution]
    actual_series: Optional[List[MonthlyDataPoint]] = None
    expected_series: Optional[List[MonthlyDataPoint]] = None
    top_3_factors: Optional[List[FactorContribution]] = None

    @model_validator(mode="after")
    def populate_household_aliases(self) -> "HouseholdExplanation":
        if self.actual_series is None:
            self.actual_series = self.monthly_actual_kwh
        if self.expected_series is None:
            self.expected_series = self.monthly_expected_kwh
        if self.top_3_factors is None:
            self.top_3_factors = self.top_factors[:3]
        return self


class ExplainInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_id: str = Field(description="Task ID (e.g. 'T17-pitch-001') or Household ID (e.g. 'HH-FAULTY-01')")


class ExplainOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    item_id: str
    item_type: str = Field(description="scheduled_job | household")
    scheduled_job_explanation: Optional[ScheduledJobExplanation] = None
    household_explanation: Optional[HouseholdExplanation] = None
    provenance: Provenance


# 12. update_from_field_report
class UpdateFromFieldReportInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    status: str = Field(description="done | deferred | needs_part | not_found")
    defer_days: Optional[int] = Field(default=None)
    new_finding: Optional[str] = Field(default=None)
    confidence: float = Field(default=1.0)
    text: Optional[str] = Field(default=None, description="Free text from technician")
    raw_text: Optional[str] = Field(default=None)


class UpdateFromFieldReportOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    status: str
    defer_days: Optional[int] = None
    new_finding: Optional[str] = None
    confidence: float
    effective_confidence: float
    needs_input: bool = False
    raw_text: Optional[str] = None
    provenance: Provenance


# 13. create_work_orders
class CreateWorkOrdersInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    schedule_id: str = Field(description="Active or confirmed schedule identifier")


class CreateWorkOrdersOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    schedule_id: str
    created_files: List[str]
    work_orders_count: int
    provenance: Provenance


# 14. send_alert
class SendAlertInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    level: str = Field(description="info | warning | critical")
    message: str
    asset_id: str


class SendAlertOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    alert_id: str
    level: str
    message: str
    asset_id: str
    logged_to: str
    provenance: Provenance


# 15. promote_schedule
class PromoteScheduleInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    draft_id: str = Field(description="Draft schedule ID to promote to active")


class PromoteScheduleOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    promoted_id: str
    previous_active_id: Optional[str] = None
    status: str = Field(default="active")
    total_mwh: float
    total_ron: float
    delta_mwh: float = 0.0
    delta_ron: float = 0.0
    provenance: Provenance

