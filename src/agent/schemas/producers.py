"""Pydantic schemas for producer tools (M1-M4 and optimizer)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from src.agent.schemas.common import Provenance


# 1. get_weather_forecast
class WeatherHourlyRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    timestamp: str
    temperature_c: float
    wind_speed_ms: Optional[float] = None  # Live: the 10 m forecast; None where the site does not fetch it
    wind_direction_deg: Optional[float] = None
    surface_solar_radiation_wm2: Optional[float] = None
    precipitation_mm: float


class GetWeatherForecastInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    site_id: str = Field(description="Site identifier, e.g. 'fantanele' or 'calugareni'")
    days: int = Field(default=7, ge=1, le=14, description="Forecast horizon in days")


class GetWeatherForecastOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    site_id: str
    hourly: List[WeatherHourlyRecord]
    stale: bool = Field(default=False, description="Flagged true if weather forecast cache was used")
    age_hours: Optional[float] = Field(default=None, description="Age in hours if stale")
    provenance: Provenance


# 2. predict_expected_production
class ProductionPrediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_id: str
    timestamp: str
    expected_kw_p10: float
    expected_kw_p50: float
    expected_kw_p90: float


class PredictExpectedProductionInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_ids: List[str] = Field(description="Turbine or inverter IDs")
    start: str = Field(description="Start ISO date/timestamp")
    end: str = Field(description="End ISO date/timestamp")


class PredictExpectedProductionOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    predictions: List[ProductionPrediction]
    provenance: Provenance


# 3. detect_underperformance
class UnderperformanceDetection(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_id: str
    pi_72h: Optional[float] = Field(default=None, description="72-hour performance index (1.0 is nominal); None if no usable hour")
    loss_mwh_per_day: float
    loss_ron_per_day: float
    cause: str = Field(description="Detected cause: pitch_fault, yaw_misalignment, icing, soiling, etc.")
    cause_probability: float
    confidence: float
    low_confidence: bool = Field(default=False, description="True when confidence < CONF_MIN")
    recommended_action: str = Field(default="repair", description="'repair' or 'inspection'")

    @model_validator(mode="after")
    def enforce_confidence_invariants(self) -> "UnderperformanceDetection":
        # Per spec: confidence below CONF_MIN (0.7) marks low_confidence: true and recommended job inspection
        if self.confidence < 0.7:
            self.low_confidence = True
            self.recommended_action = "inspection"
        return self


class DetectUnderperformanceInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_ids: Optional[List[str]] = Field(default=None, description="Specific asset IDs or all if empty")
    site_id: Optional[str] = Field(default=None, description="Optional site filter")
    as_of: Optional[str] = Field(default=None, description="Demo clock date YYYY-MM-DD")


class DetectUnderperformanceOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    detections: List[UnderperformanceDetection]
    provenance: Provenance


# 4. compute_cost_map
class CostMapEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_id: str
    timestamp: str
    date: str
    hour: int
    cost_mwh: float
    cost_ron: float
    workable: bool
    wind_speed_ms: Optional[float] = None


class ComputeCostMapInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    site_id: str = Field(description="Site identifier, e.g. 'fantanele' or 'calugareni'")
    horizon_days: int = Field(default=7, ge=1, le=14)
    as_of: Optional[str] = Field(default=None)


class ComputeCostMapOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    site_id: str
    entries: List[CostMapEntry]
    stale: bool = Field(default=False)
    age_hours: Optional[float] = None
    provenance: Provenance


# 5. list_tasks
class MaintenanceTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    asset_id: str
    job_type: str = Field(description="repair | periodic_revision | preventive | inspection")
    title: str
    description: str
    loss_mwh_per_day: float = 0.0
    loss_ron_per_day: float = 0.0
    due_date: Optional[str] = None
    due_start: Optional[str] = None
    due_end: Optional[str] = None
    estimated_hours: float = 8.0
    requires_outage: bool = True
    urgency: str = Field(default="normal", description="low | normal | high | critical")
    status: str = Field(default="open", description="open | in_progress | deferred | completed")


class ListTasksInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    site_id: Optional[str] = None
    as_of: Optional[str] = None


class ListTasksOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    tasks: List[MaintenanceTask]
    provenance: Provenance


# 6. optimize_maintenance
class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="allow")

    downtime: float
    waiting: float
    risk: float = 0.0
    total: float


class JobAlternative(BaseModel):
    model_config = ConfigDict(extra="allow")

    date: str
    total_mwh: Optional[float] = Field(default=None, description="None when the day has no feasible slot; why_not says why")
    total_ron: Optional[float] = None
    why_not: str


class ScheduledJob(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str
    asset_id: str
    date: str
    start_hour: int
    duration_hours: float
    crew: str
    cost_breakdown_mwh: CostBreakdown
    cost_breakdown_ron: CostBreakdown
    alternatives: List[JobAlternative] = Field(default_factory=list)
    trace: Dict[str, Any] = Field(default_factory=dict)


class OptimizeMaintenanceInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    site_id: Optional[str] = None
    as_of: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="e.g. {'crews_per_day': 2, 'unavailable': [{'crew': 'crew-2', 'date': '2026-11-12'}]}",
    )


class OptimizeMaintenanceOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    schedule_id: str
    status: str = Field(default="draft", description="draft | active | history")
    parent_schedule_id: Optional[str] = None
    jobs: List[ScheduledJob]
    total_mwh: float
    total_ron: float
    active_total_mwh: float
    active_total_ron: float
    delta_mwh: float = Field(description="draft_total_mwh - active_total_mwh")
    delta_ron: float = Field(description="draft_total_ron - active_total_ron")
    moved_jobs: List[str] = Field(default_factory=list)
    provenance: Provenance

