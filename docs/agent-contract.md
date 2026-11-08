# Agent Contract Specification for Module Owners

> This document defines the interface contract between the LLM Agent layer (`src/agent/`) and the underlying analytics/ML modules (`src/producers/`, `src/consumers/`, `src/decision/`).
> Linked from [docs/design.md](design.md).

## 1. Overview

The agent interacts with the analytical models exclusively through the **Tool Registry**. Each tool can be backed either by a static fixture (`backends/fixture.py`) or by a live module adapter (`backends/live.py`).

To ensure modularity and zero breakage during development:
1. Every tool expects validated typed inputs (Pydantic).
2. Live adapters wrap model functions and format results into the target Pydantic output schemas.
3. Errors must map to standard structured error codes (`unknown_asset`, `unknown_item`, `invalid_date`, `backend_contract_violation`, `unsupported_scenario`).

---

## 2. Tool Catalogue & Interface Specifications

All 15 tools are detailed below.

### Producers Tools

#### 1. `get_weather_forecast`
- **Purpose**: Fetch hourly weather forecast for wind and solar sites.
- **Live Function**: `src.data.weather.get_forecast_hourly(site_id: str, days: int) -> pd.DataFrame`
- **Expected Columns**: `['timestamp', 'temperature_c', 'wind_speed_ms', 'wind_direction_deg', 'surface_solar_radiation_wm2', 'precipitation_mm']`
- **Target Schema**: `GetWeatherForecastOutput` (records: `List[WeatherHourlyRecord]`, `stale: bool`, `age_hours: Optional[float]`)

#### 2. `predict_expected_production`
- **Purpose**: M1 model predicting healthy active power quantiles (10%, 50%, 90%).
- **Live Function**: `src.producers.m1_expected.predict_quantiles(asset_ids: list[str], start: str, end: str) -> pd.DataFrame`
- **Expected Columns**: `['asset_id', 'timestamp', 'expected_kw_p10', 'expected_kw_p50', 'expected_kw_p90']`
- **Target Schema**: `PredictExpectedProductionOutput` (records: `List[ProductionPrediction]`)

#### 3. `detect_underperformance`
- **Purpose**: M2 model detecting persistent production drops, probable cause, and daily loss rate.
- **Live Function**: `src.producers.m2_underperf.detect(asset_ids: list[str] | None, as_of: str) -> pd.DataFrame`
- **Expected Columns**: `['asset_id', 'pi_72h', 'loss_mwh_per_day', 'loss_ron_per_day', 'cause', 'cause_probability', 'confidence', 'low_confidence', 'recommended_action']`
- **Target Schema**: `DetectUnderperformanceOutput` (records: `List[UnderperformanceDetection]`)

#### 4. `compute_cost_map`
- **Purpose**: M4 model computing hourly downtime cost per asset over the forecast horizon.
- **Live Function**: `src.producers.m4_costmap.compute_cost_map(site_id: str, horizon_days: int, as_of: str) -> pd.DataFrame`
- **Expected Columns**: `['asset_id', 'timestamp', 'date', 'hour', 'cost_mwh', 'cost_ron', 'workable', 'wind_speed_ms']`
- **Target Schema**: `ComputeCostMapOutput` (records: `List[CostMapEntry]`, `stale: bool`)

#### 5. `list_tasks`
- **Purpose**: Service calendar and open maintenance task list (revisions, repairs, inspections).
- **Live Function**: `src.decision.tasks.get_open_tasks(site_id: str | None, as_of: str) -> pd.DataFrame`
- **Expected Columns**: `['task_id', 'asset_id', 'job_type', 'title', 'description', 'loss_mwh_per_day', 'loss_ron_per_day', 'due_date', 'due_start', 'due_end', 'estimated_hours', 'requires_outage', 'urgency', 'status']`
- **Target Schema**: `ListTasksOutput` (records: `List[MaintenanceTask]`)

#### 6. `optimize_maintenance`
- **Purpose**: Decision optimizer scheduling jobs while minimising downtime and waiting costs.
- **Live Function**: `src.decision.optimizer.schedule_maintenance(site_id: str | None, as_of: str, constraints: dict | None) -> dict`
- **Expected Structure**: Dict with `schedule_id`, `jobs` (list of scheduled job records), `total_mwh`, `total_ron`, `active_total_mwh`, `active_total_ron`, `delta_mwh`, `delta_ron`, `moved_jobs`.
- **Target Schema**: `OptimizeMaintenanceOutput`

---

### Consumers Tools

#### 7. `predict_expected_consumption`
- **Purpose**: C1 model predicting monthly household expected consumption quantiles.
- **Live Function**: `src.consumers.c1_expected.predict_consumption(household_ids: list[str], month: str) -> pd.DataFrame`
- **Expected Columns**: `['household_id', 'month', 'expected_kwh_p10', 'expected_kwh_p50', 'expected_kwh_p90']`
- **Target Schema**: `PredictExpectedConsumptionOutput` (records: `List[ConsumptionPrediction]`)

#### 8. `score_households`
- **Purpose**: C2 anomaly detection model identifying suspected electricity theft or faulty meters.
- **Live Function**: `src.consumers.c2_anomaly.score_all(as_of: str, top_n: int) -> pd.DataFrame`
- **Expected Columns**: `['household_id', 'area', 'theft_probability', 'monthly_loss_kwh', 'monthly_loss_ron', 'signal_type', 'confidence', 'recommended_action']`
- **Target Schema**: `ScoreHouseholdsOutput` (records: `List[HouseholdScore]`)

#### 9. `build_inspection_list`
- **Purpose**: Decision queue assigning suspect households to field crews based on recovery value.
- **Live Function**: `src.consumers.inspection_queue.build_queue(as_of: str, crews_count: int, capacity_per_crew: int) -> dict`
- **Expected Structure**: Dict with `inspections` (list of records), `total_expected_recovery_kwh`, `total_expected_recovery_ron`.
- **Target Schema**: `BuildInspectionListOutput`

---

### Shared & Action Tools

#### 10. `loss_overview`
- **Purpose**: Consolidated view of all system losses across both producer and consumer modules.
- **Live Function**: `src.agent.tools.shared_helpers.calculate_loss_overview(period: str, as_of: str) -> dict`
- **Target Schema**: `LossOverviewOutput` (`rows: List[LossOverviewRow]`, `total_mwh`, `total_ron`)

#### 11. `explain`
- **Purpose**: Detailed breakdown, peer comparisons, and alternatives for any scheduled job or household.
- **Live Function**: `src.agent.tools.shared_helpers.explain_item(item_id: str) -> dict`
- **Target Schema**: `ExplainOutput` (`scheduled_job_explanation` or `household_explanation`)

#### 12. `update_from_field_report`
- **Purpose**: Ingest and validate field technician findings, computing effective confidence.
- **Live Function**: `src.agent.actions.process_field_report(args: dict) -> dict`
- **Target Schema**: `UpdateFromFieldReportOutput`

#### 13. `create_work_orders`
- **Purpose**: Write confirmed maintenance work orders (ICS calendar & JSON files) into `results/work_orders/`.
- **Live Function**: `src.agent.actions.generate_work_orders(schedule_id: str) -> dict`
- **Target Schema**: `CreateWorkOrdersOutput`

#### 14. `send_alert`
- **Purpose**: Emit priority alert records into `results/alerts.log`.
- **Live Function**: `src.agent.actions.dispatch_alert(level: str, message: str, asset_id: str) -> dict`
- **Target Schema**: `SendAlertOutput`

#### 15. `promote_schedule`
- **Purpose**: Promote a draft maintenance schedule to active state and archive previous active schedule.
- **Live Function**: `src.agent.actions.promote_schedule_to_active(draft_id: str) -> dict`
- **Target Schema**: `PromoteScheduleOutput`

---

## 3. Data Types & Units Reference

| Quantity | Physical Unit | Currency Unit |
|---|---|---|
| Power (instantaneous / forecast) | kW | – |
| Energy (Turbine / Solar park loss) | MWh | RON (via `params.yaml` price) |
| Household Consumption | kWh | RON (via `params.yaml` price) |
| Probabilities | [0.0, 1.0] | – |
| Time series timestamps | ISO-8601 UTC string | – |

