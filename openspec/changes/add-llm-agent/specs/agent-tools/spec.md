# Spec Delta

## Purpose

Defines the single registry of tools through which the agent, the CLI, the API and the dashboard get every number and every action. Each tool has a typed contract, a declared effect class and a swappable data backend.

## ADDED Requirements

### Requirement: Complete tool catalogue
The registry SHALL expose the tools listed in README §3.3: producers (`get_weather_forecast`, `predict_expected_production`, `detect_underperformance`, `compute_cost_map`, `list_tasks`, `optimize_maintenance`), consumers (`predict_expected_consumption`, `score_households`, `build_inspection_list`) and shared (`loss_overview`, `explain`, `update_from_field_report`, `create_work_orders`, `send_alert`). It SHALL also expose `promote_schedule`, which makes a draft schedule active.

#### Scenario: Catalogue listing
- **WHEN** a client requests the list of available tools
- **THEN** the system returns all 15 tools, each with its name, description, effect class, input schema and output schema

### Requirement: Typed input and output contracts
Every tool SHALL declare a machine-readable input schema and output schema. Inputs SHALL be validated before execution, and outputs SHALL be validated before they are returned to any caller.

#### Scenario: Invalid input rejected
- **WHEN** a caller invokes `detect_underperformance` with an asset id that does not exist
- **THEN** the tool does not execute and returns a structured error with code `unknown_asset` and the list of valid asset ids for the site

#### Scenario: Past or malformed date
- **WHEN** a caller invokes `optimize_maintenance` with an unavailability date in the past or a date that cannot be parsed
- **THEN** the system returns a structured error with code `invalid_date` and does not crash

#### Scenario: Backend returns a malformed result
- **WHEN** a backend returns data that does not match the tool's output schema
- **THEN** the system returns a structured error with code `backend_contract_violation` instead of the malformed data

### Requirement: Effect classification
Every tool SHALL be classified as `read` (no state change), `what_if` (computes a draft without changing active state) or `write` (changes persisted state or emits an external effect). `update_from_field_report`, `create_work_orders`, `send_alert` and `promote_schedule` SHALL be `write`.

#### Scenario: Write tools identified
- **WHEN** the catalogue is listed
- **THEN** `update_from_field_report`, `create_work_orders`, `send_alert` and `promote_schedule` are marked `write`, `optimize_maintenance` is marked `what_if`, and all other tools are marked `read`

### Requirement: Derived figures computed by tools
Any figure the agent may need to state (differences between plans, totals, RON conversions, percentages) SHALL be returned by a tool, so the agent never has to do arithmetic.

#### Scenario: Re-plan delta
- **WHEN** `optimize_maintenance` returns a draft
- **THEN** the result includes the draft's total cost, the active plan's total cost and their difference, in MWh and RON

### Requirement: Swappable fixture and live backends
Each tool SHALL be served either by a fixture backend (versioned sample data) or by a live backend (the trained module), chosen per tool by configuration. The two SHALL be indistinguishable to callers except for a backend label.

#### Scenario: Module not yet available
- **WHEN** the consumers module is not yet implemented and `score_households` is configured with the fixture backend
- **THEN** the tool returns schema-valid sample data labelled `backend: fixture`

#### Scenario: Live backend enabled
- **WHEN** `detect_underperformance` is switched to the live backend in configuration
- **THEN** results come from the trained producers model with the same output schema, labelled `backend: live`

### Requirement: Provenance on every result
Every tool result SHALL carry provenance: tool name, backend label, data timestamp or model/forecast run identifier, and a result id that later answers and audit records can reference.

#### Scenario: Provenance present
- **WHEN** `compute_cost_map` returns
- **THEN** the result includes `result_id`, `tool`, `backend`, and the forecast run timestamp used

### Requirement: Uncertainty and staleness surfaced
Tool outputs SHALL expose model confidence or intervals where the module produces them, and SHALL flag stale inputs explicitly rather than hiding them.

#### Scenario: Stale forecast
- **WHEN** the weather API is unavailable and the cached forecast is used
- **THEN** `get_weather_forecast` and `compute_cost_map` return `stale: true` with the age of the cached forecast

#### Scenario: Low confidence detection
- **WHEN** `detect_underperformance` produces a detection with confidence below `CONF_MIN`
- **THEN** the result marks it `low_confidence: true` and the recommended job kind is `inspection`

### Requirement: Losses in physical and monetary units
`loss_overview` SHALL report losses by source for both modules in MWh and RON, using the energy price from configuration, and each row SHALL reference the result or prediction that produced it.

#### Scenario: Weekly overview
- **WHEN** `loss_overview` is called for the current week
- **THEN** it returns rows for producer underperformance, scheduled producer downtime and suspect consumer households, each with MWh, RON and a traceable source reference

### Requirement: Explanation of any decision item
`explain(item_id)` SHALL return, for a scheduled job, its cost breakdown, rejected alternatives with reasons and model trace. For a household on the inspection list, it SHALL return the actual vs. expected monthly series, the peer comparison and the top 3 contributing factors.

#### Scenario: Explain a scheduled job
- **WHEN** `explain` is called with a scheduled task id
- **THEN** it returns `cost_breakdown` (downtime, waiting, risk, total), `alternatives` with `why_not`, and `trace`

#### Scenario: Explain a household
- **WHEN** `explain` is called with a household id from the inspection list
- **THEN** it returns the monthly actual and expected series, the peer-group comparison, the theft probability, the signal type and the top 3 factors

#### Scenario: Unknown item
- **WHEN** `explain` is called with an id that matches no job or household
- **THEN** it returns a structured error with code `unknown_item`

### Requirement: Side effects confined to the results directory
`write` tools SHALL only create or modify files under `results/` and records in the agent store. No tool SHALL execute shell commands or access paths given by user text.

#### Scenario: Work orders written
- **WHEN** `create_work_orders` executes for a confirmed schedule
- **THEN** ICS/JSON files are written only to `results/work_orders/`
