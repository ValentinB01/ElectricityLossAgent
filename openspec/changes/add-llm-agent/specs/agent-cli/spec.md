# Spec Delta

## Purpose

Provides a deterministic, network-independent way to use every agent tool through explicit commands. The demo and all decisions keep working without an LLM.

## ADDED Requirements

### Requirement: One command per tool
The CLI SHALL offer one explicit command per registry tool, plus session commands (`help`, `confirm <id>`, `reject <id>`, `pending`, `promote <draft_id>`). Arguments SHALL be validated against the same input schemas as the LLM path.

#### Scenario: Plan command
- **WHEN** the user runs `plan fantanele`
- **THEN** the CLI calls `list_tasks`, `compute_cost_map` and `optimize_maintenance` for `fantanele` and prints the calendar

#### Scenario: Crew unavailable
- **WHEN** the user runs `crew-off crew-2 2026-11-12`
- **THEN** the CLI runs `optimize_maintenance` with that constraint and prints the draft and the cost difference against the active plan

#### Scenario: Bad arguments
- **WHEN** the user runs a command with a missing or invalid argument
- **THEN** the CLI prints the validation error and the command usage, and keeps running

### Requirement: Templated Romanian output
CLI output SHALL be rendered from fixed Romanian templates that only show values present in the tool results. The same input SHALL always produce the same output.

#### Scenario: Deterministic explain
- **WHEN** `explain T17-pitch-001` is run twice on the same data
- **THEN** both outputs are identical and include the cost breakdown and alternatives table

### Requirement: Same confirmation gate
Write commands in the CLI SHALL create pending actions and require `confirm <id>`, the same as the LLM path.

#### Scenario: Send alert from CLI
- **WHEN** the user runs `alert warning FAN-T17 "Pierdere 1.4 MWh/zi"`
- **THEN** a pending action id is printed, and the alert is logged only after `confirm <id>`

### Requirement: Automatic fallback mode
When no LLM key is configured, the chat interfaces (API and UI) SHALL accept CLI commands as messages and SHALL state that deterministic mode is active.

#### Scenario: UI without key
- **WHEN** the dashboard starts without an LLM key and the user types `losses week`
- **THEN** the loss overview is displayed from `loss_overview`, with a banner showing deterministic mode

