# Spec Delta

## Purpose

Controls every state-changing action the agent can trigger. Write actions always pass through human confirmation, drafts never silently replace the active plan, and every step is recorded in an audit trail.

## ADDED Requirements

### Requirement: Write actions require confirmation
When a `write` tool is requested, by the LLM or by a CLI command, the system SHALL create a pending action describing exactly what will happen, and SHALL NOT execute it until the user explicitly confirms that pending action.

#### Scenario: Agent proposes work orders
- **WHEN** the agent requests `create_work_orders` for a schedule
- **THEN** a pending action is created with a human-readable summary and the exact arguments, nothing is written to `results/work_orders/`, and the user is shown confirm/reject controls

#### Scenario: User confirms
- **WHEN** the user confirms a pending action
- **THEN** the tool executes with the stored arguments, and the result is shown and recorded

#### Scenario: User rejects
- **WHEN** the user rejects a pending action
- **THEN** the action is marked rejected, nothing executes, and the agent acknowledges it

### Requirement: Confirmation is bound to the exact action
A confirmation SHALL apply only to the pending action it was issued for, with the arguments shown to the user. A pending action SHALL execute at most once and SHALL expire after a configured time.

#### Scenario: Double confirmation
- **WHEN** the same pending action is confirmed twice
- **THEN** it executes once, and the second confirmation returns `already_executed`

#### Scenario: Expired action
- **WHEN** the user confirms a pending action after it expired
- **THEN** it does not execute, and the user is asked to request it again

#### Scenario: LLM cannot self-confirm
- **WHEN** the model output claims the user has confirmed an action
- **THEN** no action executes; only the explicit confirm control or CLI confirmation command can execute it

### Requirement: Field reports validated before application
`update_from_field_report` SHALL turn free text into the field report schema (`task_id`, `status`, `defer_days`, `new_finding`, `confidence`). Missing required fields, an unknown `task_id` or `confidence` below 0.7 SHALL require the user to complete or correct the fields before confirmation.

#### Scenario: Clear report
- **WHEN** the technician writes "Am verificat T05, rulmentul e uzat, piesa vine în 5 zile"
- **THEN** a pending action shows `status: needs_part`, `defer_days: 5` and the finding, and after confirmation the task is updated and a re-plan draft is produced

#### Scenario: Ambiguous report
- **WHEN** the report does not identify a task, or the parsed confidence is below 0.7
- **THEN** the system asks the user to choose or correct the fields before the action can be confirmed

#### Scenario: Household outcome
- **WHEN** an inspector writes "Am verificat gospodăria X: contor defect, nu furt"
- **THEN** after confirmation the case is closed with label `faulty_meter`, and the label is saved for retraining

### Requirement: Draft and active schedules
What-if runs of `optimize_maintenance` SHALL produce draft schedules that never alter the active schedule. Promoting a draft to active SHALL be a confirmed write action that records the cost difference.

#### Scenario: Draft does not change plan
- **WHEN** the user asks a what-if question that produces a new schedule
- **THEN** the active schedule shown in the dashboard is unchanged, and the draft is shown alongside with its cost difference

#### Scenario: Promote draft
- **WHEN** the user confirms promoting a draft
- **THEN** it becomes the active schedule, the previous active schedule is kept as history, and the change is recorded with its cost difference

### Requirement: Alerts are simulated and logged
`send_alert` SHALL only append to the alerts log and the audit store. It SHALL NOT contact external systems.

#### Scenario: Alert after confirmation
- **WHEN** a confirmed `send_alert` executes
- **THEN** a line with timestamp, level, asset id and message is appended to `results/alerts.log`

### Requirement: Audit trail
The system SHALL persist, per session: user messages, agent answers, every tool call with arguments, result, backend, duration and errors, grounding-check outcomes, and every pending action with its final state and timestamps.

#### Scenario: Trace an answer
- **WHEN** a reviewer opens a past answer
- **THEN** they can see every tool call and result that produced it and any grounding warnings

#### Scenario: Restart persistence
- **WHEN** the application restarts
- **THEN** previous sessions, the active schedule and pending action history are still available

### Requirement: No secrets in the audit trail
The audit store and logs SHALL NOT contain API keys or provider credentials.

#### Scenario: Provider error logged
- **WHEN** a provider call fails with an error that includes request headers
- **THEN** the stored error has credentials redacted

