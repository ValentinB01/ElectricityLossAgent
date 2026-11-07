# Spec Delta

## Purpose

Defines the user-facing interface: a chat with the agent next to the operational dashboard panels. Every answer can be traced to its data, and every action is confirmed visibly.

## ADDED Requirements

### Requirement: Chat beside dashboard
The interface SHALL show a chat panel next to dashboard panels for the loss overview, the producers map coloured by `pi_72h`, the 7-day cost map, the maintenance calendar and the inspection list. All panels SHALL be fed by the same tools as the agent.

#### Scenario: Initial load
- **WHEN** the user opens the dashboard
- **THEN** the loss overview, map, cost map, calendar and inspection list are displayed from tool results, and the chat is ready for input

### Requirement: Sources on every answer
Each agent answer SHALL include an expandable "Surse" section listing the tool calls, their arguments, backend labels and result ids it relied on.

#### Scenario: Inspect sources
- **WHEN** the user expands "Surse" under an answer
- **THEN** the tool names, arguments, backend labels and result ids for that turn are shown

### Requirement: Confirmation cards
Pending actions SHALL be rendered as cards that show the human-readable summary and exact arguments, with Confirmă and Respinge buttons. Field-report cards SHALL let the user edit parsed fields before confirming.

#### Scenario: Confirm a field report
- **WHEN** the agent produces a pending field-report action
- **THEN** a card shows the parsed fields as editable inputs, and only pressing Confirmă applies it

### Requirement: Draft vs. active plan display
When a what-if draft exists, the calendar SHALL show the draft next to the active plan, highlighting moved or added jobs and the total cost difference, with a control to promote the draft (a confirmed action).

#### Scenario: Crew missing scenario
- **WHEN** the user asks about crew-2 missing on Thursday
- **THEN** the calendar shows active and draft side by side, with changed jobs highlighted and the cost difference

### Requirement: Trust indicators
The interface SHALL visibly mark deterministic mode, fixture-backed data, stale forecasts, low-confidence items and answers that failed the grounding check.

#### Scenario: Fixture data
- **WHEN** a panel's data comes from a fixture backend
- **THEN** the panel shows a "date demonstrative" label

#### Scenario: Grounding warning
- **WHEN** an answer failed the grounding check after retry
- **THEN** the answer shows a warning badge listing the unverified values

### Requirement: Demo data disclosure and attribution
The interface SHALL state that Fântânele and Călugăreni assets replay public data under demo ids, and SHALL show "Weather data by Open-Meteo.com" wherever weather-derived data appears.

#### Scenario: Map disclosure
- **WHEN** the producers map for Fântânele is displayed
- **THEN** a note explains that turbine behaviour is replayed from Kelmarsh data under demo ids

### Requirement: Drill-down from panels to explanations
Selecting a job in the calendar, a turbine on the map or a household in the inspection list SHALL show its explanation from `explain`, without requiring the LLM.

#### Scenario: Click a household
- **WHEN** the user selects a household in the inspection list
- **THEN** the actual vs. expected monthly chart, the peer comparison and the top 3 factors are shown

