# Spec Delta

## Purpose

Exposes the agent, its tools, pending actions and sessions over a local HTTP API. The dashboard and any other client use the same backend as the CLI.

## ADDED Requirements

### Requirement: Chat endpoint
The API SHALL accept a user message for a session and return the agent's answer, the tool calls made (with result ids), any pending actions created, grounding status and the active mode (`llm` or `deterministic`).

#### Scenario: Chat turn
- **WHEN** a client posts a message to the chat endpoint with a session id
- **THEN** the response contains `answer`, `tool_calls`, `pending_actions`, `grounding` and `mode`

### Requirement: Direct tool access for read and what-if tools
The API SHALL let clients call `read` and `what_if` tools directly with schema-validated arguments, without going through the LLM. `write` tools called this way SHALL return a pending action instead of executing.

#### Scenario: Dashboard loads cost map
- **WHEN** the dashboard calls the `compute_cost_map` tool endpoint for `fantanele`
- **THEN** it receives the same schema-valid result the agent would receive

#### Scenario: Direct write call
- **WHEN** a client calls `create_work_orders` through the tool endpoint
- **THEN** the response is a pending action, and no files are written

### Requirement: Pending action endpoints
The API SHALL let clients list pending actions for a session and confirm or reject one by id.

#### Scenario: Confirm through API
- **WHEN** a client confirms a valid pending action id
- **THEN** the action executes once, and the response contains the tool result

### Requirement: Clear errors on bad input
Invalid requests SHALL return a 4xx response with a structured error body (`code`, `message`, `details`). Internal failures SHALL return a 5xx response without stack traces or secrets.

#### Scenario: Unknown tool
- **WHEN** a client calls a tool name that does not exist
- **THEN** the API returns 404 with code `unknown_tool`

#### Scenario: Schema violation
- **WHEN** a client sends arguments that fail validation
- **THEN** the API returns 422 with the failing fields

### Requirement: Local-only exposure
By default the API SHALL bind only to the loopback interface. Binding to other interfaces SHALL require an explicit configuration change.

#### Scenario: Default start
- **WHEN** `make agent` starts the service with default configuration
- **THEN** it listens only on 127.0.0.1

### Requirement: Health and mode reporting
The API SHALL expose a health endpoint reporting the LLM mode, the configured provider and model (without keys), the backend (fixture or live) of each tool, and forecast staleness.

#### Scenario: Health check
- **WHEN** a client requests the health endpoint
- **THEN** it returns mode, provider, model, per-tool backend labels and stale-forecast status

