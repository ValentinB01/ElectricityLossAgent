# Proposal: add the LLM agent layer

## Why

The project's third layer, the LLM agent (README §3.3, CLAUDE.md §10, stage E7), is specified only as a tool list and a system-prompt skeleton. Without an agreed architecture there's no contract between the agent and the producers/consumers modules. There's also no rule for how "the LLM does not decide" gets enforced or how write actions are controlled, and no plan for the UI the jury will see. The agent is worth 25 + 10 points (agent + explainability) and lands late in the 24 h (hours 18–21), so its contract has to be fixed now. That way it can be built in parallel with the ML work against fixture data.

## What Changes

- **Tool registry with typed contracts:** every tool in README §3.3, plus `promote_schedule` (draft → active plan), gets Pydantic input/output schemas, an `effect` class (`read`, `what_if`, `write`) and a backend that is either `fixture` (a committed demo scenario pack in `src/agent/fixtures/`) or `live` (the real module). The agent, the CLI, the API and the dashboard all share this one registry.
- **Orchestration loop:** a small hand-written tool-calling loop over a provider-agnostic adapter (LiteLLM, default Gemini). Romanian-only answers, a hard cap on iterations, and a fixed system prompt (CLAUDE.md §10.5).
- **Numeric grounding check:** a post-processor checks that every number in the final answer appears in a tool result from the same turn. If it fails, the loop retries once, then shows a visible warning badge.
- **Confirmation gate for state-changing actions:** `write` tools (apply a field report, create work orders, send an alert, commit a re-plan) never run directly from the LLM. Instead they create a *pending action* that the user confirms or rejects in the UI or CLI. What-if re-plans produce *draft* schedules that never replace the active plan without confirmation.
- **Audit store:** SQLite holds sessions, messages, every tool call (input, output, latency, backend) and every confirmation decision.
- **Deterministic CLI mode:** explicit commands map 1:1 to tools and render with Romanian templates. This mode is used automatically when no LLM key is set.
- **FastAPI service** (`make agent`) and a **Streamlit UI** (`make demo`): a chat panel next to the dashboard panels (map, cost map, calendar, inspection list, loss overview), with confirmation cards and "sources" for every answer.
- **Security posture for a single-user local demo:** no auth. Instead: secrets only from env, a localhost-only bind, prompt-injection containment (tool outputs and field reports are treated as data), schema validation on every tool input, and file writes limited to `results/`.

## Capabilities

### New Capabilities

- `agent-tools`: the tool registry. Typed contracts for producers, consumers and shared tools, effect classes, fixture/live backends, input validation and structured errors.
- `agent-orchestration`: the LLM loop. Provider adapter, system prompt, iteration limits, Romanian output, numeric grounding check, and low-confidence handling ("nu sunt sigur" plus a proposed check).
- `agent-actions`: the lifecycle of state-changing actions (pending → confirmed/rejected → executed), draft vs. active schedules, and the audit trail in SQLite.
- `agent-cli`: the deterministic no-LLM mode. Command grammar, 1:1 tool mapping, templated Romanian output and confirmation prompts.
- `agent-api`: the FastAPI endpoints for chat, tools, pending actions and sessions, with clear errors on bad input.
- `agent-ui`: the Streamlit interface. Chat with sources and confirmation cards, next to the dashboard panels fed by the same tools.

### Modified Capabilities

_None._ `openspec/specs/` is empty; this is the first change.

## Impact

- **New code:** `src/agent/` (`tools.py` → package `tools/`, `llm_agent.py`, `grounding.py`, `actions.py`, `store.py`, `cli.py`, `api.py`, `prompts/`), `src/agent/fixtures/demo_week/*.json`, `src/app/dashboard.py`, `tests/test_agent_*.py`. Runtime outputs: `results/agent.db`, `results/work_orders/`, `results/alerts.log`.
- **Config:** `config/agent.yaml` (provider, model, thresholds, iteration cap, grounding tolerance); reads energy price and `CONF_MIN` from `config/params.yaml`.
- **Dependencies:** `litellm`, `pydantic>=2`, `streamlit`, `fastapi`, `uvicorn` (the last three are already planned); SQLite from the stdlib.
- **Module owners:** the producers and consumers modules must expose functions that match the tool output schemas defined here (contract-first). Until they do, the fixtures stand in.
- **Docs:** `docs/design.md` gets a link to this change's design; CLAUDE.md §10 stays the build spec, and this change refines it.
- **No breaking changes:** there's no existing code yet.
