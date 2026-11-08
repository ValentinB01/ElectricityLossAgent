# Tasks

> Groups 1–4 don't depend on the ML modules and can run in parallel with producers-plan steps 1–6. Groups 5–7 are stage E7. Group 8 runs as each module lands.

## 1. Foundation and contracts

- [x] 1.1 Add `litellm`, `pydantic>=2`, `jinja2` to `requirements.txt` and create `config/agent.yaml` (provider, model, `as_of`, iteration/tool/timeout/token limits, grounding tolerance, small-count exemption, action TTL, per-tool backend map, bind host 127.0.0.1); verify `python -c "import litellm, pydantic, jinja2"` succeeds and the config loads into a typed settings object
- [x] 1.2 Write Pydantic I/O schemas in `src/agent/schemas/` for all 15 tools (incl. provenance block, `stale`, `low_confidence`, derived totals and deltas per design D5, structured error model with codes `unknown_asset`, `unknown_item`, `invalid_date`, `unknown_tool`, `backend_contract_violation`, `unsupported_scenario`); verify `tests/test_agent_contracts.py::test_schemas_export_json_schema` passes for every tool
- [x] 1.3 Document the contract for module owners in `docs/agent-contract.md` (the function each live adapter will call, expected DataFrame columns and the target schema) and link it from `docs/design.md`; verify every tool in the registry appears in the document

## 2. Fixture pack (demo world)

- [x] 2.1 Author `src/agent/fixtures/demo_week/*.json`: Fântânele + Călugăreni assets under demo ids, T17 pitch fault reproducing the 7.6 vs 19.4 MWh example, one low-confidence asset, one stale-forecast variant, households including one faulty-meter case, loss overview with MWh+RON; verify `tests/test_agent_contracts.py::test_fixtures_validate` passes for every file
- [x] 2.2 Add precomputed `optimize_maintenance` variants (baseline, crew-2 off Thursday, T05 deferred 5 days) with consistent totals and deltas; verify a test asserts `delta = draft_total − active_total` for each variant and unknown constraints return `unsupported_scenario`

## 3. Tool registry and dispatcher

- [x] 3.1 Implement `registry.py` (`@tool` decorator, effect class, input/output validation, provenance, `llm_view`, backend resolution from config) and `backends/fixture.py`; verify contract tests: bad asset id → `unknown_asset` with valid ids, past date → `invalid_date`, malformed backend output → `backend_contract_violation`
- [x] 3.2 Implement the 15 tool wrappers in `src/agent/tools/` with `llm_view` reducers (cost map → cheapest workable window per asset/day; top-N lists); verify a test checks every numeric leaf in each `llm_view` exists in the full result
- [x] 3.3 Implement `store.py` (SQLite, WAL, parameterised queries, tables from design D14, secret-redaction filter); verify tests that tool calls and results persist across a reopen and that a fake `api_key` in an error is redacted

## 4. Action gate, schedules and CLI

- [x] 4.1 Implement `actions.py`: pending action creation for `write` tools, args hash, TTL, confirm/reject/patch, execute-once, `needs_input` for field reports (effective confidence = min(LLM, rule checks) < 0.7), post-hooks (field report → what-if draft; household outcome → `labels`); verify `tests/test_agent_actions.py` covers double confirm, expiry, reject, args mismatch, needs_input→pending
- [x] 4.2 Implement schedule lifecycle (draft on every `optimize_maintenance`, `promote_schedule` swaps active→history atomically) and confine writes to `results/work_orders/` and `results/alerts.log`; verify tests that a draft never alters the active schedule and that promote is a single transaction
- [x] 4.3 Implement `cli.py` command router (grammar in design D9) and Romanian Jinja2 templates; verify golden-output tests in `tests/test_agent_cli.py` for every command on the fixture pack, including bad-argument usage messages and the `confirm`/`reject` flow

## 5. LLM orchestrator and grounding (E7)

- [x] 5.1 Implement `grounding.py` (Romanian number parser, id/date/user-number exemptions, small-count rule, tolerance by displayed precision, kWh↔MWh and fraction↔% conversions); verify `tests/test_agent_grounding.py` covers `7,6`, `1.250`, `12%`, `FAN-T17`, `2026-11-12`, and a deliberately ungrounded value
- [x] 5.2 Implement `llm_agent.py`: LiteLLM adapter (temperature 0, parallel tool calls), context assembly (design D10), system prompt in `prompts/system_ro.md`, dispatch via registry, write→pending result, iteration/tool/timeout/token limits with partial answer, one grounding retry then warning, `{{table:...}}` placeholders, provider-failure → deterministic suggestion; verify `tests/test_agent_scenarios.py` replays the 8 demo scenarios with a scripted fake LLM and asserts tool sequence, pending actions and grounding outcome
- [x] 5.3 Implement mode selection (no key → deterministic) and the injection guard (untrusted text never reaches a confirm path); verify a scenario test where a field report containing an instruction to send alerts yields no alert and only a parsed pending field report

## 6. API (E7)

- [x] 6.1 Implement `api.py` endpoints from design D12 (`/chat`, `/tools`, `/tools/{name}`, `/actions*`, `/schedules*`, `/sessions/{id}`, `/health`) with structured 4xx/5xx errors and loopback bind; verify FastAPI TestClient tests: unknown tool → 404 `unknown_tool`, bad args → 422 with fields, direct write call → pending action and no files written, confirm executes once
- [x] 6.2 Wire `make agent` (uvicorn on 127.0.0.1) and add an agent section to README §6 (env var for the key, deterministic mode, `make eval-agent`); verify `make agent` starts and `GET /health` reports mode and per-tool backends

## 7. Streamlit UI (E7–E8)

- [ ] 7.1 Implement `src/app/dashboard.py` with `AgentClient` (HTTP default, in-process flag), header trust indicators, tabs Pierderi / Producători / Calendar / Inspecții fed by `/tools/*`, footer attribution and demo-data disclosure; verify `make demo` loads all tabs with the API running and with no LLM key
- [ ] 7.2 Implement the chat panel: markdown without HTML, "Surse" expander, grounding badge, `{{table:...}}` rendering, confirmation cards with editable field-report fields, active vs draft calendar with highlighted changes and delta; verify manually the scenario "Echipa 2 lipsește joi" → draft shown → Confirmă → active calendar updated
- [ ] 7.3 Implement drill-down from map, calendar and inspection list to `explain` (cost table / actual-vs-expected chart, peer comparison, top 3 factors); verify selecting T17 and the faulty-meter household shows their explanations without the LLM

## 8. Live backends and evaluation

- [ ] 8.1 Implement `backends/live.py` adapters and switch `optimize_maintenance` to live (on fixture tasks and cost map) as soon as producers step 5 lands; verify the contract test suite passes with the live backend and the scripted scenarios still pass
- [ ] 8.2 Switch remaining producer tools and then consumer tools to live as modules land, one tool per commit; verify contract tests pass after each switch and `/health` shows `live` for that tool
- [ ] 8.3 Implement `make eval-agent` (8 scenarios + ~10 Romanian paraphrases against the real model; expected-tool recall, grounding pass rate, confirmations requested where required, latency → `results/agent_eval.json`); verify the file is produced and the results are cited in README §4
- [ ] 8.4 End-to-end demo rehearsal in both modes covering CLAUDE.md §10.4 scenarios 1–5 and the 3 consumer scenarios; verify each scenario completes and every displayed number traces to a tool result in the "Surse" panel

## Workflow follow-up

- Run `openspec validate add-llm-agent --strict` before applying.
- Archive the change after review and once the scenarios pass, so the six capabilities move to `openspec/specs/`.

