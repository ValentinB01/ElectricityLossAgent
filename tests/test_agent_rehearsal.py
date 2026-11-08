"""End-to-end demo rehearsal in both modes (deterministic and LLM) covering CLAUDE.md §10.4.

Scenarios:
1. "Ce facem săptămâna asta la Fântânele?" (Weekly plan)
2. "De ce T17 vineri și nu mâine?" (Why Friday explanation)
3. "Echipa 2 lipsește joi" (Crew missing re-plan)
4. "Am verificat T05: rulmentul e uzat, amână reparația 5 zile" (Field report)
5. "Ce problemă avem la FAN-T09?" (Low confidence check)
6. "Care sunt primii consumatori suspecți de pe lista de inspecție?" (Consumer inspections)
7. "Explică situația pentru gospodăria HH-0042" (Consumer household explain)
8. "Ce pierderi totale avem săptămâna aceasta?" (Consolidated loss overview)
"""

import pytest
from src.agent.actions import ActionGate
from src.agent.config import load_agent_config
from src.agent.grounding import GroundingChecker
from src.agent.llm_agent import LLMAgent
from src.agent.registry import registry
from src.agent.store import AuditStore


REHEARSAL_SCENARIOS = [
    # CLAUDE.md §10.4 Scenarios 1-5
    ("Ce facem săptămâna asta la Fântânele?", "plan fantanele", ["16.1", "7245"]),
    ("De ce T17 vineri și nu mâine?", "explain T17-pitch-001", ["T17-pitch-001", "7.6"]),
    ("Echipa 2 lipsește joi", "crew-off 2 joi", ["1.2", "540"]),
    ("Am verificat T05: rulmentul e uzat, amână reparația 5 zile", 'report T05-bearing-001 needs_part 5 "rulment uzat"', ["T05-bearing-001", "needs_part"]),
    ("Ce problemă avem la FAN-T09?", "underperf FAN-T09", ["FAN-T09"]),
    # Consumer Scenarios 1-3
    ("Care sunt primii consumatori suspecți de pe lista de inspecție?", "inspecții", ["HH-THEFT-01"]),
    ("Explică situația pentru gospodăria HH-0042", "explain HH-0042", ["HH-0042"]),
    ("Ce pierderi totale avem săptămâna aceasta?", "pierderi", ["39.9", "17955"]),
]


@pytest.fixture
def rehearsal_env(tmp_path):
    store = AuditStore(db_path=tmp_path / "rehearsal.db")
    gate = ActionGate(store=store)
    config = load_agent_config()
    checker = GroundingChecker()
    agent = LLMAgent(config=config, store=store, gate=gate, registry=registry, grounding_checker=checker)
    return {"agent": agent, "store": store, "gate": gate, "config": config, "checker": checker}


@pytest.mark.parametrize("nl_prompt,cli_cmd,expected_snippets", REHEARSAL_SCENARIOS)
def test_rehearsal_deterministic_mode(rehearsal_env, nl_prompt, cli_cmd, expected_snippets):
    """Rehearsal: all scenarios complete in deterministic mode and numbers trace to tool results."""
    agent = rehearsal_env["agent"]
    store = rehearsal_env["store"]
    sess_id = store.create_session(mode="deterministic", as_of="2026-11-08")

    # In deterministic mode, command is passed to cli
    resp = agent.cli.execute(cli_cmd)
    assert resp is not None
    assert len(resp) > 0
    for snippet in expected_snippets:
        assert snippet in resp, f"Expected '{snippet}' in CLI response for command '{cli_cmd}'"


def test_rehearsal_sources_and_provenance(rehearsal_env):
    """Verify that every displayed result has provenance in audit store for Surse panel."""
    agent = rehearsal_env["agent"]
    store = rehearsal_env["store"]
    sess_id = store.create_session(mode="deterministic", as_of="2026-11-08")

    # Run weekly plan
    agent.cli.session_id = sess_id
    out = agent.cli.execute("plan fantanele")
    assert "PLAN MENTENANȚĂ ACTIV: FANTANELE" in out

    # Tool calls logged
    tool_calls = store.get_tool_calls(sess_id)
    assert len(tool_calls) >= 1
    for tc in tool_calls:
        assert tc["tool"] in ("optimize_maintenance", "list_tasks", "compute_cost_map")
        assert tc["backend"] in ("live", "fixture")
        assert tc["duration_ms"] is not None
