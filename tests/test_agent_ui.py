"""Unit and integration tests for the Streamlit dashboard and AgentClient."""

import os
os.environ["AGENT_UI_MODE"] = "inproc"

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from src.agent.config import AgentConfig
from src.agent.store import DEFAULT_STORE
from src.app.client import AgentClient, AgentError
from src.app.dashboard import (
    jobs_frame,
    asset_layout,
    render_explanation,
    DISCLOSURE,
    DEMO_LABEL,
    WEATHER_ATTR,
)


@pytest.fixture(autouse=True)
def setup_inproc_env(monkeypatch):
    """Ensure tests run against in-process FastAPI app without needing external server."""
    monkeypatch.setenv("AGENT_UI_MODE", "inproc")


# ==============================================================================
# AgentClient Transport & Endpoint Tests
# ==============================================================================

def test_client_inproc_health():
    client = AgentClient(mode="inproc")
    health = client.health()
    assert health["status"] == "ok"
    assert "mode" in health
    assert "backends" in health
    assert isinstance(health["backends"], dict)
    assert len(health["backends"]) >= 15


def test_client_inproc_tool_call():
    client = AgentClient(mode="inproc")
    res = client.tool("loss_overview", {"period": "week"})
    assert "total_mwh" in res
    assert "total_ron" in res
    assert "rows" in res


def test_client_inproc_chat_turn():
    client = AgentClient(mode="inproc")
    resp = client.chat("Ce pierderi avem săptămâna aceasta?")
    assert "answer" in resp
    assert "session_id" in resp
    assert "mode" in resp
    assert "tool_calls" in resp


def test_client_inproc_action_lifecycle():
    client = AgentClient(mode="inproc")
    # 1. Create a pending action via write tool
    action = client.tool("send_alert", {"level": "warning", "asset_id": "FAN-T17", "message": "UI Test alert"})
    assert action["status"] == "pending_confirmation"
    aid = action["action_id"]

    # 2. List actions
    actions = client.actions()
    assert any(a["id"] == aid for a in actions)

    # 3. Reject action
    rej = client.reject(aid)
    assert rej["status"] == "rejected"

    # 4. Confirming rejected action raises structured AgentError
    with pytest.raises(AgentError) as exc_info:
        client.confirm(aid)
    assert exc_info.value.status == 400


def test_client_inproc_schedule_retrieval():
    client = AgentClient(mode="inproc")
    # Seed active schedule if none
    client.tool("optimize_maintenance")
    active = client.active_schedule()
    assert active is not None
    assert "total_mwh" in active
    assert active["status"] == "active"


def test_client_error_handling_unknown_tool():
    client = AgentClient(mode="inproc")
    with pytest.raises(AgentError) as exc_info:
        client.tool("nonexistent_tool_123")
    assert exc_info.value.code == "unknown_tool"
    assert exc_info.value.status == 404


# ==============================================================================
# Helper Unit Tests
# ==============================================================================

def test_jobs_frame_and_change_detection():
    active = {
        "payload": {
            "jobs": [
                {
                    "task_id": "T1",
                    "asset_id": "FAN-T17",
                    "date": "2026-11-13",
                    "start_hour": 8,
                    "duration_hours": 8.0,
                    "crew": "crew-1",
                    "cost_breakdown_mwh": {"total": 7.6},
                    "cost_breakdown_ron": {"total": 3420.0},
                }
            ]
        }
    }
    draft = {
        "payload": {
            "jobs": [
                {
                    "task_id": "T1",
                    "asset_id": "FAN-T17",
                    "date": "2026-11-14",  # moved date
                    "start_hour": 8,
                    "duration_hours": 8.0,
                    "crew": "crew-1",
                    "cost_breakdown_mwh": {"total": 8.0},
                    "cost_breakdown_ron": {"total": 3600.0},
                },
                {
                    "task_id": "T2",  # new job
                    "asset_id": "FAN-T05",
                    "date": "2026-11-15",
                    "start_hour": 8,
                    "duration_hours": 6.0,
                    "crew": "crew-2",
                    "cost_breakdown_mwh": {"total": 5.5},
                    "cost_breakdown_ron": {"total": 2475.0},
                },
            ]
        }
    }
    df = jobs_frame(draft, active, "Test")
    assert len(df) == 2
    row1 = df[df["Sarcină"] == "T1"].iloc[0]
    assert row1["Schimbare"] == "mutat"
    row2 = df[df["Sarcină"] == "T2"].iloc[0]
    assert row2["Schimbare"] == "nou"


def test_asset_layout():
    assets = ["FAN-T01", "FAN-T02", "FAN-T03", "FAN-T04", "FAN-T05", "FAN-T06", "FAN-T07"]
    df = asset_layout(assets)
    assert len(df) == 7
    assert set(df["asset_id"]) == set(assets)
    assert 0 in df["x"].values
    assert 0 in df["y"].values


# ==============================================================================
# Streamlit AppTest Suite
# ==============================================================================

DASHBOARD_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app" / "dashboard.py")


def test_dashboard_initial_render():
    """Verify that dashboard renders completely without uncaught exceptions."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    assert not at.exception

    # Check title and disclosures
    assert any("PPC Loss Agent" in str(t.value) for t in at.title)
    assert any("Weather data by Open-Meteo.com" in str(c.value) for c in at.caption)

    # Check tabs exist
    tab_labels = [t.label for t in at.tabs]
    assert "Pierderi" in tab_labels
    assert "Producători" in tab_labels
    assert "Calendar" in tab_labels
    assert "Inspecții" in tab_labels


def test_dashboard_chat_interaction():
    """Verify typing a prompt into chat_input triggers a turn and displays the message."""
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    assert not at.exception

    # Submit user prompt
    at.chat_input[0].set_value("Ce pierderi avem?").run(timeout=30)
    assert not at.exception

    # Verify messages in session
    assert len(at.session_state["messages"]) >= 2
    user_msg = at.session_state["messages"][-2]
    asst_msg = at.session_state["messages"][-1]
    assert user_msg["role"] == "user"
    assert user_msg["content"] == "Ce pierderi avem?"
    assert asst_msg["role"] == "assistant"
    assert len(asst_msg["content"]) > 0
    assert "tool_calls" in asst_msg


def test_dashboard_calendar_draft_vs_active():
    """Verify scenario: crew-2 missing on Thursday displays draft next to active calendar with deltas."""
    client = AgentClient(mode="inproc")
    # Trigger what-if scenario: crew-2 unavailable Thursday
    res = client.tool(
        "optimize_maintenance",
        {"constraints": {"unavailable": [{"crew": "crew-2", "date": "2026-11-12"}]}},
    )
    assert res["status"] == "draft"
    assert res["delta_mwh"] > 0

    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    assert not at.exception

    # Check that delta metrics and promote button are rendered
    metric_labels = [m.label for m in at.metric]
    assert any("Diferență pierderi" in label for label in metric_labels)
    assert any("Promovează draftul ca plan activ" in btn.label for btn in at.button)


def test_explain_drilldown_fixtures():
    """Verify drilldown logic for job and household produces correct structures without LLM."""
    client = AgentClient(mode="inproc")

    # 1. Explain job FAN-T17 / T17-pitch-001
    job_expl = client.tool("explain", {"item_id": "T17-pitch-001"})
    assert job_expl["item_type"] == "scheduled_job"
    assert job_expl["scheduled_job_explanation"]["task_id"] == "T17-pitch-001"
    assert "cost_breakdown_mwh" in job_expl["scheduled_job_explanation"]
    assert len(job_expl["scheduled_job_explanation"]["alternatives"]) >= 1

    # 2. Explain household HH-FAULTY-01
    hh_expl = client.tool("explain", {"item_id": "HH-FAULTY-01"})
    assert hh_expl["item_type"] == "household"
    exp = hh_expl["household_explanation"]
    assert exp["household_id"] == "HH-FAULTY-01"
    assert exp["signal_type"] == "faulty_meter"
    assert len(exp["top_factors"]) == 3
    assert exp["peer_comparison"]["deviation_pct"] == -100.0

