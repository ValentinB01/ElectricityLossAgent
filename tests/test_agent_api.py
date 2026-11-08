"""FastAPI TestClient tests verifying the HTTP API endpoints, contracts, and error handling."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.agent.actions import ActionGate
from src.agent.api import create_app
from src.agent.config import AgentConfig
from src.agent.llm_agent import AgentResponse, LLMAgent
from src.agent.registry import ToolRegistry, registry as default_registry
from src.agent.schemas.common import ErrorCode
from src.agent.store import AuditStore
import src.agent.tools  # Register tools


class FakeCompletionResponse:
    def __init__(self, content: str = "Răspuns test în limba română."):
        class Choice:
            def __init__(self, c):
                class Message:
                    content = c
                    tool_calls = []
                self.message = Message()
        self.choices = [Choice(content)]


def mock_completion(*args, **kwargs):
    return FakeCompletionResponse()


@pytest.fixture
def test_client(tmp_path):
    """Create isolated TestClient with clean SQLite database and mock LLMAgent."""
    db_file = tmp_path / "api_test_agent.db"
    store = AuditStore(db_path=db_file)
    gate = ActionGate(store=store)
    cfg = AgentConfig(as_of="2026-11-08")

    reg = ToolRegistry()
    for name, defn in default_registry._tools.items():
        reg._tools[name] = defn
    reg.set_store(store)

    agent = LLMAgent(
        config=cfg,
        store=store,
        gate=gate,
        registry=reg,
        completion_fn=mock_completion,
    )

    app = create_app(
        agent=agent,
        store=store,
        gate=gate,
        registry=reg,
        config=cfg,
    )

    client = TestClient(app, raise_server_exceptions=False)
    return {
        "client": client,
        "store": store,
        "gate": gate,
        "registry": reg,
        "config": cfg,
    }


# ==============================================================================
# Health and Mode Reporting
# ==============================================================================
def test_get_health(test_client):
    """Verify /health reports mode, provider, model (no keys), and per-tool backends."""
    client = test_client["client"]
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["mode"] in ("llm", "deterministic")
    assert data["provider"] == "gemini"
    assert data["model"] == "gemini/gemini-2.5-flash"
    assert "api_key" not in str(data).lower()
    assert data["as_of"] == "2026-11-08"
    assert data["stale_forecast"] is False
    assert isinstance(data["backends"], dict)
    assert len(data["backends"]) >= 15
    assert data["backends"]["get_weather_forecast"] == "fixture"


# ==============================================================================
# Chat Turn Endpoint
# ==============================================================================
def test_post_chat_turn(test_client):
    """Verify /chat processes a message, creates a session if needed, and returns structured answer."""
    client = test_client["client"]
    payload = {
        "message": "Ce facem săptămâna asta la Fântânele?",
        "session_id": "test-session-123",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "answer" in data
    assert "tool_calls" in data
    assert "pending_actions" in data
    assert "mode" in data
    assert data["session_id"] == "test-session-123"
    assert "tool_result_ids" in data

    # Verify session and messages are persisted in store
    store = test_client["store"]
    sess = store.get_session("test-session-123")
    assert sess is not None
    msgs = store.get_messages("test-session-123")
    assert len(msgs) >= 2  # user and assistant


# ==============================================================================
# Tool Catalogue Endpoints
# ==============================================================================
def test_list_tools(test_client):
    """Verify GET /tools returns full catalogue of 15 registered tools with schemas."""
    client = test_client["client"]
    resp = client.get("/tools")
    assert resp.status_code == 200
    tools = resp.json()

    assert len(tools) == 15
    tool_names = {t["name"] for t in tools}
    assert "get_weather_forecast" in tool_names
    assert "compute_cost_map" in tool_names
    assert "optimize_maintenance" in tool_names
    assert "create_work_orders" in tool_names

    for t in tools:
        assert "name" in t
        assert "description" in t
        assert "effect" in t
        assert "input_schema" in t
        assert "output_schema" in t


def test_get_tool_definition(test_client):
    """Verify GET /tools/{name} returns schema for valid tool and 404 for unknown tool."""
    client = test_client["client"]

    # 1. Valid tool
    resp = client.get("/tools/compute_cost_map")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "compute_cost_map"
    assert data["effect"] == "read"
    assert "input_schema" in data

    # 2. Unknown tool -> 404 unknown_tool
    resp_bad = client.get("/tools/unknown_nonexistent_tool")
    assert resp_bad.status_code == 404
    err = resp_bad.json()
    assert err["code"] == ErrorCode.UNKNOWN_TOOL.value


# ==============================================================================
# Direct Tool Execution
# ==============================================================================
def test_direct_tool_call_read(test_client):
    """Verify POST /tools/{name} executes read tools directly and returns valid schema output."""
    client = test_client["client"]
    resp = client.post("/tools/compute_cost_map", json={"site_id": "fantanele", "horizon_days": 7})
    assert resp.status_code == 200
    data = resp.json()

    assert data["site_id"] == "fantanele"
    assert "entries" in data
    assert "provenance" in data


def test_direct_tool_call_write_intercepted(test_client, tmp_path):
    """Verify POST /tools/{name} for write tool produces pending action and writes no files."""
    client = test_client["client"]

    # Ensure work orders directory is clean
    work_orders_dir = Path("results/work_orders")
    existing_files_before = set(work_orders_dir.glob("*.csv")) if work_orders_dir.exists() else set()

    # Call direct write tool create_work_orders
    resp = client.post(
        "/tools/create_work_orders",
        json={"schedule_id": "sched-active-001"},
        params={"session_id": "test-write-session"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Must return pending action payload
    assert data["status"] == "pending_confirmation"
    assert "action_id" in data
    assert data["tool"] == "create_work_orders"

    # Verify NO files were created on disk
    existing_files_after = set(work_orders_dir.glob("*.csv")) if work_orders_dir.exists() else set()
    assert existing_files_after == existing_files_before


def test_direct_tool_call_unknown_tool_404(test_client):
    """Verify POST /tools/{name} for nonexistent tool returns 404 with code unknown_tool."""
    client = test_client["client"]
    resp = client.post("/tools/ghost_tool", json={})
    assert resp.status_code == 404
    err = resp.json()
    assert err["code"] == ErrorCode.UNKNOWN_TOOL.value


def test_direct_tool_call_schema_violation_422(test_client):
    """Verify POST /tools/{name} with invalid parameters returns 422 with failing fields."""
    client = test_client["client"]
    # days is below minimum (minimum is 1)
    resp = client.post("/tools/get_weather_forecast", json={"site_id": "fantanele", "days": -3})
    assert resp.status_code == 422
    err = resp.json()
    assert err["code"] == ErrorCode.BACKEND_CONTRACT_VIOLATION.value
    assert "details" in err


# ==============================================================================
# Pending Action Endpoints (List, Confirm, Reject, Patch)
# ==============================================================================
def test_action_lifecycle_confirm_executes_once(test_client):
    """Verify creating pending action, listing, confirming executes once, and second confirm fails."""
    client = test_client["client"]

    # 1. Create pending action via direct call
    create_resp = client.post(
        "/tools/create_work_orders",
        json={"schedule_id": "sched-active-001"},
        params={"session_id": "session-action-1"},
    )
    action_data = create_resp.json()
    action_id = action_data["action_id"]

    # 2. List actions filtered by session
    list_resp = client.get("/actions", params={"session_id": "session-action-1"})
    assert list_resp.status_code == 200
    actions = list_resp.json()
    assert len(actions) == 1
    assert actions[0]["id"] == action_id
    assert actions[0]["state"] == "pending"

    # 3. Confirm action executes once
    confirm_resp = client.post(f"/actions/{action_id}/confirm")
    assert confirm_resp.status_code == 200
    confirmed = confirm_resp.json()
    assert confirmed["status"] in ("executed", "confirmed")
    assert confirmed["action_id"] == action_id

    # 4. Confirming again MUST fail (executes-once guard)
    second_confirm = client.post(f"/actions/{action_id}/confirm")
    assert second_confirm.status_code == 400
    err = second_confirm.json()
    assert err["code"] in ("already_executed", "action_already_decided")


def test_action_reject(test_client):
    """Verify rejecting a pending action updates state and prevents subsequent confirmation."""
    client = test_client["client"]

    # Create pending action
    create_resp = client.post(
        "/tools/send_alert",
        json={"level": "warning", "asset_id": "FAN-T17", "message": "Test"},
    )
    action_id = create_resp.json()["action_id"]

    # Reject action
    reject_resp = client.post(f"/actions/{action_id}/reject")
    assert reject_resp.status_code == 200
    rejected = reject_resp.json()
    assert rejected["status"] == "rejected"

    # Attempt to confirm rejected action fails
    confirm_resp = client.post(f"/actions/{action_id}/confirm")
    assert confirm_resp.status_code == 400


def test_action_patch_edits_fields(test_client):
    """Verify PATCH /actions/{id} modifies args of pending action."""
    client = test_client["client"]

    # Create pending action
    create_resp = client.post(
        "/tools/update_from_field_report",
        json={
            "task_id": "T05-bearing-001",
            "status": "needs_part",
            "defer_days": 2,
        },
    )
    action_id = create_resp.json()["action_id"]

    # Patch defer_days to 5
    patch_resp = client.patch(
        f"/actions/{action_id}",
        json={"edits": {"defer_days": 5, "status": "needs_part"}},
    )
    assert patch_resp.status_code == 200
    patched = patch_resp.json()
    assert patched["args"]["defer_days"] == 5

    # Verify store has updated args
    stored = test_client["store"].get_pending_action(action_id)
    assert stored["args"]["defer_days"] == 5


# ==============================================================================
# Schedules and Session Endpoints
# ==============================================================================
def test_schedules_endpoints(test_client):
    """Verify GET /schedules/active and GET /schedules/{id}."""
    client = test_client["client"]
    store = test_client["store"]

    # Seed an active schedule
    store.store_schedule(
        schedule_id="sched-test-active",
        status="active",
        payload={"schedule_id": "sched-test-active", "jobs": []},
        total_mwh=16.1,
        total_ron=7245.0,
    )

    # 1. GET /schedules/active
    active_resp = client.get("/schedules/active")
    assert active_resp.status_code == 200
    data = active_resp.json()
    assert data["id"] == "sched-test-active"
    assert data["status"] == "active"
    assert data["total_mwh"] == 16.1

    # 2. GET /schedules/{id}
    sched_resp = client.get("/schedules/sched-test-active")
    assert sched_resp.status_code == 200
    assert sched_resp.json()["id"] == "sched-test-active"

    # 3. GET /schedules/{id} unknown -> 404
    bad_resp = client.get("/schedules/non_existent_sched")
    assert bad_resp.status_code == 404


def test_get_session_transcript(test_client):
    """Verify GET /sessions/{id} returns session record, messages and tool calls."""
    client = test_client["client"]
    store = test_client["store"]

    # Create session with messages and a tool call
    sess_id = store.create_session(mode="llm", as_of="2026-11-08")
    store.add_message(sess_id, "user", "Întrebare test")
    store.add_message(sess_id, "assistant", "Răspuns test")
    store.log_tool_call(
        session_id=sess_id,
        tool="list_tasks",
        effect="read",
        args={"site_id": "fantanele"},
        result_id="res-001",
    )

    resp = client.get(f"/sessions/{sess_id}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["session"]["id"] == sess_id
    assert len(data["messages"]) == 2
    assert len(data["tool_calls"]) == 1
    assert data["tool_calls"][0]["tool"] == "list_tasks"


# ==============================================================================
# Security & Secret Sanitization
# ==============================================================================
def test_server_error_sanitization(test_client, monkeypatch):
    """Verify internal errors (500) return generic code without stack trace or secrets."""
    client = test_client["client"]

    # Force an unexpected internal exception inside an endpoint handler
    def crashing_list_actions(*args, **kwargs):
        raise RuntimeError("Internal crash with fake sk-SECRET12345678901234567890 key")

    monkeypatch.setattr(test_client["store"], "list_pending_actions", crashing_list_actions)

    resp = client.get("/actions")
    assert resp.status_code == 500
    err = resp.json()

    assert err["code"] == "internal_server_error"
    assert "sk-SECRET" not in str(err)
    assert "traceback" not in str(err).lower()


# ==============================================================================
# Extended Edge Case & Compliance Tests
# ==============================================================================
def test_direct_tool_call_without_body(test_client):
    """Verify calling a tool with optional parameters without an HTTP body succeeds."""
    client = test_client["client"]
    resp = client.post("/tools/loss_overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_mwh" in data
    assert "total_ron" in data


def test_direct_tool_call_unknown_item_404(test_client):
    """Verify explain tool with unknown item_id returns 404 unknown_item."""
    client = test_client["client"]
    resp = client.post("/tools/explain", json={"item_id": "nonexistent_item_999"})
    assert resp.status_code == 404
    err = resp.json()
    assert err["code"] == ErrorCode.UNKNOWN_ITEM.value
    assert "details" in err


def test_action_confirm_with_args_hash_validation(test_client):
    """Verify action confirmation validates args_hash when supplied."""
    client = test_client["client"]
    create_resp = client.post(
        "/tools/send_alert",
        json={"level": "warning", "asset_id": "FAN-T17", "message": "Test hash"},
    )
    action_data = create_resp.json()
    action_id = action_data["action_id"]
    correct_hash = action_data["args_hash"]

    # 1. Mismatched args_hash fails with 400 args_mismatch
    bad_resp = client.post(f"/actions/{action_id}/confirm", json={"args_hash": "wrong_hash_val"})
    assert bad_resp.status_code == 400
    assert bad_resp.json()["code"] == "args_mismatch"

    # 2. Matching args_hash succeeds
    ok_resp = client.post(f"/actions/{action_id}/confirm", json={"args_hash": correct_hash})
    assert ok_resp.status_code == 200
    assert ok_resp.json()["status"] == "executed"


def test_action_patch_flat_body(test_client):
    """Verify PATCH /actions/{id} supports flat body without 'edits' key."""
    client = test_client["client"]
    create_resp = client.post(
        "/tools/update_from_field_report",
        json={
            "task_id": "T05-bearing-001",
            "status": "needs_part",
            "defer_days": 2,
        },
    )
    action_id = create_resp.json()["action_id"]

    # Send flat payload
    patch_resp = client.patch(
        f"/actions/{action_id}",
        json={"defer_days": 10, "status": "deferred"},
    )
    assert patch_resp.status_code == 200
    patched = patch_resp.json()
    assert patched["args"]["defer_days"] == 10


def test_cors_headers_streamlit_origin(test_client):
    """Verify Streamlit origin receives permissive CORS headers."""
    client = test_client["client"]
    headers = {
        "Origin": "http://localhost:8501",
        "Access-Control-Request-Method": "POST",
    }
    resp = client.options("/chat", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8501"


def test_health_stale_forecast_reporting(tmp_path):
    """Verify GET /health reports stale_forecast when enabled in config."""
    cfg = AgentConfig(stale_forecast=True)
    app = create_app(config=cfg)
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["stale_forecast"] is True
