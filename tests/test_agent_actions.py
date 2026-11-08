"""Tests for Action Gate, confirmation lifecycle, schedules, and audit store."""

import json
from pathlib import Path
import pytest

from src.agent.actions import ActionGate, compute_args_hash
from src.agent.registry import registry
import src.agent.tools  # Register tools
from src.agent.schemas import OptimizeMaintenanceOutput
from src.agent.store import AuditStore, redact_secrets


@pytest.fixture
def tmp_store(tmp_path):
    db_file = tmp_path / "test_agent.db"
    return AuditStore(db_path=db_file)


@pytest.fixture
def tmp_gate(tmp_store):
    return ActionGate(store=tmp_store)


def test_write_action_intercepted_and_pending(tmp_gate):
    """Verify that calling a write tool creates a pending action and performs no side effects."""
    args = {"schedule_id": "sched-active-001"}
    res = tmp_gate.create_pending_action("create_work_orders", args)

    assert res["status"] == "pending_confirmation"
    assert res["state"] == "pending"
    assert "action_id" in res
    assert "Generare ordine de lucru" in res["summary"]

    # Verify action stored in SQLite
    action_in_db = tmp_gate.store.get_pending_action(res["action_id"])
    assert action_in_db is not None
    assert action_in_db["tool"] == "create_work_orders"
    assert action_in_db["state"] == "pending"


def test_double_confirm_executes_once(tmp_gate):
    """Verify execute-once semantics: second confirmation returns already_executed."""
    args = {"level": "warning", "message": "High temp test", "asset_id": "FAN-T17"}
    pending = tmp_gate.create_pending_action("send_alert", args)
    act_id = pending["action_id"]

    # 1. First confirmation
    res1 = tmp_gate.confirm_action(act_id)
    assert res1["status"] == "executed"
    assert res1["result"]["asset_id"] == "FAN-T17"

    # Verify side-effect: alert appended to results/alerts.log with ISO timestamp
    log_file = Path("results/alerts.log")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "[WARNING]" in content
    assert "asset=FAN-T17" in content
    assert "High temp test" in content
    assert "[202" in content  # ISO timestamp bracket

    # 2. Second confirmation
    res2 = tmp_gate.confirm_action(act_id)
    assert res2["status"] == "error"
    assert res2["code"] == "already_executed"


def test_expired_action_rejected(tmp_gate):
    """Verify that an expired pending action returns expired and does not execute."""
    args = {"level": "info", "message": "Test expire", "asset_id": "FAN-T17"}
    pending = tmp_gate.create_pending_action("send_alert", args)
    act_id = pending["action_id"]

    # Manually expire action in store
    past_iso = "2020-01-01T00:00:00Z"
    with tmp_gate.store._get_connection() as conn:
        conn.execute("UPDATE pending_actions SET expires_at = ? WHERE id = ?", (past_iso, act_id))
        conn.commit()

    confirm_res = tmp_gate.confirm_action(act_id)
    assert confirm_res["status"] == "error"
    assert confirm_res["code"] == "expired"


def test_user_rejects_action(tmp_gate):
    """Verify that rejecting a pending action marks it rejected without executing."""
    args = {"level": "info", "message": "Test reject", "asset_id": "FAN-T17"}
    pending = tmp_gate.create_pending_action("send_alert", args)
    act_id = pending["action_id"]

    rej_res = tmp_gate.reject_action(act_id)
    assert rej_res["status"] == "rejected"

    action_db = tmp_gate.store.get_pending_action(act_id)
    assert action_db["state"] == "rejected"

    # Trying to confirm rejected action fails
    confirm_res = tmp_gate.confirm_action(act_id)
    assert confirm_res["status"] == "error"
    assert confirm_res["code"] == "already_rejected"


def test_args_hash_mismatch(tmp_gate):
    """Verify that confirmation with mismatched args hash is rejected."""
    args = {"schedule_id": "sched-active-001"}
    pending = tmp_gate.create_pending_action("create_work_orders", args)
    act_id = pending["action_id"]

    fake_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    res = tmp_gate.confirm_action(act_id, args_hash=fake_hash)
    assert res["status"] == "error"
    assert res["code"] == "args_mismatch"


def test_field_report_clear_triggers_what_if_draft(tmp_gate):
    """Verify clear field report updates task and triggers what-if maintenance draft post-hook."""
    args = {
        "task_id": "T05-bearing-001",
        "status": "needs_part",
        "defer_days": 5,
        "new_finding": "Rulment uzat, piesa vine in 5 zile",
        "confidence": 0.95,
        "raw_text": "Am verificat T05, rulmentul e uzat, piesa vine in 5 zile",
    }
    pending = tmp_gate.create_pending_action("update_from_field_report", args)
    assert pending["state"] == "pending"

    confirm_res = tmp_gate.confirm_action(pending["action_id"])
    assert confirm_res["status"] == "executed"
    assert "what_if_draft" in confirm_res["post_hooks"]
    draft_info = confirm_res["post_hooks"]["what_if_draft"]
    assert draft_info["schedule_id"] == "sched-draft-t05-defer"
    assert draft_info["delta_mwh"] == 2.5


def test_field_report_needs_input_and_patch(tmp_gate):
    """Verify ambiguous report enters needs_input, user patches fields, then confirms."""
    # Unknown task_id and missing raw_text
    args = {
        "task_id": "UNKNOWN-TASK",
        "status": "deferred",
        "defer_days": 3,
        "confidence": 0.5,
    }
    pending = tmp_gate.create_pending_action("update_from_field_report", args)
    act_id = pending["action_id"]
    assert pending["state"] == "needs_input"

    # Confirmation blocked while in needs_input
    blocked = tmp_gate.confirm_action(act_id)
    assert blocked["status"] == "error"
    assert blocked["code"] == "needs_input"

    # User patches valid task and confidence
    patch_res = tmp_gate.patch_action(act_id, {
        "task_id": "T05-bearing-001",
        "confidence": 0.9,
        "raw_text": "piesa vine in 3 zile",
    })
    assert patch_res["status"] == "patched"
    assert patch_res["state"] == "pending"

    # Confirmation now succeeds
    success = tmp_gate.confirm_action(act_id)
    assert success["status"] == "executed"


def test_household_outcome_post_hook(tmp_gate):
    """Verify confirmed inspection report writes label to labels table in store."""
    args = {
        "task_id": "T01-rev-001",
        "status": "done",
        "confidence": 1.0,
        "raw_text": "Am verificat gospodăria HH-FAULTY-01: contor defect, nu furt",
    }
    pending = tmp_gate.create_pending_action("update_from_field_report", args)
    confirm_res = tmp_gate.confirm_action(pending["action_id"])
    assert confirm_res["status"] == "executed"

    labels = tmp_gate.store.get_labels("HH-FAULTY-01")
    assert len(labels) >= 1
    assert labels[0]["label"] == "faulty_meter"
    assert labels[0]["household_id"] == "HH-FAULTY-01"

    # Also test generic "gospodăria X: contor defect, nu furt" defaults to HH-FAULTY-01
    args_x = {
        "task_id": "T01-rev-001",
        "status": "done",
        "confidence": 1.0,
        "raw_text": "Am verificat gospodăria X: contor defect, nu furt",
    }
    pending_x = tmp_gate.create_pending_action("update_from_field_report", args_x)
    confirm_x = tmp_gate.confirm_action(pending_x["action_id"])
    assert confirm_x["status"] == "executed"
    assert confirm_x["post_hooks"]["household_label"]["household_id"] == "HH-FAULTY-01"
    assert confirm_x["post_hooks"]["household_label"]["label"] == "faulty_meter"


def test_schedule_lifecycle_draft_does_not_alter_active(tmp_store):
    """Verify what-if maintenance draft does not alter active schedule and promote is atomic with cost delta."""
    # 1. Seed active schedule
    tmp_store.store_schedule(
        schedule_id="sched-active-001",
        status="active",
        payload={"baseline": True},
        total_mwh=16.1,
        total_ron=7245.0,
    )
    active_before = tmp_store.get_active_schedule()
    assert active_before["id"] == "sched-active-001"

    # 2. Store draft schedule
    tmp_store.store_schedule(
        schedule_id="sched-draft-thursday",
        status="draft",
        payload={"variant": "crew_2_off_thursday"},
        total_mwh=17.3,
        total_ron=7785.0,
        parent_id="sched-active-001",
    )

    # Active schedule remains unchanged
    active_still = tmp_store.get_active_schedule()
    assert active_still["id"] == "sched-active-001"
    assert active_still["total_mwh"] == 16.1

    # 3. Promote draft to active in single atomic transaction with delta calculation
    promoted = tmp_store.promote_schedule_atomic("sched-draft-thursday", action_id="act-prom-01")
    assert promoted["id"] == "sched-draft-thursday"
    assert promoted["status"] == "active"
    assert promoted["previous_active_id"] == "sched-active-001"
    assert promoted["delta_mwh"] == 1.2
    assert promoted["delta_ron"] == 540.0

    # Verify active is now the promoted schedule
    active_after = tmp_store.get_active_schedule()
    assert active_after["id"] == "sched-draft-thursday"
    assert active_after["total_mwh"] == 17.3

    # Verify previous active was demoted to history
    old_sched = tmp_store.get_schedule("sched-active-001")
    assert old_sched["status"] == "history"


def test_secret_redaction():
    """Verify secrets and API keys are redacted from logs, errors, and stored records."""
    raw_error = "Failed with headers {'Authorization': 'Bearer secret_token_xyz123', 'x-api-key': 'AIzaSyFakeKey12345678901234567890123'}"
    redacted = redact_secrets(raw_error)
    assert "secret_token_xyz123" not in redacted
    assert "AIzaSyFakeKey" not in redacted
    assert "[REDACTED]" in redacted

    dict_with_secret = {
        "api_key": "secret_abc_123",
        "model": "gemini-2.5-flash",
        "nested": {"client_secret": "top_secret_999"},
    }
    redacted_dict = redact_secrets(dict_with_secret)
    assert redacted_dict["api_key"] == "[REDACTED]"
    assert redacted_dict["nested"]["client_secret"] == "[REDACTED]"
    assert redacted_dict["model"] == "gemini-2.5-flash"


def test_audit_store_persistence_across_reopen(tmp_path):
    """Verify that sessions, messages, tool calls, and results persist across database reopen."""
    db_file = tmp_path / "persist_agent.db"

    # Open 1: Write records
    store1 = AuditStore(db_path=db_file)
    sess_id = store1.create_session(mode="llm", as_of="2026-11-08")
    msg_id = store1.add_message(sess_id, "user", "Ce pierderi avem?")
    call_id = store1.log_tool_call(
        session_id=sess_id,
        tool="loss_overview",
        effect="read",
        args={"period": "week"},
        result_id="res-001",
    )
    store1.store_tool_result("res-001", {"total_mwh": 39.9}, {"total_mwh": 39.9}, {"backend": "fixture"})

    # Open 2: New store instance on same file
    store2 = AuditStore(db_path=db_file)
    sess = store2.get_session(sess_id)
    assert sess is not None
    assert sess["mode"] == "llm"

    msgs = store2.get_messages(sess_id)
    assert len(msgs) == 1
    assert msgs[0]["content"] == "Ce pierderi avem?"

    res = store2.get_tool_result("res-001")
    assert res is not None
    assert res["payload"]["total_mwh"] == 39.9


def test_field_report_updates_task_status_and_reflection(tmp_gate):
    """Verify that after confirming a field report, subsequent list_tasks calls reflect updated task."""
    args = {
        "task_id": "T05-bearing-001",
        "status": "needs_part",
        "defer_days": 5,
        "new_finding": "Rulment uzat, comanda piesa",
        "confidence": 0.95,
        "raw_text": "Am verificat T05, rulmentul e uzat, piesa vine in 5 zile",
    }
    pending = tmp_gate.create_pending_action("update_from_field_report", args)
    confirm_res = tmp_gate.confirm_action(pending["action_id"])
    assert confirm_res["status"] == "executed"

    # Query list_tasks and verify T05-bearing-001 has status 'needs_part'
    tasks_res = registry.execute("list_tasks")
    assert not tasks_res.is_error
    t05 = next(t for t in tasks_res.full_result.tasks if t.task_id == "T05-bearing-001")
    assert t05.status == "needs_part"


def test_action_confirmation_and_rejection_logs_system_message(tmp_gate):
    """Verify confirmed and rejected actions append Romanian system messages to the session audit trail."""
    sess_id = tmp_gate.store.create_session(mode="deterministic", as_of="2026-11-08")

    # 1. Confirmation note
    args1 = {"schedule_id": "sched-active-001"}
    act1 = tmp_gate.create_pending_action("create_work_orders", args1, session_id=sess_id)
    confirm_res = tmp_gate.confirm_action(act1["action_id"])
    assert confirm_res["status"] == "executed"

    # 2. Rejection note
    args2 = {"level": "info", "message": "Test reject", "asset_id": "FAN-T17"}
    act2 = tmp_gate.create_pending_action("send_alert", args2, session_id=sess_id)
    rej_res = tmp_gate.reject_action(act2["action_id"])
    assert rej_res["status"] == "rejected"

    msgs = tmp_gate.store.get_messages(sess_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "confirmată și executată" in msgs[0]["content"]
    assert msgs[1]["role"] == "system"
    assert "respinsă de utilizator" in msgs[1]["content"]


def test_tool_execution_logs_duration_and_errors_to_audit_store(tmp_store):
    """Verify registry execution records execution duration and structured errors in SQLite audit store."""
    sess_id = tmp_store.create_session(mode="deterministic", as_of="2026-11-08")

    # Successful call
    res_ok = registry.execute("loss_overview", {"period": "week"}, session_id=sess_id, store=tmp_store)
    assert not res_ok.is_error

    # Erroneous call (unknown tool)
    res_err = registry.execute("non_existent_tool", {}, session_id=sess_id, store=tmp_store)
    assert res_err.is_error

    # Erroneous call (unknown asset)
    res_asset_err = registry.execute("detect_underperformance", {"site_id": "fantanele", "asset_id": "UNKNOWN-99"}, session_id=sess_id, store=tmp_store)
    assert res_asset_err.is_error

    with tmp_store._get_connection() as conn:
        rows = conn.execute("SELECT * FROM tool_calls WHERE session_id = ? ORDER BY created_at ASC", (sess_id,)).fetchall()
        assert len(rows) == 3

        # Row 1: loss_overview succeeded with measured duration
        assert rows[0]["tool"] == "loss_overview"
        assert rows[0]["duration_ms"] >= 0.0
        assert rows[0]["error_code"] is None

        # Row 2: non_existent_tool logged error_code
        assert rows[1]["tool"] == "non_existent_tool"
        assert rows[1]["error_code"] == "unknown_tool"

        # Row 3: detect_underperformance logged error_code unknown_asset
        assert rows[2]["tool"] == "detect_underperformance"
        assert rows[2]["error_code"] == "unknown_asset"


def test_promote_schedule_tool_execution(tmp_gate):
    """Verify promote_schedule tool updates active schedule and returns cost delta."""
    # Seed active schedule in store
    tmp_gate.store.store_schedule(
        schedule_id="sched-active-001",
        status="active",
        payload={"name": "baseline"},
        total_mwh=16.1,
        total_ron=7245.0,
    )
    # Seed draft schedule
    tmp_gate.store.store_schedule(
        schedule_id="sched-draft-thursday",
        status="draft",
        payload={"name": "thursday_variant"},
        total_mwh=17.3,
        total_ron=7785.0,
        parent_id="sched-active-001",
    )

    pending = tmp_gate.create_pending_action("promote_schedule", {"draft_id": "sched-draft-thursday"})
    confirm_res = tmp_gate.confirm_action(pending["action_id"])
    assert confirm_res["status"] == "executed"
    post = confirm_res["post_hooks"]["promoted_schedule"]
    assert post["schedule_id"] == "sched-draft-thursday"
    assert post["status"] == "active"
    assert post["previous_active_id"] == "sched-active-001"
    assert post["delta_mwh"] == 1.2
    assert post["delta_ron"] == 540.0


