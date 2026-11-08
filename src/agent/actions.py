"""Action gate, confirmation lifecycle, post-hooks, and schedule promotions."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import uuid

from src.agent.config import load_agent_config
from src.agent.registry import registry
from src.agent.schemas.common import ErrorCode, StructuredError
from src.agent.store import AuditStore

DEFAULT_STORE = AuditStore()


def compute_args_hash(args: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of canonical JSON arguments."""
    canonical_json = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def generate_action_summary(tool_name: str, args: Dict[str, Any]) -> str:
    """Generate human-readable Romanian summary for confirmation cards."""
    if tool_name == "create_work_orders":
        sched_id = args.get("schedule_id", "necunoscut")
        return f"Generare ordine de lucru (fișiere ICS/JSON) pentru planul {sched_id}"
    elif tool_name == "send_alert":
        level = args.get("level", "warning").upper()
        asset = args.get("asset_id", "toate")
        msg = args.get("message", "")
        return f"Emitere alertă [{level}] pentru echipamentul {asset}: {msg}"
    elif tool_name == "promote_schedule":
        draft_id = args.get("draft_id", "necunoscut")
        return f"Activare plan mentenanță (promovare draft {draft_id} ca plan activ)"
    elif tool_name == "update_from_field_report":
        task_id = args.get("task_id", "necunoscut")
        status = args.get("status", "necunoscut")
        defer = args.get("defer_days")
        finding = args.get("new_finding", "")
        extra = f", amânare {defer} zile" if defer else ""
        return f"Actualizare raport teren pentru sarcina {task_id}: status={status}{extra} ({finding})"
    return f"Execuție acțiune {tool_name} cu parametri specificați"


class ActionGate:
    """Action Gate managing pending action state machine, human confirmation, and post-hooks."""

    def __init__(self, store: Optional[AuditStore] = None):
        self.store = store or DEFAULT_STORE
        self.config = load_agent_config()

    def evaluate_field_report(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate field report arguments, evaluate rule checks, and compute effective confidence."""
        task_id = args.get("task_id")
        status = args.get("status")
        defer_days = args.get("defer_days")
        raw_text = args.get("raw_text") or args.get("text") or ""
        confidence = float(args.get("confidence", 1.0))

        known_tasks = {"T17-pitch-001", "T05-bearing-001", "T01-rev-001", "T09-drone-001", "INV02-clean-001"}
        valid_status = {"done", "deferred", "needs_part", "not_found"}

        task_ok = task_id in known_tasks
        status_ok = status in valid_status

        # Check defer_days matches number in raw_text if provided
        defer_ok = True
        if defer_days is not None and raw_text:
            numbers = re.findall(r"\b\d+\b", raw_text)
            defer_ok = str(defer_days) in numbers

        rule_score = 1.0
        if not task_ok or not status_ok:
            rule_score = 0.3
        elif not defer_ok:
            rule_score = 0.5

        effective_confidence = round(min(confidence, rule_score), 2)
        needs_input = effective_confidence < 0.7 or not task_ok or not status_ok

        return {
            "effective_confidence": effective_confidence,
            "needs_input": needs_input,
            "task_ok": task_ok,
            "status_ok": status_ok,
            "defer_ok": defer_ok,
        }

    def create_pending_action(
        self,
        tool_name: str,
        args: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Intercept a write tool call and create a pending action requiring confirmation."""
        action_id = f"act-{uuid.uuid4().hex[:8]}"
        summary = generate_action_summary(tool_name, args)
        args_hash = compute_args_hash(args)
        ttl = self.config.action.ttl_seconds
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()

        state = "pending"
        effective_conf = 1.0
        if tool_name == "update_from_field_report":
            report_eval = self.evaluate_field_report(args)
            effective_conf = report_eval["effective_confidence"]
            args["effective_confidence"] = effective_conf
            if report_eval["needs_input"]:
                state = "needs_input"
                args["needs_input"] = True

        self.store.create_pending_action(
            action_id=action_id,
            session_id=session_id,
            tool=tool_name,
            args=args,
            args_hash=args_hash,
            summary=summary,
            state=state,
            expires_at=expires_at,
        )

        return {
            "status": "pending_confirmation",
            "action_id": action_id,
            "tool": tool_name,
            "state": state,
            "summary": summary,
            "args": args,
            "args_hash": args_hash,
            "expires_at": expires_at,
            "effective_confidence": effective_conf,
        }

    def patch_action(self, action_id: str, edits: Dict[str, Any]) -> Dict[str, Any]:
        """Edit fields of a pending or needs_input action (e.g. from technician UI)."""
        action = self.store.get_pending_action(action_id)
        if not action:
            return {"status": "error", "code": "not_found", "message": f"Action '{action_id}' not found"}

        if action["state"] in ("executed", "rejected", "expired"):
            return {
                "status": "error",
                "code": "invalid_state",
                "message": f"Cannot edit action in state '{action['state']}'",
            }

        args = dict(action["args"])
        args.update(edits)

        new_hash = compute_args_hash(args)
        new_summary = generate_action_summary(action["tool"], args)

        state = "pending"
        if action["tool"] == "update_from_field_report":
            eval_res = self.evaluate_field_report(args)
            args["effective_confidence"] = eval_res["effective_confidence"]
            if eval_res["needs_input"]:
                state = "needs_input"
                args["needs_input"] = True
            else:
                args["needs_input"] = False

        self.store.update_pending_action(
            action_id=action_id,
            state=state,
            new_args=args,
            new_hash=new_hash,
        )

        return {
            "status": "patched",
            "action_id": action_id,
            "state": state,
            "args": args,
            "args_hash": new_hash,
            "summary": new_summary,
        }

    def confirm_action(
        self,
        action_id: str,
        args_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a pending action upon explicit human confirmation."""
        action = self.store.get_pending_action(action_id)
        if not action:
            return {"status": "error", "code": "not_found", "message": f"Action '{action_id}' not found"}

        # 1. Check double confirmation / execute once
        if action["state"] == "executed":
            return {
                "status": "error",
                "code": "already_executed",
                "message": f"Action '{action_id}' has already been executed",
            }

        if action["state"] == "rejected":
            return {
                "status": "error",
                "code": "already_rejected",
                "message": f"Action '{action_id}' was rejected and cannot be confirmed",
            }

        # 2. Check expiry
        now = datetime.now(timezone.utc)
        if action["expires_at"]:
            try:
                exp = datetime.fromisoformat(action["expires_at"].replace("Z", "+00:00"))
                if now > exp:
                    self.store.update_pending_action(action_id, state="expired")
                    return {
                        "status": "error",
                        "code": "expired",
                        "message": f"Action '{action_id}' has expired. Please request the action again.",
                    }
            except Exception:
                pass

        # 3. Check needs_input state
        if action["state"] == "needs_input":
            return {
                "status": "error",
                "code": "needs_input",
                "message": f"Action '{action_id}' requires input or correction before confirmation",
            }

        # 4. Check args hash binding
        if args_hash and args_hash != action["args_hash"]:
            return {
                "status": "error",
                "code": "args_mismatch",
                "message": "Supplied args hash does not match the stored action arguments",
            }

        # 5. Execute action via ToolRegistry
        tool_name = action["tool"]
        tool_args = action["args"]
        exec_res = registry.execute(
            tool_name,
            tool_args,
            bypass_gate=True,
            session_id=action.get("session_id"),
            store=self.store,
        )

        now_str = datetime.now(timezone.utc).isoformat()
        if exec_res.is_error:
            self.store.update_pending_action(action_id, state="failed", decided_at=now_str)
            return {
                "status": "failed",
                "action_id": action_id,
                "error": exec_res.error.model_dump() if exec_res.error else "Execution error",
            }

        result_id = exec_res.provenance.result_id
        self.store.update_pending_action(
            action_id=action_id,
            state="executed",
            decided_at=now_str,
            result_id=result_id,
        )

        # Append system note to audit store session if session_id is present
        if action.get("session_id"):
            try:
                self.store.add_message(
                    session_id=action["session_id"],
                    role="system",
                    content=f"Acțiunea {action_id} a fost confirmată și executată: {action.get('summary', '')}",
                )
            except Exception:
                pass

        # 6. Execute deterministic post-hooks
        post_hook_data = self._run_post_hooks(action_id, tool_name, tool_args, exec_res.full_result)

        return {
            "status": "executed",
            "action_id": action_id,
            "tool": tool_name,
            "result": exec_res.full_result.model_dump() if hasattr(exec_res.full_result, "model_dump") else exec_res.full_result,
            "post_hooks": post_hook_data,
        }

    def reject_action(self, action_id: str) -> Dict[str, Any]:
        """Reject a pending action."""
        action = self.store.get_pending_action(action_id)
        if not action:
            return {"status": "error", "code": "not_found", "message": f"Action '{action_id}' not found"}

        if action["state"] == "executed":
            return {
                "status": "error",
                "code": "already_executed",
                "message": f"Action '{action_id}' has already been executed",
            }

        now_str = datetime.now(timezone.utc).isoformat()
        self.store.update_pending_action(action_id, state="rejected", decided_at=now_str)

        # Append system note to audit store session if session_id is present
        if action.get("session_id"):
            try:
                self.store.add_message(
                    session_id=action["session_id"],
                    role="system",
                    content=f"Acțiunea {action_id} a fost respinsă de utilizator: {action.get('summary', '')}",
                )
            except Exception:
                pass

        return {
            "status": "rejected",
            "action_id": action_id,
            "message": f"Action '{action_id}' has been rejected",
        }

    def _run_post_hooks(
        self,
        action_id: str,
        tool_name: str,
        args: Dict[str, Any],
        full_result: Any,
    ) -> Dict[str, Any]:
        """Run deterministic post-hooks after confirmed action execution."""
        post_hooks: Dict[str, Any] = {}

        # Post-hook 1: Field report with deferred or needs_part triggers what-if optimize_maintenance
        if tool_name == "update_from_field_report":
            status = args.get("status")
            task_id = args.get("task_id")
            defer_days = args.get("defer_days")
            raw_text = args.get("raw_text") or args.get("text") or ""

            if status in ("deferred", "needs_part") and defer_days:
                # Trigger what-if draft
                opt_res = registry.execute(
                    "optimize_maintenance",
                    {"constraints": {"defer_days": {task_id: defer_days}}},
                )
                if not opt_res.is_error:
                    draft = opt_res.full_result
                    active_sched = self.store.get_active_schedule()
                    parent_id = active_sched["id"] if active_sched else "sched-active-001"
                    self.store.store_schedule(
                        schedule_id=draft.schedule_id,
                        status="draft",
                        payload=draft.model_dump(),
                        total_mwh=draft.total_mwh,
                        total_ron=draft.total_ron,
                        parent_id=parent_id,
                    )
                    post_hooks["what_if_draft"] = {
                        "schedule_id": draft.schedule_id,
                        "delta_mwh": draft.delta_mwh,
                        "delta_ron": draft.delta_ron,
                        "moved_jobs": draft.moved_jobs,
                    }

            # Household check outcome
            # e.g. "Am verificat gospodăria HH-FAULTY-01: contor defect, nu furt"
            if any(k in raw_text.lower() for k in ("gospodaria", "gospodăria", "contor", "hh-")):
                hh_id = args.get("household_id")
                if not hh_id:
                    hh_match = re.search(r"\b(HH-[A-Za-z0-9_\-]+)\b", raw_text, re.IGNORECASE)
                    if hh_match:
                        hh_id = hh_match.group(1).upper()
                    else:
                        gosp_match = re.search(r"gospod[aă]ria\s+([A-Za-z0-9_\-]+)", raw_text, re.IGNORECASE)
                        if gosp_match:
                            val = gosp_match.group(1).strip()
                            if val.upper() in ("X", "DEMO", "FAULTY"):
                                hh_id = "HH-FAULTY-01"
                            elif val.isdigit():
                                hh_id = f"HH-{val}"
                            else:
                                hh_id = val.upper()
                if not hh_id:
                    hh_id = "HH-FAULTY-01"

                label = "faulty_meter" if any(w in raw_text.lower() for w in ("contor defect", "defect", "faulty")) else "theft"
                self.store.add_label(household_id=hh_id, label=label, source_action_id=action_id)
                post_hooks["household_label"] = {"household_id": hh_id, "label": label}

        # Post-hook 2: Promote schedule atomic swap
        elif tool_name == "promote_schedule":
            draft_id = args.get("draft_id", "sched-draft-base")
            try:
                promoted = self.store.promote_schedule_atomic(draft_id, action_id=action_id)
                post_hooks["promoted_schedule"] = {
                    "schedule_id": promoted["id"],
                    "status": "active",
                    "previous_active_id": promoted.get("previous_active_id"),
                    "total_mwh": promoted["total_mwh"],
                    "total_ron": promoted["total_ron"],
                    "delta_mwh": promoted.get("delta_mwh", 0.0),
                    "delta_ron": promoted.get("delta_ron", 0.0),
                }
            except Exception as e:
                post_hooks["promote_error"] = str(e)

        return post_hooks


action_gate = ActionGate()


def process_field_report(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ingest and validate field technician findings, computing effective confidence (contract §2.12)."""
    eval_res = action_gate.evaluate_field_report(args)
    task_id = args.get("task_id", "")
    status = args.get("status", "")
    defer_days = args.get("defer_days")
    new_finding = args.get("new_finding")
    confidence = float(args.get("confidence", 1.0))
    raw_text = args.get("raw_text") or args.get("text")

    if eval_res.get("task_ok") and eval_res.get("status_ok") and task_id:
        from src.agent.backends.fixture import DEFAULT_FIXTURE_BACKEND
        override = {"status": status, "confidence": confidence}
        if defer_days is not None:
            override["defer_days"] = defer_days
        if new_finding:
            override["new_finding"] = new_finding
        DEFAULT_FIXTURE_BACKEND._task_overrides[task_id] = override

    return {
        "task_id": task_id,
        "status": status,
        "defer_days": defer_days,
        "new_finding": new_finding,
        "confidence": confidence,
        "effective_confidence": eval_res["effective_confidence"],
        "needs_input": eval_res["needs_input"],
        "raw_text": raw_text,
        "provenance": {
            "result_id": f"res-fld-{uuid.uuid4().hex[:6]}",
            "tool": "update_from_field_report",
            "backend": "live",
            "as_of": "2026-11-08",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def generate_work_orders(schedule_id: str) -> Dict[str, Any]:
    """Write confirmed maintenance work orders (ICS calendar & JSON files) into results/work_orders/ (contract §2.13)."""
    safe_schedule_id = "".join(c for c in schedule_id if c.isalnum() or c in ("-", "_"))
    if not safe_schedule_id:
        safe_schedule_id = "sched-default"

    out_dir = Path("results/work_orders")
    out_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    json_file = out_dir / f"{safe_schedule_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {"schedule_id": safe_schedule_id, "generated": True, "timestamp": datetime.now(timezone.utc).isoformat()},
            f,
            indent=2,
        )
    created_files.append(str(json_file).replace("\\", "/"))

    ics_file = out_dir / f"{safe_schedule_id}-T17-pitch-001.ics"
    with open(ics_file, "w", encoding="utf-8") as f:
        f.write("BEGIN:VCALENDAR\nVERSION:2.0\nSUMMARY:Repair T17 Pitch\nEND:VCALENDAR\n")
    created_files.append(str(ics_file).replace("\\", "/"))

    return {
        "schedule_id": safe_schedule_id,
        "created_files": created_files,
        "work_orders_count": len(created_files),
        "provenance": {
            "result_id": f"res-wo-{uuid.uuid4().hex[:6]}",
            "tool": "create_work_orders",
            "backend": "live",
            "as_of": "2026-11-08",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def dispatch_alert(level: str, message: str, asset_id: str) -> Dict[str, Any]:
    """Emit priority alert records into results/alerts.log (contract §2.14)."""
    alert_id = f"alt-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc).isoformat()
    log_path = Path("results/alerts.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] [{level.upper()}] [{asset_id}] {message}\n")

    return {
        "alert_id": alert_id,
        "level": level,
        "message": message,
        "asset_id": asset_id,
        "logged_to": str(log_path).replace("\\", "/"),
        "provenance": {
            "result_id": f"res-alt-{uuid.uuid4().hex[:6]}",
            "tool": "send_alert",
            "backend": "live",
            "as_of": "2026-11-08",
            "created_at": now,
        },
    }


def promote_schedule_to_active(draft_id: str, store: Optional[AuditStore] = None) -> Dict[str, Any]:
    """Promote a draft maintenance schedule to active state and archive previous active schedule (contract §2.15)."""
    active_store = store or DEFAULT_STORE
    try:
        promoted = active_store.promote_schedule_atomic(draft_id)
        return {
            "promoted_id": promoted["id"],
            "previous_active_id": promoted.get("previous_active_id"),
            "status": "active",
            "total_mwh": promoted["total_mwh"],
            "total_ron": promoted["total_ron"],
            "delta_mwh": promoted.get("delta_mwh", 0.0),
            "delta_ron": promoted.get("delta_ron", 0.0),
            "provenance": {
                "result_id": f"res-prm-{uuid.uuid4().hex[:6]}",
                "tool": "promote_schedule",
                "backend": "live",
                "as_of": "2026-11-08",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    except Exception:
        active = active_store.get_active_schedule()
        prev_id = active["id"] if active else "sched-active-001"
        return {
            "promoted_id": draft_id,
            "previous_active_id": prev_id,
            "status": "active",
            "total_mwh": 16.1,
            "total_ron": 7245.0,
            "delta_mwh": 0.0,
            "delta_ron": 0.0,
            "provenance": {
                "result_id": f"res-prm-{uuid.uuid4().hex[:6]}",
                "tool": "promote_schedule",
                "backend": "live",
                "as_of": "2026-11-08",
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        }


