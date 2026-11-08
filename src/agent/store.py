"""SQLite audit store with WAL mode, parameterized queries, and secret redaction."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, List, Optional
import uuid

DEFAULT_DB_PATH = Path("results/agent.db")
DEFAULT_STORE: Optional["AuditStore"] = None


def get_default_store() -> "AuditStore":
    global DEFAULT_STORE
    if DEFAULT_STORE is None:
        DEFAULT_STORE = AuditStore()
    return DEFAULT_STORE

# Regex patterns for sensitive secrets to redact
SECRET_PATTERNS = [
    re.compile(r'(?i)((?:[a-zA-Z0-9_\-]*api[_-]?key|token|secret|password|authorization)\s*[:=]\s*["\']?)([^"\'\s,;}]+)(["\']?)'),
    re.compile(r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{10,})'),
    re.compile(r'(AIzaSy[a-zA-Z0-9_\-]{20,})'),  # Google API key pattern
    re.compile(r'(sk-[a-zA-Z0-9_\-]{20,})'),  # OpenAI key pattern
]


def redact_secrets(val: Any) -> Any:
    """Recursively redact secrets from strings, dictionaries, and lists."""
    if isinstance(val, str):
        redacted = val
        # 1. Key-value pairs: e.g. api_key: "secret" or Bearer token
        redacted = re.sub(
            r'(?i)((?:[a-zA-Z0-9_\-]*api[_-]?key|token|secret|password|authorization)\s*[:=]\s*["\']?)([^"\'\s,;}]+)(["\']?)',
            r"\1[REDACTED]\3",
            redacted,
        )
        redacted = re.sub(r'(?i)(bearer\s+)([a-zA-Z0-9_\-\.]{10,})', r"\1[REDACTED]", redacted)
        # 2. Standalone key patterns
        redacted = re.sub(r'AIzaSy[a-zA-Z0-9_\-]{20,}', '[REDACTED]', redacted)
        redacted = re.sub(r'sk-[a-zA-Z0-9_\-]{20,}', '[REDACTED]', redacted)
        return redacted
    elif isinstance(val, dict):
        new_dict = {}
        for k, v in val.items():
            if any(s in k.lower() for s in ["api_key", "secret", "password", "token", "authorization"]):
                new_dict[k] = "[REDACTED]"
            else:
                new_dict[k] = redact_secrets(v)
        return new_dict
    elif isinstance(val, list):
        return [redact_secrets(item) for item in val]
    return val


class AuditStore:
    """Audit store backed by SQLite with WAL mode, parameterized queries, and secret redaction."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self) -> None:
        """Create tables from design D14 if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                as_of TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                message_id TEXT,
                tool TEXT NOT NULL,
                effect TEXT NOT NULL,
                args_json TEXT NOT NULL,
                result_id TEXT,
                backend TEXT NOT NULL,
                duration_ms REAL DEFAULT 0.0,
                error_code TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tool_results (
                result_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                llm_view_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_actions (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                tool TEXT NOT NULL,
                args_json TEXT NOT NULL,
                args_hash TEXT NOT NULL,
                summary TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                decided_at TEXT,
                expires_at TEXT,
                result_id TEXT
            );

            CREATE TABLE IF NOT EXISTS schedules (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                parent_id TEXT,
                payload_json TEXT NOT NULL,
                total_mwh REAL NOT NULL,
                total_ron REAL NOT NULL,
                created_at TEXT NOT NULL,
                promoted_by_action TEXT
            );

            CREATE TABLE IF NOT EXISTS labels (
                household_id TEXT NOT NULL,
                label TEXT NOT NULL,
                source_action_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS grounding_events (
                message_id TEXT PRIMARY KEY,
                passed INTEGER NOT NULL,
                retried INTEGER NOT NULL,
                unmatched_json TEXT NOT NULL
            );
            """)
            conn.commit()

    # --- Sessions & Messages ---

    def create_session(
        self,
        session_id: Optional[str] = None,
        mode: str = "llm",
        as_of: str = "2026-11-08",
    ) -> str:
        s_id = session_id or f"sess-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO sessions (id, created_at, mode, as_of) VALUES (?, ?, ?, ?)",
                (s_id, now, mode, as_of),
            )
            conn.commit()
        return s_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        message_id: Optional[str] = None,
    ) -> str:
        m_id = message_id or f"msg-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        safe_content = redact_secrets(content)
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (m_id, session_id, role, safe_content, now),
            )
            conn.commit()
        return m_id

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # --- Tool Calls & Results ---

    def log_tool_call(
        self,
        session_id: Optional[str],
        tool: str,
        effect: str,
        args: Dict[str, Any],
        result_id: Optional[str] = None,
        backend: str = "fixture",
        duration_ms: float = 0.0,
        error_code: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> str:
        call_id = f"tc-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        safe_args = redact_secrets(args)
        args_json = json.dumps(safe_args, sort_keys=True)
        safe_error = redact_secrets(error_code) if error_code else None

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO tool_calls 
                (id, session_id, message_id, tool, effect, args_json, result_id, backend, duration_ms, error_code, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (call_id, session_id, message_id, tool, effect, args_json, result_id, backend, duration_ms, safe_error, now),
            )
            conn.commit()
        return call_id

    def store_tool_result(
        self,
        result_id: str,
        payload: Dict[str, Any],
        llm_view: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> None:
        safe_payload = redact_secrets(payload)
        safe_llm_view = redact_secrets(llm_view)
        safe_prov = redact_secrets(provenance)

        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO tool_results (result_id, payload_json, llm_view_json, provenance_json)
                VALUES (?, ?, ?, ?)""",
                (result_id, json.dumps(safe_payload), json.dumps(safe_llm_view), json.dumps(safe_prov)),
            )
            conn.commit()

    def get_tool_result(self, result_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM tool_results WHERE result_id = ?", (result_id,)).fetchone()
            if not row:
                return None
            return {
                "result_id": row["result_id"],
                "payload": json.loads(row["payload_json"]),
                "llm_view": json.loads(row["llm_view_json"]),
                "provenance": json.loads(row["provenance_json"]),
            }

    def get_tool_calls(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve tool calls from the audit store, optionally filtered by session_id."""
        query = "SELECT * FROM tool_calls"
        params: List[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY created_at ASC"

        with self._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            calls = []
            for r in rows:
                item = dict(r)
                item["args"] = json.loads(item["args_json"])
                calls.append(item)
            return calls

    # --- Pending Actions ---

    def create_pending_action(
        self,
        action_id: str,
        session_id: Optional[str],
        tool: str,
        args: Dict[str, Any],
        args_hash: str,
        summary: str,
        state: str = "pending",
        expires_at: Optional[str] = None,
    ) -> str:
        now = datetime.now(timezone.utc).isoformat()
        safe_args = redact_secrets(args)
        args_json = json.dumps(safe_args, sort_keys=True)
        safe_summary = redact_secrets(summary)

        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO pending_actions 
                (id, session_id, tool, args_json, args_hash, summary, state, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (action_id, session_id, tool, args_json, args_hash, safe_summary, state, now, expires_at),
            )
            conn.commit()
        return action_id

    def update_pending_action(
        self,
        action_id: str,
        state: str,
        decided_at: Optional[str] = None,
        result_id: Optional[str] = None,
        new_args: Optional[Dict[str, Any]] = None,
        new_hash: Optional[str] = None,
    ) -> None:
        updates = ["state = ?"]
        params: List[Any] = [state]

        if decided_at is not None:
            updates.append("decided_at = ?")
            params.append(decided_at)
        if result_id is not None:
            updates.append("result_id = ?")
            params.append(result_id)
        if new_args is not None:
            updates.append("args_json = ?")
            params.append(json.dumps(redact_secrets(new_args), sort_keys=True))
        if new_hash is not None:
            updates.append("args_hash = ?")
            params.append(new_hash)

        params.append(action_id)
        query = f"UPDATE pending_actions SET {', '.join(updates)} WHERE id = ?"

        with self._get_connection() as conn:
            conn.execute(query, tuple(params))
            conn.commit()

    def get_pending_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM pending_actions WHERE id = ?", (action_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            res["args"] = json.loads(res["args_json"])
            return res

    def list_pending_actions(
        self,
        session_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM pending_actions"
        conditions = []
        params: List[Any] = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if state:
            conditions.append("state = ?")
            params.append(state)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["args"] = json.loads(item["args_json"])
                results.append(item)
            return results

    get_pending_actions = list_pending_actions

    # --- Schedules ---

    def store_schedule(
        self,
        schedule_id: str,
        status: str,
        payload: Dict[str, Any],
        total_mwh: float,
        total_ron: float,
        parent_id: Optional[str] = None,
        promoted_by_action: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO schedules 
                (id, status, parent_id, payload_json, total_mwh, total_ron, created_at, promoted_by_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (schedule_id, status, parent_id, json.dumps(payload), total_mwh, total_ron, now, promoted_by_action),
            )
            conn.commit()

    def get_schedule(self, schedule_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM schedules WHERE id = ?", (schedule_id,)).fetchone()
            if not row:
                return None
            res = dict(row)
            res["payload"] = json.loads(res["payload_json"])
            return res

    def get_active_schedule(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM schedules WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            res = dict(row)
            res["payload"] = json.loads(res["payload_json"])
            return res

    def get_latest_draft_schedule(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM schedules WHERE status = 'draft' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            res = dict(row)
            res["payload"] = json.loads(res["payload_json"])
            return res

    def promote_schedule_atomic(self, draft_id: str, action_id: Optional[str] = None) -> Dict[str, Any]:
        """Atomically demote currently active schedule to history and promote draft to active."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Fetch draft schedule
            row = cursor.execute("SELECT * FROM schedules WHERE id = ?", (draft_id,)).fetchone()
            if not row:
                raise ValueError(f"Draft schedule '{draft_id}' not found")
            draft_sched = dict(row)

            # 2. Fetch current active schedule to calculate cost delta
            active_row = cursor.execute(
                "SELECT * FROM schedules WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            previous_active_id = active_row["id"] if active_row else None
            active_mwh = active_row["total_mwh"] if active_row else draft_sched["total_mwh"]
            active_ron = active_row["total_ron"] if active_row else draft_sched["total_ron"]
            delta_mwh = round(draft_sched["total_mwh"] - active_mwh, 2)
            delta_ron = round(draft_sched["total_ron"] - active_ron, 2)

            # 3. Demote current active schedules to history
            cursor.execute("UPDATE schedules SET status = 'history' WHERE status = 'active'")

            # 4. Promote draft to active
            cursor.execute(
                "UPDATE schedules SET status = 'active', promoted_by_action = ? WHERE id = ?",
                (action_id, draft_id),
            )
            conn.commit()

            draft_sched["status"] = "active"
            draft_sched["promoted_by_action"] = action_id
            draft_sched["previous_active_id"] = previous_active_id
            draft_sched["delta_mwh"] = delta_mwh
            draft_sched["delta_ron"] = delta_ron
            draft_sched["payload"] = json.loads(draft_sched["payload_json"])
            return draft_sched

    # --- Labels ---

    def add_label(self, household_id: str, label: str, source_action_id: Optional[str] = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "INSERT INTO labels (household_id, label, source_action_id, created_at) VALUES (?, ?, ?, ?)",
                (household_id, label, source_action_id, now),
            )
            conn.commit()

    def get_labels(self, household_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM labels"
        params: List[Any] = []
        if household_id:
            query += " WHERE household_id = ?"
            params.append(household_id)
        query += " ORDER BY created_at DESC"
        with self._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [dict(r) for r in rows]

    # --- Grounding Events ---

    def record_grounding_event(
        self,
        message_id: str,
        passed: bool,
        retried: bool,
        unmatched: List[Any],
    ) -> None:
        safe_unmatched = redact_secrets(unmatched)
        with self._get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO grounding_events (message_id, passed, retried, unmatched_json)
                VALUES (?, ?, ?, ?)""",
                (message_id, 1 if passed else 0, 1 if retried else 0, json.dumps(safe_unmatched)),
            )
            conn.commit()


DEFAULT_STORE = AuditStore()
