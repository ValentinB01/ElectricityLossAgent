"""Thin client used by the dashboard to talk to the agent service (design D12).

Default transport is HTTP against ``make agent``. ``mode="inproc"`` (or env
``AGENT_UI_MODE=inproc``) mounts the FastAPI app inside this process as a
fallback when a second process is a risk on the demo machine.
"""

import os
from typing import Any, Dict, List, Optional

import requests

DEFAULT_URL = "http://127.0.0.1:8000"


class AgentError(Exception):
    """Structured error returned by the agent service."""

    def __init__(self, code: str, message: str, status: int = 0, payload: Optional[Dict[str, Any]] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status = status
        self.payload = payload or {}


class AgentClient:
    def __init__(self, base_url: Optional[str] = None, mode: Optional[str] = None, timeout: float = 90.0):
        self.mode = (mode or os.environ.get("AGENT_UI_MODE", "http")).lower()
        self.base_url = (base_url or os.environ.get("AGENT_URL", DEFAULT_URL)).rstrip("/")
        self.timeout = timeout
        self._inproc = None
        if self.mode == "inproc":
            from fastapi.testclient import TestClient
            from src.agent.api import app

            self._inproc = TestClient(app)

    # --- transport ---

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            if self._inproc is not None:
                resp = self._inproc.request(method, path, **kwargs)
            else:
                resp = requests.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise AgentError("service_unreachable", f"Serviciul agent nu răspunde la {self.base_url} ({exc.__class__.__name__})")
        except Exception as exc:
            raise AgentError("service_unreachable", f"Eroare client intern: {str(exc)}")
        try:
            body = resp.json()
        except ValueError:
            body = {"message": resp.text}
        if resp.status_code >= 400:
            reason = getattr(resp, "reason_phrase", getattr(resp, "reason", "HTTP Error"))
            if isinstance(body, dict):
                raise AgentError(str(body.get("code", "http_error")), str(body.get("message", reason)), resp.status_code, body)
            raise AgentError("http_error", str(body), resp.status_code)
        return body

    # --- endpoints ---

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def tool(self, name: str, args: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"session_id": session_id} if session_id else None
        return self._request("POST", f"/tools/{name}", json=args or {}, params=params)

    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        return self._request("POST", "/chat", json={"message": message, "session_id": session_id})

    def actions(self, session_id: Optional[str] = None, state: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {k: v for k, v in (("session_id", session_id), ("state", state)) if v}
        return self._request("GET", "/actions", params=params)

    def confirm(self, action_id: str, args_hash: Optional[str] = None) -> Dict[str, Any]:
        return self._request("POST", f"/actions/{action_id}/confirm", json={"args_hash": args_hash})

    def reject(self, action_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/actions/{action_id}/reject")

    def patch_action(self, action_id: str, edits: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("PATCH", f"/actions/{action_id}", json={"edits": edits})

    def active_schedule(self) -> Optional[Dict[str, Any]]:
        try:
            return self._request("GET", "/schedules/active")
        except AgentError as exc:
            if exc.status == 404:
                return None
            raise

    def draft_schedule(self) -> Optional[Dict[str, Any]]:
        try:
            return self._request("GET", "/schedules/draft")
        except AgentError as exc:
            if exc.status == 404:
                return None
            raise

