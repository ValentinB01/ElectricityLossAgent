"""FastAPI application exposing the agent, tools, action gate, and schedules over local HTTP."""

import logging
from typing import Any, Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import src.agent.tools  # Ensure all 15 tools are registered
from src.agent.actions import ActionGate, action_gate as default_gate
from src.agent.config import AgentConfig, load_agent_config
from src.agent.llm_agent import LLMAgent
from src.agent.registry import ToolRegistry, registry as default_registry
from src.agent.schemas.common import ErrorCode, StructuredError
from src.agent.store import AuditStore, DEFAULT_STORE

logger = logging.getLogger(__name__)


# --- Request and Response Models ---

class ChatRequest(BaseModel):
    message: str = Field(description="User message text in Romanian or English")
    session_id: Optional[str] = Field(default=None, description="Optional existing session id")


class ChatResponse(BaseModel):
    answer: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    pending_actions: List[Dict[str, Any]] = Field(default_factory=list)
    drafts: List[Dict[str, Any]] = Field(default_factory=list)
    grounding: Optional[Any] = None
    mode: str
    session_id: str
    tool_result_ids: List[str] = Field(default_factory=list)


class PatchActionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    edits: Optional[Dict[str, Any]] = Field(default=None, description="Fields to update on the pending action")

    def get_edits(self) -> Dict[str, Any]:
        """Extract edits either from the 'edits' key or from top-level body fields."""
        if self.edits is not None:
            return self.edits
        extra = getattr(self, "__pydantic_extra__", {}) or {}
        return dict(extra)


class ConfirmActionRequest(BaseModel):
    args_hash: Optional[str] = Field(default=None, description="Optional expected args hash for validation")


class HealthResponse(BaseModel):
    status: str = "ok"
    mode: str
    provider: str
    model: str
    as_of: str
    backends: Dict[str, str]
    stale_forecast: bool = False


# --- Application Factory ---

def create_app(
    agent: Optional[LLMAgent] = None,
    store: Optional[AuditStore] = None,
    gate: Optional[ActionGate] = None,
    registry: Optional[ToolRegistry] = None,
    config: Optional[AgentConfig] = None,
) -> FastAPI:
    """Create and configure FastAPI instance with state-isolated dependencies."""
    app_config = config or load_agent_config()
    app_store = store or DEFAULT_STORE
    app_gate = gate or ActionGate(store=app_store)
    app_registry = registry or default_registry
    app_registry.set_store(app_store)

    app_agent = agent or LLMAgent(
        config=app_config,
        store=app_store,
        gate=app_gate,
        registry=app_registry,
    )

    app = FastAPI(
        title="PPC Electricity Loss Agent API",
        description="Local HTTP API for loss reduction, maintenance planning, and numeric grounding.",
        version="0.1.0",
    )

    # Security: loopback-only demo CORS policy per design D15
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Exception Handlers ---

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": ErrorCode.BACKEND_CONTRACT_VIOLATION.value,
                "message": "Invalid request parameters",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict):
            body = dict(exc.detail)
            if "code" not in body:
                body["code"] = "http_error"
            if "message" not in body:
                body["message"] = str(exc.detail)
            if "details" not in body:
                body["details"] = None
            return JSONResponse(status_code=exc.status_code, content=body)
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": "http_error", "message": str(exc.detail), "details": None},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled server error: {exc}", exc_info=True)
        # Never leak secrets or stack traces to client
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_server_error",
                "message": "An internal error occurred",
                "details": None,
            },
        )

    # --- Routes ---

    @app.get("/health", response_model=HealthResponse)
    def get_health() -> HealthResponse:
        """Report server status, mode, provider/model without secrets, and per-tool backends."""
        mode = "llm" if app_agent.has_llm_api_key() else "deterministic"
        backend_map: Dict[str, str] = {}
        for t in app_registry.list_tools():
            t_name = t["name"]
            backend_map[t_name] = app_config.backends.get(t_name, "fixture")

        return HealthResponse(
            status="ok",
            mode=mode,
            provider=app_config.provider,
            model=app_config.model,
            as_of=app_config.as_of,
            backends=backend_map,
            stale_forecast=app_config.stale_forecast,
        )

    @app.post("/chat", response_model=ChatResponse)
    def post_chat(req: ChatRequest) -> ChatResponse:
        """Execute a user conversation turn with tools, grounding, and pending actions."""
        resp = app_agent.chat(user_message=req.message, session_id=req.session_id)
        grounding_data = resp.grounding
        if hasattr(grounding_data, "is_grounded"):
            grounding_data = {
                "is_grounded": grounding_data.is_grounded,
                "tool_result_ids": grounding_data.tool_result_ids,
                "warning_message": grounding_data.warning_message,
                "unmatched_numbers": [u.raw_text for u in grounding_data.unmatched_numbers],
            }

        return ChatResponse(
            answer=resp.answer,
            tool_calls=resp.tool_calls,
            pending_actions=resp.pending_actions,
            drafts=resp.drafts,
            grounding=grounding_data,
            mode=resp.mode,
            session_id=resp.session_id,
            tool_result_ids=resp.tool_result_ids,
        )

    @app.get("/tools")
    def list_tools() -> List[Dict[str, Any]]:
        """Return catalogue of all registered tools with schemas and effect classes."""
        return app_registry.list_tools()

    @app.get("/tools/{name}")
    def get_tool_definition(name: str) -> Dict[str, Any]:
        """Return definition and schemas for a single registered tool."""
        defn = app_registry.get_tool(name)
        if not defn:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": ErrorCode.UNKNOWN_TOOL.value,
                    "message": f"Tool '{name}' is not registered in catalogue",
                    "details": None,
                },
            )
        return {
            "name": defn.name,
            "description": defn.description,
            "effect": defn.effect.value,
            "input_schema": defn.input_schema.model_json_schema(),
            "output_schema": defn.output_schema.model_json_schema(),
        }

    @app.post("/tools/{name}")
    def execute_tool_direct(
        name: str,
        args: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = Query(default=None),
    ) -> Any:
        """Call a tool directly. Read and what-if tools execute; write tools produce a pending action."""
        defn = app_registry.get_tool(name)
        if not defn:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": ErrorCode.UNKNOWN_TOOL.value,
                    "message": f"Tool '{name}' is not registered in catalogue",
                    "details": None,
                },
            )

        exec_res = app_registry.execute(
            tool_name=name,
            raw_args=args or {},
            session_id=session_id,
            store=app_store,
        )

        if exec_res.is_error:
            err = exec_res.error
            err_code = err.code.value if hasattr(err.code, "value") else str(err.code)
            status_code = 400
            if err_code in (
                ErrorCode.UNKNOWN_TOOL.value,
                ErrorCode.UNKNOWN_ASSET.value,
                ErrorCode.UNKNOWN_ITEM.value,
            ):
                status_code = 404
            elif err_code in (
                ErrorCode.BACKEND_CONTRACT_VIOLATION.value,
                ErrorCode.INVALID_DATE.value,
                ErrorCode.UNSUPPORTED_SCENARIO.value,
            ):
                status_code = 422

            return JSONResponse(status_code=status_code, content=err.model_dump())

        # Success: return full result or pending action payload
        res = exec_res.full_result
        if hasattr(res, "model_dump"):
            return res.model_dump()
        return res

    @app.get("/actions")
    def list_actions(
        session_id: Optional[str] = Query(default=None),
        state: Optional[str] = Query(default=None),
    ) -> List[Dict[str, Any]]:
        """List pending actions and their history, optionally filtered by session or state."""
        return app_store.list_pending_actions(session_id=session_id, state=state)

    @app.post("/actions/{action_id}/confirm")
    def confirm_action(
        action_id: str,
        req: Optional[ConfirmActionRequest] = None,
        args_hash: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        """Confirm and execute a pending action through the Action Gate. Executes once."""
        action = app_store.get_pending_action(action_id)
        if not action:
            raise HTTPException(
                status_code=404,
                detail={"code": "action_not_found", "message": f"Action '{action_id}' not found", "details": None},
            )

        target_hash = (req.args_hash if req else None) or args_hash
        res = app_gate.confirm_action(action_id=action_id, args_hash=target_hash)
        if isinstance(res, dict) and res.get("status") == "error":
            code = res.get("code", "action_error")
            status_code = 404 if code == "not_found" else 400
            raise HTTPException(
                status_code=status_code,
                detail={"code": code, "message": res.get("message", "Error confirming action"), "details": None},
            )
        return res

    @app.post("/actions/{action_id}/reject")
    def reject_action(action_id: str) -> Dict[str, Any]:
        """Reject a pending action through the Action Gate."""
        action = app_store.get_pending_action(action_id)
        if not action:
            raise HTTPException(
                status_code=404,
                detail={"code": "action_not_found", "message": f"Action '{action_id}' not found", "details": None},
            )

        res = app_gate.reject_action(action_id=action_id)
        if isinstance(res, dict) and res.get("status") == "error":
            code = res.get("code", "action_error")
            status_code = 404 if code == "not_found" else 400
            raise HTTPException(
                status_code=status_code,
                detail={"code": code, "message": res.get("message", "Error rejecting action"), "details": None},
            )
        return res

    @app.patch("/actions/{action_id}")
    def patch_action(action_id: str, req: PatchActionRequest) -> Dict[str, Any]:
        """Edit fields of a pending action (e.g. from technician UI)."""
        action = app_store.get_pending_action(action_id)
        if not action:
            raise HTTPException(
                status_code=404,
                detail={"code": "action_not_found", "message": f"Action '{action_id}' not found", "details": None},
            )

        edits = req.get_edits()
        res = app_gate.patch_action(action_id=action_id, edits=edits)
        if isinstance(res, dict) and res.get("status") == "error":
            code = res.get("code", "action_error")
            status_code = 404 if code == "not_found" else 400
            raise HTTPException(
                status_code=status_code,
                detail={"code": code, "message": res.get("message", "Error patching action"), "details": None},
            )
        return res

    @app.get("/schedules/active")
    def get_active_schedule() -> Dict[str, Any]:
        """Get the currently active maintenance calendar schedule."""
        sched = app_store.get_active_schedule()
        if not sched:
            raise HTTPException(
                status_code=404,
                detail={"code": "not_found", "message": "No active schedule found in database", "details": None},
            )
        return sched

    @app.get("/schedules/{schedule_id}")
    def get_schedule(schedule_id: str) -> Dict[str, Any]:
        """Get a specific schedule by ID (active, draft, or history)."""
        sched = app_store.get_schedule(schedule_id)
        if not sched:
            raise HTTPException(
                status_code=404,
                detail={"code": "not_found", "message": f"Schedule '{schedule_id}' not found", "details": None},
            )
        return sched

    @app.get("/sessions/{session_id}")
    def get_session_transcript(session_id: str) -> Dict[str, Any]:
        """Get transcript with messages and tool calls for session audit."""
        sess = app_store.get_session(session_id)
        if not sess:
            raise HTTPException(
                status_code=404,
                detail={"code": "not_found", "message": f"Session '{session_id}' not found", "details": None},
            )
        messages = app_store.get_messages(session_id)
        tool_calls = app_store.get_tool_calls(session_id)
        return {
            "session": sess,
            "messages": messages,
            "tool_calls": tool_calls,
        }

    return app


app = create_app()


def run_server():
    """Start local uvicorn server binding strictly to configured loopback interface."""
    cfg = load_agent_config()
    uvicorn.run(
        "src.agent.api:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
    )


if __name__ == "__main__":
    run_server()
