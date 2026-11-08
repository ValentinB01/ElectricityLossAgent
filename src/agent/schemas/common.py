"""Common schemas, error models, and provenance for the agent tools."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional
import uuid
from pydantic import BaseModel, ConfigDict, Field


class EffectClass(str, Enum):
    READ = "read"
    WHAT_IF = "what_if"
    WRITE = "write"


class ErrorCode(str, Enum):
    UNKNOWN_ASSET = "unknown_asset"
    UNKNOWN_ITEM = "unknown_item"
    INVALID_DATE = "invalid_date"
    UNKNOWN_TOOL = "unknown_tool"
    BACKEND_CONTRACT_VIOLATION = "backend_contract_violation"
    UNSUPPORTED_SCENARIO = "unsupported_scenario"


class StructuredError(BaseModel):
    """Structured error returned by tools and dispatcher per agent-tools spec."""
    model_config = ConfigDict(extra="allow")

    code: ErrorCode
    message: str
    valid_ids: Optional[List[str]] = None
    item_id: Optional[str] = None
    date_str: Optional[str] = None
    tool_name: Optional[str] = None
    details: Optional[Any] = None


class Provenance(BaseModel):
    """Provenance block attached to every tool result."""
    model_config = ConfigDict(extra="allow")

    result_id: str = Field(default_factory=lambda: f"res-{uuid.uuid4().hex[:8]}")
    tool: str
    backend: str = "fixture"  # "fixture" or "live"
    as_of: Optional[str] = None
    forecast_run: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

