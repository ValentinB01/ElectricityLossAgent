import os
from pathlib import Path
from typing import Dict, Literal
import yaml
from pydantic import BaseModel, Field


class LimitsConfig(BaseModel):
    max_iterations: int = Field(default=6, ge=1)
    max_tool_calls: int = Field(default=12, ge=1)
    timeout_seconds: int = Field(default=60, ge=1)
    max_tokens: int = Field(default=4096, ge=1)


class GroundingConfig(BaseModel):
    tolerance: float = Field(default=0.05, ge=0.0)
    small_count_exemption: int = Field(default=10, ge=0)


class ActionConfig(BaseModel):
    ttl_seconds: int = Field(default=3600, ge=1)


class ServerConfig(BaseModel):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)


BackendType = Literal["fixture", "live"]


class AgentConfig(BaseModel):
    provider: str = Field(default="gemini")
    model: str = Field(default="gemini/gemini-2.5-flash")
    as_of: str = Field(default="2026-11-08")
    stale_forecast: bool = Field(default=False)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    backends: Dict[str, BackendType] = Field(default_factory=dict)


def load_agent_config(config_path: Path | str = "config/agent.yaml") -> AgentConfig:
    path = Path(config_path)
    if not path.exists():
        return AgentConfig()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # AGENT_BACKENDS=fixture runs every tool on the demo fixtures (tests, an offline demo); =live forces the live backend.
    forced = os.environ.get("AGENT_BACKENDS", "").strip().lower()
    if forced in ("fixture", "live"):
        data["backends"] = {name: forced for name in (data.get("backends") or {})}
    return AgentConfig.model_validate(data)

