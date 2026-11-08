"""Tool registry and dispatcher pipeline with typed contracts, effect gating, and provenance."""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Set, Type
from pydantic import BaseModel, ValidationError

from src.agent.backends.fixture import DEFAULT_FIXTURE_BACKEND, FixtureBackend
from src.agent.config import load_agent_config
from src.agent.schemas.common import EffectClass, ErrorCode, Provenance, StructuredError

ALLOWED_DAY_NAMES: Set[str] = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "luni", "marți", "marti", "miercuri", "joi", "vineri", "sâmbătă", "sambata", "duminică", "duminica",
}
WIND_ASSETS: Set[str] = {f"FAN-T{i:02d}" for i in range(1, 21)}
PV_ASSETS: Set[str] = {f"CAL-INV{i:02d}" for i in range(1, 9)}
ALL_VALID_ASSETS: Set[str] = WIND_ASSETS | PV_ASSETS


@dataclass
class ToolDefinition:
    name: str
    description: str
    effect: EffectClass
    input_schema: Type[BaseModel]
    output_schema: Type[BaseModel]
    llm_view: Optional[Callable[[Any], Dict[str, Any]]] = None
    handler: Optional[Callable[..., Any]] = None


@dataclass
class ToolExecutionResult:
    tool: str
    full_result: Any
    llm_view: Dict[str, Any]
    provenance: Provenance
    is_error: bool = False
    error: Optional[StructuredError] = None


class ToolRegistry:
    """Central registry for all tools, managing effects, schema validation, and dispatch."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._fixture_backend = DEFAULT_FIXTURE_BACKEND
        self._config = load_agent_config()
        from src.agent.store import DEFAULT_STORE
        self._store = DEFAULT_STORE

    def register(
        self,
        name: str,
        description: str,
        effect: EffectClass | str,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        llm_view: Optional[Callable[[Any], Dict[str, Any]]] = None,
    ) -> Callable:
        """Decorator to register a tool implementation."""
        eff = EffectClass(effect) if isinstance(effect, str) else effect

        def decorator(fn: Callable) -> Callable:
            tool_def = ToolDefinition(
                name=name,
                description=description,
                effect=eff,
                input_schema=input_schema,
                output_schema=output_schema,
                llm_view=llm_view,
                handler=fn,
            )
            self._tools[name] = tool_def

            from functools import wraps

            @wraps(fn)
            def wrapper(args: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
                bypass = kwargs.pop("bypass_gate", False)
                session_id = kwargs.pop("session_id", None)
                call_args = dict(args) if isinstance(args, dict) else {}
                call_args.update(kwargs)
                exec_res = self.execute(name, call_args, bypass_gate=bypass, session_id=session_id)
                if exec_res.is_error:
                    return exec_res.error
                return exec_res.full_result

            wrapper.__tool_definition__ = tool_def
            return wrapper

        return decorator

    def set_store(self, store: Any) -> None:
        """Set or swap the active audit store (e.g. for isolated testing)."""
        self._store = store

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return catalogue of all registered tools with schemas and effect classes."""
        catalogue = []
        for name, defn in self._tools.items():
            catalogue.append({
                "name": name,
                "description": defn.description,
                "effect": defn.effect.value,
                "input_schema": defn.input_schema.model_json_schema(),
                "output_schema": defn.output_schema.model_json_schema(),
            })
        return catalogue

    def _validate_dates(self, args: Dict[str, Any], as_of_str: str) -> Optional[StructuredError]:
        """Validate date strings against format and ensure constraint dates are not in the past."""
        as_of_date = None
        try:
            as_of_date = datetime.strptime(as_of_str, "%Y-%m-%d").date()
        except Exception:
            pass

        # Check constraint dates in optimize_maintenance
        constraints = args.get("constraints") or {}
        if isinstance(constraints, dict):
            # Check direct constraint dates or unavailable lists
            candidate_dates = []
            if "date" in constraints:
                candidate_dates.append(str(constraints["date"]).strip())
            unavail = constraints.get("unavailable", [])
            if isinstance(unavail, list):
                for item in unavail:
                    if isinstance(item, dict) and "date" in item:
                        candidate_dates.append(str(item["date"]).strip())
                    elif isinstance(item, str):
                        candidate_dates.append(item.strip())

            for raw_date in candidate_dates:
                if raw_date.lower() in ALLOWED_DAY_NAMES:
                    continue
                try:
                    d = datetime.strptime(raw_date, "%Y-%m-%d").date()
                    if as_of_date and d < as_of_date:
                        return StructuredError(
                            code=ErrorCode.INVALID_DATE,
                            message=f"Date '{raw_date}' is in the past relative to as_of '{as_of_str}'",
                            date_str=raw_date,
                        )
                except ValueError:
                    return StructuredError(
                        code=ErrorCode.INVALID_DATE,
                        message=f"Malformed date '{raw_date}', expected YYYY-MM-DD",
                        date_str=raw_date,
                    )

        # Check start / end dates if present
        for key in ["start", "end", "due_date"]:
            if key in args and args[key] is not None:
                val = str(args[key])
                try:
                    # Try ISO parsing
                    datetime.fromisoformat(val.replace("Z", "+00:00"))
                except Exception:
                    try:
                        datetime.strptime(val, "%Y-%m-%d")
                    except Exception:
                        return StructuredError(
                            code=ErrorCode.INVALID_DATE,
                            message=f"Malformed date '{val}' in field '{key}'",
                            date_str=val,
                        )
        return None

    def _validate_assets(self, args: Dict[str, Any], tool_name: str) -> Optional[StructuredError]:
        """Validate asset ids against portfolio assets."""
        asset_id = args.get("asset_id")
        asset_ids = args.get("asset_ids")
        check_ids: List[str] = []
        if asset_id:
            check_ids.append(str(asset_id))
        if asset_ids and isinstance(asset_ids, list):
            check_ids.extend([str(a) for a in asset_ids])

        site_id = args.get("site_id")
        if site_id == "calugareni":
            valid_set = PV_ASSETS
        elif site_id == "fantanele":
            valid_set = WIND_ASSETS
        else:
            valid_set = ALL_VALID_ASSETS

        for a_id in check_ids:
            target_valid_set = valid_set
            if not site_id:
                if a_id.startswith("FAN-"):
                    target_valid_set = WIND_ASSETS
                elif a_id.startswith("CAL-"):
                    target_valid_set = PV_ASSETS
            if a_id not in target_valid_set:
                site_name = site_id or ("fantanele" if a_id.startswith("FAN-") else ("calugareni" if a_id.startswith("CAL-") else "portfolio"))
                return StructuredError(
                    code=ErrorCode.UNKNOWN_ASSET,
                    message=f"Asset '{a_id}' does not exist for site '{site_name}'",
                    valid_ids=sorted(list(target_valid_set)),
                )
        return None

    def execute(
        self,
        tool_name: str,
        raw_args: Optional[Dict[str, Any]] = None,
        bypass_gate: bool = False,
        session_id: Optional[str] = None,
        store: Optional[Any] = None,
    ) -> ToolExecutionResult:
        """Run the full dispatch pipeline for a tool call."""
        import time
        start_time = time.perf_counter()
        active_store = store or self._store
        raw_args = raw_args or {}
        defn = self._tools.get(tool_name)

        def _log_error(err: StructuredError, backend: str, effect: str = "read") -> ToolExecutionResult:
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            err_code = err.code.value if hasattr(err.code, "value") else str(err.code)
            try:
                active_store.log_tool_call(
                    session_id=session_id,
                    tool=tool_name,
                    effect=effect,
                    args=raw_args,
                    result_id=None,
                    backend=backend,
                    duration_ms=duration_ms,
                    error_code=err_code,
                )
            except Exception:
                pass
            return ToolExecutionResult(
                tool=tool_name,
                full_result=err,
                llm_view=err.model_dump(),
                provenance=Provenance(tool=tool_name, backend=backend),
                is_error=True,
                error=err,
            )

        if not defn:
            err = StructuredError(
                code=ErrorCode.UNKNOWN_TOOL,
                message=f"Tool '{tool_name}' is not registered in catalogue",
                tool_name=tool_name,
            )
            return _log_error(err, backend="error", effect="read")

        # 1. Validate inputs via Pydantic
        try:
            validated_input = defn.input_schema.model_validate(raw_args)
            input_dict = validated_input.model_dump()
        except ValidationError as e:
            err = StructuredError(
                code=ErrorCode.BACKEND_CONTRACT_VIOLATION,
                message=f"Invalid input arguments for tool '{tool_name}': {str(e)}",
                details=e.errors(),
            )
            return _log_error(err, backend="error", effect=defn.effect.value)

        # 2. Check asset ids
        asset_err = self._validate_assets(input_dict, tool_name)
        if asset_err:
            return _log_error(asset_err, backend="validation", effect=defn.effect.value)

        # 3. Check dates
        as_of_str = input_dict.get("as_of") or self._config.as_of
        date_err = self._validate_dates(input_dict, as_of_str)
        if date_err:
            return _log_error(date_err, backend="validation", effect=defn.effect.value)

        # 4. Action gate: Intercept write tools unless confirmation has bypassed the gate
        if defn.effect == EffectClass.WRITE and not bypass_gate:
            from src.agent.actions import action_gate
            pending_action = action_gate.create_pending_action(tool_name, input_dict, session_id=session_id)
            duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            active_store.log_tool_call(
                session_id=session_id,
                tool=tool_name,
                effect="write",
                args=input_dict,
                result_id=pending_action["action_id"],
                backend="action_gate",
                duration_ms=duration_ms,
            )
            return ToolExecutionResult(
                tool=tool_name,
                full_result=pending_action,
                llm_view=pending_action,
                provenance=Provenance(tool=tool_name, backend="action_gate"),
                is_error=False,
                error=None,
            )

        # 5. Resolve backend (fixture or live)
        backend_label = self._config.backends.get(tool_name, "fixture")

        # 5. Call backend
        try:
            if defn.handler:
                raw_output = defn.handler(input_dict, backend=backend_label)
            else:
                raw_output = self._fixture_backend.call(tool_name, input_dict)
        except Exception as ex:
            err = StructuredError(
                code=ErrorCode.BACKEND_CONTRACT_VIOLATION,
                message=f"Exception during backend call for '{tool_name}': {str(ex)}",
            )
            return _log_error(err, backend=backend_label, effect=defn.effect.value)

        # Check if backend returned a StructuredError directly
        if isinstance(raw_output, StructuredError):
            return _log_error(raw_output, backend=backend_label, effect=defn.effect.value)

        # 6. Validate output against output schema
        try:
            if isinstance(raw_output, defn.output_schema):
                validated_output = raw_output
            elif isinstance(raw_output, dict):
                validated_output = defn.output_schema.model_validate(raw_output)
            else:
                validated_output = defn.output_schema.model_validate(
                    getattr(raw_output, "model_dump", lambda: raw_output)()
                )
        except ValidationError as e:
            err = StructuredError(
                code=ErrorCode.BACKEND_CONTRACT_VIOLATION,
                message=f"Backend output violated output schema for tool '{tool_name}': {str(e)}",
                details=e.errors(),
            )
            return _log_error(err, backend=backend_label, effect=defn.effect.value)

        # 7. Ensure provenance is attached and set
        if hasattr(validated_output, "provenance") and validated_output.provenance:
            validated_output.provenance.tool = tool_name
            validated_output.provenance.backend = backend_label
            if as_of_str and not validated_output.provenance.as_of:
                validated_output.provenance.as_of = as_of_str
            prov = validated_output.provenance
        else:
            prov = Provenance(tool=tool_name, backend=backend_label, as_of=as_of_str)

        # 8. Compute llm_view
        if defn.llm_view:
            llm_v = defn.llm_view(validated_output)
        else:
            llm_v = validated_output.model_dump()

        # Store result and log call in audit store
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        dumped_payload = validated_output.model_dump() if hasattr(validated_output, "model_dump") else validated_output
        dumped_prov = prov.model_dump() if hasattr(prov, "model_dump") else prov
        active_store.store_tool_result(
            result_id=prov.result_id,
            payload=dumped_payload,
            llm_view=llm_v,
            provenance=dumped_prov,
        )
        active_store.log_tool_call(
            session_id=session_id,
            tool=tool_name,
            effect=defn.effect.value,
            args=input_dict,
            result_id=prov.result_id,
            backend=backend_label,
            duration_ms=duration_ms,
        )

        return ToolExecutionResult(
            tool=tool_name,
            full_result=validated_output,
            llm_view=llm_v,
            provenance=prov,
            is_error=False,
            error=None,
        )


registry = ToolRegistry()
tool = registry.register

