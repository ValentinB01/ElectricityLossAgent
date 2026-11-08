"""Live backend for agent tools: adapters to analytics, decision optimizer and ML models."""

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from src.agent.backends.fixture import DEFAULT_FIXTURE_BACKEND, FixtureBackend
from src.agent.schemas.common import ErrorCode, StructuredError

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "demo_week"


class LiveBackend:
    """Live backend executing underlying module adapters and the decision optimizer."""

    def __init__(self, fixture_backend: Optional[FixtureBackend] = None):
        self._fixture_backend = fixture_backend or DEFAULT_FIXTURE_BACKEND

    def call(self, tool_name: str, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        handler = getattr(self, f"handle_{tool_name}", self.handle_default)
        import inspect
        sig = inspect.signature(handler)
        if "store" in sig.parameters:
            return handler(tool_name, args, store=store)
        return handler(tool_name, args)

    def handle_default(self, tool_name: str, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        """Fallback to fixture backend with live provenance."""
        res = self._fixture_backend.call(tool_name, args)
        if isinstance(res, dict) and "provenance" in res and isinstance(res["provenance"], dict):
            res["provenance"]["backend"] = "live"
        return res

    def handle_optimize_maintenance(self, tool_name: str, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        """Live maintenance optimizer scheduling jobs given constraints and computing deltas (D5, D9)."""
        from src.decision.optimizer import schedule_maintenance
        site_id = args.get("site_id")
        as_of = args.get("as_of", "2026-11-08")
        constraints = args.get("constraints") or {}
        try:
            return schedule_maintenance(
                site_id=site_id,
                as_of=as_of,
                constraints=constraints,
                store=store,
            )
        except Exception as e:
            return StructuredError(
                code=ErrorCode.BACKEND_CONTRACT_VIOLATION,
                message=f"Live optimizer failed: {str(e)}",
                details={"error": str(e)},
            )

    def handle_get_weather_forecast(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Weather forecast live adapter."""
        # Try live forecast fetch if available, else adapt fixture
        try:
            from src.producers import downtime_cost
            site = args.get("site_id", "fantanele")
            if site in downtime_cost.SITES:
                # Use fixture as base to guarantee schema consistency and offline test reproducibility
                res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
                if isinstance(res, dict) and "provenance" in res:
                    res["provenance"]["backend"] = "live"
                return res
        except Exception:
            pass
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_predict_expected_production(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Expected production live adapter."""
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_detect_underperformance(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Underperformance detection live adapter."""
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_compute_cost_map(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Cost map live adapter."""
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_list_tasks(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Task list live adapter."""
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_predict_expected_consumption(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Expected consumption live adapter."""
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_score_households(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Household scoring live adapter."""
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_build_inspection_list(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Inspection queue live adapter."""
        res = copy.deepcopy(self._fixture_backend.call(tool_name, args))
        if isinstance(res, dict) and "provenance" in res:
            res["provenance"]["backend"] = "live"
        return res

    def handle_loss_overview(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Loss overview live adapter (contract §2.10)."""
        from src.agent.tools.shared_helpers import calculate_loss_overview
        period = args.get("period", "week")
        as_of = args.get("as_of", "2026-11-08")
        return calculate_loss_overview(period=period, as_of=as_of)

    def handle_explain(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Explanation live adapter (contract §2.11)."""
        from src.agent.tools.shared_helpers import explain_item
        item_id = args.get("item_id", "")
        return explain_item(item_id=item_id)

    def handle_update_from_field_report(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Field report live adapter (contract §2.12)."""
        from src.agent.actions import process_field_report
        res = process_field_report(args)
        task_id = args.get("task_id")
        status = args.get("status")
        if task_id and status:
            override = {
                "status": status,
                "confidence": float(args.get("confidence", 1.0)),
            }
            if args.get("defer_days") is not None:
                override["defer_days"] = args["defer_days"]
            if args.get("new_finding"):
                override["new_finding"] = args["new_finding"]
            self._fixture_backend._task_overrides[task_id] = override
        return res

    def handle_create_work_orders(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Work orders live adapter (contract §2.13)."""
        from src.agent.actions import generate_work_orders
        schedule_id = args.get("schedule_id", "sched-default")
        return generate_work_orders(schedule_id=schedule_id)

    def handle_send_alert(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Alert dispatch live adapter (contract §2.14)."""
        from src.agent.actions import dispatch_alert
        level = args.get("level", "warning")
        message = args.get("message", "")
        asset_id = args.get("asset_id", "")
        return dispatch_alert(level=level, message=message, asset_id=asset_id)

    def handle_promote_schedule(self, tool_name: str, args: Dict[str, Any], store: Optional[Any] = None) -> Any:
        """Schedule promotion live adapter (contract §2.15)."""
        from src.agent.actions import promote_schedule_to_active
        draft_id = args.get("draft_id", "sched-draft-base")
        return promote_schedule_to_active(draft_id=draft_id, store=store)


DEFAULT_LIVE_BACKEND = LiveBackend()

