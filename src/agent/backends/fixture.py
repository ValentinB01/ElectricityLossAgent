"""Fixture backend for agent tools: serves versioned demo data from src/agent/fixtures/demo_week/."""

import copy
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict
from src.agent.schemas.common import ErrorCode, StructuredError

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "demo_week"


def _load_fixture(tool_name: str) -> Dict[str, Any]:
    file_path = FIXTURE_DIR / f"{tool_name}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _is_crew_2(value: Any) -> bool:
    """crew-2, crew_2, "2", "echipa 2": any spelling that names crew 2 (the demo's precomputed scenario)."""
    m = re.search(r"\d+", str(value or ""))
    return bool(m) and int(m.group()) == 2


class FixtureBackend:
    """Fixture backend serving static demo_week JSON data."""

    def __init__(self, fixture_dir: Path = FIXTURE_DIR):
        self.fixture_dir = fixture_dir
        self._task_overrides: Dict[str, Dict[str, Any]] = {}

    def call(self, tool_name: str, args: Dict[str, Any]) -> Any:
        handler = getattr(self, f"handle_{tool_name}", self.handle_default)
        return handler(tool_name, args)

    def handle_default(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return _load_fixture(tool_name)

    def handle_get_weather_forecast(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name)
        site_id = args.get("site_id")
        if site_id:
            data["site_id"] = site_id
        if args.get("stale"):
            data["stale"] = True
            data["age_hours"] = float(args.get("age_hours", 12.0))
        return data

    def handle_predict_expected_production(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name)
        asset_ids = args.get("asset_ids")
        if asset_ids:
            asset_set = set(asset_ids)
            data["predictions"] = [
                p for p in data.get("predictions", [])
                if p.get("asset_id") in asset_set
            ]
        return data

    def handle_detect_underperformance(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name)
        asset_ids = args.get("asset_ids")
        if asset_ids:
            asset_set = set(asset_ids)
            data["detections"] = [
                d for d in data.get("detections", [])
                if d.get("asset_id") in asset_set
            ]
        return data

    def handle_compute_cost_map(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name)
        site_id = args.get("site_id")
        if site_id:
            data["site_id"] = site_id
        if args.get("stale"):
            data["stale"] = True
            data["age_hours"] = float(args.get("age_hours", 12.0))
        return data

    def handle_list_tasks(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = copy.deepcopy(_load_fixture(tool_name))
        for task in data.get("tasks", []):
            tid = task.get("task_id")
            if tid in self._task_overrides:
                task.update(self._task_overrides[tid])
        return data

    def handle_optimize_maintenance(self, tool_name: str, args: Dict[str, Any]) -> Any:
        data = _load_fixture(tool_name)
        variants = data.get("variants", {})
        constraints = args.get("constraints") or {}

        # Default or standard baseline constraint
        if not constraints or constraints == {"crews_per_day": 2}:
            return variants.get("baseline", data)

        # Detect crew 2 off Thursday constraint
        # E.g. {"unavailable": [{"crew": "crew-2", "date": "2026-11-12"}]} or "Thursday" or "joi"
        unavail = constraints.get("unavailable", [])
        crew_off_thursday = False
        thursday_tokens = {"2026-11-12", "thursday", "joi"}
        if isinstance(unavail, list):
            for u in unavail:
                if isinstance(u, dict) and (_is_crew_2(u.get("crew")) or _is_crew_2(u.get("crew_id"))):
                    d_str = str(u.get("date", u.get("day", ""))).lower()
                    if any(tok in d_str for tok in thursday_tokens):
                        crew_off_thursday = True
                elif isinstance(u, str) and any(tok in u.lower() for tok in thursday_tokens):
                    crew_off_thursday = True
        
        crew_key = constraints.get("crew") or constraints.get("crew_id")
        if _is_crew_2(crew_key):
            d_str = str(constraints.get("date", constraints.get("day", ""))).lower()
            if any(tok in d_str for tok in thursday_tokens):
                crew_off_thursday = True

        if crew_off_thursday:
            return variants.get("crew_2_off_thursday", variants["baseline"])

        # Detect T05 deferred by 5 days
        # E.g. {"defer_days": {"T05-bearing-001": 5}} or {"task_id": "T05-bearing-001", "defer_days": 5}
        defer_days = constraints.get("defer_days")
        if (
            (isinstance(defer_days, dict) and defer_days.get("T05-bearing-001") == 5)
            or (constraints.get("task_id") == "T05-bearing-001" and defer_days == 5)
            or defer_days == 5
        ):
            return variants.get("t05_deferred_5_days", variants["baseline"])

        # Unknown constraint scenario
        return StructuredError(
            code=ErrorCode.UNSUPPORTED_SCENARIO,
            message="Fixture backend only supports precomputed scenarios: baseline, crew-2 off Thursday, T05 deferred 5 days",
            details={"constraints": constraints},
        )

    def handle_predict_expected_consumption(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name)
        household_ids = args.get("household_ids")
        if household_ids:
            hh_set = set(household_ids)
            data["predictions"] = [
                p for p in data.get("predictions", [])
                if p.get("household_id") in hh_set
            ]
        return data

    def handle_score_households(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name)
        top_n = args.get("top_n", 50)
        data["households"] = data.get("households", [])[:top_n]
        return data

    def handle_build_inspection_list(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return _load_fixture(tool_name)

    def handle_loss_overview(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name).copy()
        period = args.get("period", "week")
        if period == "month":
            data["period"] = "month"
            data["total_mwh"] = round(data["total_mwh"] * 4.0, 2)
            data["total_ron"] = round(data["total_ron"] * 4.0, 2)
            monthly_rows = []
            for r in data.get("rows", []):
                rc = r.copy()
                rc["mwh"] = round(rc["mwh"] * 4.0, 2)
                rc["ron"] = round(rc["ron"] * 4.0, 2)
                monthly_rows.append(rc)
            data["rows"] = monthly_rows
        return data

    def handle_explain(self, tool_name: str, args: Dict[str, Any]) -> Any:
        data = _load_fixture(tool_name)
        item_id = args.get("item_id")
        items = data.get("items", {})

        if item_id in items:
            return items[item_id]

        if data.get("item_id") == item_id:
            return data

        return StructuredError(
            code=ErrorCode.UNKNOWN_ITEM,
            message=f"Item '{item_id}' not found in maintenance tasks or household queues",
            item_id=item_id,
        )

    def handle_update_from_field_report(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name).copy()
        task_id = args.get("task_id", data.get("task_id"))
        status = args.get("status", data.get("status"))
        defer_days = args.get("defer_days", data.get("defer_days"))
        new_finding = args.get("new_finding", data.get("new_finding"))
        confidence = float(args.get("confidence", 1.0))
        raw_text = args.get("raw_text") or args.get("text") or data.get("raw_text")

        # Rule check: task_id must be known, status in enum
        known_tasks = {"T17-pitch-001", "T05-bearing-001", "T01-rev-001", "T09-drone-001", "INV02-clean-001"}
        valid_status = {"done", "deferred", "needs_part", "not_found"}

        rule_ok = (task_id in known_tasks) and (status in valid_status)
        rule_score = 1.0 if rule_ok else 0.4
        effective_confidence = min(confidence, rule_score)
        needs_input = effective_confidence < 0.7

        if rule_ok and task_id:
            override = {"status": status, "confidence": confidence}
            if defer_days is not None:
                override["defer_days"] = defer_days
            if new_finding:
                override["new_finding"] = new_finding
            self._task_overrides[task_id] = override

        data.update({
            "task_id": task_id,
            "status": status,
            "defer_days": defer_days,
            "new_finding": new_finding,
            "confidence": confidence,
            "effective_confidence": effective_confidence,
            "needs_input": needs_input,
            "raw_text": raw_text,
        })
        return data

    def handle_create_work_orders(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name).copy()
        raw_id = args.get("schedule_id", data.get("schedule_id"))
        # Security: sanitize schedule_id to prevent directory traversal
        safe_schedule_id = "".join(c for c in raw_id if c.isalnum() or c in ("-", "_"))
        if not safe_schedule_id:
            safe_schedule_id = "sched-default"
        data["schedule_id"] = safe_schedule_id

        # Write files strictly under results/work_orders/ per side-effect confinement requirement
        out_dir = Path("results/work_orders")
        out_dir.mkdir(parents=True, exist_ok=True)

        created_files = []
        # JSON summary
        json_file = out_dir / f"{safe_schedule_id}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({"schedule_id": safe_schedule_id, "generated": True}, f, indent=2)
        created_files.append(str(json_file).replace("\\", "/"))

        # Sample ICS file
        ics_file = out_dir / f"{safe_schedule_id}-T17-pitch-001.ics"
        with open(ics_file, "w", encoding="utf-8") as f:
            f.write("BEGIN:VCALENDAR\nVERSION:2.0\nSUMMARY:Repair T17 Pitch\nEND:VCALENDAR\n")
        created_files.append(str(ics_file).replace("\\", "/"))

        data["created_files"] = created_files
        data["work_orders_count"] = len(created_files)
        return data

    def handle_send_alert(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name).copy()
        level = args.get("level", "warning")
        message = args.get("message", "Alert message")
        asset_id = args.get("asset_id", "FAN-T17")

        out_dir = Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = out_dir / "alerts.log"
        now_iso = datetime.now(timezone.utc).isoformat()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{now_iso}] [{level.upper()}] asset={asset_id} msg={message}\n")

        data.update({
            "level": level,
            "message": message,
            "asset_id": asset_id,
            "logged_to": str(log_file).replace("\\", "/"),
        })
        return data

    def handle_promote_schedule(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        data = _load_fixture(tool_name).copy()
        draft_id = args.get("draft_id", "sched-draft-base")
        data["promoted_id"] = draft_id
        data["status"] = "active"
        from src.agent.store import DEFAULT_STORE
        active = DEFAULT_STORE.get_active_schedule()
        draft = DEFAULT_STORE.get_schedule(draft_id)
        if draft:
            data["total_mwh"] = draft["total_mwh"]
            data["total_ron"] = draft["total_ron"]
        if active and active.get("id") != draft_id:
            data["previous_active_id"] = active["id"]
            if draft:
                data["delta_mwh"] = round(draft["total_mwh"] - active["total_mwh"], 2)
                data["delta_ron"] = round(draft["total_ron"] - active["total_ron"], 2)
            else:
                data["delta_mwh"] = round(data.get("total_mwh", 16.1) - active["total_mwh"], 2)
                data["delta_ron"] = round(data.get("total_ron", 7245.0) - active["total_ron"], 2)
        else:
            data.setdefault("delta_mwh", 0.0)
            data.setdefault("delta_ron", 0.0)
        return data


DEFAULT_FIXTURE_BACKEND = FixtureBackend()

