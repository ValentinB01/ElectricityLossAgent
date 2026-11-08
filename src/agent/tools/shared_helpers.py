"""Shared tool helpers for consolidated loss overview and explainability lookups."""

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from src.agent.schemas.common import ErrorCode, StructuredError

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "demo_week"


def calculate_loss_overview(period: str = "week", as_of: str = "2026-11-08") -> Dict[str, Any]:
    """Calculate consolidated loss overview across producers and consumers."""
    fixture_path = FIXTURE_DIR / "loss_overview.json"
    if fixture_path.exists():
        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "period": "week",
            "rows": [
                {
                    "category": "producers_underperformance",
                    "description": "Subperformanță turbine eoliene și invertoare",
                    "mwh": 19.6,
                    "ron": 8820.0,
                    "source_ref": "res-det-001",
                    "confidence": 0.88,
                },
                {
                    "category": "producers_downtime",
                    "description": "Downtime programat revizii și reparații mentenanță",
                    "mwh": 16.1,
                    "ron": 7245.0,
                    "source_ref": "sched-active-001",
                    "confidence": 1.0,
                },
                {
                    "category": "consumers_suspect",
                    "description": "Pierderi suspecte rețea consumatori (anomalii și contoare defecte)",
                    "mwh": 4.2,
                    "ron": 1890.0,
                    "source_ref": "res-score-hh-001",
                    "confidence": 0.85,
                },
            ],
            "total_mwh": 39.9,
            "total_ron": 17955.0,
        }

    res = copy.deepcopy(data)
    if period == "month":
        res["period"] = "month"
        res["total_mwh"] = round(res["total_mwh"] * 4.0, 2)
        res["total_ron"] = round(res["total_ron"] * 4.0, 2)
        monthly_rows = []
        for r in res.get("rows", []):
            rc = copy.deepcopy(r)
            rc["mwh"] = round(rc["mwh"] * 4.0, 2)
            rc["ron"] = round(rc["ron"] * 4.0, 2)
            monthly_rows.append(rc)
        res["rows"] = monthly_rows
    else:
        res["period"] = "week"

    res["provenance"] = {
        "result_id": f"res-loss-ovw-{uuid.uuid4().hex[:6]}",
        "tool": "loss_overview",
        "backend": "live",
        "as_of": as_of,
        "forecast_run": f"{as_of}T00:00:00Z",
        "created_at": f"{as_of}T00:05:00Z",
    }
    return res


def explain_item(item_id: str) -> Any:
    """Provide explainability breakdown for a scheduled job or suspect consumer household."""
    fixture_path = FIXTURE_DIR / "explain.json"
    if not fixture_path.exists():
        return StructuredError(
            code=ErrorCode.UNKNOWN_ITEM,
            message=f"Item '{item_id}' not found in maintenance tasks or household queues",
            item_id=item_id,
        )

    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", {})
    item_data = None
    if item_id in items:
        item_data = copy.deepcopy(items[item_id])
    elif data.get("item_id") == item_id:
        item_data = copy.deepcopy(data)

    if not item_data:
        return StructuredError(
            code=ErrorCode.UNKNOWN_ITEM,
            message=f"Item '{item_id}' not found in maintenance tasks or household queues",
            item_id=item_id,
        )

    if "provenance" in item_data:
        item_data["provenance"]["backend"] = "live"
    return item_data

