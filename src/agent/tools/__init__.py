"""Agent tools package: import all 15 tools to ensure registration in registry."""

from src.agent.tools.producers import (
    compute_cost_map,
    detect_underperformance,
    get_weather_forecast,
    list_tasks,
    optimize_maintenance,
    predict_expected_production,
)
from src.agent.tools.consumers import (
    build_inspection_list,
    predict_expected_consumption,
    score_households,
)
from src.agent.tools.shared import (
    create_work_orders,
    explain,
    loss_overview,
    promote_schedule,
    send_alert,
    update_from_field_report,
)

__all__ = [
    # Producers
    "get_weather_forecast",
    "predict_expected_production",
    "detect_underperformance",
    "compute_cost_map",
    "list_tasks",
    "optimize_maintenance",
    # Consumers
    "predict_expected_consumption",
    "score_households",
    "build_inspection_list",
    # Shared
    "loss_overview",
    "explain",
    "update_from_field_report",
    "create_work_orders",
    "send_alert",
    "promote_schedule",
]

