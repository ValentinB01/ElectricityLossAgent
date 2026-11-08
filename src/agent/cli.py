"""CLI command router and deterministic execution engine with Romanian Jinja2 templates."""

import json
from pathlib import Path
import shlex
from typing import Any, Dict, List, Optional

from src.agent.actions import ActionGate, action_gate as default_gate
from src.agent.config import load_agent_config
from src.agent.registry import registry as default_registry
from src.agent.schemas.common import StructuredError
from src.agent.store import AuditStore, DEFAULT_STORE
from src.agent.templates import render_template

DETERMINISTIC_BANNER = "[MOD DETERMINIST ACTIV - Fără cheie LLM configurată]"


class AgentCLI:
    """Deterministic command router mapping user commands to registry tools and Jinja2 templates."""

    def __init__(
        self,
        gate: Optional[ActionGate] = None,
        store: Optional[AuditStore] = None,
        session_id: Optional[str] = None,
    ):
        self.store = store or DEFAULT_STORE
        self.gate = gate or default_gate
        self.registry = default_registry
        self.config = load_agent_config()
        self.session_id = session_id or self.store.create_session(
            mode="deterministic",
            as_of=self.config.as_of,
        )

    def execute(self, command_line: str, banner: bool = False) -> str:
        """Parse and execute a command string, returning formatted deterministic Romanian text."""
        raw_cmd = command_line.strip()
        if not raw_cmd:
            return ""

        try:
            tokens = shlex.split(raw_cmd)
        except ValueError as err:
            return render_template(
                "error",
                command=raw_cmd,
                message=f"Eroare de sintaxă la parsarea comenzii: {str(err)}",
                usage="Verificați ghilimelele sau caracterele speciale.",
            )

        if not tokens:
            return ""

        cmd = tokens[0].lower()
        args = tokens[1:]

        # Dispatch command
        handler_name = f"_handle_{cmd.replace('-', '_')}"
        handler = getattr(self, handler_name, None)

        if not handler:
            # Check if command is a direct registry tool name
            if cmd in self.registry._tools:
                output = self._handle_direct_tool(cmd, args)
            else:
                output = render_template(
                    "error",
                    command=cmd,
                    message=f"Comandă necunoscută '{cmd}'. Tastați 'help' pentru lista comenzilor disponibile.",
                    usage="help",
                )
        else:
            try:
                output = handler(args)
            except Exception as e:
                output = render_template(
                    "error",
                    command=cmd,
                    message=f"A apărut o eroare neașteptată la rularea comenzii: {str(e)}",
                    usage=f"help {cmd}",
                )

        if banner:
            output = f"{DETERMINISTIC_BANNER}\n\n{output}"

        return output

    # --- Command Handlers ---

    def _handle_plan(self, args: List[str]) -> str:
        """Plan command: list_tasks -> compute_cost_map -> optimize_maintenance for site."""
        if not args:
            return render_template(
                "error",
                command="plan",
                message="Lipsește parametrul obligatoriu <site>.",
                usage="plan <site> (ex: plan fantanele, plan calugareni)",
            )

        site_id = args[0].lower()
        if site_id not in ("fantanele", "calugareni"):
            return render_template(
                "error",
                command="plan",
                message=f"Site-ul '{site_id}' nu este recunoscut. Site-uri valide: fantanele, calugareni.",
                usage="plan <site>",
            )

        # 1. list_tasks
        tasks_res = self.registry.execute("list_tasks", {"site_id": site_id}, session_id=self.session_id, store=self.store)
        if tasks_res.is_error:
            return self._format_error("plan", tasks_res.error, "plan <site>")

        # 2. compute_cost_map
        cost_res = self.registry.execute("compute_cost_map", {"site_id": site_id}, session_id=self.session_id, store=self.store)
        if cost_res.is_error:
            return self._format_error("plan", cost_res.error, "plan <site>")

        # 3. optimize_maintenance
        opt_res = self.registry.execute(
            "optimize_maintenance",
            {"site_id": site_id, "constraints": {"crews_per_day": 2}},
            session_id=self.session_id,
            store=self.store,
        )
        if opt_res.is_error:
            return self._format_error("plan", opt_res.error, "plan <site>")

        plan = opt_res.full_result
        return render_template(
            "plan",
            site_id=site_id,
            schedule_id=plan.schedule_id,
            total_mwh=plan.total_mwh,
            total_ron=plan.total_ron,
            as_of=opt_res.provenance.as_of or self.config.as_of,
            jobs=[j.model_dump() for j in plan.jobs],
        )

    def _handle_explain(self, args: List[str]) -> str:
        """Explain command: explain decision for job or household."""
        if not args:
            return render_template(
                "error",
                command="explain",
                message="Lipsește identificatorul (item_id).",
                usage="explain <item_id> (ex: explain T17-pitch-001 sau explain HH-FAULTY-01)",
            )

        item_id = args[0]
        res = self.registry.execute("explain", {"item_id": item_id}, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("explain", res.error, "explain <item_id>")

        expl = res.full_result
        if expl.scheduled_job_explanation:
            return render_template("explain_job", **expl.scheduled_job_explanation.model_dump())
        elif expl.household_explanation:
            return render_template("explain_household", **expl.household_explanation.model_dump())

        return f"Explicare disponibilă pentru {item_id}, dar nu există detalii configurate."

    def _handle_crew_off(self, args: List[str]) -> str:
        """Crew-off command: optimize_maintenance with crew unavailable constraint."""
        if len(args) < 2:
            return render_template(
                "error",
                command="crew-off",
                message="Comanda necesită ambii parametri: <crew> și <date>.",
                usage="crew-off <crew> <date> (ex: crew-off crew-2 2026-11-12 sau crew-off crew-2 joi)",
            )

        crew = args[0]
        date_str = args[1]
        constraints = {"unavailable": [{"crew": crew, "date": date_str}]}

        res = self.registry.execute(
            "optimize_maintenance",
            {"constraints": constraints},
            session_id=self.session_id,
            store=self.store,
        )
        if res.is_error:
            return self._format_error("crew-off", res.error, "crew-off <crew> <date>")

        draft = res.full_result
        return render_template(
            "crew_off",
            schedule_id=draft.schedule_id,
            crew=crew,
            date=date_str,
            total_mwh=draft.total_mwh,
            total_ron=draft.total_ron,
            delta_mwh=draft.delta_mwh,
            delta_ron=draft.delta_ron,
            moved_jobs=draft.moved_jobs,
        )

    def _handle_losses(self, args: List[str]) -> str:
        """Losses command: loss_overview for period."""
        period = args[0].lower() if args else "week"
        if period not in ("week", "month"):
            return render_template(
                "error",
                command="losses",
                message=f"Perioada '{period}' este invalidă. Valori permise: week, month.",
                usage="losses [week|month]",
            )

        res = self.registry.execute("loss_overview", {"period": period}, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("losses", res.error, "losses [week|month]")

        data = res.full_result
        return render_template(
            "losses",
            period=data.period,
            total_mwh=data.total_mwh,
            total_ron=data.total_ron,
            rows=[r.model_dump() for r in data.rows],
        )

    def _handle_underperf(self, args: List[str]) -> str:
        """Underperf command: detect_underperformance for site or assets."""
        if not args:
            return render_template(
                "error",
                command="underperf",
                message="Lipsește ținta analizei (site sau cod echipament).",
                usage="underperf <site|asset...> (ex: underperf fantanele sau underperf FAN-T17)",
            )

        target = args[0]
        if target.lower() in ("fantanele", "calugareni"):
            call_args = {"site_id": target.lower()}
        else:
            call_args = {"asset_ids": args}

        res = self.registry.execute("detect_underperformance", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("underperf", res.error, "underperf <site|asset...>")

        data = res.full_result
        return render_template(
            "underperf",
            site_id=target,
            detections=[d.model_dump() for d in data.detections],
        )

    def _handle_costmap(self, args: List[str]) -> str:
        """Costmap command: compute_cost_map for site."""
        if not args:
            return render_template(
                "error",
                command="costmap",
                message="Lipsește parametrul obligatoriu <site>.",
                usage="costmap <site> [days] (ex: costmap fantanele 7)",
            )

        site_id = args[0].lower()
        call_args: Dict[str, Any] = {"site_id": site_id}
        days_val = 7
        if len(args) > 1:
            try:
                days_val = int(args[1])
                call_args["horizon_days"] = days_val
            except ValueError:
                call_args["horizon_days"] = args[1]
        else:
            call_args["horizon_days"] = 7

        res = self.registry.execute("compute_cost_map", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("costmap", res.error, "costmap <site> [days]")

        data = res.full_result
        from src.agent.tools.producers import cost_map_llm_view
        windows = cost_map_llm_view(data).get("cheapest_workable_windows", [])
        return render_template(
            "costmap",
            site_id=data.site_id,
            days=days_val,
            stale=data.stale,
            age_hours=data.age_hours,
            windows=windows,
        )

    def _handle_weather(self, args: List[str]) -> str:
        """Weather command: get_weather_forecast for site."""
        if not args:
            return render_template(
                "error",
                command="weather",
                message="Lipsește parametrul obligatoriu <site>.",
                usage="weather <site> [days] (ex: weather fantanele 7)",
            )

        site_id = args[0].lower()
        call_args: Dict[str, Any] = {"site_id": site_id}
        days_val = 7
        if len(args) > 1:
            try:
                days_val = int(args[1])
                call_args["days"] = days_val
            except ValueError:
                call_args["days"] = args[1]
        else:
            call_args["days"] = 7

        res = self.registry.execute("get_weather_forecast", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("weather", res.error, "weather <site> [days]")

        data = res.full_result
        return render_template(
            "weather",
            site_id=data.site_id,
            days=days_val,
            stale=data.stale,
            age_hours=data.age_hours,
            hourly=[h.model_dump() for h in data.hourly],
        )

    def _handle_households(self, args: List[str]) -> str:
        """Households command: score_households top N."""
        call_args: Dict[str, Any] = {}
        if args:
            try:
                call_args["top_n"] = int(args[0])
            except ValueError:
                call_args["top_n"] = args[0]

        res = self.registry.execute("score_households", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("households", res.error, "households [top_N]")

        data = res.full_result
        return render_template(
            "households",
            households=[h.model_dump() for h in data.households],
        )

    def _handle_optimize(self, args: List[str]) -> str:
        """Optimize command: optimize_maintenance with optional JSON constraints."""
        constraints = None
        if args:
            raw = " ".join(args)
            try:
                constraints = json.loads(raw)
            except Exception:
                return render_template(
                    "error",
                    command="optimize",
                    message="Constrângerile trebuie să fie un obiect JSON valid.",
                    usage="optimize '{\"crews_per_day\": 2}'",
                )
        res = self.registry.execute(
            "optimize_maintenance",
            {"constraints": constraints or {"crews_per_day": 2}},
            session_id=self.session_id,
            store=self.store,
        )
        if res.is_error:
            return self._format_error("optimize", res.error, "optimize [constraints_json]")

        plan = res.full_result
        return render_template(
            "plan",
            site_id="fantanele",
            schedule_id=plan.schedule_id,
            total_mwh=plan.total_mwh,
            total_ron=plan.total_ron,
            as_of=res.provenance.as_of or self.config.as_of,
            jobs=[j.model_dump() for j in plan.jobs],
        )

    def _handle_inspections(self, args: List[str]) -> str:
        """Inspections command: build_inspection_list for crew."""
        crew_filter = args[0] if args else None
        res = self.registry.execute("build_inspection_list", session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("inspections", res.error, "inspections [crew]")

        data = res.full_result
        if crew_filter:
            target = crew_filter.lower()
            if target in ("crew-1", "echipa-1"):
                target = "echipa-teren-1"
            elif target in ("crew-2", "echipa-2", "metrologie"):
                target = "echipa-metrologie"
            filtered = [i.model_dump() for i in data.inspections if target in i.assigned_crew.lower() or i.assigned_crew.lower() in target]
        else:
            filtered = [i.model_dump() for i in data.inspections]

        return render_template(
            "inspections",
            crew=crew_filter or "toate",
            inspections=filtered,
        )

    def _handle_tasks(self, args: List[str]) -> str:
        """Tasks command: list_tasks."""
        site_id = args[0] if args else None
        call_args = {"site_id": site_id} if site_id else {}
        res = self.registry.execute("list_tasks", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("tasks", res.error, "tasks [site]")

        data = res.full_result
        return render_template("tasks", tasks=[t.model_dump() for t in data.tasks])

    def _handle_production(self, args: List[str]) -> str:
        """Production command: predict_expected_production."""
        site = args[0].lower() if args else "fantanele"
        if site == "fantanele":
            asset_ids = [f"FAN-T{i:02d}" for i in range(1, 21)]
        elif site == "calugareni":
            asset_ids = [f"CAL-INV{i:02d}" for i in range(1, 9)]
        else:
            asset_ids = args
        call_args = {
            "asset_ids": asset_ids,
            "start": "2026-11-08T00:00:00Z",
            "end": "2026-11-15T00:00:00Z",
        }
        res = self.registry.execute("predict_expected_production", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("production", res.error, "production <site>")

        data = res.full_result
        return render_template("production", predictions=[p.model_dump() for p in data.predictions])

    def _handle_consumption(self, args: List[str]) -> str:
        """Consumption command: predict_expected_consumption."""
        hh_id = args[0] if args else "HH-FAULTY-01"
        month = args[1] if len(args) > 1 else "2026-11"
        call_args = {"household_ids": [hh_id], "month": month}
        res = self.registry.execute("predict_expected_consumption", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("consumption", res.error, "consumption <household_id> [month]")

        data = res.full_result
        return render_template("consumption", predictions=[p.model_dump() for p in data.predictions])

    # --- Write Commands (Action Gate) ---

    def _handle_report(self, args: List[str]) -> str:
        """Report command: update_from_field_report (write action)."""
        if len(args) < 3:
            return render_template(
                "error",
                command="report",
                message="Sunt necesari cel puțin 3 parametri: <task_id>, <status> și textul raportului.",
                usage="report <task_id> <status> [defer_days] \"<text>\" (ex: report T05-bearing-001 needs_part 5 \"piesa vine in 5 zile\")",
            )

        task_id = args[0]
        status = args[1]
        defer_days = None
        raw_text_idx = 2

        if len(args) > 3 and args[2].isdigit():
            defer_days = int(args[2])
            raw_text_idx = 3

        text = " ".join(args[raw_text_idx:])
        call_args = {
            "task_id": task_id,
            "status": status,
            "defer_days": defer_days,
            "text": text,
            "raw_text": text,
        }

        res = self.registry.execute("update_from_field_report", call_args, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("report", res.error, "report <task_id> <status> [defer_days] \"<text>\"")

        action = res.full_result
        return render_template("action_pending", **action)

    def _handle_workorders(self, args: List[str]) -> str:
        """Workorders command: create_work_orders (write action)."""
        if not args:
            return render_template(
                "error",
                command="workorders",
                message="Lipsește identificatorul planului <schedule_id>.",
                usage="workorders <schedule_id> (ex: workorders sched-active-001)",
            )

        schedule_id = args[0]
        res = self.registry.execute("create_work_orders", {"schedule_id": schedule_id}, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("workorders", res.error, "workorders <schedule_id>")

        action = res.full_result
        return render_template("action_pending", **action)

    def _handle_alert(self, args: List[str]) -> str:
        """Alert command: send_alert (write action)."""
        if len(args) < 3:
            return render_template(
                "error",
                command="alert",
                message="Comanda necesită 3 parametri: <level>, <asset> și \"<mesaj>\".",
                usage="alert <level> <asset> \"<mesaj>\" (ex: alert warning FAN-T17 \"Pierdere 1.4 MWh/zi\")",
            )

        level = args[0].lower()
        asset_id = args[1]
        message = " ".join(args[2:])

        res = self.registry.execute(
            "send_alert",
            {"level": level, "asset_id": asset_id, "message": message},
            session_id=self.session_id,
            store=self.store,
        )
        if res.is_error:
            return self._format_error("alert", res.error, "alert <level> <asset> \"<mesaj>\"")

        action = res.full_result
        return render_template("action_pending", **action)

    def _handle_promote(self, args: List[str]) -> str:
        """Promote command: promote_schedule (write action)."""
        if not args:
            return render_template(
                "error",
                command="promote",
                message="Lipsește identificatorul planului draft <draft_id>.",
                usage="promote <draft_id> (ex: promote sched-draft-base)",
            )

        draft_id = args[0]
        res = self.registry.execute("promote_schedule", {"draft_id": draft_id}, session_id=self.session_id, store=self.store)
        if res.is_error:
            return self._format_error("promote", res.error, "promote <draft_id>")

        action = res.full_result
        return render_template("action_pending", **action)

    # --- Session & Gate Management ---

    def _handle_pending(self, args: List[str]) -> str:
        """Pending command: list open pending actions."""
        pending_items = self.store.list_pending_actions(state="pending")
        needs_input_items = self.store.list_pending_actions(state="needs_input")
        all_open = pending_items + needs_input_items
        return render_template("pending_list", actions=all_open)

    def _handle_confirm(self, args: List[str]) -> str:
        """Confirm command: execute a pending action."""
        if not args:
            return render_template(
                "error",
                command="confirm",
                message="Lipsește identificatorul acțiunii <action_id>.",
                usage="confirm <action_id> (ex: confirm act-12345678)",
            )

        action_id = args[0]
        res = self.gate.confirm_action(action_id)

        if res["status"] == "error":
            return render_template(
                "action_error",
                action_id=action_id,
                code=res.get("code", "error"),
                message=res.get("message", "Eroare la confirmare"),
            )

        action_db = self.store.get_pending_action(action_id) or {}
        summary = action_db.get("summary", res.get("tool", "acțiune"))
        return render_template(
            "action_confirmed",
            action_id=action_id,
            tool=res.get("tool", ""),
            summary=summary,
            post_hooks=res.get("post_hooks", {}),
        )

    def _handle_reject(self, args: List[str]) -> str:
        """Reject command: reject a pending action."""
        if not args:
            return render_template(
                "error",
                command="reject",
                message="Lipsește identificatorul acțiunii <action_id>.",
                usage="reject <action_id> (ex: reject act-12345678)",
            )

        action_id = args[0]
        res = self.gate.reject_action(action_id)

        if res["status"] == "error":
            return render_template(
                "action_error",
                action_id=action_id,
                code=res.get("code", "error"),
                message=res.get("message", "Eroare la respingere"),
            )

        return render_template(
            "action_rejected",
            action_id=action_id,
            message=res.get("message", f"Acțiunea {action_id} a fost respinsă."),
        )

    def _handle_help(self, args: List[str]) -> str:
        """Help command: display reference manual."""
        return render_template("help")

    # --- Direct Tool Fallback ---

    def _handle_direct_tool(self, tool_name: str, args: List[str]) -> str:
        """Fallback router when user types exact tool name from the registry."""
        tool_alias_map = {
            "get_weather_forecast": self._handle_weather,
            "predict_expected_production": self._handle_production,
            "detect_underperformance": self._handle_underperf,
            "compute_cost_map": self._handle_costmap,
            "list_tasks": self._handle_tasks,
            "optimize_maintenance": self._handle_plan,
            "predict_expected_consumption": self._handle_consumption,
            "score_households": self._handle_households,
            "build_inspection_list": self._handle_inspections,
            "loss_overview": self._handle_losses,
            "explain": self._handle_explain,
            "update_from_field_report": self._handle_report,
            "create_work_orders": self._handle_workorders,
            "send_alert": self._handle_alert,
            "promote_schedule": self._handle_promote,
        }
        handler = tool_alias_map.get(tool_name)
        if handler:
            return handler(args)

        return render_template(
            "error",
            command=tool_name,
            message=f"Unealta '{tool_name}' este înregistrată, dar nu are adaptor direct în CLI.",
            usage="help",
        )

    def _format_error(self, cmd: str, err: Optional[StructuredError], usage: str) -> str:
        """Format a StructuredError into readable Romanian template error output."""
        if not err:
            return render_template("error", command=cmd, message="Eroare necunoscută.", usage=usage)

        msg = err.message
        if err.valid_ids:
            msg += f"\nValori acceptate: {', '.join(err.valid_ids[:10])}"
        return render_template("error", command=cmd, message=msg, usage=usage)


def run_cli(command: str, banner: bool = False, session_id: Optional[str] = None) -> str:
    """Convenience functional interface for executing deterministic CLI commands."""
    cli = AgentCLI(session_id=session_id)
    return cli.execute(command, banner=banner)


def main():
    """Interactive command-line entry point."""
    import sys

    cli = AgentCLI()
    if len(sys.argv) > 1:
        cmd_str = " ".join(sys.argv[1:])
        print(cli.execute(cmd_str))
    else:
        print("=== PPC Loss Agent CLI (Mod determinist) ===")
        print("Tastați 'help' pentru ajutor, 'exit' pentru a încheia.")
        while True:
            try:
                line = input("ppc-agent> ").strip()
                if not line:
                    continue
                if line.lower() in ("exit", "quit", "q"):
                    break
                print(cli.execute(line))
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
