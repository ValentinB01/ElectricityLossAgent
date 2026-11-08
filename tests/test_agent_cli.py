"""Golden-output and functional tests for deterministic CLI command router."""

from pathlib import Path
import pytest

from src.agent.actions import ActionGate
from src.agent.cli import AgentCLI, DETERMINISTIC_BANNER, run_cli
import src.agent.tools  # Register tools
from src.agent.store import AuditStore


@pytest.fixture
def tmp_cli(tmp_path):
    db_file = tmp_path / "cli_test_agent.db"
    store = AuditStore(db_path=db_file)
    gate = ActionGate(store=store)
    return AgentCLI(gate=gate, store=store)


def test_cli_plan_fantanele(tmp_cli):
    """Verify 'plan fantanele' calls tasks, costmap and optimizer and renders calendar."""
    out = tmp_cli.execute("plan fantanele")
    assert "=== PLAN MENTENANȚĂ ACTIV: FANTANELE ===" in out
    assert "Plan ID:" in out
    assert "Cost total oportunitate:" in out
    assert "Program intervenții echipe:" in out
    assert "FAN-T17" in out
    assert "T17-pitch-001" in out


def test_cli_explain_deterministic(tmp_cli):
    """Verify 'explain T17-pitch-001' is deterministic and renders cost breakdown and alternatives."""
    out1 = tmp_cli.execute("explain T17-pitch-001")
    out2 = tmp_cli.execute("explain T17-pitch-001")

    # Identical output on repeated runs
    assert out1 == out2
    assert "=== EXPLICARE DECIZIE INTERVENȚIE: T17-pitch-001 ===" in out1
    assert "FAN-T17" in out1
    assert "Defalcare costuri:" in out1
    assert "Pierdere producție (oportunitate):" in out1
    assert "Alternative analizate:" in out1
    assert "| Dată | Fereastră | Echipă |" in out1


def test_cli_explain_household(tmp_cli):
    """Verify 'explain HH-FAULTY-01' renders peer comparison and top factors."""
    out = tmp_cli.execute("explain HH-FAULTY-01")
    assert "=== EXPLICARE SUSPICIUNE ANOMALIE: HH-FAULTY-01 ===" in out
    assert "Probabilitate fraudă:" in out
    assert "Comparație grup similar" in out
    assert "Factori principali de influență:" in out


def test_cli_crew_off(tmp_cli):
    """Verify 'crew-off crew-2 2026-11-12' produces draft and cost differences."""
    out = tmp_cli.execute("crew-off crew-2 2026-11-12")
    assert "=== PLAN MENTENANȚĂ DRAFT (ECHIPĂ INDISPONIBILĂ) ===" in out
    assert "crew-2 indisponibilă la 2026-11-12" in out
    assert "Diferență față de planul activ:" in out
    assert "Variație pierdere producție: 1.2 MWh" in out
    assert "Variație cost: 540.0 RON" in out
    assert "Intervenții reprogramate:" in out
    assert "promote" in out


def test_cli_losses_overview(tmp_cli):
    """Verify 'losses' and 'losses week' / 'losses month' render categories and contributors."""
    out_default = tmp_cli.execute("losses")
    assert "=== PRIVIRE DE ANSAMBLU ASUPRA PIERDERILOR (WEEK) ===" in out_default
    assert "Subperformanță turbine eoliene" in out_default
    assert "Downtime programat" in out_default
    assert "Pierderi suspecte rețea consumatori" in out_default

    out_month = tmp_cli.execute("losses month")
    assert "=== PRIVIRE DE ANSAMBLU ASUPRA PIERDERILOR (MONTH) ===" in out_month


def test_cli_underperf(tmp_cli):
    """Verify 'underperf fantanele' and 'underperf FAN-T17' detect underperforming assets."""
    out_site = tmp_cli.execute("underperf fantanele")
    assert "=== DETECȚIE SUBPERFORMANȚĂ ECHIPAMENTE ===" in out_site
    assert "FAN-T17" in out_site
    assert "pitch_fault" in out_site
    assert "REPAIR" in out_site

    out_asset = tmp_cli.execute("underperf FAN-T17")
    assert "FAN-T17" in out_asset


def test_cli_costmap(tmp_cli):
    """Verify 'costmap fantanele 7' renders cheapest windows."""
    out = tmp_cli.execute("costmap fantanele 7")
    assert "=== HARTĂ COSTURI DE OPORTUNITATE (FERESTRE OPTIME) ===" in out
    assert "fantanele" in out
    assert "Cele mai avantajoase ferestre de intervenție:" in out
    assert "FAN-T17" in out


def test_cli_weather(tmp_cli):
    """Verify 'weather fantanele 7' renders weather summary."""
    out = tmp_cli.execute("weather fantanele 7")
    assert "=== PROGNOZĂ METEOROLOGICĂ: FANTANELE ===" in out
    assert "Rezumat intervale orare:" in out
    assert "Vânt" in out


def test_cli_households(tmp_cli):
    """Verify 'households 5' renders top scored households."""
    out = tmp_cli.execute("households 5")
    assert "=== CLASAMENT CONSUMATORI CU RISC DE PIERDERI / FRAUDĂ ===" in out
    assert "Gospodării cu scor ridicat:" in out
    assert "HH-" in out


def test_cli_inspections(tmp_cli):
    """Verify 'inspections crew-1' renders inspection queue."""
    out = tmp_cli.execute("inspections crew-1")
    assert "=== LISTĂ INSPECȚII PE TEREN" in out
    assert "Program inspecții:" in out
    assert "HH-THEFT-01" in out


def test_cli_tasks(tmp_cli):
    """Verify 'tasks' renders maintenance task list."""
    out = tmp_cli.execute("tasks")
    assert "=== SARCINI DE MENTENANȚĂ ÎNREGISTRATE ===" in out
    assert "T17-pitch-001" in out


def test_cli_production_and_consumption(tmp_cli):
    """Verify 'production fantanele' and 'consumption HH-FAULTY-01'."""
    prod_out = tmp_cli.execute("production fantanele")
    assert "=== PRODUCȚIE AȘTEPTATĂ (PROGNOZĂ) ===" in prod_out
    assert "P50 =" in prod_out

    cons_out = tmp_cli.execute("consumption HH-FAULTY-01")
    assert "=== CONSUM AȘTEPTAT (PROGNOZĂ CONSUMATORI) ===" in cons_out
    assert "HH-FAULTY-01" in cons_out


def test_cli_action_gate_flow(tmp_cli):
    """Verify write command creates pending action, confirm executes once, and reject cancels."""
    # 1. Propose alert
    alert_out = tmp_cli.execute('alert warning FAN-T17 "Pierdere 1.4 MWh/zi"')
    assert "=== ACȚIUNE ÎN AȘTEPTARE (NECESITĂ CONFIRMARE) ===" in alert_out
    assert "ID Acțiune: act-" in alert_out
    assert "send_alert" in alert_out

    # Extract action ID
    lines = alert_out.splitlines()
    id_line = next(l for l in lines if l.startswith("ID Acțiune:"))
    act_id = id_line.split(":", 1)[1].strip()

    # 2. Check pending list
    pend_out = tmp_cli.execute("pending")
    assert "=== ACȚIUNI ÎN AȘTEPTARE ===" in pend_out
    assert act_id in pend_out

    # 3. Confirm action
    conf_out = tmp_cli.execute(f"confirm {act_id}")
    assert "=== ACȚIUNE CONFIRMATĂ ȘI EXECUTATĂ ===" in conf_out
    assert act_id in conf_out

    # Verify side effect: alert logged to results/alerts.log with ISO timestamp
    log_content = Path("results/alerts.log").read_text(encoding="utf-8")
    assert "Pierdere 1.4 MWh/zi" in log_content
    assert "[WARNING]" in log_content

    # 4. Confirming again returns already executed error
    double_conf = tmp_cli.execute(f"confirm {act_id}")
    assert "=== EROARE ACȚIUNE ===" in double_conf
    assert "already_executed" in double_conf

    # 5. Propose second action and reject it
    alert2_out = tmp_cli.execute('alert info FAN-T01 "Test reject"')
    id_line2 = next(l for l in alert2_out.splitlines() if l.startswith("ID Acțiune:"))
    act_id2 = id_line2.split(":", 1)[1].strip()

    rej_out = tmp_cli.execute(f"reject {act_id2}")
    assert "=== ACȚIUNE RESPINSĂ ===" in rej_out
    assert act_id2 in rej_out


def test_cli_field_report_and_what_if_draft(tmp_cli):
    """Verify report command creates pending action and confirmation triggers what-if draft."""
    cmd = 'report T05-bearing-001 needs_part 5 "Am verificat T05, rulmentul e uzat, piesa vine in 5 zile"'
    report_out = tmp_cli.execute(cmd)
    assert "=== ACȚIUNE ÎN AȘTEPTARE (NECESITĂ CONFIRMARE) ===" in report_out
    assert "update_from_field_report" in report_out

    act_id = next(l for l in report_out.splitlines() if l.startswith("ID Acțiune:")).split(":", 1)[1].strip()

    conf_out = tmp_cli.execute(f"confirm {act_id}")
    assert "=== ACȚIUNE CONFIRMATĂ ȘI EXECUTATĂ ===" in conf_out
    assert "Plan draft generat automat în urma raportului:" in conf_out
    assert "sched-draft-t05-defer" in conf_out


def test_cli_bad_arguments_handling(tmp_cli):
    """Verify invalid or missing arguments display validation error and syntax usage."""
    # Missing site in plan
    err_plan = tmp_cli.execute("plan")
    assert "Eroare la comanda 'plan':" in err_plan
    assert "Sintaxă corectă:" in err_plan
    assert "plan <site>" in err_plan

    # Bad site
    err_site = tmp_cli.execute("plan non_existent_site")
    assert "nu este recunoscut" in err_site

    # Missing date in crew-off
    err_crew = tmp_cli.execute("crew-off crew-2")
    assert "Eroare la comanda 'crew-off':" in err_crew
    assert "crew-off <crew> <date>" in err_crew

    # Missing args in alert
    err_alert = tmp_cli.execute("alert warning")
    assert "Eroare la comanda 'alert':" in err_alert

    # Unknown command
    err_unknown = tmp_cli.execute("non_existent_command_xyz")
    assert "Comandă necunoscută" in err_unknown

    # Schema range validation error (horizon_days > 14)
    err_horizon = tmp_cli.execute("costmap fantanele 99")
    assert "Eroare la comanda 'costmap':" in err_horizon
    assert "less than or equal to 14" in err_horizon

    # Past date validation error
    err_past = tmp_cli.execute("crew-off crew-2 2020-01-01")
    assert "Eroare la comanda 'crew-off':" in err_past
    assert "is in the past" in err_past

    # Unknown asset validation error
    err_asset = tmp_cli.execute("underperf FAN-UNKNOWN-99")
    assert "Eroare la comanda 'underperf':" in err_asset
    assert "Asset 'FAN-UNKNOWN-99' does not exist" in err_asset


def test_cli_optimize_command(tmp_cli):
    """Verify 'optimize' command runs optimize_maintenance directly."""
    out = tmp_cli.execute("optimize")
    assert "=== PLAN MENTENANȚĂ ACTIV: FANTANELE ===" in out
    assert "Plan ID:" in out


def test_cli_help(tmp_cli):
    """Verify 'help' renders full command reference manual."""
    help_out = tmp_cli.execute("help")
    assert "=== ASISTENT PPC REDUCERE PIERDERI - COMENZI DETERMINISTE ===" in help_out
    assert "plan <site>" in help_out
    assert "explain <item_id>" in help_out
    assert "crew-off <crew> <date>" in help_out
    assert "losses [week|month]" in help_out
    assert "confirm <action_id>" in help_out


def test_cli_deterministic_mode_banner(tmp_cli):
    """Verify banner inclusion when running in deterministic mode."""
    out = tmp_cli.execute("losses week", banner=True)
    assert DETERMINISTIC_BANNER in out
    assert "=== PRIVIRE DE ANSAMBLU ASUPRA PIERDERILOR (WEEK) ===" in out


def test_cli_direct_tool_name_aliases(tmp_cli):
    """Verify that entering exact tool names routes to the proper command."""
    weather_out = tmp_cli.execute("get_weather_forecast fantanele")
    assert "=== PROGNOZĂ METEOROLOGICĂ: FANTANELE ===" in weather_out

    tasks_out = tmp_cli.execute("list_tasks")
    assert "=== SARCINI DE MENTENANȚĂ ÎNREGISTRATE ===" in tasks_out
