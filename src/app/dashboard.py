"""Streamlit dashboard: operational panels beside the agent chat (design D13).

Run the service first (`make agent`), then `make demo`. Panels call `/tools/*`
directly, so they keep working with no LLM key.
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.app.client import AgentClient, AgentError  # noqa: E402

SITES = {"fantanele": ("Fântânele", "FAN-"), "calugareni": ("Călugăreni", "CAL-")}
DEMO_LABEL = "date demonstrative"
WEATHER_ATTR = "Weather data by Open-Meteo.com"
DISCLOSURE = {
    "fantanele": "Comportamentul turbinelor din Fântânele este redat din datele Kelmarsh, sub ID-uri demo (`FAN-T..`).",
    "calugareni": "Datele Călugăreni sunt date publice redate sub ID-uri demo (`CAL-INV..`).",
}
OPEN_STATES = ("pending", "needs_input")
FIELD_STATUSES = ["done", "deferred", "needs_part", "not_found"]


# --- Client and data helpers ---------------------------------------------------------


@st.cache_resource
def get_client() -> AgentClient:
    return AgentClient()


def call_tool(name: str, **args: Any) -> Tuple[Optional[Dict[str, Any]], Optional[AgentError]]:
    """Call a tool through the service; never raises so one failing panel does not break the page."""
    try:
        return get_client().tool(name, args, session_id=st.session_state.get("session_id")), None
    except AgentError as exc:
        return None, exc


def show_error(err: AgentError) -> None:
    extra = ""
    valid = err.payload.get("valid_ids") if err.payload else None
    if valid:
        extra = f" Valori valide: {', '.join(valid[:8])}"
    st.error(f"Eroare `{err.code}`: {err.message}.{extra}")


def trust_caption(res: Dict[str, Any], weather: bool = False) -> None:
    """Trust indicators for a panel: fixture label, stale forecast, provenance, weather attribution."""
    prov = res.get("provenance", {}) or {}
    parts = []
    if prov.get("backend") == "fixture":
        parts.append(f"🧪 {DEMO_LABEL}")
    if res.get("stale"):
        age = res.get("age_hours")
        parts.append("⏳ prognoză veche" + (f" ({age:g} h)" if age is not None else ""))
    parts.append(f"sursă: `{prov.get('tool', '?')}` · `{prov.get('result_id', '?')}` · as_of {prov.get('as_of', '?')}")
    if weather:
        parts.append(WEATHER_ATTR)
    st.caption(" · ".join(parts))


def badge(label: str, color: str) -> None:
    st.badge(label, color=color)


# --- Drill-down (7.3): explanations come from `explain`, never from the LLM ----------------


def render_explanation(item_id: str, key: str) -> None:
    res, err = call_tool("explain", item_id=item_id)
    if err:
        show_error(err)
        return
    st.markdown(f"#### Explicație: `{item_id}`")
    job = res.get("scheduled_job_explanation")
    hh = res.get("household_explanation")
    if job:
        _explain_job(job)
    elif hh:
        _explain_household(hh)
    else:
        st.info("Nu există explicație pentru acest element.")
    trust_caption(res)


def _explain_job(job: Dict[str, Any]) -> None:
    st.markdown(f"Sarcină **{job['task_id']}** pe **{job['asset_id']}**, {job['date']}, echipa `{job['crew']}`")
    mwh, ron = job.get("cost_breakdown_mwh", {}), job.get("cost_breakdown_ron", {})
    rows = [
        {"Componentă": label, "MWh": mwh.get(k), "RON": ron.get(k)}
        for k, label in (("downtime", "Pierdere prin oprire"), ("waiting", "Pierdere prin așteptare"), ("risk", "Risc"), ("total", "Total"))
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    alts = job.get("alternatives") or []
    if alts:
        st.markdown("**Alternative analizate**")
        df = pd.DataFrame(alts).rename(columns={"date": "Data", "total_mwh": "Total MWh", "total_ron": "Total RON", "why_not": "De ce nu"})
        st.dataframe(df, hide_index=True, use_container_width=True)
    if job.get("trace"):
        with st.expander("Urmă de calcul"):
            st.json(job["trace"])


def _explain_household(hh: Dict[str, Any]) -> None:
    sig = {"theft": "posibil consum neînregistrat", "faulty_meter": "posibil contor defect", "uncertain": "semnal incert"}.get(hh["signal_type"], hh["signal_type"])
    st.markdown(f"Gospodăria **{hh['household_id']}** ({hh['area']}): {sig}, probabilitate {hh['theft_probability']:.0%}")
    st.caption("Se programează o verificare; semnalul nu reprezintă o acuzație.")
    actual = pd.DataFrame(hh.get("actual_series") or hh["monthly_actual_kwh"]).assign(serie="Real")
    expected = pd.DataFrame(hh.get("expected_series") or hh["monthly_expected_kwh"]).assign(serie="Așteptat")
    chart_df = pd.concat([actual, expected])
    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:N", title="Luna"),
            y=alt.Y("kwh:Q", title="kWh"),
            color=alt.Color("serie:N", title=None),
            strokeDash=alt.StrokeDash("serie:N", legend=None),
            tooltip=["serie", "month", "kwh"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
    peer = hh["peer_comparison"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Gospodărie ({peer['peer_group_name']})", f"{peer['household_kwh']:g} kWh")
    c2.metric("Media grupului", f"{peer['avg_kwh']:g} kWh")
    c3.metric("Abatere", f"{peer['deviation_pct']:g}%")
    factors = (hh.get("top_3_factors") or hh["top_factors"])[:3]
    st.markdown("**Primii 3 factori**")
    st.dataframe(
        pd.DataFrame(factors).rename(columns={"factor": "Factor", "importance": "Importanță", "description": "Descriere"}),
        hide_index=True,
        use_container_width=True,
        column_config={"Importanță": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0)},
    )


def selected_row(event: Any, df: pd.DataFrame) -> Optional[pd.Series]:
    rows = getattr(getattr(event, "selection", None), "rows", None) or []
    return df.iloc[rows[0]] if rows else None


# --- Panels ---------------------------------------------------------------------------


def panel_losses() -> None:
    period = st.radio("Perioadă", ["week", "month"], format_func=lambda p: {"week": "Săptămână", "month": "Lună"}[p], horizontal=True)
    res, err = call_tool("loss_overview", period=period)
    if err:
        return show_error(err)
    c1, c2 = st.columns(2)
    c1.metric("Total pierderi", f"{res['total_mwh']:,.1f} MWh")
    c2.metric("Valoare", f"{res['total_ron']:,.0f} RON")
    df = pd.DataFrame(res["rows"])
    names = {
        "producers_underperformance": "Producători: subperformanță",
        "producers_downtime": "Producători: opriri",
        "consumers_suspect": "Consumatori: pierderi suspecte",
    }
    df["Categorie"] = df["category"].map(names).fillna(df["category"])
    st.dataframe(
        df[["Categorie", "description", "mwh", "ron", "confidence", "source_ref"]].rename(
            columns={"description": "Descriere", "mwh": "MWh", "ron": "RON", "confidence": "Încredere", "source_ref": "Sursă"}
        ),
        hide_index=True,
        use_container_width=True,
        column_config={"MWh": st.column_config.NumberColumn(format="%.1f"), "RON": st.column_config.NumberColumn(format="%.0f")},
    )
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(x=alt.X("mwh:Q", title="MWh"), y=alt.Y("Categorie:N", title=None, sort="-x"), tooltip=["Categorie", "mwh", "ron"]),
        use_container_width=True,
    )
    low = df[df["confidence"].fillna(1.0) < 0.7]
    if not low.empty:
        st.warning("⚠️ Încredere scăzută pentru: " + ", ".join(low["Categorie"]))
    trust_caption(res)


def asset_layout(assets: List[str]) -> pd.DataFrame:
    """Schematic grid layout: demo assets carry no coordinates, so this is not a geographic map."""
    cols = 6
    return pd.DataFrame([{"asset_id": a, "x": i % cols, "y": -(i // cols)} for i, a in enumerate(sorted(assets))])


def panel_producers() -> None:
    site = st.radio("Parc", list(SITES), format_func=lambda s: SITES[s][0], horizontal=True)
    prefix = SITES[site][1]
    det, err = call_tool("detect_underperformance", site_id=site)
    cost, cerr = call_tool("compute_cost_map", site_id=site, horizon_days=7)
    tasks, terr = call_tool("list_tasks", site_id=site)
    for e in (err, cerr, terr):
        if e:
            show_error(e)
    st.info(DISCLOSURE[site])

    if det:
        detections = [d for d in det["detections"] if d["asset_id"].startswith(prefix)]
        known = {d["asset_id"]: d for d in detections}
        assets = set(known)
        if cost:
            assets |= {e["asset_id"] for e in cost["entries"] if e["asset_id"].startswith(prefix)}
        if tasks:
            assets |= {t["asset_id"] for t in tasks["tasks"] if t["asset_id"].startswith(prefix)}
        layout = asset_layout(sorted(assets))
        layout["pi_72h"] = layout["asset_id"].map(lambda a: known[a]["pi_72h"] if a in known else None)
        layout["cause"] = layout["asset_id"].map(lambda a: known[a]["cause"] if a in known else "fără detecție")
        layout["low_confidence"] = layout["asset_id"].map(lambda a: bool(known[a]["low_confidence"]) if a in known else False)
        layout["loss_mwh_per_day"] = layout["asset_id"].map(lambda a: known[a]["loss_mwh_per_day"] if a in known else None)
        layout["label"] = layout.apply(lambda r: r["asset_id"].split("-", 1)[1] + (" ?" if r["low_confidence"] else ""), axis=1)

        st.markdown("**Parc colorat după `pi_72h`** (schemă, nu hartă geografică; `?` = încredere scăzută)")
        pick = alt.selection_point(name="asset", fields=["asset_id"])
        base = alt.Chart(layout).encode(x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None))
        circles = base.mark_circle(size=900).encode(
            color=alt.Color(
                "pi_72h:Q",
                scale=alt.Scale(domain=[0.8, 1.0], scheme="redyellowgreen", clamp=True),
                legend=alt.Legend(title="pi_72h"),
            ),
            stroke=alt.condition("datum.low_confidence", alt.value("black"), alt.value("white")),
            strokeWidth=alt.condition("datum.low_confidence", alt.value(4), alt.value(1)),
            tooltip=["asset_id", "pi_72h", "cause", "loss_mwh_per_day", "low_confidence"],
        )
        text = base.mark_text(dy=1, fontSize=11, color="black").encode(text="label:N")
        event = st.altair_chart((circles + text).add_params(pick), use_container_width=True, on_select="rerun", key=f"map_{site}")
        if det:
            trust_caption(det)

        picked = None
        sel = getattr(event, "selection", None) or {}
        chosen = (sel.get("asset") if hasattr(sel, "get") else None) or []
        if chosen:
            picked = chosen[0].get("asset_id")
        if picked is None:
            picked = st.selectbox("Sau alege o turbină/invertor", [""] + sorted(assets), key=f"asset_pick_{site}") or None
        if picked:
            if picked in known:
                d = known[picked]
                st.markdown(
                    f"**{picked}**: {d['cause']} (probabilitate {d['cause_probability']:.0%}, încredere {d['confidence']:.0%}), "
                    f"{d['loss_mwh_per_day']:g} MWh/zi, acțiune recomandată: {d['recommended_action']}"
                )
                if d["low_confidence"]:
                    st.warning("Nu sunt sigur: propun o verificare (dronă, citire la distanță sau inginer).")
            asset_tasks = [t for t in (tasks["tasks"] if tasks else []) if t["asset_id"] == picked]
            if asset_tasks:
                tid = asset_tasks[0]["task_id"]
                if len(asset_tasks) > 1:
                    tid = st.selectbox("Sarcină", [t["task_id"] for t in asset_tasks], key=f"asset_task_{site}")
                render_explanation(tid, key=f"exp_map_{site}")
            else:
                st.info("Nicio sarcină de mentenanță asociată acestui echipament.")

    if cost:
        st.markdown("**Harta costurilor, 7 zile** (cea mai ieftină fereastră lucrabilă pe echipament și zi)")
        cdf = pd.DataFrame(cost["entries"])
        cdf = cdf[cdf["asset_id"].str.startswith(prefix)]
        if cdf.empty:
            st.info("Nu există intrări de cost pentru acest parc.")
        else:
            best = cdf[cdf["workable"]].sort_values("cost_mwh").groupby(["asset_id", "date"], as_index=False).first()
            heat = (
                alt.Chart(best)
                .mark_rect()
                .encode(
                    x=alt.X("date:O", title="Data"),
                    y=alt.Y("asset_id:N", title=None),
                    color=alt.Color("cost_mwh:Q", title="MWh", scale=alt.Scale(scheme="orangered")),
                    tooltip=["asset_id", "date", "hour", "cost_mwh", "cost_ron", "wind_speed_ms"],
                )
            )
            st.altair_chart(heat, use_container_width=True)
        trust_caption(cost, weather=True)


def jobs_frame(schedule: Optional[Dict[str, Any]], other: Optional[Dict[str, Any]], label: str) -> pd.DataFrame:
    sched_dict = schedule or {}
    payload = sched_dict.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    elif not isinstance(payload, dict):
        payload = {}
    jobs = payload.get("jobs") or sched_dict.get("jobs", [])

    other_dict = other or {}
    other_payload = other_dict.get("payload")
    if isinstance(other_payload, str):
        try:
            other_payload = json.loads(other_payload)
        except Exception:
            other_payload = {}
    elif not isinstance(other_payload, dict):
        other_payload = {}
    other_jobs = other_payload.get("jobs") or other_dict.get("jobs", [])
    ref = {j["task_id"]: j for j in other_jobs if isinstance(j, dict) and "task_id" in j}

    rows = []
    for j in jobs:
        if not isinstance(j, dict):
            continue
        change = ""
        if other is not None:
            r = ref.get(j.get("task_id", ""))
            if r is None:
                change = "nou"
            elif (r.get("date"), r.get("start_hour"), r.get("crew")) != (j.get("date"), j.get("start_hour"), j.get("crew")):
                change = "mutat"

        mwh_total = j.get("cost_breakdown_mwh", {}).get("total", 0.0) if isinstance(j.get("cost_breakdown_mwh"), dict) else j.get("cost_breakdown_mwh", 0.0)
        ron_total = j.get("cost_breakdown_ron", {}).get("total", 0.0) if isinstance(j.get("cost_breakdown_ron"), dict) else j.get("cost_breakdown_ron", 0.0)

        rows.append(
            {
                "Schimbare": change,
                "Sarcină": j.get("task_id", ""),
                "Echipament": j.get("asset_id", ""),
                "Data": j.get("date", ""),
                "Ora": j.get("start_hour", 0),
                "Durată (h)": j.get("duration_hours", 0.0),
                "Echipa": j.get("crew", ""),
                "Cost MWh": mwh_total,
                "Cost RON": ron_total,
            }
        )
    return pd.DataFrame(rows)


def highlight_changes(row: pd.Series) -> List[str]:
    color = {"mutat": "background-color: #fff3bf", "nou": "background-color: #d3f9d8"}.get(row["Schimbare"], "")
    return [color] * len(row)


def panel_calendar() -> None:
    client = get_client()
    try:
        active = client.active_schedule()
        if active is None:  # seed the baseline plan, then reload
            call_tool("optimize_maintenance")
            active = client.active_schedule()
        draft = client.draft_schedule()
    except AgentError as exc:
        return show_error(exc)
    if active is None:
        st.info("Nu există un plan activ.")
        return
    if draft is not None and draft["id"] == active["id"]:
        draft = None
    if draft is not None:
        changed = jobs_frame(draft, active, "")
        if draft["total_mwh"] == active["total_mwh"] and (changed.empty or (changed["Schimbare"] == "").all()):
            draft = None  # identical to the active plan: nothing to compare

    colcfg = {
        "Cost MWh": st.column_config.NumberColumn(format="%.1f"),
        "Cost RON": st.column_config.NumberColumn(format="%.0f"),
        "Ora": st.column_config.NumberColumn(format="%d:00"),
    }
    sel_df, sel_event = None, None

    def table(title: str, sched: Dict[str, Any], other: Optional[Dict[str, Any]], key: str):
        df = jobs_frame(sched, other, title)
        st.markdown(f"**{title}** `{sched['id']}`: {sched['total_mwh']:.1f} MWh · {sched['total_ron']:,.0f} RON")
        ev = st.dataframe(
            df.style.apply(highlight_changes, axis=1),
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            key=key,
            column_config=colcfg,
        )
        return df, ev

    if draft is None:
        st.caption("Nu există un draft de comparat. Întreabă agentul, de exemplu „Echipa 2 lipsește joi”.")
        df_a, ev_a = table("Plan activ", active, None, "cal_active")
        sel_df, sel_event = df_a, ev_a
    else:
        dm, dr = draft["total_mwh"] - active["total_mwh"], draft["total_ron"] - active["total_ron"]
        m1, m2 = st.columns(2)
        m1.metric("Diferență pierderi (draft − activ)", f"{dm:+.1f} MWh", delta=f"{dm:+.1f}", delta_color="inverse")
        m2.metric("Diferență valoare", f"{dr:+,.0f} RON", delta=f"{dr:+,.0f}", delta_color="inverse")
        left, right = st.columns(2)
        with left:
            df_a, ev_a = table("Plan activ", active, None, "cal_active")
        with right:
            df_d, ev_d = table("Draft", draft, active, "cal_draft")
        st.caption("Galben = lucrare mutată · verde = lucrare nouă")
        sel_df, sel_event = (df_d, ev_d) if selected_row(ev_d, df_d) is not None else (df_a, ev_a)
        if st.button("Promovează draftul ca plan activ", key=f"promote_{draft['id']}"):
            res, err = call_tool("promote_schedule", draft_id=draft["id"])
            if err:
                show_error(err)
            else:
                st.session_state["notice"] = "Acțiune de promovare creată. Confirmă-o în panoul de chat."
                st.rerun()

    st.caption(f"Costurile depind de prognoza meteo. {WEATHER_ATTR}.")
    trust_caption(active.get("payload", {}))
    row = selected_row(sel_event, sel_df)
    if row is not None:
        render_explanation(str(row["Sarcină"]), key="exp_cal")


def panel_inspections() -> None:
    res, err = call_tool("build_inspection_list")
    if err:
        return show_error(err)
    c1, c2 = st.columns(2)
    c1.metric("Recuperare estimată", f"{res['total_expected_recovery_kwh']:,.0f} kWh")
    c2.metric("Valoare estimată", f"{res['total_expected_recovery_ron']:,.0f} RON")
    df = pd.DataFrame(res["inspections"])
    if df.empty:
        return st.info("Nu există inspecții programate.")
    signals = {"theft": "consum neînregistrat", "faulty_meter": "contor defect?", "uncertain": "incert"}
    df["signal"] = df["signal_type"].map(signals).fillna(df["signal_type"])
    show = df[["rank", "household_id", "area", "assigned_crew", "signal", "theft_probability", "monthly_loss_kwh", "expected_recovery_ron", "priority"]]
    event = st.dataframe(
        show.rename(
            columns={
                "rank": "Loc",
                "household_id": "Gospodărie",
                "area": "Zonă",
                "assigned_crew": "Echipa",
                "signal": "Semnal",
                "theft_probability": "Probabilitate",
                "monthly_loss_kwh": "kWh/lună",
                "expected_recovery_ron": "Recuperare RON",
                "priority": "Prioritate",
            }
        ),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="insp_table",
        column_config={"Probabilitate": st.column_config.ProgressColumn(format="%.2f", min_value=0.0, max_value=1.0)},
    )
    st.caption("Se programează verificări; niciun semnal nu reprezintă o acuzație.")
    trust_caption(res)
    row = selected_row(event, df)
    if row is not None:
        render_explanation(str(row["household_id"]), key="exp_insp")
    else:
        st.caption("Selectează o gospodărie pentru explicație.")


# --- Chat (7.2) ------------------------------------------------------------------------


def render_sources(msg: Dict[str, Any], backends: Dict[str, str]) -> None:
    calls = msg.get("tool_calls") or []
    with st.expander(f"Surse ({len(calls)} unelte)"):
        if not calls:
            st.caption("Răspuns fără apeluri de unelte.")
        for tc in calls:
            tool = tc.get("tool", "?")
            st.markdown(
                f"- `{tool}` · argumente `{json.dumps(tc.get('args', {}), ensure_ascii=False)}` · backend `{backends.get(tool, '?')}` · rezultat `{tc.get('result_id') or '-'}`"
                + (" · ⚠️ eroare" if tc.get("is_error") else "")
            )
        extra = [r for r in msg.get("tool_result_ids", []) if r not in {tc.get("result_id") for tc in calls}]
        if extra:
            st.caption("Alte rezultate: " + ", ".join(f"`{r}`" for r in extra))


def render_grounding(g: Any) -> None:
    if isinstance(g, dict) and g.get("is_grounded") is False:
        values = ", ".join(g.get("unmatched_numbers") or []) or "necunoscute"
        st.warning(f"⚠️ Valori neverificate în răspuns: {values}")
    elif isinstance(g, dict) and g.get("is_grounded"):
        st.caption("✅ Cifrele din răspuns provin din rezultatele uneltelor")


def render_action_card(action: Dict[str, Any]) -> None:
    aid, tool, state = action["id"], action["tool"], action["state"]
    args = action.get("args") or {}
    with st.container(border=True):
        st.markdown(f"**Acțiune în așteptare `{aid}`** · `{tool}`")
        st.write(action.get("summary", ""))
        if tool == "update_from_field_report":
            if state == "needs_input":
                st.warning(f"Încredere efectivă {args.get('effective_confidence', '?')}: verifică și corectează câmpurile.")
            with st.form(f"form_{aid}"):
                task_id = st.text_input("Sarcină", args.get("task_id", ""))
                cur = args.get("status", "done")
                status = st.selectbox("Status", FIELD_STATUSES, index=FIELD_STATUSES.index(cur) if cur in FIELD_STATUSES else 0)
                defer = st.number_input("Zile amânare", min_value=0, value=int(args.get("defer_days") or 0), step=1)
                finding = st.text_input("Constatare", args.get("new_finding") or "")
                if st.form_submit_button("Aplică corecțiile"):
                    edits: Dict[str, Any] = {"task_id": task_id, "status": status, "new_finding": finding or None}
                    edits["defer_days"] = int(defer) if defer else None
                    try:
                        get_client().patch_action(aid, edits)
                        st.rerun()
                    except AgentError as exc:
                        show_error(exc)
        with st.expander("Argumente exacte"):
            st.code(json.dumps(args, ensure_ascii=False, indent=2), language="json")
        c1, c2 = st.columns(2)
        if c1.button("Confirmă", key=f"ok_{aid}", type="primary", disabled=state == "needs_input"):
            try:
                res = get_client().confirm(aid, action.get("args_hash"))
                note = f"Acțiunea `{aid}` a fost executată."
                hook = (res.get("post_hooks") or {}).get("what_if_draft")
                if hook:
                    note += f" Draft nou `{hook['schedule_id']}`: {hook['delta_mwh']:+g} MWh, {hook['delta_ron']:+g} RON."
                promoted = (res.get("post_hooks") or {}).get("promoted_schedule")
                if promoted:
                    note += f" Planul `{promoted['schedule_id']}` a fost promovat ca plan activ ({promoted.get('delta_mwh', 0.0):+g} MWh, {promoted.get('delta_ron', 0.0):+g} RON)."
                st.session_state["notice"] = note
            except AgentError as exc:
                st.session_state["notice"] = f"Confirmarea a eșuat ({exc.code}): {exc.message}"
            st.rerun()
        if c2.button("Respinge", key=f"no_{aid}"):
            try:
                get_client().reject(aid)
                st.session_state["notice"] = f"Acțiunea `{aid}` a fost respinsă."
            except AgentError as exc:
                st.session_state["notice"] = f"Respingerea a eșuat ({exc.code}): {exc.message}"
            st.rerun()


def panel_chat(health: Dict[str, Any]) -> None:
    st.subheader("Chat")
    if notice := st.session_state.pop("notice", None):
        st.success(notice)
    if health["mode"] != "llm":
        st.caption("Mod determinist: folosește comenzi, de ex. `help`, `losses`, `plan fantanele`, `crew-off crew-2 2026-11-12`.")
    box = st.container(height=520)
    with box:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=False)  # markdown only, no raw HTML
                if msg["role"] == "assistant":
                    render_grounding(msg.get("grounding"))
                    render_sources(msg, health["backends"])
        try:
            sess_id = st.session_state.get("session_id")
            client_actions = get_client().actions(sess_id) if sess_id else get_client().actions()
            if not client_actions and sess_id:
                client_actions = get_client().actions()
            open_actions = [a for a in client_actions if a.get("state") in OPEN_STATES]
        except AgentError:
            open_actions = []
        for action in open_actions:
            render_action_card(action)
    if prompt := st.chat_input("Scrie un mesaj..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        try:
            with st.spinner("Agentul lucrează..."):
                resp = get_client().chat(prompt, st.session_state.get("session_id"))
            st.session_state["session_id"] = resp["session_id"]
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": resp["answer"],
                    "tool_calls": resp.get("tool_calls", []),
                    "tool_result_ids": resp.get("tool_result_ids", []),
                    "grounding": resp.get("grounding"),
                    "mode": resp.get("mode"),
                }
            )
        except AgentError as exc:
            st.session_state["messages"].append({"role": "assistant", "content": f"Eroare `{exc.code}`: {exc.message}"})
        st.rerun()


# --- Page ------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="PPC Loss Agent", layout="wide")
    st.session_state.setdefault("session_id", f"sess_{uuid.uuid4().hex[:8]}")
    st.session_state.setdefault("messages", [])
    try:
        health = get_client().health()
    except AgentError as exc:
        st.title("PPC Loss Agent")
        st.error(f"Serviciul agent nu este disponibil ({exc.message}). Pornește-l cu `make agent`, apoi reîncarcă pagina.")
        st.stop()
        return

    h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
    h1.title("PPC Loss Agent")
    with h2:
        badge("mod: LLM" if health["mode"] == "llm" else "mod: determinist", "green" if health["mode"] == "llm" else "orange")
    with h3:
        if "fixture" in health["backends"].values():
            badge(DEMO_LABEL, "violet")
    with h4:
        badge("prognoză: veche" if health.get("stale_forecast") else "prognoză: ok", "red" if health.get("stale_forecast") else "green")

    left, right = st.columns([3, 2], gap="large")
    with left:
        t_loss, t_prod, t_cal, t_insp = st.tabs(["Pierderi", "Producători", "Calendar", "Inspecții"])
        with t_loss:
            panel_losses()
        with t_prod:
            panel_producers()
        with t_cal:
            panel_calendar()
        with t_insp:
            panel_inspections()
    with right:
        panel_chat(health)

    st.divider()
    st.caption(f"{WEATHER_ATTR} · Fântânele/Călugăreni: date publice redate sub ID-uri demo")


if __name__ == "__main__":
    main()
