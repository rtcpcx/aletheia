"""
Aletheia — enterprise Streamlit decision-intelligence UI.

This UI deliberately translates backend/statistical concepts into business
language while preserving Aletheia's analytical boundaries:

- raw/mart/analysis data remain read-only from the dashboard;
- weak driver candidates remain visible rather than being silently removed;
- external retrieval is presented as supporting context, never causal proof;
- relative softmax weights are labelled as relative evidence weights;
- complex KPI interactions are not forced into misleading percentage splits;
- margin data remain restricted to Executive / Finance roles.

Navigation model
----------------
1. Regional Command Center (default): one region dropdown, all KPIs at once.
2. KPI Detail: click any KPI card to open the existing deep analysis experience.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd
import streamlit as st

from src import repository

try:  # Optional dependency. The UI still works without Plotly.
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover - deployment dependent
    go = None  # type: ignore[assignment]
    PLOTLY_AVAILABLE = False


# =============================================================================
# App configuration & RBAC
# =============================================================================

st.set_page_config(
    page_title="Aletheia · Decision Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROLES: tuple[str, ...] = ("Executive", "Growth analyst", "Finance")
MARGIN_VISIBLE_ROLES: frozenset[str] = frozenset({"Executive", "Finance"})
FALLBACK_KPIS: tuple[str, ...] = (
    "revenue",
    "conversion_rate",
    "customer_acquisition_cost",
    "stock_availability",
    "churn_rate",
)
FALLBACK_REGIONS: tuple[str, ...] = ("North", "South")

DashboardData = Mapping[str, pd.DataFrame]


KPI_DESCRIPTIONS: dict[str, str] = {
    "revenue": "Total sales value generated. Aletheia explains whether a change came through sales volume, selling price, or their interaction.",
    "conversion_rate": "Share of customer sessions that became orders. Useful for understanding demand quality and funnel performance.",
    "customer_acquisition_cost": "Marketing spend required to acquire one new customer. Lower is generally more efficient, but context matters.",
    "stock_availability": "How consistently products were available relative to their reorder threshold. Drops can constrain fulfilled sales.",
    "churn_rate": "Share of active customers who churned. Aletheia can connect changes to operational signals such as support load and uptime.",
}

KPI_BUSINESS_LABELS: dict[str, str] = {
    "revenue": "Revenue",
    "conversion_rate": "Conversion rate",
    "customer_acquisition_cost": "Customer acquisition cost",
    "stock_availability": "Stock availability",
    "churn_rate": "Churn rate",
}


@dataclass(frozen=True)
class HeaderState:
    confidence: str
    change_label: str
    change_tone: str
    data_status: str
    data_tone: str
    incident_date: str


@dataclass(frozen=True)
class KpiSummary:
    kpi: str
    label: str
    status: str
    status_tone: str
    change: str
    change_tone: str
    confidence: str
    confidence_tone: str
    incident_date: str
    top_driver: str
    recommendation: str
    description: str
    has_detail: bool


@dataclass(frozen=True)
class KpiStructure:
    formula: str
    component_a: str
    component_b: str | None
    decomposition_type: str


# =============================================================================
# Design system
# =============================================================================


def inject_design_system() -> None:
    """Inject a bespoke enterprise visual system using only Streamlit + CSS."""
    st.markdown(
        """
        <style>
        :root {
            --al-bg: #f7f8fb;
            --al-surface: #ffffff;
            --al-surface-subtle: #f9fafb;
            --al-border: #e5e7eb;
            --al-border-strong: #d1d5db;
            --al-text: #111827;
            --al-text-2: #475467;
            --al-text-3: #667085;
            --al-brand: #4f46e5;
            --al-brand-soft: #eef2ff;
            --al-green: #137a4d;
            --al-green-soft: #ecfdf3;
            --al-red: #b42318;
            --al-red-soft: #fef3f2;
            --al-amber: #b54708;
            --al-amber-soft: #fffaeb;
            --al-blue: #175cd3;
            --al-blue-soft: #eff8ff;
            --al-gray-soft: #f2f4f7;
            --al-radius: 10px;
            --al-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }

        html, body, [class*="css"] {
            font-family: Inter, Geist, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                         "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 28% -12%, rgba(79,70,229,0.055), transparent 28rem),
                var(--al-bg);
            color: var(--al-text);
        }
        [data-testid="stHeader"] { background: rgba(247,248,251,.88); }
        [data-testid="stSidebar"] {
            background: #fbfbfd;
            border-right: 1px solid var(--al-border);
        }
        .block-container {
            max-width: 1560px;
            padding-top: 1.25rem;
            padding-bottom: 3rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .25rem;
            background: rgba(255,255,255,.78);
            border: 1px solid var(--al-border);
            padding: .30rem;
            border-radius: 12px;
            box-shadow: var(--al-shadow);
        }
        .stTabs [data-baseweb="tab"] {
            height: 2.55rem;
            padding: 0 .95rem;
            border-radius: 8px;
            color: var(--al-text-2);
            font-weight: 600;
            font-size: .88rem;
        }
        .stTabs [aria-selected="true"] {
            color: var(--al-text) !important;
            background: var(--al-surface) !important;
            box-shadow: 0 1px 2px rgba(0,0,0,.06);
        }

        div[data-baseweb="select"] > div,
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
            border-radius: 9px !important;
            border-color: var(--al-border) !important;
        }
        [data-testid="stForm"] {
            background: var(--al-surface);
            border: 1px solid var(--al-border);
            border-radius: 12px;
            padding: 1rem 1rem .35rem 1rem;
            box-shadow: var(--al-shadow);
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--al-border);
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 1px 2px rgba(0,0,0,.03);
        }
        div.stButton > button {
            border-radius: 9px;
            font-weight: 650;
            border-color: var(--al-border-strong);
        }

        .al-brand { display:flex; align-items:center; gap:.7rem; margin:.25rem 0 1.05rem 0; }
        .al-brand-mark {
            width:34px; height:34px; border-radius:9px; display:grid; place-items:center;
            background:linear-gradient(145deg,#4338ca,#6366f1); color:white; font-weight:800;
            box-shadow:0 4px 12px rgba(79,70,229,.20);
        }
        .al-brand-name { font-weight:780; letter-spacing:-.02em; color:var(--al-text); }
        .al-brand-sub { font-size:.72rem; color:var(--al-text-3); margin-top:-.12rem; }

        .al-header {
            background:linear-gradient(135deg,rgba(255,255,255,.97),rgba(249,250,251,.97));
            border:1px solid var(--al-border); border-radius:14px; padding:1.05rem 1.2rem;
            box-shadow:var(--al-shadow); margin-bottom:1rem;
        }
        .al-breadcrumb { color:var(--al-text-3); font-size:.78rem; font-weight:560; margin-bottom:.42rem; }
        .al-header-row { display:flex; gap:1rem; align-items:center; justify-content:space-between; flex-wrap:wrap; }
        .al-title { font-size:1.48rem; line-height:1.2; font-weight:760; letter-spacing:-.03em; color:var(--al-text); }
        .al-subtitle { color:var(--al-text-2); font-size:.82rem; margin-top:.28rem; max-width:820px; line-height:1.5; }
        .al-chips { display:flex; gap:.45rem; align-items:center; flex-wrap:wrap; }
        .al-chip {
            display:inline-flex; align-items:center; gap:.34rem; border:1px solid transparent;
            border-radius:999px; padding:.31rem .58rem; font-size:.70rem; line-height:1;
            font-weight:720; letter-spacing:.025em; white-space:nowrap;
        }
        .al-chip-neutral { background:var(--al-gray-soft); color:#344054; border-color:#eaecf0; }
        .al-chip-green { background:var(--al-green-soft); color:var(--al-green); border-color:#d1fadf; }
        .al-chip-red { background:var(--al-red-soft); color:var(--al-red); border-color:#fee4e2; }
        .al-chip-amber { background:var(--al-amber-soft); color:var(--al-amber); border-color:#fef0c7; }
        .al-chip-blue { background:var(--al-blue-soft); color:var(--al-blue); border-color:#d1e9ff; }
        .al-chip-brand { background:var(--al-brand-soft); color:#4338ca; border-color:#e0e7ff; }

        .al-section-head {
            display:flex; align-items:flex-end; justify-content:space-between; gap:1rem;
            margin:1.05rem 0 .55rem 0;
        }
        .al-section-title { font-size:1.02rem; font-weight:720; letter-spacing:-.015em; color:var(--al-text); }
        .al-section-kicker { font-size:.75rem; color:var(--al-text-3); margin-top:.12rem; line-height:1.45; }

        .al-card {
            height:100%; background:var(--al-surface); border:1px solid var(--al-border);
            border-radius:12px; padding:1rem 1.05rem; box-shadow:var(--al-shadow);
        }
        .al-metric-label { color:var(--al-text-3); font-size:.67rem; font-weight:720; letter-spacing:.09em; text-transform:uppercase; }
        .al-metric-value { color:var(--al-text); font-size:1.54rem; line-height:1.18; font-weight:760; letter-spacing:-.035em; margin-top:.38rem; }
        .al-metric-note { color:var(--al-text-3); font-size:.73rem; margin-top:.32rem; min-height:1.1rem; line-height:1.45; }
        .al-metric-pill { margin-top:.55rem; }

        .al-kpi-card {
            background:var(--al-surface); border:1px solid var(--al-border); border-radius:13px;
            padding:1.05rem 1.08rem .95rem 1.08rem; box-shadow:var(--al-shadow); min-height:248px;
            margin-bottom:.42rem;
        }
        .al-kpi-top { display:flex; align-items:flex-start; justify-content:space-between; gap:.8rem; }
        .al-kpi-name { font-size:1.04rem; font-weight:740; color:var(--al-text); letter-spacing:-.02em; }
        .al-kpi-desc { color:var(--al-text-3); font-size:.75rem; line-height:1.45; margin:.34rem 0 .8rem 0; min-height:2.2rem; }
        .al-kpi-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.48rem; }
        .al-kpi-cell { background:#fafbfc; border:1px solid #eef0f2; border-radius:9px; padding:.58rem .62rem; }
        .al-kpi-cell-label { color:var(--al-text-3); font-size:.61rem; text-transform:uppercase; letter-spacing:.07em; font-weight:700; }
        .al-kpi-cell-value { color:var(--al-text); font-size:.86rem; margin-top:.19rem; font-weight:680; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .al-kpi-action { margin-top:.7rem; color:var(--al-text-2); font-size:.74rem; line-height:1.42; }

        .al-callout {
            border-radius:10px; border:1px solid var(--al-border); padding:.78rem .9rem;
            margin:.55rem 0 .75rem 0; display:flex; gap:.68rem; align-items:flex-start;
        }
        .al-callout-title { font-weight:700; font-size:.82rem; color:var(--al-text); }
        .al-callout-body { font-size:.78rem; color:var(--al-text-2); margin-top:.12rem; line-height:1.48; }
        .al-callout-info { background:#f8fbff; border-color:#dbeafe; }
        .al-callout-warning { background:#fffcf5; border-color:#fde68a; }
        .al-callout-danger { background:#fff8f7; border-color:#fecaca; }
        .al-callout-success { background:#f7fdf9; border-color:#bbf7d0; }

        .al-empty {
            border:1px dashed var(--al-border-strong); border-radius:12px; background:rgba(255,255,255,.58);
            padding:1.35rem; text-align:center; color:var(--al-text-3); margin:.5rem 0;
        }
        .al-empty-title { color:var(--al-text-2); font-weight:680; font-size:.88rem; margin-bottom:.2rem; }

        .al-narrative {
            background:linear-gradient(135deg,#ffffff,#fafaff); border:1px solid #e4e7ec;
            border-left:3px solid var(--al-brand); border-radius:12px; padding:1.05rem 1.15rem;
            box-shadow:var(--al-shadow);
        }
        .al-narrative-headline { color:var(--al-text); font-size:1.05rem; font-weight:730; letter-spacing:-.015em; margin-bottom:.42rem; }
        .al-narrative-body { color:var(--al-text-2); line-height:1.58; font-size:.88rem; }

        .al-recommendation { background:#111827; color:#f9fafb; border-radius:12px; padding:1rem 1.05rem; box-shadow:var(--al-shadow); }
        .al-recommendation .al-metric-label { color:#9ca3af; }
        .al-recommendation-text { margin-top:.42rem; font-size:.88rem; line-height:1.55; }

        .al-definition {
            background:#fff; border:1px solid var(--al-border); border-radius:10px; padding:.78rem .85rem;
            margin:.4rem 0; box-shadow:0 1px 2px rgba(0,0,0,.03);
        }
        .al-definition-term { font-weight:700; font-size:.8rem; color:var(--al-text); }
        .al-definition-text { font-size:.75rem; color:var(--al-text-2); margin-top:.16rem; line-height:1.46; }

        .al-skeleton {
            height:86px; border-radius:12px; border:1px solid var(--al-border);
            background:linear-gradient(90deg,#f3f4f6 25%,#fafafa 37%,#f3f4f6 63%);
            background-size:400% 100%; animation:al-shimmer 1.3s ease infinite;
        }
        @keyframes al-shimmer { 0% {background-position:100% 0} 100% {background-position:0 0} }
        .al-footnote { color:var(--al-text-3); font-size:.70rem; line-height:1.45; }

        hr { border-color:var(--al-border) !important; }
        h1,h2,h3 { letter-spacing:-.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Utility & terminology helpers
# =============================================================================


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _friendly_name(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return str(value).replace("_", " ").strip().title()


def _kpi_label(kpi: str) -> str:
    return KPI_BUSINESS_LABELS.get(kpi, _friendly_name(kpi))


def _fmt_number(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if abs(numeric) >= 1_000_000:
        return f"{numeric / 1_000_000:.2f}M"
    if abs(numeric) >= 1_000:
        return f"{numeric / 1_000:.2f}K"
    return f"{numeric:.{digits}g}"


def _fmt_percent(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{digits}%}"
    except (TypeError, ValueError):
        return "—"


def _safe_list(callable_obj: Any, fallback: Sequence[str]) -> list[str]:
    try:
        values = list(callable_obj())
        return values if values else list(fallback)
    except Exception:
        return list(fallback)


def _latest_row(frame: pd.DataFrame) -> pd.Series | None:
    return None if frame.empty else frame.iloc[-1]


def _safe_columns(frame: pd.DataFrame, desired: Sequence[str]) -> list[str]:
    return [column for column in desired if column in frame.columns]


def _role_can_see_margin(role: str) -> bool:
    return role in MARGIN_VISIBLE_ROLES


def _persona_for_role(role: str) -> str:
    return "Executive" if role in ("Executive", "Finance") else "Growth analyst"


def _confidence_tone(confidence: str) -> str:
    return {"high": "green", "medium": "amber", "low": "red"}.get(confidence.lower(), "neutral")


def _confidence_plain_english(confidence: str) -> str:
    mapping = {
        "high": "Evidence is comparatively strong and the main data/model guardrails passed.",
        "medium": "The explanation is usable, but at least one source of uncertainty remains.",
        "low": "Do not treat the ranking as a decision by itself. Data quality, model sufficiency, or ambiguity requires validation first.",
    }
    return mapping.get(confidence.lower(), "No confidence assessment is available yet.")


def _evidence_mode_label(value: Any) -> str:
    mapping = {
        "historical_relationship": "Historical pattern",
        "structural_break": "New structural shift",
        "insufficient_evidence": "Not enough evidence",
    }
    return mapping.get(str(value), _friendly_name(value))


def _model_status_label(value: Any) -> str:
    mapping = {
        "fitted": "Model fitted",
        "insufficient_history": "Not enough history",
        "historical_variance_unavailable": "No useful historical variation",
    }
    return mapping.get(str(value), _friendly_name(value))


def _safe_contract_structure(kpi: str, decomposition_type: str = "") -> KpiStructure:
    """Read the real KPI contract when available; fall back to business-safe labels."""
    fallback_components: dict[str, tuple[str, str | None, str]] = {
        "revenue": ("Units sold", "Average selling price", "Revenue = units sold × average selling price"),
        "conversion_rate": ("Orders", "Sessions", "Conversion rate = orders ÷ sessions"),
        "customer_acquisition_cost": ("Marketing spend", "New customers", "Customer acquisition cost = marketing spend ÷ new customers"),
        "stock_availability": ("Stock availability", None, "Stock availability is analyzed directly"),
        "churn_rate": ("Churned customers", "Active customers", "Churn rate = churned customers ÷ active customers"),
    }
    a, b, formula = fallback_components.get(kpi, ("Primary component", "Secondary component", _kpi_label(kpi)))

    try:
        from src.contracts import get_contract

        contract = get_contract(kpi)
        names = [_friendly_name(component.name) for component in contract.components]
        if names:
            a = names[0]
        if len(names) > 1:
            b = names[1]
        elif len(names) == 1:
            b = None
        formula = str(contract.formula)
    except Exception:
        pass

    return KpiStructure(
        formula=formula,
        component_a=a,
        component_b=b,
        decomposition_type=decomposition_type,
    )


def chip(text: str, tone: str = "neutral") -> str:
    allowed = {"neutral", "green", "red", "amber", "blue", "brand"}
    safe_tone = tone if tone in allowed else "neutral"
    return f'<span class="al-chip al-chip-{safe_tone}">{_esc(text)}</span>'


def render_callout(title: str, body: str, *, tone: str = "info", icon: str = "i") -> None:
    allowed = {"info", "warning", "danger", "success"}
    safe_tone = tone if tone in allowed else "info"
    st.markdown(
        f"""
        <div class="al-callout al-callout-{safe_tone}">
            <div aria-hidden="true"><strong>{_esc(icon)}</strong></div>
            <div>
                <div class="al-callout-title">{_esc(title)}</div>
                <div class="al-callout-body">{_esc(body)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="al-empty">
            <div class="al-empty-title">{_esc(title)}</div>
            <div>{_esc(body)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, kicker: str = "", trailing: str = "") -> None:
    trailing_html = f"<div>{trailing}</div>" if trailing else ""
    st.markdown(
        f"""
        <div class="al-section-head">
            <div>
                <div class="al-section-title">{_esc(title)}</div>
                <div class="al-section-kicker">{_esc(kicker)}</div>
            </div>
            {trailing_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    *,
    note: str = "",
    pill_text: str | None = None,
    pill_tone: str = "neutral",
) -> None:
    pill = f'<div class="al-metric-pill">{chip(pill_text, pill_tone)}</div>' if pill_text else ""
    st.markdown(
        f"""
        <div class="al-card">
            <div class="al-metric-label">{_esc(label)}</div>
            <div class="al-metric-value">{_esc(value)}</div>
            <div class="al-metric-note">{_esc(note)}</div>
            {pill}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_definition(term: str, explanation: str) -> None:
    st.markdown(
        f"""
        <div class="al-definition">
            <div class="al-definition-term">{_esc(term)}</div>
            <div class="al-definition-text">{_esc(explanation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Navigation & state
# =============================================================================


def render_sidebar() -> tuple[str, str, list[str]]:
    """The region remains the only analysis selector; KPIs are chosen from the home screen."""
    st.sidebar.markdown(
        """
        <div class="al-brand">
            <div class="al-brand-mark">A</div>
            <div>
                <div class="al-brand-name">Aletheia</div>
                <div class="al-brand-sub">Decision Intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.caption("VIEW CONTEXT")
    role = st.sidebar.selectbox("Persona", list(ROLES), index=0, key="aletheia_role")
    regions = _safe_list(repository.list_regions, FALLBACK_REGIONS)
    kpis = _safe_list(repository.list_kpis, FALLBACK_KPIS)
    region = st.sidebar.selectbox("Region", regions, key="aletheia_region")

    previous_region = st.session_state.get("_aletheia_previous_region")
    if previous_region is not None and previous_region != region:
        st.session_state["selected_kpi_detail"] = None
    st.session_state["_aletheia_previous_region"] = region

    selected = st.session_state.get("selected_kpi_detail")
    if selected and selected not in kpis:
        st.session_state["selected_kpi_detail"] = None
        selected = None

    if selected:
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        if st.sidebar.button("← Regional overview", use_container_width=True):
            st.session_state["selected_kpi_detail"] = None
            st.rerun()
        st.sidebar.caption(f"OPEN KPI · {_kpi_label(str(selected)).upper()}")
    else:
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        st.sidebar.caption("REGIONAL COMMAND CENTER")
        st.sidebar.write("All KPIs are summarized on the home screen. Open a KPI only when you need the detailed explanation.")

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    render_callout(
        "How to read Aletheia",
        "Aletheia ranks evidence-supported explanations. A high evidence weight means a driver ranks strongly among the candidates tested; it does not mean causality has been mathematically proven.",
        tone="info",
        icon="◇",
    )
    return role, region, kpis


def open_kpi_detail(kpi: str) -> None:
    st.session_state["selected_kpi_detail"] = kpi
    st.rerun()


# =============================================================================
# Data loading helpers
# =============================================================================


def _top_evidence_row(data: DashboardData) -> pd.Series | None:
    """Return the business-priority driver for the latest incident.

    The action layer first chooses the KPI component that contributed most to
    the KPI movement, then ranks drivers within that component. This avoids
    comparing component-local evidence weights as if they were globally
    comparable. Older bundles without action_context fall back to the previous
    evidence ordering.
    """
    evidence = data.get("driver_evidence", pd.DataFrame())
    if evidence.empty:
        return None

    working = evidence.copy()
    if "window_start" in working.columns:
        latest_window = working["window_start"].max()
        working = working[working["window_start"] == latest_window]

    bundle = _load_latest_bundle(data)
    if isinstance(bundle, dict):
        decision = bundle.get("decision", {})
        action = decision.get("action_context", {}) if isinstance(decision, dict) else {}
        if isinstance(action, dict):
            explanation_driver = str(
                action.get("leading_explanation_driver")
                or action.get("primary_driver")
                or ""
            ).strip()
            primary_component = str(action.get("primary_component") or "").strip()
            if explanation_driver and "driver_name" in working.columns:
                match = working[working["driver_name"].astype(str) == explanation_driver]
                if primary_component and "explains_component" in match.columns:
                    component_match = match[
                        match["explains_component"].astype(str) == primary_component
                    ]
                    if not component_match.empty:
                        match = component_match
                if not match.empty:
                    return match.iloc[0]

    if "softmax_probability" in working.columns:
        working = working.sort_values("softmax_probability", ascending=False)
    return working.iloc[0] if not working.empty else None


def _load_kpi_data(kpi: str, region: str) -> dict[str, pd.DataFrame] | None:
    try:
        return repository.dashboard_data(kpi, region)
    except Exception:
        return None


def _pipeline_status_for_region(region: str) -> pd.DataFrame:
    try:
        status_df = repository.pipeline_status()
    except Exception:
        return pd.DataFrame()
    if status_df.empty or "region" not in status_df.columns:
        return status_df
    return status_df[status_df["region"] == region].copy()


def _status_for_kpi(status_df: pd.DataFrame, kpi: str) -> str:
    if status_df.empty or not {"kpi", "status"}.issubset(status_df.columns):
        return "unknown"
    row = status_df[status_df["kpi"] == kpi]
    return str(row.iloc[0]["status"]) if not row.empty else "unknown"


def _build_kpi_summary(kpi: str, region: str, status_df: pd.DataFrame) -> KpiSummary:
    pipeline_status = _status_for_kpi(status_df, kpi)
    data = _load_kpi_data(kpi, region)
    description = KPI_DESCRIPTIONS.get(kpi, f"Business analysis for {_kpi_label(kpi)}.")

    if data is None:
        return KpiSummary(
            kpi=kpi,
            label=_kpi_label(kpi),
            status="Data unavailable",
            status_tone="red",
            change="—",
            change_tone="neutral",
            confidence="—",
            confidence_tone="neutral",
            incident_date="—",
            top_driver="—",
            recommendation="Dashboard data could not be loaded for this KPI.",
            description=description,
            has_detail=False,
        )

    packets = data.get("decision_packets", pd.DataFrame())
    top = _top_evidence_row(data)

    if packets.empty:
        if pipeline_status == "no changepoint detected":
            return KpiSummary(
                kpi=kpi,
                label=_kpi_label(kpi),
                status="Stable — no structural incident",
                status_tone="green",
                change="No incident",
                change_tone="neutral",
                confidence="Not required",
                confidence_tone="neutral",
                incident_date="—",
                top_driver="—",
                recommendation="No root-cause investigation is required because the KPI did not show a structural shift.",
                description=description,
                has_detail=True,
            )
        if pipeline_status == "changepoint found, evidence not yet computed":
            return KpiSummary(
                kpi=kpi,
                label=_kpi_label(kpi),
                status="Analysis in progress",
                status_tone="amber",
                change="Shift detected",
                change_tone="amber",
                confidence="Pending",
                confidence_tone="amber",
                incident_date="—",
                top_driver="—",
                recommendation="A structural change was detected, but the root-cause evidence has not been computed yet.",
                description=description,
                has_detail=True,
            )
        return KpiSummary(
            kpi=kpi,
            label=_kpi_label(kpi),
            status="No decision available",
            status_tone="neutral",
            change="—",
            change_tone="neutral",
            confidence="—",
            confidence_tone="neutral",
            incident_date="—",
            top_driver="—",
            recommendation="The pipeline has not produced a decision for this KPI yet.",
            description=description,
            has_detail=True,
        )

    latest = packets.iloc[-1]
    pct = latest.get("percent_change")
    pct_value = float(pct) if pct is not None and pd.notna(pct) else None
    if pct_value is None:
        change = "Baseline near zero"
        change_tone = "neutral"
    else:
        change = f"{pct_value:+.1%}"
        change_tone = "green" if pct_value > 0 else "red" if pct_value < 0 else "neutral"

    confidence = str(latest.get("confidence_level", "Unknown")).title()
    caveat = latest.get("freshness_caveat")
    has_caveat = caveat is not None and pd.notna(caveat) and str(caveat).strip() != ""
    status = "Needs validation" if confidence.lower() == "low" or has_caveat else "Investigation available"
    status_tone = "amber" if has_caveat else "red" if confidence.lower() == "low" else "blue"

    top_driver = _friendly_name(top.get("driver_name")) if top is not None else "—"
    recommendation = latest.get("recommended_action")
    recommendation_text = (
        str(recommendation)
        if recommendation is not None and pd.notna(recommendation) and str(recommendation).strip()
        else "No recommended action was recorded."
    )

    return KpiSummary(
        kpi=kpi,
        label=_kpi_label(kpi),
        status=status,
        status_tone=status_tone,
        change=change,
        change_tone=change_tone,
        confidence=confidence,
        confidence_tone=_confidence_tone(confidence),
        incident_date=str(latest.get("window_start", "—")),
        top_driver=top_driver,
        recommendation=recommendation_text,
        description=description,
        has_detail=True,
    )


# =============================================================================
# Home screen — Regional Command Center
# =============================================================================


def render_home_header(region: str, role: str) -> None:
    st.markdown(
        f"""
        <div class="al-header">
            <div class="al-breadcrumb">Aletheia &nbsp;/&nbsp; Regional Command Center &nbsp;/&nbsp; {_esc(region)}</div>
            <div class="al-header-row">
                <div>
                    <div class="al-title">What is happening across {_esc(region)}?</div>
                    <div class="al-subtitle">One screen summarizes every KPI. Open a KPI only when you want its full evidence, change breakdown, external context, and audit trail.</div>
                </div>
                <div class="al-chips">
                    {chip(f'Persona: {role}', 'brand')}
                    {chip(f'Region: {region}', 'neutral')}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_summary_card(summary: KpiSummary, *, key_suffix: str = "") -> None:
    st.markdown(
        f"""
        <div class="al-kpi-card">
            <div class="al-kpi-top">
                <div class="al-kpi-name">{_esc(summary.label)}</div>
                <div>{chip(summary.status, summary.status_tone)}</div>
            </div>
            <div class="al-kpi-desc">{_esc(summary.description)}</div>
            <div class="al-kpi-grid">
                <div class="al-kpi-cell">
                    <div class="al-kpi-cell-label">Latest change</div>
                    <div class="al-kpi-cell-value">{_esc(summary.change)}</div>
                </div>
                <div class="al-kpi-cell">
                    <div class="al-kpi-cell-label">Evidence confidence</div>
                    <div class="al-kpi-cell-value">{_esc(summary.confidence)}</div>
                </div>
                <div class="al-kpi-cell">
                    <div class="al-kpi-cell-label">Leading explanation</div>
                    <div class="al-kpi-cell-value" title="{_esc(summary.top_driver)}">{_esc(summary.top_driver)}</div>
                </div>
            </div>
            <div class="al-kpi-action"><strong>What Aletheia recommends:</strong> {_esc(summary.recommendation)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if summary.has_detail:
        if st.button(
            f"Open {_esc(summary.label)} analysis →",
            key=f"open_{summary.kpi}_{key_suffix}",
            use_container_width=True,
        ):
            open_kpi_detail(summary.kpi)


def render_regional_command_center(region: str, role: str, kpis: Sequence[str]) -> None:
    render_home_header(region, role)
    render_callout(
        "This page is the regional health check",
        "A structural incident means the KPI shifted into a materially different regime. 'Stable' is a valid result. 'Low confidence' means validate the evidence or source data before acting.",
        tone="info",
        icon="◇",
    )

    loading = st.empty()
    loading.markdown(_loading_skeleton(), unsafe_allow_html=True)
    status_df = _pipeline_status_for_region(region)
    summaries = [_build_kpi_summary(kpi, region, status_df) for kpi in kpis]
    loading.empty()

    active = sum(s.status in {"Investigation available", "Needs validation"} for s in summaries)
    stable = sum(s.status == "Stable — no structural incident" for s in summaries)
    low_conf = sum(s.confidence.lower() == "low" for s in summaries)
    pending = sum(s.status in {"Analysis in progress", "No decision available", "Data unavailable"} for s in summaries)

    render_section_header("Regional health at a glance", "A summary of the current analytical state across every configured KPI.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("KPIs monitored", str(len(summaries)), note="Configured business measures")
    with c2:
        render_metric_card("Active investigations", str(active), note="KPI shifts with decision packets", pill_text="Review", pill_tone="blue")
    with c3:
        render_metric_card("Stable KPIs", str(stable), note="No structural incident detected", pill_text="No action", pill_tone="green")
    with c4:
        render_metric_card("Need attention", str(low_conf + pending), note="Low confidence, pending, or unavailable", pill_text="Validate", pill_tone="amber" if low_conf + pending else "green")

    render_section_header(
        "All KPI summaries",
        "Click any KPI to open its detailed explanation. You no longer need to inspect every KPI one-by-one just to understand the region.",
    )

    for idx in range(0, len(summaries), 2):
        cols = st.columns(2)
        for offset, col in enumerate(cols):
            pos = idx + offset
            if pos >= len(summaries):
                continue
            with col:
                render_kpi_summary_card(summaries[pos], key_suffix=str(pos))

    with st.expander("What do the home-screen labels mean?", expanded=False):
        render_definition("Structural incident", "Aletheia detected that the KPI moved into a materially different regime. This does not automatically mean something went wrong; it means the change is worth explaining.")
        render_definition("Leading explanation", "The highest-ranked driver among the candidates permitted by the KPI contract. It is the strongest supported explanation in the current evidence set, not proof of causality.")
        render_definition("Evidence confidence", "How reliable the analytical evidence is, considering model sufficiency, evidence concentration, statistical validation, and source health. It is separate from whether a business action is ready to execute.")
        render_definition("Stable — no structural incident", "The signal detector did not find a meaningful regime change. No RCA is required for that KPI at the moment.")


# =============================================================================
# Detail page header & overview
# =============================================================================


def _derive_header_state(data: DashboardData) -> HeaderState:
    packets = data.get("decision_packets", pd.DataFrame())
    if packets.empty:
        return HeaderState("NO DECISION", "No incident packet", "neutral", "Awaiting analysis", "neutral", "—")

    latest = packets.iloc[-1]
    confidence = str(latest.get("confidence_level", "Unknown")).upper()
    pct = latest.get("percent_change")
    if pct is not None and pd.notna(pct):
        pct_value = float(pct)
        change_label = f"{pct_value:+.1%}"
        change_tone = "green" if pct_value > 0 else "red" if pct_value < 0 else "neutral"
    else:
        change_label = "Baseline near zero"
        change_tone = "neutral"

    caveat = latest.get("freshness_caveat")
    has_caveat = caveat is not None and pd.notna(caveat) and str(caveat).strip() != ""
    return HeaderState(
        confidence=confidence,
        change_label=change_label,
        change_tone=change_tone,
        data_status="Data needs validation" if has_caveat else "Source data healthy",
        data_tone="amber" if has_caveat else "green",
        incident_date=str(latest.get("window_start", "—")),
    )


def render_detail_header(data: DashboardData, role: str, kpi: str, region: str) -> None:
    state = _derive_header_state(data)
    st.markdown(
        f"""
        <div class="al-header">
            <div class="al-breadcrumb">Aletheia &nbsp;/&nbsp; {_esc(region)} &nbsp;/&nbsp; {_esc(_kpi_label(kpi))}</div>
            <div class="al-header-row">
                <div>
                    <div class="al-title">{_esc(_kpi_label(kpi))} · {_esc(region)}</div>
                    <div class="al-subtitle">Detailed root-cause analysis: what changed, how the KPI moved, which explanations have the strongest evidence, and what should be validated before action.</div>
                </div>
                <div class="al-chips">
                    {chip(f'Persona: {role}', 'brand')}
                    {chip(f'Evidence confidence: {state.confidence}', _confidence_tone(state.confidence))}
                    {chip(f'Change: {state.change_label}', state.change_tone)}
                    {chip(state.data_status, state.data_tone)}
                    {chip(f'Incident: {state.incident_date}', 'neutral')}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_no_data_reason(kpi: str, region: str) -> None:
    status_df = _pipeline_status_for_region(region)
    status = _status_for_kpi(status_df, kpi)
    if status == "no changepoint detected":
        render_empty_state(
            "No structural incident detected",
            f"{_kpi_label(kpi)} in {region} stayed within its normal regime over the available history. This is a valid result, so there is no root cause to investigate.",
        )
        return
    if status == "changepoint found, evidence not yet computed":
        render_callout(
            "A change was detected, but the explanation is still pending",
            "The signal engine found a structural shift, but the evidence and change breakdown have not been computed yet. Run `python -m src.pipeline` to finish the analysis.",
            tone="warning",
            icon="!",
        )
        return
    render_empty_state("Analysis not available", "No decision packet exists yet for this KPI and region.")


def render_decision_summary(data: DashboardData, kpi: str, region: str) -> None:
    packets = data.get("decision_packets", pd.DataFrame())
    if packets.empty:
        _render_no_data_reason(kpi, region)
        return

    latest = packets.iloc[-1]
    top_driver = _top_evidence_row(data)
    evidence = data.get("driver_evidence", pd.DataFrame())
    pct = latest.get("percent_change")
    pct_value = float(pct) if pct is not None and pd.notna(pct) else None
    confidence = str(latest.get("confidence_level", "Unknown"))

    bundle = _load_latest_bundle(data)
    action_context: dict[str, Any] = {}
    if isinstance(bundle, dict):
        decision_payload = bundle.get("decision", {})
        if isinstance(decision_payload, dict) and isinstance(decision_payload.get("action_context"), dict):
            action_context = decision_payload["action_context"]
    action_readiness = str(action_context.get("action_readiness") or "Validate first")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "What changed?",
            _fmt_percent(pct_value),
            note="Relative change between the previous regime and the detected incident regime.",
            pill_text="Increase" if pct_value and pct_value > 0 else "Decrease" if pct_value and pct_value < 0 else "No net movement",
            pill_tone="green" if pct_value and pct_value > 0 else "red" if pct_value and pct_value < 0 else "neutral",
        )
    with c2:
        render_metric_card(
            "Evidence confidence",
            confidence.upper(),
            note=_confidence_plain_english(confidence),
            pill_text="Evidence quality",
            pill_tone=_confidence_tone(confidence),
        )
    with c3:
        readiness_tone = (
            "green" if action_readiness.lower().startswith("act")
            else "amber" if any(word in action_readiness.lower() for word in ("validate", "investigate", "fix"))
            else "neutral"
        )
        render_metric_card(
            "Action readiness",
            action_readiness,
            note="Whether the evidence supports acting now, validating first, fixing data, or simply monitoring.",
            pill_text="Decision state",
            pill_tone=readiness_tone,
        )
    with c4:
        driver_name = _friendly_name(top_driver.get("driver_name")) if top_driver is not None else "—"
        weight = top_driver.get("softmax_probability") if top_driver is not None else None
        note = (
            f"Relative evidence weight: {_fmt_percent(weight)}. This ranks candidates; it is not a causal probability."
            if weight is not None and pd.notna(weight)
            else f"{len(evidence)} candidate evidence rows were evaluated."
        )
        render_metric_card("Leading explanation", driver_name, note=note, pill_text="Evidence-ranked", pill_tone="blue")

    recommendation = latest.get("recommended_action")
    recommendation_text = (
        str(recommendation)
        if recommendation is not None and pd.notna(recommendation) and str(recommendation).strip()
        else "No recommended action was recorded."
    )

    bundle = _load_latest_bundle(data)
    action_context: dict[str, Any] = {}
    if isinstance(bundle, dict):
        decision_payload = bundle.get("decision", {})
        if isinstance(decision_payload, dict) and isinstance(decision_payload.get("action_context"), dict):
            action_context = decision_payload["action_context"]

    if action_context and any(action_context.get(key) for key in ("finding", "why_it_matters", "next_check", "action_if_confirmed")):
        level = str(action_context.get("action_level") or "").replace("_", " ").title()
        owner = str(action_context.get("owner") or "").strip()
        trailing = " ".join(
            item for item in (chip(level, "amber" if "validate" in level.lower() else "green" if level.lower() == "act" else "neutral"), chip(f"Owner: {owner}", "brand") if owner else "") if item
        )
        render_section_header(
            "Recommended decision plan",
            "Aletheia separates what was observed from the intervention. Business action is conditional on the operational check being confirmed.",
            trailing,
        )
        row1 = st.columns(2)
        with row1[0]:
            render_action_step("What we found", str(action_context.get("finding") or ""), accent="finding")
        with row1[1]:
            render_action_step("Why it matters", str(action_context.get("why_it_matters") or ""), accent="why")
        row2 = st.columns(2)
        with row2[0]:
            render_action_step("Check now", str(action_context.get("next_check") or ""), accent="check")
        with row2[1]:
            render_action_step("If confirmed", str(action_context.get("action_if_confirmed") or ""), accent="act")
        secondary_check = str(action_context.get("secondary_check") or "").strip()
        if secondary_check:
            render_callout("Keep this alternative open", secondary_check, tone="info", icon="↔")
    else:
        st.markdown(
            f"""
            <div class="al-recommendation">
                <div class="al-metric-label">What should the business do next?</div>
                <div class="al-recommendation-text">{_esc(recommendation_text)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    caveat = latest.get("freshness_caveat")
    if caveat is not None and pd.notna(caveat) and str(caveat).strip():
        render_callout("Data-quality warning", str(caveat), tone="warning", icon="!")


def _load_latest_bundle(data: DashboardData) -> dict[str, Any] | None:
    bundles = data.get("evidence_bundle", pd.DataFrame())
    if bundles.empty:
        return None
    raw = bundles.iloc[-1].get("bundle_json")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def render_narration(data: DashboardData, role: str) -> None:
    render_section_header(
        "Business summary",
        "A persona-specific explanation generated from the stored evidence bundle. It summarizes the analysis; it does not replace the underlying evidence.",
        chip(_persona_for_role(role), "brand"),
    )
    bundle = _load_latest_bundle(data)
    if bundle is None:
        render_empty_state("Business summary unavailable", "No readable evidence bundle has been generated yet.")
        return

    persona = _persona_for_role(role)
    narration = bundle.get("narration", {}).get(persona)
    if not isinstance(narration, dict):
        render_empty_state("Business summary pending", f"No summary has been generated for the {persona} persona yet.")
        return

    headline = str(narration.get("headline", "Evidence summary"))
    narrative = str(narration.get("narrative", ""))
    st.markdown(
        f"""
        <div class="al-narrative">
            <div class="al-narrative-headline">{_esc(headline)}</div>
            <div class="al-narrative-body">{_esc(narrative)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    caveats = narration.get("caveats", [])
    if isinstance(caveats, list) and caveats:
        with st.expander(f"Important caveats ({len(caveats)})", expanded=False):
            for caveat in caveats:
                st.markdown(f"- {caveat}")


def _driver_source_cadence_days(kpi: str, driver_name: Any) -> int:
    try:
        from src.contracts import get_contract
        contract = get_contract(kpi)
        name = str(driver_name or "")
        spec = next((d for d in contract.root_drivers if d.name == name), None)
        return max(1, int(getattr(spec, "source_cadence_days", 1) or 1)) if spec else 1
    except Exception:
        return 1


def _historical_lead_label(kpi: str, driver_name: Any, lag_value: Any) -> str:
    if lag_value is None or pd.isna(lag_value):
        return "—"
    lag = int(lag_value)
    cadence = _driver_source_cadence_days(kpi, driver_name)
    if cadence > 1:
        periods = lag / cadence
        period_text = f"{periods:g} source period" + ("" if abs(periods - 1.0) < 1e-9 else "s")
        return f"{period_text} (~{lag} days; {cadence}-day resolution)"
    return f"{lag} days"


def render_overview_glossary() -> None:
    with st.expander("How to read this page", expanded=False):
        render_definition("Relative change", "How much the KPI changed between the previous stable regime and the detected incident regime. It is not a forecast.")
        render_definition("Leading explanation", "The highest-priority driver for the KPI component that contributed most to the detected change. Aletheia ranks drivers within that component rather than comparing unrelated component rankings.")
        render_definition("Relative evidence weight", "A ranking weight among the candidate explanations that were assessed. 70% means stronger relative support than the other candidates in that comparison; it does NOT mean a 70% probability of causality.")
        render_definition("Evidence confidence", "A guardrail-aware label for the strength and reliability of the analytical evidence. Action readiness is shown separately because strong evidence can still require an operational validation step before intervention.")
        render_definition("Detected incident window", "The date Aletheia identified a change in the KPI regime. It should not be interpreted as an exact causal timestamp.")


def render_overview(data: DashboardData, role: str, kpi: str, region: str) -> None:
    render_section_header("Executive answer", "Start here: what changed, the strongest current explanation, how sure the system is, and the next action.")
    render_decision_summary(data, kpi, region)
    render_overview_glossary()
    render_narration(data, role)

    top = _top_evidence_row(data)
    if top is not None:
        render_section_header("Why this explanation is leading", "A simple preview of the strongest evidence row. Full statistical detail remains in the Evidence tab.")
        cols = st.columns(4)
        with cols[0]:
            render_metric_card("Driver", _friendly_name(top.get("driver_name")), note="Highest-ranked candidate in the latest incident window.")
        with cols[1]:
            render_metric_card("What it affects", _friendly_name(top.get("explains_component")), note="The KPI component this driver is allowed to explain under the KPI contract.")
        with cols[2]:
            lag_value = top.get("best_lag_days")
            lag_label = _historical_lead_label(kpi, top.get("driver_name"), lag_value)
            render_metric_card("Historical lead time", lag_label, note="Historical alignment shown at the native cadence of the source. A weekly source cannot support exact daily timing.")
        with cols[3]:
            render_metric_card("Relative evidence weight", _fmt_percent(top.get("softmax_probability")), note="Relative rank among assessed candidates; not probability of causality.", pill_text="Ranking only", pill_tone="neutral")


# =============================================================================
# KPI change breakdown (decomposition)
# =============================================================================


def _decomposition_component_labels(kpi: str, latest: pd.Series) -> tuple[str, str | None, str]:
    decomposition_type = str(latest.get("decomposition_type", ""))
    structure = _safe_contract_structure(kpi, decomposition_type)
    return structure.component_a, structure.component_b, structure.formula


def _decomposition_figure(latest: pd.Series, component_a: str, component_b: str | None) -> Any | None:
    if not PLOTLY_AVAILABLE or go is None:
        return None

    labels = [f"{component_a} contribution"]
    values = [float(latest.get("effect_a", 0.0) or 0.0)]
    if component_b:
        labels.append(f"{component_b} contribution")
        values.append(float(latest.get("effect_b", 0.0) or 0.0))
    interaction = float(latest.get("interaction_effect", 0.0) or 0.0)
    if abs(interaction) > 1e-12:
        labels.append("Combined interaction")
        values.append(interaction)

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": ["#4f46e5" if v >= 0 else "#d92d20" for v in values]},
            text=[_fmt_number(v, 4) for v in values],
            textposition="outside",
            hovertemplate="%{y}<br>Contribution to KPI change: %{x:.6g}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#98a2b3")
    fig.update_layout(
        height=max(260, 70 + len(labels) * 55),
        margin={"l": 10, "r": 30, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Contribution to the KPI change (in KPI units)",
        yaxis_title=None,
        showlegend=False,
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": "#344054"},
    )
    fig.update_xaxes(gridcolor="#eaecf0", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
    return fig


def render_decomposition(data: DashboardData, kpi: str) -> None:
    render_section_header(
        "How did the KPI change?",
        "This is a mathematical KPI breakdown. It tells you which KPI components contributed to the change before Aletheia evaluates upstream business drivers.",
    )
    decomp = data.get("decomposition", pd.DataFrame())
    if decomp.empty:
        render_empty_state("No KPI change breakdown", "No decomposition has been recorded for this case.")
        return

    latest = decomp.iloc[-1]
    component_a, component_b, formula = _decomposition_component_labels(kpi, latest)
    decomposition_type = str(latest.get("decomposition_type", ""))
    complex_interaction = str(latest.get("narrative_mode", "")) == "complex_interaction"
    volatile = bool(latest.get("is_volatile", False))

    render_callout(
        "KPI formula used",
        f"{formula}. Contribution values below are changes in the KPI's own units, not automatically percentages.",
        tone="info",
        icon="ƒ",
    )

    chips: list[str] = []
    if complex_interaction:
        chips.append(chip("Components moved together", "amber"))
    if volatile:
        chips.append(chip("Attribution needs caution", "amber"))
    if not chips:
        chips.append(chip("Breakdown mathematically stable", "green"))
    st.markdown(f'<div class="al-chips">{"".join(chips)}</div>', unsafe_allow_html=True)

    if complex_interaction:
        render_callout(
            "Why Aletheia does not show a forced percentage split",
            "More than one KPI component moved at the same time. A single isolated percentage for each component would imply more certainty than the mathematics supports, so Aletheia shows the raw contribution values instead.",
            tone="warning",
            icon="!",
        )

    metric_cols = st.columns(4 if component_b else 2)
    with metric_cols[0]:
        render_metric_card(
            f"{component_a} contribution",
            _fmt_number(latest.get("effect_a"), 4),
            note=f"How much the movement in {component_a.lower()} contributed to the KPI change, holding the decomposition ordering/rule fixed.",
        )
    if component_b:
        with metric_cols[1]:
            render_metric_card(
                f"{component_b} contribution",
                _fmt_number(latest.get("effect_b"), 4),
                note=f"How much the movement in {component_b.lower()} contributed to the KPI change under the decomposition rule.",
            )
        with metric_cols[2]:
            render_metric_card(
                "Combined interaction",
                _fmt_number(latest.get("interaction_effect"), 4),
                note="Extra change created because both components moved together. For ratio KPIs this may be zero or absorbed by the symmetric decomposition rule.",
            )
        with metric_cols[3]:
            render_metric_card(
                "Total KPI change",
                _fmt_number(latest.get("total_change"), 4),
                note="The total raw change reconstructed by the decomposition. Component contributions plus interaction and residual should reconcile to this value.",
            )
    else:
        with metric_cols[1]:
            render_metric_card(
                "Total KPI change",
                _fmt_number(latest.get("total_change"), 4),
                note="For a single-metric KPI, the component itself is the KPI, so there is no second component to separate.",
            )

    fig = _decomposition_figure(latest, component_a, component_b)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        chart_labels = [f"{component_a} contribution"]
        chart_values = [latest.get("effect_a", 0.0)]
        if component_b:
            chart_labels.append(f"{component_b} contribution")
            chart_values.append(latest.get("effect_b", 0.0))
        st.bar_chart(pd.DataFrame({"Contribution": chart_values}, index=chart_labels))

    if volatile:
        render_callout(
            "Why the breakdown is marked as needing caution",
            "The residual tolerance was exceeded or a denominator became unstable. The total KPI movement is still shown, but precise component attribution should not be treated as exact business truth.",
            tone="warning",
            icon="!",
        )

    with st.expander("Definitions used in this change breakdown", expanded=False):
        render_definition(f"{component_a} contribution", f"The portion of the KPI's raw change mathematically associated with movement in {component_a}. It is not the same as saying {component_a} was the root cause.")
        if component_b:
            render_definition(f"{component_b} contribution", f"The portion of the KPI's raw change mathematically associated with movement in {component_b}.")
            render_definition("Combined interaction", "The extra change created when both KPI components move together. In multiplicative KPIs such as revenue, simultaneous movement can create an interaction beyond the two isolated component effects.")
        render_definition("Residual", "Any tiny amount left after reconstructing the KPI change from the defined component effects. A large residual or unstable denominator triggers a caution flag.")
        render_definition("This breakdown is not the root-cause ranking", "The breakdown explains HOW the KPI changed. The driver evidence below explains WHICH upstream business signals are most supported as explanations for those component movements.")

    with st.expander("All breakdown windows · technical audit", expanded=False):
        display = decomp.copy()
        rename_map = {
            "effect_a": f"{component_a} contribution",
            "effect_b": f"{component_b} contribution" if component_b else "Secondary component contribution",
            "interaction_effect": "Combined interaction",
            "residual": "Residual / unreconciled amount",
            "total_change": "Total KPI change",
            "is_volatile": "Needs caution",
            "narrative_mode": "Explanation mode",
            "window_start": "Incident window",
        }
        display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})
        st.dataframe(display, use_container_width=True, hide_index=True)


# =============================================================================
# Driver evidence — simple first, statistics second
# =============================================================================


def _prepare_business_evidence_table(evidence: pd.DataFrame, kpi: str) -> pd.DataFrame:
    display = evidence.copy()
    if "window_start" in display.columns:
        latest_window = display["window_start"].max()
        display = display[display["window_start"] == latest_window]
    if "softmax_probability" in display.columns:
        display = display.sort_values("softmax_probability", ascending=False)

    output = pd.DataFrame(index=display.index)
    if "driver_name" in display.columns:
        output["Possible driver"] = display["driver_name"].map(_friendly_name)
    if "explains_component" in display.columns:
        output["What it can explain"] = display["explains_component"].map(_friendly_name)
    if "evidence_mode" in display.columns:
        output["Evidence type"] = display["evidence_mode"].map(_evidence_mode_label)
    if "best_lag_days" in display.columns:
        output["Historical lead time"] = [
            _historical_lead_label(kpi, driver, lag)
            for driver, lag in zip(display.get("driver_name", pd.Series(index=display.index, dtype=object)), display["best_lag_days"])
        ]
    if "baseline_value" in display.columns:
        output["Typical historical level"] = pd.to_numeric(display["baseline_value"], errors="coerce")
    if "incident_value" in display.columns:
        output["Incident level"] = pd.to_numeric(display["incident_value"], errors="coerce")
    if "softmax_probability" in display.columns:
        output["Relative evidence weight (%)"] = pd.to_numeric(display["softmax_probability"], errors="coerce") * 100.0
    if "is_significant" in display.columns:
        output["Statistical check passed"] = display["is_significant"].astype(bool)
    if "model_status" in display.columns:
        output["Evidence availability"] = display["model_status"].map(_model_status_label)
    return output


def render_evidence_drilldown(data: DashboardData, role: str, kpi: str) -> None:
    render_section_header(
        "Which business drivers have the strongest evidence?",
        "The first table is intentionally business-readable. Aletheia keeps every candidate visible, including weak or unsupported ones.",
        chip("Relative ranking, not causal proof", "neutral"),
    )
    evidence = data.get("driver_evidence", pd.DataFrame())
    if evidence.empty:
        render_empty_state("No driver evidence", "No candidate-driver evidence has been recorded yet.")
        return

    working = evidence.copy()
    if not _role_can_see_margin(role):
        working = working[[c for c in working.columns if "margin" not in c.lower()]]

    business_table = _prepare_business_evidence_table(working, kpi)
    config: dict[str, Any] = {}
    if "Historical lead time (days)" in business_table.columns:
        config["Historical lead time (days)"] = st.column_config.NumberColumn(
            "Historical lead time (days)",
            help="The lag that best aligned this driver with the KPI component using historical data only. Precision is limited by the source cadence.",
            format="%d",
        )
    if "Relative evidence weight (%)" in business_table.columns:
        config["Relative evidence weight (%)"] = st.column_config.ProgressColumn(
            "Relative evidence weight",
            help="How strongly this candidate ranks relative to the other assessed candidates. This is not a probability that the driver caused the KPI change.",
            min_value=0.0,
            max_value=100.0,
            format="%.1f%%",
        )
    if "Statistical check passed" in business_table.columns:
        config["Statistical check passed"] = st.column_config.CheckboxColumn(
            "Statistical check passed",
            help="For historical-pattern evidence, the separate inferential validity check passed the configured threshold. A failed check does not automatically mean the driver is irrelevant; it means the statistical evidence is weaker.",
        )
    for col in ("Typical historical level", "Incident level"):
        if col in business_table.columns:
            config[col] = st.column_config.NumberColumn(col, format="%.4g")

    st.dataframe(business_table, use_container_width=True, hide_index=True, column_config=config)

    render_callout(
        "How to interpret the ranking",
        "A driver can rank highly because it has a stable historical relationship with the KPI component and became unusually strong during the incident, or because it experienced a meaningful structural break. A high rank is evidence to investigate, not mathematical proof of causality.",
        tone="info",
        icon="◇",
    )

    with st.expander("What do the evidence-table terms mean?", expanded=False):
        render_definition("Possible driver", "A contract-approved business signal that is allowed to explain movement in a specific KPI component. Aletheia does not search every column indiscriminately.")
        render_definition("What it can explain", "The KPI component the driver is mapped to. Example: stock availability may explain units sold, while competitor pricing may explain average selling price.")
        render_definition("Historical pattern", "The driver had enough historical variation to estimate a lagged relationship using the pre-incident history.")
        render_definition("New structural shift", "The driver was historically too flat for a meaningful regression, but it changed materially during the incident. Aletheia keeps this as separate structural-break evidence instead of pretending the regression proved zero effect.")
        render_definition("Historical lead time", "The lag selected from historical data before the incident. It represents the best historical alignment available, not an exact causal delay.")
        render_definition("Typical historical level", "The recent historical baseline used to contextualize the incident value.")
        render_definition("Incident level", "The average level of the driver during the incident window.")
        render_definition("Statistical check passed", "A separate validity check passed for the historical relationship. Ridge regression itself is not assigned fake p-values.")
        render_definition("Relative evidence weight", "A softmax-based ranking weight among the candidates assessed together. It must never be read as a calibrated probability of causality.")

    with st.expander("Advanced statistical audit · for technical reviewers", expanded=False):
        desired = [
            "window_start",
            "driver_name",
            "explains_component",
            "evidence_mode",
            "model_status",
            "best_lag_days",
            "baseline_value",
            "incident_value",
            "historical_coefficient",
            "holdout_correlation",
            "p_value",
            "is_significant",
            "coefficient_stability",
            "driver_zscore",
            "structural_break_score",
            "evidence_score",
            "normalized_score",
            "softmax_probability",
        ]
        display_cols = _safe_columns(working, desired)
        technical = working[display_cols].copy()
        if "softmax_probability" in technical.columns:
            technical["relative_evidence_weight_pct"] = pd.to_numeric(technical.pop("softmax_probability"), errors="coerce") * 100.0
        technical = technical.rename(
            columns={
                "window_start": "Incident window",
                "driver_name": "Driver",
                "explains_component": "Explains component",
                "evidence_mode": "Evidence mode",
                "model_status": "Model status",
                "best_lag_days": "Lag (days)",
                "baseline_value": "Baseline",
                "incident_value": "Incident",
                "historical_coefficient": "Historical coefficient",
                "holdout_correlation": "Holdout diagnostic",
                "p_value": "Corrected p-value",
                "is_significant": "Statistical check passed",
                "coefficient_stability": "Coefficient stability",
                "driver_zscore": "Incident severity (z)",
                "structural_break_score": "Structural-break score",
                "evidence_score": "Raw evidence score",
                "normalized_score": "Normalized evidence score",
                "relative_evidence_weight_pct": "Relative evidence weight (%)",
            }
        )
        st.dataframe(technical, use_container_width=True, hide_index=True)

        render_definition("Historical coefficient", "The regularized Ridge relationship coefficient after features are standardized. It is used as relationship-strength evidence, not as a causal effect estimate.")
        render_definition("Corrected p-value", "A separate lag-locked inferential check, corrected for multiple candidate testing. It is not a Ridge p-value.")
        render_definition("Coefficient stability", "How consistently the relationship direction/magnitude survives historical resampling. Higher values indicate a more stable historical relationship.")
        render_definition("Incident severity (z)", "How unusual the incident level is relative to historical variation, expressed in standard-deviation units.")
        render_definition("Structural-break score", "Used when historical variation is unavailable. It measures how materially the incident level shifted relative to the baseline/incident magnitudes.")
        render_definition("Holdout diagnostic", "A time-ordered out-of-sample diagnostic of model behavior. It is not proof of causality and should not be treated as the business answer.")


def render_causal_evidence(data: DashboardData, role: str, kpi: str) -> None:
    render_decomposition(data, kpi)
    render_evidence_drilldown(data, role, kpi)


# =============================================================================
# External retrieval & intelligence
# =============================================================================


def _probability_shift_figure(updates: pd.DataFrame) -> Any | None:
    if not PLOTLY_AVAILABLE or go is None or updates.empty:
        return None
    required = {"driver_name", "probability_before", "probability_after"}
    if not required.issubset(updates.columns):
        return None

    latest = updates.copy()
    if "window_start" in latest.columns:
        latest_window = latest["window_start"].max()
        latest = latest[latest["window_start"] == latest_window]
    latest = latest.sort_values("probability_after", ascending=True)

    fig = go.Figure()
    for _, row in latest.iterrows():
        y = _friendly_name(row["driver_name"])
        before = float(row["probability_before"] or 0.0)
        after = float(row["probability_after"] or 0.0)
        fig.add_trace(go.Scatter(x=[before, after], y=[y, y], mode="lines", line={"color": "#d0d5dd", "width": 3}, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=pd.to_numeric(latest["probability_before"], errors="coerce"), y=latest["driver_name"].map(_friendly_name), mode="markers", name="Before external context", marker={"size": 10, "color": "#98a2b3"}, hovertemplate="Before: %{x:.1%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=pd.to_numeric(latest["probability_after"], errors="coerce"), y=latest["driver_name"].map(_friendly_name), mode="markers", name="After external context", marker={"size": 11, "color": "#4f46e5"}, hovertemplate="After: %{x:.1%}<extra></extra>"))
    fig.update_layout(
        height=max(280, 60 + 42 * len(latest)),
        margin={"l": 10, "r": 20, "t": 10, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": 1.08},
        xaxis={"tickformat": ".0%", "range": [0, 1], "gridcolor": "#eaecf0", "title": "Relative evidence weight"},
        yaxis={"title": None},
        font={"family": "Inter, Segoe UI, sans-serif", "size": 12, "color": "#344054"},
    )
    return fig


def render_retrieval_drilldown(data: DashboardData) -> None:
    render_section_header(
        "External context check",
        "This section appears only when Aletheia needs externally resolvable context. External information can support an existing explanation, but it cannot rewrite the deterministic analysis.",
        chip("External context ≠ causal proof", "blue"),
    )
    retrieved = data.get("retrieved_context", pd.DataFrame())
    updates = data.get("orchestrator_updates", pd.DataFrame())

    if retrieved.empty and updates.empty:
        bundle = _load_latest_bundle(data) or {}
        plan = bundle.get("retrieval_plan")
        plan = plan if isinstance(plan, dict) else None

        if plan is None:
            render_empty_state(
                "No external context was needed",
                "The deterministic evidence did not require Stage 4 web retrieval for this KPI and region. This is normal when the internal evidence is sufficiently clear.",
            )
            return

        target = str(plan.get("retrieval_target") or "none").strip().lower()
        query = str(plan.get("retrieval_query") or "").strip()
        reason = str(plan.get("planner_reason") or "").strip()

        if target == "web":
            detail = (
                "Stage 4 requested public context, but no usable source rows were stored. "
                "Check network/retrieval availability and the System & Audit tab."
            )
            if query:
                detail += f" Search used: {query}"
            render_empty_state("External context was requested but returned no usable evidence", detail)
        elif target == "internal":
            detail = (
                "Aletheia identified an evidence gap, but the missing fact belongs in company systems or operational records rather than on the public web."
            )
            if reason:
                detail += f" Routing note: {reason}"
            render_empty_state("Clarification needed · internal evidence required", detail)
        else:
            detail = (
                "Aletheia identified ambiguity but could not form a safe, grounded public-web retrieval plan, so it abstained rather than inventing external context."
            )
            if reason:
                detail += f" Routing note: {reason}"
            render_empty_state("Clarification considered · web retrieval abstained", detail)
        return

    render_callout(
        "Safety boundary",
        "External sources are stored separately from the business data. They may only strengthen or weaken an already-existing hypothesis within a bounded adjustment; they cannot create a new deterministic driver, change the detected incident date, or rewrite the model fit.",
        tone="info",
        icon="◇",
    )

    if not retrieved.empty:
        desired = ["window_start", "hypothesis", "retrieval_query", "source_title", "source_url", "retrieval_support", "retrieval_confidence", "retrieved_at"]
        display = retrieved[_safe_columns(retrieved, desired)].copy()
        display = display.rename(
            columns={
                "window_start": "Incident window",
                "hypothesis": "Explanation being checked",
                "retrieval_query": "Search used",
                "source_title": "Source title",
                "source_url": "Source link",
                "retrieval_support": "External support score",
                "retrieval_confidence": "Source-scoring confidence",
                "retrieved_at": "Retrieved at",
            }
        )
        config: dict[str, Any] = {}
        if "Source link" in display.columns:
            config["Source link"] = st.column_config.LinkColumn("Source", display_text="Open source")
        if "External support score" in display.columns:
            config["External support score"] = st.column_config.NumberColumn(
                "External support score",
                help="Positive values support the existing explanation; negative values weaken it. The score is bounded and is not a probability.",
                format="%.3f",
            )
        if "Source-scoring confidence" in display.columns:
            config["Source-scoring confidence"] = st.column_config.ProgressColumn(
                "Source-scoring confidence",
                help="How confident the retrieval scorer is in its interpretation of the source text. It is not confidence that the source proves causality.",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            )
        st.dataframe(display, use_container_width=True, hide_index=True, column_config=config)

    if not updates.empty:
        render_section_header(
            "How much did external context change the ranking?",
            "The before/after comparison makes the bounded influence of external evidence visible.",
        )
        fig = _probability_shift_figure(updates)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        desired_updates = ["window_start", "driver_name", "probability_before", "retrieval_support", "probability_after", "updated_at"]
        display_updates = updates[_safe_columns(updates, desired_updates)].copy()
        if "probability_before" in display_updates.columns:
            display_updates["Evidence weight before (%)"] = pd.to_numeric(display_updates.pop("probability_before"), errors="coerce") * 100.0
        if "probability_after" in display_updates.columns:
            display_updates["Evidence weight after (%)"] = pd.to_numeric(display_updates.pop("probability_after"), errors="coerce") * 100.0
        display_updates = display_updates.rename(
            columns={
                "window_start": "Incident window",
                "driver_name": "Explanation",
                "retrieval_support": "External support score",
                "updated_at": "Updated at",
            }
        )
        st.dataframe(
            display_updates,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Evidence weight before (%)": st.column_config.ProgressColumn("Evidence weight · before", min_value=0.0, max_value=100.0, format="%.1f%%"),
                "External support score": st.column_config.NumberColumn("External support score", format="%.3f"),
                "Evidence weight after (%)": st.column_config.ProgressColumn("Evidence weight · after", min_value=0.0, max_value=100.0, format="%.1f%%"),
            },
        )

    with st.expander("What do the external-intelligence terms mean?", expanded=False):
        render_definition("External support score", "A bounded score describing whether a retrieved source supports or weakens an existing explanation. It cannot create evidence from nothing.")
        render_definition("Source-scoring confidence", "Confidence that the local retrieval-scoring model interpreted the source correctly. It is not business-decision confidence.")
        render_definition("Evidence weight before / after", "The relative hypothesis ranking before and after bounded external-context fusion. Unassessed hypotheses are left unchanged and total assessed evidence mass is preserved.")
        render_definition("Why external context is separate", "Aletheia prevents web retrieval from contaminating raw business facts, KPI marts, detected changepoints, lag selection, or regression fits.")


def render_margin_panel(role: str, region: str) -> None:
    if not _role_can_see_margin(role):
        render_callout("Restricted financial metric", "Margin detail is available only to Executive and Finance personas.", tone="info", icon="◇")
        return

    render_section_header("Margin intelligence", "Recent margin context for authorized Executive / Finance users.", chip("RBAC protected", "brand"))
    try:
        from src import database

        margin_df = database.query_df(
            "SELECT metric_date, margin_pct FROM mart.daily_kpi_evidence WHERE region = %s ORDER BY metric_date DESC LIMIT 30",
            params=(region,),
        )
    except Exception as exc:
        render_callout("Margin data unavailable", f"Authorized query failed: {exc}", tone="danger", icon="!")
        return

    if margin_df.empty:
        render_empty_state("No margin observations", "No recent margin records are available for this region.")
        return

    ordered = margin_df.sort_values("metric_date")
    if PLOTLY_AVAILABLE and go is not None:
        fig = go.Figure(go.Scatter(x=ordered["metric_date"], y=ordered["margin_pct"], mode="lines+markers", line={"color": "#4f46e5", "width": 2}, marker={"size": 5}, hovertemplate="%{x}<br>Margin: %{y:.1%}<extra></extra>"))
        fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"tickformat": ".0%", "gridcolor": "#eaecf0"}, xaxis={"gridcolor": "rgba(0,0,0,0)"}, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    margin_display = margin_df.copy()
    margin_display["margin_pct"] = pd.to_numeric(margin_display["margin_pct"], errors="coerce") * 100.0
    margin_display = margin_display.rename(columns={"metric_date": "Date", "margin_pct": "Margin (%)"})
    st.dataframe(
        margin_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Margin (%)": st.column_config.NumberColumn("Margin", format="%.2f%%"),
        },
    )


def render_retrieval_intelligence(data: DashboardData, role: str, region: str) -> None:
    render_retrieval_drilldown(data)
    render_margin_panel(role, region)


# =============================================================================
# System & pipeline health
# =============================================================================


def _plain_pipeline_status(value: Any) -> str:
    mapping = {
        "ready": "Complete analysis available",
        "changepoint found, evidence not yet computed": "Change detected · explanation pending",
        "no changepoint detected": "Stable · no structural incident",
    }
    return mapping.get(str(value), _friendly_name(value))


def render_pipeline_health(kpi: str, region: str) -> None:
    render_section_header(
        "Analysis coverage",
        "Shows whether each KPI/region has a complete analysis, a detected change still waiting for explanation, or no structural incident.",
    )
    try:
        status_df = repository.pipeline_status()
    except Exception as exc:
        render_callout("Coverage status unavailable", f"Coverage could not be loaded: {exc}", tone="danger", icon="!")
        return

    if status_df.empty:
        render_empty_state("No coverage information", "The repository returned no pipeline-coverage rows.")
        return

    statuses = status_df.get("status", pd.Series(dtype=str)).astype(str)
    ready = int((statuses == "ready").sum())
    waiting = int((statuses == "changepoint found, evidence not yet computed").sum())
    stable = int((statuses == "no changepoint detected").sum())
    total = len(status_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("KPI × region combinations", str(total), note="Total combinations monitored by the configured KPI contracts.")
    with c2:
        render_metric_card("Complete analyses", str(ready), note="A decision packet is available.", pill_text="Ready", pill_tone="green")
    with c3:
        render_metric_card("Explanations pending", str(waiting), note="A structural change was found but evidence is not complete.", pill_text="Needs processing", pill_tone="amber")
    with c4:
        render_metric_card("Stable combinations", str(stable), note="No structural incident was detected.", pill_text="Valid outcome", pill_tone="neutral")

    display = status_df.copy()
    if {"kpi", "region"}.issubset(display.columns):
        display.insert(0, "Current selection", (display["kpi"] == kpi) & (display["region"] == region))
    if "kpi" in display.columns:
        display["kpi"] = display["kpi"].map(_kpi_label)
    if "status" in display.columns:
        display["status"] = display["status"].map(_plain_pipeline_status)
    display = display.rename(
        columns={
            "kpi": "KPI",
            "region": "Region",
            "changepoints_detected": "Detected shifts",
            "decision_packets": "Completed analyses",
            "status": "Analysis status",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("What does the analysis status mean?", expanded=False):
        render_definition("Complete analysis available", "A structural incident was processed through the evidence/decomposition pipeline and a decision packet is available.")
        render_definition("Change detected · explanation pending", "The signal detector found a structural KPI shift, but the downstream RCA evidence has not yet been completed.")
        render_definition("Stable · no structural incident", "The signal detector did not find a meaningful regime shift. This is not a pipeline failure; it means there is no incident to explain.")


def render_analysis_audit_snapshot(data: DashboardData) -> None:
    render_section_header("Analysis audit snapshot", "Counts of the analytical records stored for this KPI and region. Intended mainly for audit/technical review.")
    artifact_labels = {
        "changepoints": "Detected KPI shifts",
        "decomposition": "KPI change breakdowns",
        "driver_evidence": "Driver evidence rows",
        "retrieved_context": "External sources retrieved",
        "orchestrator_updates": "External-context ranking updates",
        "decision_packets": "Decision packets",
        "evidence_bundle": "Evidence bundles",
    }
    audit = pd.DataFrame([{"Stored artifact": label, "Records": len(data.get(key, pd.DataFrame()))} for key, label in artifact_labels.items()])
    st.dataframe(audit, use_container_width=True, hide_index=True, column_config={"Records": st.column_config.NumberColumn("Records", format="%d")})


def render_runtime_audit_log() -> None:
    render_section_header("Runtime telemetry", "Recent operational events. This measures application behavior, not business performance.")
    try:
        from src import database

        telemetry = database.query_df(
            "SELECT created_at, operation_name, latency_ms, llm_calls, estimated_cost_usd, status FROM app.runtime_telemetry ORDER BY created_at DESC LIMIT 100"
        )
    except Exception as exc:
        render_callout("Runtime telemetry unavailable", f"Telemetry could not be read: {exc}", tone="warning", icon="!")
        return

    if telemetry.empty:
        render_empty_state("No runtime telemetry", "No operational telemetry has been recorded yet.")
        return

    telemetry = telemetry.rename(
        columns={
            "created_at": "Time",
            "operation_name": "Operation",
            "latency_ms": "Latency (ms)",
            "llm_calls": "LLM calls",
            "estimated_cost_usd": "Estimated cost (USD)",
            "status": "Status",
        }
    )
    st.dataframe(telemetry, use_container_width=True, hide_index=True)


def render_feedback(role: str, region: str) -> None:
    render_section_header("Feedback", "Feedback is stored separately from the analytical evidence and does not alter the current RCA result.")
    with st.form("feedback_form", clear_on_submit=True):
        disposition = st.radio("Was this analysis useful?", ["Helpful", "Not helpful", "Unclear"], horizontal=True)
        comment = st.text_area("Comments (optional)", placeholder="What was clear, missing, or misleading?", max_chars=2000)
        submitted = st.form_submit_button("Submit feedback", type="primary")
        if submitted:
            try:
                repository.save_feedback(role, region, disposition, comment)
                render_callout("Feedback recorded", "Thank you. The feedback was stored successfully.", tone="success", icon="✓")
            except Exception as exc:
                render_callout("Feedback could not be recorded", str(exc), tone="danger", icon="!")


def render_system_health(data: DashboardData, role: str, kpi: str, region: str) -> None:
    render_pipeline_health(kpi, region)
    left, right = st.columns([1, 1])
    with left:
        render_analysis_audit_snapshot(data)
    with right:
        render_runtime_audit_log()
    render_feedback(role, region)


# =============================================================================
# Main detailed KPI screen
# =============================================================================


def render_kpi_detail(role: str, region: str, kpi: str) -> None:
    if st.button("← Back to regional overview", key="detail_back_top"):
        st.session_state["selected_kpi_detail"] = None
        st.rerun()

    loading = st.empty()
    loading.markdown(_loading_skeleton(), unsafe_allow_html=True)
    try:
        with repository.timed_operation("dashboard_load"):
            data = repository.dashboard_data(kpi, region)
    except Exception as exc:
        loading.empty()
        render_detail_header({}, role, kpi, region)
        render_callout("Dashboard data could not be loaded", str(exc), tone="danger", icon="!")
        st.code(
            "Check ALETHEIA_DB_HOST / ALETHEIA_DB_USER / ALETHEIA_DB_PASSWORD and confirm the pipeline has run: python -m src.pipeline",
            language="text",
        )
        return
    finally:
        loading.empty()

    render_detail_header(data, role, kpi, region)

    overview_tab, evidence_tab, retrieval_tab, system_tab = st.tabs(
        [
            "Overview & Business Summary",
            "KPI Change & Driver Evidence",
            "External Context & Intelligence",
            "System & Audit",
        ]
    )

    with overview_tab:
        render_overview(data, role, kpi, region)

    with evidence_tab:
        render_causal_evidence(data, role, kpi)

    with retrieval_tab:
        render_retrieval_intelligence(data, role, region)

    with system_tab:
        render_system_health(data, role, kpi, region)


# =============================================================================
# App entry point
# =============================================================================


def _loading_skeleton() -> str:
    return """
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin:.6rem 0 1rem 0;">
            <div class="al-skeleton"></div><div class="al-skeleton"></div>
            <div class="al-skeleton"></div><div class="al-skeleton"></div>
        </div>
    """


def main() -> None:
    inject_design_system()
    role, region, kpis = render_sidebar()

    selected_kpi = st.session_state.get("selected_kpi_detail")
    if selected_kpi:
        render_kpi_detail(role, region, str(selected_kpi))
    else:
        render_regional_command_center(region, role, kpis)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div class="al-footnote" style="text-align:center;">'
        'Aletheia · Evidence-first decision intelligence · Relative evidence is not causal proof · External context never rewrites deterministic business evidence.'
        '</div>',
        unsafe_allow_html=True,
    )


def render_action_step(label: str, text: str, *, accent: str = "neutral") -> None:
    """Compact plain-language action card used in the executive decision plan."""
    if not text:
        return
    tone = {
        "finding": "blue",
        "why": "neutral",
        "check": "amber",
        "act": "green",
        "warning": "red",
    }.get(accent, "neutral")
    st.markdown(
        f"""
        <div class="al-card">
            <div class="al-metric-label">{_esc(label)}</div>
            <div class="al-metric-note" style="font-size:0.94rem;line-height:1.55;color:var(--al-text);margin-top:0.35rem;">{_esc(text)}</div>
            <div style="margin-top:0.65rem;">{chip(label, tone)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
