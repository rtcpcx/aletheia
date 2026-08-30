"""
Aletheia — src/guardrails_engine.py

Guardrails that gate the pipeline away from confidently wrong answers:
entity history sufficiency, regression-window sufficiency, ambiguous-context
detection, and final confidence labeling.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Iterable, Literal

from src import database

MIN_ENTITY_HISTORY_DAYS = 60

MIN_ABSOLUTE_OBSERVATIONS = 30
MIN_OBSERVATIONS_PER_FEATURE = 10

TOP_PROBABILITY_GAP_THRESHOLD = 0.10
TOP2_PROBABILITY_MASS_THRESHOLD = 0.55
FRESHNESS_STALE_DAYS = 3
MIN_SOURCE_COMPLETENESS_PCT = 95.0

# Raw-source lineage for KPI measurements and deterministic driver evidence.
_SOURCE_BY_METRIC = {
    # sales
    "revenue": "sales",
    "units_sold": "sales",
    "average_selling_price": "sales",
    "average_discount_pct": "sales",
    "stock_availability": "sales",
    "margin_pct": "sales",

    # marketing
    "marketing_spend": "marketing",
    "sessions": "marketing",
    "orders": "marketing",
    "new_customers": "marketing",
    "conversion_rate": "marketing",
    "customer_acquisition_cost": "marketing",

    # market context
    "competitor_price_index": "market_context",
    "weather_index": "market_context",

    # customer success
    "total_active_customers": "customer_success",
    "active_customers": "customer_success",
    "churned_customers": "customer_success",
    "churn_rate": "customer_success",
    "support_ticket_volume": "customer_success",
    "platform_uptime_pct": "customer_success",
}


def check_history_sufficiency(product_id: str | None, region: str) -> dict:
    """
    Fewer than 60 observed days:
        sparse_history

    Otherwise:
        sufficient_history

    When product_id is None, sufficiency is evaluated at the region level
    (used by the signal engine, which detects changepoints per KPI/region,
    not per product).
    """
    if product_id is not None:
        sql = """
            SELECT COUNT(DISTINCT sale_date) AS n_days
            FROM raw.sales
            WHERE region = %s AND product_id = %s
        """
        params = (region, product_id)
    else:
        sql = """
            SELECT COUNT(DISTINCT metric_date) AS n_days
            FROM mart.daily_kpi_evidence
            WHERE region = %s
        """
        params = (region,)

    df = database.query_df(sql, params=params)
    n_days = int(df["n_days"].iloc[0]) if not df.empty else 0

    status = "sparse_history" if n_days < MIN_ENTITY_HISTORY_DAYS else "sufficient_history"
    return {"status": status, "observed_days": n_days}


def check_model_sufficiency(n_observations: int, n_features: int) -> dict:
    """
    Separate guardrail from entity history sufficiency: an entity can have
    >= 60 days of raw history and STILL not have enough usable rows to fit
    a regression once lag alignment, differencing, and NaN removal have
    reduced the row count.

        N < max(MIN_ABSOLUTE_OBSERVATIONS,
                MIN_OBSERVATIONS_PER_FEATURE * n_features)
        => insufficient
    """
    required = max(
        MIN_ABSOLUTE_OBSERVATIONS, MIN_OBSERVATIONS_PER_FEATURE * max(n_features, 1)
    )
    if n_observations < required:
        return {
            "status": "insufficient_history",
            "required": required,
            "observed": n_observations,
        }
    return {
        "status": "sufficient",
        "required": required,
        "observed": n_observations,
    }



def source_dependencies_for_analysis(
    kpi: str,
    driver_names: Iterable[str] | None = None,
) -> list[str]:
    """Return raw sources whose health can affect this RCA result."""
    sources: set[str] = set()

    kpi_source = _SOURCE_BY_METRIC.get(str(kpi))
    if kpi_source:
        sources.add(kpi_source)

    for name in driver_names or []:
        source = _SOURCE_BY_METRIC.get(str(name))
        if source:
            sources.add(source)

    return sorted(sources)


def check_source_health(
    region: str,
    as_of_date: dt.date,
    source_names: Iterable[str] | None = None,
) -> dict:
    """
    Read source-health facts available ON OR BEFORE the incident date.

    Do not use mart.source_health_latest for historical RCA because it can
    expose source-health facts that occurred after the incident.
    """
    requested = sorted(
        {str(source).strip() for source in (source_names or []) if str(source).strip()}
    )

    source_filter = ""
    query_params: list[object] = [region, as_of_date, region]
    if requested:
        placeholders = ", ".join(["%s"] * len(requested))
        source_filter = f" AND h.source_name IN ({placeholders})"
        query_params.extend(requested)

    sql = f"""
        SELECT
            h.health_date,
            h.source_name,
            h.expected_cadence_hours,
            h.observed_lag_hours,
            h.completeness_pct,
            h.status,
            h.notes
        FROM raw.source_health h
        JOIN (
            SELECT source_name, MAX(health_date) AS health_date
            FROM raw.source_health
            WHERE region = %s
              AND health_date <= %s
            GROUP BY source_name
        ) latest
          ON latest.source_name = h.source_name
         AND latest.health_date = h.health_date
        WHERE h.region = %s
        {source_filter}
        ORDER BY h.source_name
    """

    df = database.query_df(sql, params=tuple(query_params))

    if df.empty:
        return {
            "status": "source_health_unavailable",
            "healthy": False,
            "as_of_date": as_of_date,
            "required_sources": requested,
            "observed_sources": [],
            "missing_sources": requested,
            "stale_sources": [],
            "freshness_gap_days": FRESHNESS_STALE_DAYS + 1,
            "details": [],
        }

    details: list[dict] = []
    stale_sources: list[str] = []
    observed_sources: set[str] = set()
    max_lag_hours = 0.0

    for _, row in df.iterrows():
        source_name = str(row["source_name"])
        observed_sources.add(source_name)

        lag_hours = float(row["observed_lag_hours"] or 0.0)
        completeness = float(row["completeness_pct"] or 0.0)
        declared_status = str(row["status"] or "").strip().lower()

        max_lag_hours = max(max_lag_hours, lag_hours)

        stale_by_status = declared_status not in {"healthy", "ok", "fresh"}
        stale_by_lag = lag_hours > FRESHNESS_STALE_DAYS * 24
        stale_by_completeness = completeness < MIN_SOURCE_COMPLETENESS_PCT
        unhealthy = stale_by_status or stale_by_lag or stale_by_completeness

        if unhealthy:
            stale_sources.append(source_name)

        details.append(
            {
                "source_name": source_name,
                "health_date": row["health_date"],
                "expected_cadence_hours": float(row["expected_cadence_hours"]),
                "observed_lag_hours": lag_hours,
                "completeness_pct": completeness,
                "declared_status": declared_status,
                "healthy": not unhealthy,
                "notes": row.get("notes"),
            }
        )

    missing_sources = sorted(set(requested) - observed_sources)
    if missing_sources:
        stale_sources.extend(missing_sources)

    freshness_gap_days = int(math.ceil(max_lag_hours / 24.0))
    healthy = not stale_sources

    return {
        "status": "healthy_sources" if healthy else "stale_or_incomplete_source",
        "healthy": healthy,
        "as_of_date": as_of_date,
        "required_sources": requested,
        "observed_sources": sorted(observed_sources),
        "missing_sources": missing_sources,
        "stale_sources": sorted(set(stale_sources)),
        "max_observed_lag_hours": max_lag_hours,
        "freshness_gap_days": freshness_gap_days,
        "details": details,
    }

def check_ambiguous_context(region: str, week_start: dt.date) -> dict | None:
    """
    Detect ambiguous_local_event context rows for a given region/week.

    Abstain when:
    - deterministic evidence is weak
    - top hypotheses are insufficiently separated
    - no significant corroborating evidence exists

    Retrieval may be triggered downstream by the orchestrator based on this
    result. Raw or mart data is never modified here — this is a read-only
    check.
    """
    sql = """
        SELECT event_type, source_type
        FROM raw.market_context
        WHERE region = %s AND week_start = %s
    """
    df = database.query_df(sql, params=(region, week_start))
    if df.empty:
        return None

    row = df.iloc[0]
    if row.get("event_type") == "ambiguous_local_event":
        return {
            "region": region,
            "week_start": week_start,
            "event_type": row["event_type"],
            "requires_clarification": True,
        }
    return None


def compute_confidence(
    decision: str,
    top2_probability_mass: float,
    top_probability_gap: float,
    top_driver_significant: bool,
    model_sufficient: bool,
    freshness_gap_days: int,
    source_health_ok: bool = True,
) -> Literal["High", "Medium", "Low"]:
    """
    High:
        sufficient model
        strong probability mass
        sufficient probability separation
        valid top-driver evidence
        fresh data

    Medium:
        adequate evidence but one non-critical weakness

    Low:
        monitor decision
        OR insufficient model
        OR weak evidence
        OR ambiguous top hypotheses
        OR stale data
    """
    if (
        decision == "monitor"
        or not model_sufficient
        or not source_health_ok
        or freshness_gap_days > FRESHNESS_STALE_DAYS
    ):
        return "Low"

    weaknesses = 0
    if top2_probability_mass < TOP2_PROBABILITY_MASS_THRESHOLD:
        weaknesses += 1
    if top_probability_gap < TOP_PROBABILITY_GAP_THRESHOLD:
        weaknesses += 1
    if not top_driver_significant:
        weaknesses += 1
    if weaknesses == 0:
        return "High"
    if weaknesses == 1:
        return "Medium"
    return "Low"
