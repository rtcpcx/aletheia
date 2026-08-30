"""
Aletheia — src/repository.py

The dashboard's only path into the database. Reads analysis tables and
deterministic mart views; writes only app.user_feedback and
app.runtime_telemetry (never raw.*, mart.*, or analysis.*).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import pandas as pd

from src import database


def dashboard_data(kpi: str, region: str) -> dict:
    """
    Read analysis tables and deterministic mart views. Exposes:
    - changepoints
    - decomposition
    - all driver candidates (including weak ones — never filtered out)
    - model sufficiency / lag selection / validity (embedded in driver_evidence)
    - retrieved evidence, kept separate from deterministic evidence
    - orchestrator before/after probability updates
    """
    changepoints = database.query_df(
        """
        SELECT changepoint_date, segment_mean_before, segment_mean_after,
               shift_magnitude, method
        FROM analysis.detected_changepoints
        WHERE kpi = %s AND region = %s
        ORDER BY changepoint_date
        """,
        params=(kpi, region),
    )

    decomposition = database.query_df(
        """
        SELECT window_start, decomposition_type, effect_a, effect_b,
               interaction_effect, residual, total_change,
               is_volatile, narrative_mode
        FROM analysis.pvm_decomposition
        WHERE kpi = %s AND region = %s
        ORDER BY window_start
        """,
        params=(kpi, region),
    )

    driver_evidence = database.query_df(
        """
        SELECT
            window_start,
            driver_name,
            explains_component,
            evidence_mode,
            model_status,
            best_lag_days,
            baseline_value,
            incident_value,
            driver_zscore,
            structural_break_score,
            historical_coefficient,
            holdout_correlation,
            p_value,
            is_significant,
            coefficient_stability,
            evidence_score,
            normalized_score,
            softmax_probability
        FROM analysis.driver_evidence
        WHERE kpi = %s AND region = %s
        ORDER BY window_start, softmax_probability DESC
        """,
        params=(kpi, region),
    )

    retrieved_context = database.query_df(
        """
        SELECT window_start, hypothesis, retrieval_query, source_title,
               source_url, retrieval_support, retrieval_confidence,
               retrieved_at
        FROM analysis.retrieved_context
        WHERE kpi = %s AND region = %s
        ORDER BY window_start, retrieved_at
        """,
        params=(kpi, region),
    )

    orchestrator_updates = database.query_df(
        """
        SELECT window_start, driver_name, probability_before,
               retrieval_query, retrieval_support, probability_after,
               updated_at
        FROM analysis.orchestrator_updates
        WHERE kpi = %s AND region = %s
        ORDER BY window_start, updated_at
        """,
        params=(kpi, region),
    )

    decision_packets = database.query_df(
        """
        SELECT window_start, percent_change, confidence_level,
               recommended_action, freshness_caveat
        FROM analysis.decision_packets
        WHERE kpi = %s AND region = %s
        ORDER BY window_start
        """,
        params=(kpi, region),
    )

    evidence_bundle = database.query_df(
        """
        SELECT window_start, bundle_json, generated_at
        FROM analysis.evidence_bundle
        WHERE kpi = %s AND region = %s
        ORDER BY window_start
        """,
        params=(kpi, region),
    )

    return {
        "changepoints": changepoints,
        "decomposition": decomposition,
        "driver_evidence": driver_evidence,
        "retrieved_context": retrieved_context,
        "orchestrator_updates": orchestrator_updates,
        "decision_packets": decision_packets,
        "evidence_bundle": evidence_bundle,
    }


def list_kpis() -> list[str]:
    """
    Returns every KPI defined in config/kpi_contracts.yaml, regardless of
    whether the pipeline has produced results for it yet.

    Deliberately does NOT query analysis.decision_packets for this: a KPI
    with zero detected changepoints (or one the pipeline simply hasn't
    reached yet) would otherwise disappear from the selector entirely
    instead of being selectable and showing "no data yet".
    """
    from src.contracts import load_contracts

    contracts = load_contracts()
    return sorted(contracts.keys())


def list_regions() -> list[str]:
    df = database.query_df(
        "SELECT DISTINCT region FROM mart.daily_kpi_evidence ORDER BY region"
    )
    return df["region"].tolist()


def pipeline_status() -> pd.DataFrame:
    """
    One row per (kpi, region) showing how far the pipeline has gotten for
    each combination, for the sidebar status panel in app.py. This is what
    actually explains why a given KPI shows "no data yet" instead of
    leaving the person guessing.
    """
    from src.contracts import load_contracts

    contracts = load_contracts()
    regions = list_regions()

    changepoints = database.query_df(
        "SELECT kpi, region, COUNT(*) AS n_changepoints "
        "FROM analysis.detected_changepoints GROUP BY kpi, region"
    )
    packets = database.query_df(
        "SELECT kpi, region, COUNT(*) AS n_decision_packets "
        "FROM analysis.decision_packets GROUP BY kpi, region"
    )

    rows = []
    for kpi in sorted(contracts.keys()):
        for region in regions:
            cp_row = changepoints[
                (changepoints["kpi"] == kpi) & (changepoints["region"] == region)
            ]
            pk_row = packets[
                (packets["kpi"] == kpi) & (packets["region"] == region)
            ]
            n_cp = int(cp_row["n_changepoints"].iloc[0]) if not cp_row.empty else 0
            n_pk = int(pk_row["n_decision_packets"].iloc[0]) if not pk_row.empty else 0

            if n_pk > 0:
                status = "ready"
            elif n_cp > 0:
                status = "changepoint found, evidence not yet computed"
            else:
                status = "no changepoint detected"

            rows.append(
                {
                    "kpi": kpi,
                    "region": region,
                    "changepoints_detected": n_cp,
                    "decision_packets": n_pk,
                    "status": status,
                }
            )

    return pd.DataFrame(rows)


def save_feedback(persona: str, region: str, disposition: str, comment: str) -> None:
    """Insert user feedback into app.user_feedback."""
    database.execute(
        """
        INSERT INTO app.user_feedback (persona, region, disposition, comment_text)
        VALUES (%s, %s, %s, %s)
        """,
        (persona, region, disposition, comment),
    )


def log_telemetry(
    operation: str,
    latency_ms: int,
    status: str = "success",
    llm_calls: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Write runtime telemetry into app.runtime_telemetry."""
    database.execute(
        """
        INSERT INTO app.runtime_telemetry (
            operation_name, latency_ms, llm_calls, estimated_cost_usd, status
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        (operation, latency_ms, llm_calls, cost_usd, status),
    )


@contextmanager
def timed_operation(operation_name: str, llm_calls: int = 0, cost_usd: float = 0.0) -> Iterator[None]:
    """Convenience context manager: times a block and logs telemetry for it."""
    start = time.perf_counter()
    status = "success"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        log_telemetry(operation_name, elapsed_ms, status=status, llm_calls=llm_calls, cost_usd=cost_usd)
