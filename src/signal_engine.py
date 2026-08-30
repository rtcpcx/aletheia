"""
Aletheia — src/signal_engine.py

Stage 1: Signal. Did something change, and when?

Uses ruptures.Pelt with an rbf cost model to find changepoints in a daily
KPI series, then filters candidates using a minimum segment length and a
minimum gap policy derived from the KPI contract's maximum driver lag (so a
changepoint can never be closer to the series edges, or to another
changepoint, than the longest lag we might need to search around it).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import ruptures as rpt

from src import database
from src.contracts import KpiContract, get_contract, load_contracts
from src.guardrails_engine import check_history_sufficiency

MIN_SEGMENT_LENGTH = 10
MIN_OBSERVATIONS_FOR_PELT = 30
DEFAULT_MIN_GAP_BUFFER_DAYS = 7


@dataclass
class Changepoint:
    kpi: str
    region: str
    changepoint_date: dt.date
    segment_mean_before: float
    segment_mean_after: float
    shift_magnitude: float
    method: str = "pelt_rbf"


def min_gap_days_for(contract: KpiContract) -> int:
    """
    min_gap_days = max(max_lag across root drivers) + 7

    This ensures a changepoint is never flagged so close to another one (or
    to the edge of available data) that there isn't enough runway to search
    for a driver lag around it.
    """
    return contract.max_lag_overall() + DEFAULT_MIN_GAP_BUFFER_DAYS


def detect_changepoints(
    kpi: str,
    region: str,
    series: pd.DataFrame,
    min_gap_days: int,
) -> list[dict]:
    """
    Run ruptures.Pelt(model='rbf').

    Enforce:
    - enough observations
    - minimum segment length
    - minimum gap between changepoints

    `series` must have columns ['metric_date', 'value'], sorted ascending
    by metric_date, with no gaps in the date index (the caller is
    responsible for resampling to daily frequency).

    Returns a list of dicts with:
        changepoint_date, mean_before, mean_after, shift_magnitude
    """
    series = series.dropna(subset=["value"]).sort_values("metric_date")
    values = series["value"].to_numpy(dtype=float)
    dates = series["metric_date"].to_numpy()

    if len(values) < MIN_OBSERVATIONS_FOR_PELT:
        return []

    std = np.std(values)
    if np.isclose(std, 0.0):
        # A perfectly flat series has no changepoints by definition.
        return []

    # Standardize for detection so the penalty is scale-invariant across
    # KPIs with very different units (dollars vs. rates). Segment means /
    # shift_magnitude below are still computed from the original `values`.
    standardized = (values - np.mean(values)) / std

    algo = rpt.Pelt(model="l2", min_size=MIN_SEGMENT_LENGTH).fit(standardized)
    penalty = 3 * np.log(len(values)) * np.var(standardized)  # var ≈ 1.0 now
    breakpoints = algo.predict(pen=penalty)

    # ruptures returns breakpoint indices with the final index == len(values)
    # as a sentinel; drop it.
    breakpoints = [b for b in breakpoints if b < len(values)]

    # 1. Collect all candidates and compute magnitude using the original unstandardized values
    candidates: list[dict] = []
    
    for idx in breakpoints:
        if idx < MIN_SEGMENT_LENGTH or (len(values) - idx) < MIN_SEGMENT_LENGTH:
            continue

        before = values[max(0, idx - MIN_SEGMENT_LENGTH):idx]
        after = values[idx: idx + MIN_SEGMENT_LENGTH]
        if len(before) == 0 or len(after) == 0:
            continue

        mean_before = float(np.mean(before))
        mean_after = float(np.mean(after))

        candidates.append(
            {
                "idx": idx,
                "changepoint_date": pd.Timestamp(dates[idx]).date(),
                "mean_before": mean_before,
                "mean_after": mean_after,
                "shift_magnitude": mean_after - mean_before,
            }
        )

    # 2. Sort by absolute magnitude so we prioritize keeping the largest shifts
    candidates.sort(key=lambda c: abs(c["shift_magnitude"]), reverse=True)

    # 3. Greedily accept changepoints in order of magnitude, rejecting any within min_gap_days
    results: list[dict] = []
    kept_indices: list[int] = []
    
    for c in candidates:
        idx = c["idx"]
        
        # Check against already accepted breakpoints
        if any(abs(idx - kept_idx) < min_gap_days for kept_idx in kept_indices):
            continue

        kept_indices.append(idx)
        results.append(
            {
                "changepoint_date": c["changepoint_date"],
                "mean_before": c["mean_before"],
                "mean_after": c["mean_after"],
                "shift_magnitude": c["shift_magnitude"],
            }
        )

    # Return chronological order
    results.sort(key=lambda r: r["changepoint_date"])

    return results


def _load_kpi_series(kpi_column: str, region: str) -> pd.DataFrame:
    sql = """
        SELECT metric_date, %s AS value
        FROM mart.daily_kpi_evidence
        WHERE region = %%s
        ORDER BY metric_date
    """ % kpi_column
    return database.query_df(sql, params=(region,))


def run_signal_engine_for_all() -> None:
    """
    Evaluate each KPI/region.

    Skip sparse-history cases (fewer than 60 observed days per
    guardrails_engine.check_history_sufficiency).

    Persist to analysis.detected_changepoints.
    """
    contracts = load_contracts()
    regions_df = database.query_df(
        "SELECT DISTINCT region FROM mart.daily_kpi_evidence"
    )
    regions = regions_df["region"].tolist()

    insert_sql = """
        REPLACE INTO analysis.detected_changepoints (
            kpi, region, changepoint_date,
            segment_mean_before, segment_mean_after,
            shift_magnitude, method
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    rows_to_insert: list[tuple] = []

    for kpi_name, contract in contracts.items():
        gap_days = min_gap_days_for(contract)

        for region in regions:
            sufficiency = check_history_sufficiency(product_id=None, region=region)
            if sufficiency["status"] == "sparse_history":
                continue

            # KPI value itself may be a raw mart column (single-metric KPIs)
            # or must be computed from its components (multiply/divide KPIs).
            kpi_series = _resolve_kpi_series(kpi_name, contract, region)
            if kpi_series is None or kpi_series.empty:
                continue

            changepoints = detect_changepoints(
                kpi=kpi_name,
                region=region,
                series=kpi_series,
                min_gap_days=gap_days,
            )

            for cp in changepoints:
                rows_to_insert.append(
                    (
                        kpi_name,
                        region,
                        cp["changepoint_date"],
                        cp["mean_before"],
                        cp["mean_after"],
                        cp["shift_magnitude"],
                        "pelt_rbf",
                    )
                )

    if rows_to_insert:
        database.executemany(insert_sql, rows_to_insert)


def _resolve_kpi_series(
    kpi_name: str, contract: KpiContract, region: str
) -> pd.DataFrame | None:
    """
    Build the actual KPI value series for changepoint detection.

    Single-metric KPIs (e.g. stock_availability) read one mart column
    directly. Multiply/divide KPIs are computed from their two component
    columns so the signal engine watches the KPI itself, not a component.
    """
    from src.contracts import detect_formula_operator

    op = detect_formula_operator(contract.formula)

    if op == "single":
        col = contract.components[0].column
        return _load_kpi_series(col, region)

    if len(contract.components) != 2:
        return None

    col_a = contract.components[0].column
    col_b = contract.components[1].column

    sql = """
        SELECT metric_date, %s AS a, %s AS b
        FROM mart.daily_kpi_evidence
        WHERE region = %%s
        ORDER BY metric_date
    """ % (col_a, col_b)
    df = database.query_df(sql, params=(region,))
    if df.empty:
        return None

    if op == "multiply":
        df["value"] = df["a"] * df["b"]
    else:  # divide
        df["value"] = df["a"] / df["b"].replace({0: np.nan})

    return df[["metric_date", "value"]]