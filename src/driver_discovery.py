"""
Aletheia — src/driver_discovery.py

Stage 2: Driver discovery. Mechanical and contract-driven — no scoring
happens here. Builds three temporally separated datasets around a detected
changepoint, per Section 15's temporal split policy:

    |------ lag_selection_window ------| gap |--- incident_window ---|
                                        T

`model_training_window` reuses the same historical span as
`lag_selection_window` (a sufficiently long continuous historical run
ending strictly before the incident), so a single historical pull serves
both lag search and coefficient fitting without ever touching incident
rows.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from src import database
from src.contracts import KpiContract, RootDriver

HISTORICAL_LOOKBACK_DAYS = 90
VALIDATION_GAP_DAYS = 7
DEFAULT_INCIDENT_LENGTH_DAYS = 10


@dataclass
class DriverWindows:
    kpi: str
    region: str
    changepoint_date: dt.date
    lag_selection_window: pd.DataFrame
    model_training_window: pd.DataFrame
    incident_window: pd.DataFrame
    component_columns: dict[str, str]
    root_drivers: list[RootDriver]


def _window_bounds(
    changepoint_date: dt.date, incident_length_days: int
) -> dict[str, tuple[dt.date, dt.date]]:
    hist_end = changepoint_date - dt.timedelta(days=VALIDATION_GAP_DAYS + 1)
    hist_start = hist_end - dt.timedelta(days=HISTORICAL_LOOKBACK_DAYS)

    incident_start = changepoint_date
    incident_end = changepoint_date + dt.timedelta(days=incident_length_days)

    return {
        "historical": (hist_start, hist_end),
        "incident": (incident_start, incident_end),
    }


def _binarize_event_columns(df: pd.DataFrame, contract: KpiContract) -> pd.DataFrame:
    """
    Any root driver whose underlying database column is the categorical
    'event_type' column (values are either NULL or a specific event label,
    e.g. 'ambiguous_local_event') is converted here into a numeric 0.0/1.0
    indicator: 1.0 where event_type equals the driver's own name, else 0.0.

    This lets event-flag drivers flow through the same numeric evidence
    pipeline (stationarity, correlation, Ridge) as every other driver,
    instead of leaking a raw string column into code that expects floats.
    """
    if df.empty:
        return df

    for d in contract.root_drivers:
        if d.column == "event_type" and d.name in df.columns:
            df[d.name] = (df[d.name] == d.name).astype(float)

    return df


def discover_candidates(
    kpi: str,
    region: str,
    changepoint_date: dt.date,
    contract: KpiContract,
    incident_length_days: int = DEFAULT_INCIDENT_LENGTH_DAYS,
) -> DriverWindows:
    """
    Retrieve data according to the KPI contract.

    Build three temporally separated datasets:

    1. lag_selection_window
       Historical data ending before the incident (with a validation gap
       buffer immediately preceding the incident, per Section 15).

    2. model_training_window
       The same continuous historical span as (1) — a sufficiently long
       longitudinal historical window for fitting the Ridge/OLS models.

    3. incident_window
       The post-changepoint evaluation period.

    Returns components and root-driver series pulled from
    mart.daily_kpi_evidence. No scoring occurs here.
    """
    bounds = _window_bounds(changepoint_date, incident_length_days)
    hist_start, hist_end = bounds["historical"]
    incident_start, incident_end = bounds["incident"]

    columns_needed = {c.name: c.column for c in contract.components}
    for d in contract.root_drivers:
        columns_needed[d.name] = d.column

    # Alias every selected column to its driver/component NAME, not its
    # raw database column. Downstream code (evidence_engine.py,
    # pipeline.py) always indexes the resulting DataFrame by name — e.g.
    # historical_data['ambiguous_local_event'] — but that driver's
    # underlying column is 'event_type'. Without aliasing, the DataFrame
    # would only have a column literally called 'event_type' and every
    # such name/column mismatch would raise a KeyError.
    select_parts = sorted(
        f"{col} AS `{name}`" for name, col in columns_needed.items()
    )
    select_cols = "metric_date, " + ", ".join(select_parts)

    sql = f"""
        SELECT {select_cols}
        FROM mart.daily_kpi_evidence
        WHERE region = %s
          AND metric_date BETWEEN %s AND %s
        ORDER BY metric_date
    """

    historical_df = database.query_df(sql, params=(region, hist_start, hist_end))
    incident_df = database.query_df(sql, params=(region, incident_start, incident_end))

    historical_df = _binarize_event_columns(historical_df, contract)
    incident_df = _binarize_event_columns(incident_df, contract)

    return DriverWindows(
        kpi=kpi,
        region=region,
        changepoint_date=changepoint_date,
        lag_selection_window=historical_df,
        model_training_window=historical_df,
        incident_window=incident_df,
        component_columns=columns_needed,
        root_drivers=contract.root_drivers,
    )


def assert_no_incident_leakage(windows: DriverWindows) -> None:
    """
    Hard invariant check used by the leakage test in
    eval/validate_against_truth.py: no row date in the historical windows
    may be >= the changepoint date, and no row date in the incident window
    may be < the changepoint date.
    """
    if not windows.lag_selection_window.empty:
        max_hist_date = windows.lag_selection_window["metric_date"].max()
        if pd.Timestamp(max_hist_date).date() >= windows.changepoint_date:
            raise AssertionError(
                f"Temporal leakage: historical window for {windows.kpi}/"
                f"{windows.region} contains a row on or after the "
                f"changepoint date {windows.changepoint_date}."
            )

    if not windows.incident_window.empty:
        min_incident_date = windows.incident_window["metric_date"].min()
        if pd.Timestamp(min_incident_date).date() < windows.changepoint_date:
            raise AssertionError(
                f"Temporal leakage: incident window for {windows.kpi}/"
                f"{windows.region} contains a row before the changepoint "
                f"date {windows.changepoint_date}."
            )
