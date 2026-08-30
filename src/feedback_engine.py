"""
Aletheia — src/feedback_engine.py

Governed downstream feedback calibration.

Feedback never changes deterministic RCA evidence, selected lags, changepoints,
model coefficients, decomposition, or driver ranking. It only adds an auditable,
bounded calibration signal to the action context produced downstream of RCA.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

from src import database


DISPOSITION_VALUE = {
    "helpful": 1.0,
    "unclear": 0.0,
    "not helpful": -1.0,
}

BASE_READINESS_BY_ACTION = {
    "act": 0.85,
    "validate": 0.60,
    "validate_unexplained": 0.45,
    "monitor": 0.35,
    "data_quality_first": 0.20,
}

PRIOR_STRENGTH = 5.0
MAX_ABSOLUTE_ADJUSTMENT = 0.10
CROSS_REGION_WEIGHT = 0.50


@dataclass(frozen=True)
class FeedbackSignal:
    applied: bool
    sample_count: int
    same_region_count: int
    effective_sample_size: float
    weighted_mean: float
    reliability: float
    adjustment: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def base_readiness_score(action_level: str | None) -> float:
    return BASE_READINESS_BY_ACTION.get(str(action_level or "").strip().lower(), 0.50)


def _include_demo_feedback() -> bool:
    raw = os.getenv("ALETHEIA_FEEDBACK_INCLUDE_DEMO", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _analysis_cutoff_epoch(value: dt.datetime) -> float:
    """Convert the Python analysis cutoff to an absolute Unix timestamp.

    Aletheia internally uses naive UTC datetimes for run timestamps. MySQL
    TIMESTAMP values, however, are displayed/compared in the active MySQL
    session timezone. Passing a naive UTC datetime directly can therefore make
    valid feedback appear to be "from the future" on non-UTC servers.

    Comparing against FROM_UNIXTIME(epoch) keeps both sides in the MySQL
    session timezone while preserving the same absolute instant.
    """
    if value.tzinfo is None:
        aware = value.replace(tzinfo=dt.timezone.utc)
    else:
        aware = value.astimezone(dt.timezone.utc)
    return aware.timestamp()


def _no_signal(reason: str) -> FeedbackSignal:
    return FeedbackSignal(
        applied=False,
        sample_count=0,
        same_region_count=0,
        effective_sample_size=0.0,
        weighted_mean=0.0,
        reliability=0.0,
        adjustment=0.0,
        reason=reason,
    )


def load_feedback_signal(
    *,
    kpi: str,
    region: str,
    primary_driver: str | None,
    action_level: str | None,
    analysis_started_at: dt.datetime,
) -> FeedbackSignal:
    """Load historical feedback for a comparable decision and compute a bounded signal.

    Matching is deliberately narrow: same KPI, same primary driver and same action
    level. Same-region feedback receives full weight; cross-region feedback receives
    half weight. Only feedback created strictly before this analysis execution may be
    used, preventing the current output from rewriting itself in-place.
    """
    driver = str(primary_driver or "").strip()
    level = str(action_level or "").strip()
    if not driver or not level:
        return _no_signal("No primary driver/action level is available for feedback matching.")

    demo_clause = "" if _include_demo_feedback() else "AND COALESCE(is_demo, 0) = 0"
    sql = f"""
        SELECT disposition, region, created_at, COALESCE(is_demo, 0) AS is_demo
        FROM app.user_feedback
        WHERE kpi = %s
          AND primary_driver = %s
          AND action_level = %s
          AND created_at < FROM_UNIXTIME(%s)
          {demo_clause}
        ORDER BY created_at
    """
    cutoff_epoch = _analysis_cutoff_epoch(analysis_started_at)
    try:
        frame = database.query_df(
            sql,
            params=(kpi, driver, level, cutoff_epoch),
        )
    except Exception as exc:
        # Feedback is a downstream calibration layer. A feedback-store outage must
        # never break or alter deterministic RCA; fail closed to an exact no-op.
        return _no_signal(f"Feedback store unavailable; calibration skipped ({type(exc).__name__}).")

    if frame.empty:
        return _no_signal("No eligible historical feedback exists for this decision pattern.")

    weighted_sum = 0.0
    total_weight = 0.0
    same_region_count = 0
    valid_count = 0

    for _, row in frame.iterrows():
        disposition = str(row.get("disposition") or "").strip().lower()
        if disposition not in DISPOSITION_VALUE:
            continue
        row_region = str(row.get("region") or "").strip()
        same_region = row_region == str(region)
        weight = 1.0 if same_region else CROSS_REGION_WEIGHT
        weighted_sum += DISPOSITION_VALUE[disposition] * weight
        total_weight += weight
        valid_count += 1
        if same_region:
            same_region_count += 1

    if valid_count == 0 or total_weight <= 0:
        return _no_signal("Historical feedback rows were present but contained no valid dispositions.")

    weighted_mean = weighted_sum / total_weight
    reliability = total_weight / (total_weight + PRIOR_STRENGTH)
    adjustment = _clamp(
        MAX_ABSOLUTE_ADJUSTMENT * weighted_mean * reliability,
        -MAX_ABSOLUTE_ADJUSTMENT,
        MAX_ABSOLUTE_ADJUSTMENT,
    )

    return FeedbackSignal(
        applied=True,
        sample_count=valid_count,
        same_region_count=same_region_count,
        effective_sample_size=round(total_weight, 6),
        weighted_mean=round(weighted_mean, 6),
        reliability=round(reliability, 6),
        adjustment=round(adjustment, 6),
        reason="Bounded historical user-feedback calibration applied downstream of deterministic RCA.",
    )


def apply_feedback_to_action_context(
    *,
    kpi: str,
    region: str,
    window_start: dt.date,
    action_context: dict[str, Any],
    analysis_started_at: dt.datetime,
) -> dict[str, Any]:
    """Return an enriched copy of action_context with bounded feedback calibration.

    The function intentionally preserves action_level, primary_driver,
    recommended_action and every deterministic evidence field. The only mutation is
    the addition of ``feedback_calibration`` metadata.
    """
    enriched = dict(action_context or {})
    action_level = str(enriched.get("action_level") or "")
    primary_driver = enriched.get("primary_driver")
    base = base_readiness_score(action_level)

    signal = load_feedback_signal(
        kpi=kpi,
        region=region,
        primary_driver=str(primary_driver) if primary_driver is not None else None,
        action_level=action_level,
        analysis_started_at=analysis_started_at,
    )
    adjusted = _clamp(base + signal.adjustment, 0.0, 1.0)

    calibration = signal.to_dict()
    calibration.update(
        {
            "kpi": kpi,
            "region": region,
            "window_start": str(window_start),
            "matched_primary_driver": primary_driver,
            "matched_action_level": action_level or None,
            "base_readiness_score": round(base, 6),
            "adjusted_readiness_score": round(adjusted, 6),
            "analysis_cutoff": analysis_started_at.isoformat(timespec="seconds"),
            "demo_feedback_included": _include_demo_feedback(),
            "governance": (
                "Feedback calibrates downstream action readiness only; deterministic "
                "RCA evidence, ranking, lag, decomposition and retrieval evidence are unchanged."
            ),
        }
    )
    enriched["feedback_calibration"] = calibration
    return enriched


def record_feedback(
    *,
    persona: str,
    region: str,
    kpi: str,
    window_start: str | dt.date | None,
    primary_driver: str | None,
    action_level: str | None,
    disposition: str,
    comment: str = "",
    is_demo: bool = False,
) -> None:
    """Persist feedback with enough decision context to support future calibration."""
    normalized = str(disposition or "").strip().lower()
    if normalized not in DISPOSITION_VALUE:
        raise ValueError("disposition must be Helpful, Not helpful, or Unclear")

    database.execute(
        """
        INSERT INTO app.user_feedback (
            persona, region, kpi, window_start, primary_driver, action_level,
            disposition, comment_text, is_demo
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            persona,
            region,
            kpi,
            window_start,
            primary_driver,
            action_level,
            disposition,
            comment,
            1 if is_demo else 0,
        ),
    )
