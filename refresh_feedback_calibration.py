"""
Refresh only the downstream feedback-calibration metadata in existing evidence bundles.

This utility does NOT rerun changepoint detection, lag search, evidence fitting,
decomposition, retrieval, or deterministic driver ranking.

Usage:
    python refresh_feedback_calibration.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import pandas as pd

from src import database, feedback_engine


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return number if math.isfinite(number) else None


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def main() -> None:
    cutoff = dt.datetime.utcnow()
    frame = database.query_df(
        """
        SELECT kpi, region, window_start, bundle_json
        FROM analysis.evidence_bundle
        ORDER BY kpi, region, window_start
        """
    )
    updated = 0
    skipped = 0

    for _, row in frame.iterrows():
        bundle = _loads(row.get("bundle_json"))
        decision = bundle.get("decision")
        if not isinstance(decision, dict):
            skipped += 1
            continue
        action_context = decision.get("action_context")
        if not isinstance(action_context, dict):
            skipped += 1
            continue

        kpi = str(row["kpi"])
        region = str(row["region"])
        window_start = pd.Timestamp(row["window_start"]).date()
        calibrated = feedback_engine.apply_feedback_to_action_context(
            kpi=kpi,
            region=region,
            window_start=window_start,
            action_context=action_context,
            analysis_started_at=cutoff,
        )
        decision["action_context"] = calibrated
        decision["feedback_calibration"] = calibrated.get("feedback_calibration", {})
        bundle["decision"] = decision

        payload = json.dumps(
            _json_safe(bundle),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        database.execute(
            """
            UPDATE analysis.evidence_bundle
            SET bundle_json = %s
            WHERE kpi = %s AND region = %s AND window_start = %s
            """,
            (payload, kpi, region, window_start),
        )
        updated += 1

    print(f"Feedback calibration refresh complete: updated={updated}, skipped={skipped}")
    print("Deterministic RCA tables were not recomputed or modified.")


if __name__ == "__main__":
    main()
