"""Validate that feedback is used downstream without mutating deterministic RCA evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src import database

SNAPSHOT_PATH = Path("data/generated/feedback_demo_snapshot.json")


def _loads(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalized_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.where(pd.notna(frame), None)
    records: list[dict[str, Any]] = []
    for item in clean.to_dict(orient="records"):
        converted: dict[str, Any] = {}
        for key, value in item.items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            converted[key] = value
        records.append(converted)
    return records


def main() -> None:
    if not SNAPSHOT_PATH.exists():
        raise SystemExit(
            "Missing feedback demo snapshot. Run: "
            "python data/generate_feedback_demo.py --sentiment positive --count 8 --snapshot"
        )

    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    target = snapshot["target"]

    bundle_df = database.query_df(
        """
        SELECT bundle_json FROM analysis.evidence_bundle
        WHERE kpi = %s AND region = %s AND window_start = %s
        """,
        params=(target["kpi"], target["region"], target["window_start"]),
    )
    if bundle_df.empty:
        raise SystemExit("Target evidence bundle no longer exists.")

    bundle = _loads(bundle_df.iloc[0]["bundle_json"])
    decision = bundle.get("decision", {}) if isinstance(bundle.get("decision"), dict) else {}
    action = decision.get("action_context", {}) if isinstance(decision.get("action_context"), dict) else {}
    calibration = action.get("feedback_calibration", {}) if isinstance(action.get("feedback_calibration"), dict) else {}

    evidence = database.query_df(
        """
        SELECT driver_name, explains_component, evidence_mode, model_status,
               best_lag_days, baseline_value, incident_value,
               historical_coefficient, p_value, is_significant,
               coefficient_stability, driver_zscore, evidence_score,
               normalized_score, softmax_probability
        FROM analysis.driver_evidence
        WHERE kpi = %s AND region = %s AND window_start = %s
        ORDER BY driver_name, explains_component
        """,
        params=(target["kpi"], target["region"], target["window_start"]),
    )
    decomp = database.query_df(
        """
        SELECT decomposition_type, effect_a, effect_b, interaction_effect,
               residual, total_change, is_volatile, narrative_mode
        FROM analysis.pvm_decomposition
        WHERE kpi = %s AND region = %s AND window_start = %s
        """,
        params=(target["kpi"], target["region"], target["window_start"]),
    )

    evidence_same = _normalized_records(evidence) == snapshot.get("driver_evidence", [])
    decomp_same = _normalized_records(decomp) == snapshot.get("decomposition", [])
    applied = bool(calibration.get("applied"))
    sample_count = int(calibration.get("sample_count", 0) or 0)
    adjustment = float(calibration.get("adjustment", 0.0) or 0.0)

    print("Feedback integration validation")
    print(f"Target: {target['kpi']} / {target['region']} / {target['window_start']}")
    print(f"Feedback applied: {applied}")
    print(f"Feedback samples: {sample_count}")
    print(f"Feedback adjustment: {adjustment:+.4f}")
    print(f"Base readiness: {calibration.get('base_readiness_score')}")
    print(f"Adjusted readiness: {calibration.get('adjusted_readiness_score')}")
    print(f"Driver evidence unchanged: {evidence_same}")
    print(f"Decomposition unchanged: {decomp_same}")

    failures = []
    if not applied or sample_count <= 0:
        failures.append("feedback was not applied")
    if abs(adjustment) > 0.1000001:
        failures.append("feedback adjustment exceeded the ±0.10 bound")
    if not evidence_same:
        failures.append("deterministic driver evidence changed")
    if not decomp_same:
        failures.append("deterministic decomposition changed")

    if failures:
        print("FAIL:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("PASS: feedback changed only downstream readiness calibration; deterministic RCA remained unchanged.")


if __name__ == "__main__":
    main()
