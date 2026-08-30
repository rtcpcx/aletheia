"""
Generate clearly labelled demonstration feedback for an existing Aletheia decision.

Usage:
    python data/generate_feedback_demo.py --sentiment positive --count 8 --snapshot
    python data/generate_feedback_demo.py --sentiment negative --count 8 --snapshot
    python data/generate_feedback_demo.py --clear

Demo rows are marked is_demo=1. They are ignored by the pipeline unless
ALETHEIA_FEEDBACK_INCLUDE_DEMO=1 is explicitly enabled.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Allow direct execution from the repository root:
#     python data/generate_feedback_demo.py ...
# Python otherwise places only data/ on sys.path for a path-executed script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import database, feedback_engine

SNAPSHOT_PATH = Path("data/generated/feedback_demo_snapshot.json")


def _json_load(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _candidate_bundles() -> list[dict[str, Any]]:
    frame = database.query_df(
        """
        SELECT kpi, region, window_start, bundle_json
        FROM analysis.evidence_bundle
        ORDER BY window_start DESC, kpi, region
        """
    )
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        bundle = _json_load(row.get("bundle_json"))
        decision = bundle.get("decision", {}) if isinstance(bundle.get("decision"), dict) else {}
        action = decision.get("action_context", {}) if isinstance(decision.get("action_context"), dict) else {}
        driver = str(action.get("primary_driver") or "").strip()
        level = str(action.get("action_level") or "").strip()
        if driver and level:
            rows.append(
                {
                    "kpi": str(row["kpi"]),
                    "region": str(row["region"]),
                    "window_start": str(row["window_start"]),
                    "primary_driver": driver,
                    "action_level": level,
                }
            )
    return rows


def _choose_target(args: argparse.Namespace) -> dict[str, Any]:
    candidates = _candidate_bundles()
    if args.kpi:
        candidates = [c for c in candidates if c["kpi"] == args.kpi]
    if args.region:
        candidates = [c for c in candidates if c["region"] == args.region]
    if args.window_start:
        candidates = [c for c in candidates if c["window_start"] == args.window_start]
    if not candidates:
        raise RuntimeError(
            "No completed evidence bundle with a primary driver/action level was found. "
            "Run `python -u -m src.pipeline` first."
        )
    return candidates[0]


def _deterministic_snapshot(target: dict[str, Any]) -> dict[str, Any]:
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

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        clean = frame.where(pd.notna(frame), None)
        result = []
        for item in clean.to_dict(orient="records"):
            converted = {}
            for key, value in item.items():
                if isinstance(value, (dt.date, dt.datetime, pd.Timestamp)):
                    converted[key] = str(value)
                elif hasattr(value, "item"):
                    try:
                        converted[key] = value.item()
                    except Exception:
                        converted[key] = value
                else:
                    converted[key] = value
            result.append(converted)
        return result

    return {
        "target": target,
        "created_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
        "driver_evidence": records(evidence),
        "decomposition": records(decomp),
    }


def _write_snapshot(target: dict[str, Any]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(
        json.dumps(_deterministic_snapshot(target), indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Wrote deterministic snapshot: {SNAPSHOT_PATH}")


def _clear() -> None:
    database.execute("DELETE FROM app.user_feedback WHERE COALESCE(is_demo, 0) = 1")
    print("Cleared demonstration feedback rows.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentiment", choices=["positive", "negative", "mixed"], default="positive")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--kpi")
    parser.add_argument("--region")
    parser.add_argument("--window-start")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if args.clear:
        _clear()
        return
    if args.count < 1:
        raise ValueError("--count must be >= 1")

    target = _choose_target(args)
    if args.snapshot:
        _write_snapshot(target)

    dispositions: list[str]
    if args.sentiment == "positive":
        dispositions = ["Helpful"] * args.count
    elif args.sentiment == "negative":
        dispositions = ["Not helpful"] * args.count
    else:
        cycle = ["Helpful", "Unclear", "Not helpful"]
        dispositions = [cycle[i % len(cycle)] for i in range(args.count)]

    for i, disposition in enumerate(dispositions, start=1):
        feedback_engine.record_feedback(
            persona="Demo evaluator",
            region=target["region"],
            kpi=target["kpi"],
            window_start=target["window_start"],
            primary_driver=target["primary_driver"],
            action_level=target["action_level"],
            disposition=disposition,
            comment=f"FEEDBACK_DEMO_{args.sentiment.upper()}_{i:03d}",
            is_demo=True,
        )

    print("Inserted demo feedback:")
    print(json.dumps({**target, "sentiment": args.sentiment, "count": args.count}, indent=2))
    print("To allow the pipeline/refresh to consume demo rows in this shell:")
    print('  $env:ALETHEIA_FEEDBACK_INCLUDE_DEMO="1"')


if __name__ == "__main__":
    main()
