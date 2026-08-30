"""Validate Aletheia's decision-language layer against internal consistency rules.

Run after refresh_decision_language.py:
    python -m eval.validate_action_language

This is not a causal-accuracy benchmark. It checks whether the presentation/action
layer contradicts its own stored evidence semantics.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from src import database, narrator


def _bundle(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw) if raw else {}
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def main() -> None:
    df = database.query_df(
        """
        SELECT kpi, region, window_start, bundle_json
        FROM analysis.evidence_bundle
        ORDER BY region, kpi, window_start
        """
    )

    failures: list[str] = []
    levels: Counter[str] = Counter()
    checked = 0

    for _, row in df.iterrows():
        bundle = _bundle(row.get("bundle_json"))
        decision = bundle.get("decision", {}) if isinstance(bundle.get("decision"), dict) else {}
        action = decision.get("action_context", {}) if isinstance(decision.get("action_context"), dict) else {}
        if not action:
            continue

        checked += 1
        case = f"{row['kpi']}/{row['region']}/{row['window_start']}"
        level = str(action.get("action_level") or "unknown")
        levels[level] += 1
        direction = str(action.get("driver_direction") or "")
        movement = str(action.get("movement_status") or "")
        alignment = str(action.get("direction_alignment") or "")
        recommendation = str(action.get("recommended_action") or "")

        if level == "act" and movement != "material_movement":
            failures.append(f"{case}: direct action without material driver movement")
        if level == "act" and direction in {"unchanged", "unknown", ""}:
            failures.append(f"{case}: direct action targets a non-moved/unknown driver ({direction})")
        if level == "act" and alignment == "contradictory":
            failures.append(f"{case}: direct action targets directionally contradictory evidence")
        if level == "validate_unexplained" and recommendation.lower().startswith(("work on ", "prioritize ")):
            failures.append(f"{case}: unexplained case still recommends a direct lever")
        if level == "data_quality_first" and "rerun" not in recommendation.lower():
            failures.append(f"{case}: data-quality action does not tell the user to rerun RCA")

        readiness = str(action.get("action_readiness") or "")
        if not readiness:
            failures.append(f"{case}: action_readiness is missing")
        if level == "act" and not action.get("action_target_driver"):
            failures.append(f"{case}: act case has no explicit action_target_driver")
        if level in {"validate", "validate_unexplained"} and readiness.lower().startswith("act"):
            failures.append(f"{case}: validation case is mislabeled as action-ready")

        explanation = str(action.get("leading_explanation_driver") or "")
        action_target = str(action.get("action_target_driver") or "")
        differs = bool(action.get("explanation_differs_from_action_target"))
        if differs and (not explanation or not action_target or explanation == action_target):
            failures.append(f"{case}: explanation/action-target distinction metadata is inconsistent")

        narrations = bundle.get("narration", {}) if isinstance(bundle.get("narration"), dict) else {}
        for persona, payload in narrations.items():
            if not isinstance(payload, dict):
                continue
            text = " ".join(
                [
                    str(payload.get("headline") or ""),
                    str(payload.get("narrative") or ""),
                    " ".join(str(v) for v in payload.get("caveats", []) if v),
                ]
            ).lower()
            for term in narrator.FORBIDDEN_NARRATIVE_TERMS:
                if term in text:
                    failures.append(f"{case}/{persona}: narration leaked forbidden term '{term}'")
                    break

    print("Action-language validation")
    print(f"Checked bundles: {checked}")
    print("Action levels:", dict(sorted(levels.items())))
    print(f"Consistency failures: {len(failures)}")
    for item in failures:
        print("FAIL", item)

    if failures:
        raise SystemExit(1)
    print("PASS: no action/narration consistency violations found.")


if __name__ == "__main__":
    main()
