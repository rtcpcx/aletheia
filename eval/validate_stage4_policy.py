"""Read-only validation/report for Aletheia Stage-4 retrieval policy.

Run after a full V3 pipeline rerun:
    python -m eval.validate_stage4_policy

This is not a causal-accuracy benchmark. It checks that stored retrieval plans
respect routing/grounding policy and prints the cases where Stage 4 actually ran.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from src import database, orchestrator


def _obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def main() -> None:
    df = database.query_df(
        """
        SELECT kpi, region, window_start, bundle_json
        FROM analysis.evidence_bundle
        ORDER BY region, kpi, window_start
        """
    )

    failures: list[str] = []
    warnings: list[str] = []
    routes: Counter[str] = Counter()
    planned = 0

    print("Stage-4 policy validation")
    print("case | action_level | target | query | retrieved_rows")

    for _, row in df.iterrows():
        bundle = _obj(row.get("bundle_json"))
        decision = bundle.get("decision") if isinstance(bundle.get("decision"), dict) else {}
        action = decision.get("action_context") if isinstance(decision.get("action_context"), dict) else {}
        ranked = (
            bundle.get("deterministic_evidence", {}).get("ranked_drivers", [])
            if isinstance(bundle.get("deterministic_evidence"), dict)
            else []
        )
        ranked = ranked if isinstance(ranked, list) else []
        plan_raw = bundle.get("retrieval_plan")
        plan = plan_raw if isinstance(plan_raw, dict) else None
        retrieved = bundle.get("retrieved_evidence")
        retrieved = retrieved if isinstance(retrieved, list) else []

        case = f"{row['kpi']}/{row['region']}/{row['window_start']}"
        action_level = str(action.get("action_level") or "unknown")

        if plan is None:
            routes["deterministic_only"] += 1
            if retrieved:
                failures.append(f"{case}: retrieved evidence exists without a stored retrieval plan")
            continue

        planned += 1
        target = str(plan.get("retrieval_target") or "").strip().lower()
        query = plan.get("retrieval_query")
        question = str(plan.get("clarification_question") or "").strip()
        routes[target or "invalid"] += 1

        print(f"{case} | {action_level} | {target or '<missing>'} | {query or '-'} | {len(retrieved)}")

        if action_level in {"monitor", "data_quality_first"}:
            failures.append(f"{case}: Stage 4 ran for action level {action_level}")

        if target not in {"web", "internal", "none"}:
            failures.append(f"{case}: invalid retrieval target '{target}'")
            continue

        if target == "web" and not query:
            failures.append(f"{case}: web plan has no query")
        if target != "web" and query:
            failures.append(f"{case}: non-web plan contains a web query")
        if target != "web" and retrieved:
            failures.append(f"{case}: non-web plan has retrieved public evidence")

        if question.count("?") != 1:
            failures.append(f"{case}: clarification question must contain exactly one question")

        # Re-run the same deterministic grounding validator used by Stage 4.
        if target == "web" and query and ranked:
            context = orchestrator.build_retrieval_context(decision, ranked)
            eligible_public = [
                c
                for c in context.get("candidate_hypotheses", [])
                if isinstance(c, dict) and c.get("public_web_eligible") is True
            ]
            if not eligible_public:
                failures.append(
                    f"{case}: web route has no decision-relevant public candidate"
                )
            stored_plan = orchestrator.RetrievalPlan(
                clarification_question=question,
                retrieval_query=str(query),
                retrieval_target="web",
                planner_status=str(plan.get("planner_status") or "stored"),
                planner_reason=plan.get("planner_reason"),
            )
            valid, reason = orchestrator._validate_plan(stored_plan, context)
            if not valid:
                failures.append(f"{case}: stored web plan fails grounding policy: {reason}")

        if target == "web" and not retrieved:
            warnings.append(
                f"{case}: web retrieval was requested but no usable source rows were stored "
                "(network/search failure is allowed; inspect the query if persistent)"
            )

    print(f"Bundles checked: {len(df)}")
    print(f"Stage-4 plans: {planned}")
    print("Routes:", dict(sorted(routes.items())))
    print(f"Policy failures: {len(failures)}")
    for item in failures:
        print("FAIL", item)
    print(f"Warnings: {len(warnings)}")
    for item in warnings:
        print("WARN", item)

    if failures:
        raise SystemExit(1)
    print("PASS: stored Stage-4 plans respect routing, grounding, and retrieval-precision policy.")


if __name__ == "__main__":
    main()
