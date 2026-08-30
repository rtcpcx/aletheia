"""Refresh only Aletheia Stage-4 retrieval/fusion from existing deterministic RCA.

Use this after changing orchestrator.py / retrieval_hypotheses.yaml when the
statistical evidence, changepoints, lags, decomposition and source-health logic
have NOT changed.

What this script DOES:
- clears only Stage-4 derived tables:
    analysis.retrieved_context
    analysis.orchestrator_updates
    analysis.clarification_requests
- re-evaluates the Stage-4 gate and retrieval plan from the deterministic
  evidence already stored in analysis.evidence_bundle
- reruns public retrieval only for newly-approved web routes
- reapplies bounded fusion
- rebuilds the final action recommendation and persona narration
- updates evidence_bundle and decision_packets.recommended_action

What this script DOES NOT do:
- no changepoint detection
- no lag search
- no regression / ADF / stationarity work
- no decomposition recomputation
- no raw.* or mart.* writes

Run from the repository root:
    python refresh_stage4_policy.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
from typing import Any

import pandas as pd

from src import action_engine, database, evidence_fusion, narrator, orchestrator, retrieval


def _obj(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        value = json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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


def _json_dumps(value: Any) -> str:
    return json.dumps(
        _json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )


def _clear_stage4_tables() -> None:
    for table in (
        "analysis.retrieved_context",
        "analysis.orchestrator_updates",
        "analysis.clarification_requests",
    ):
        database.execute(f"TRUNCATE TABLE {table}")


def _persist_clarification(
    *,
    kpi: str,
    region: str,
    window_start: dt.date,
    reason: str,
    question: str,
) -> None:
    database.execute(
        """
        INSERT INTO analysis.clarification_requests (
            kpi, region, window_start, reason, requested_clarification, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        params=(kpi, region, window_start, reason, question, dt.datetime.utcnow()),
    )


def _stage4_reason(decision: dict[str, Any], ranked: list[dict[str, Any]]) -> str:
    if orchestrator.requires_external_verification(decision, ranked):
        return "decision-relevant external hypothesis requires independent public corroboration"
    action = decision.get("action_context")
    if isinstance(action, dict) and action.get("is_ambiguous") is True:
        return "deterministic evidence remains ambiguous"
    if str(decision.get("confidence_level") or "").lower() == "low":
        return "overall confidence is Low"
    try:
        if float(decision.get("top_probability_gap", 1.0)) < orchestrator.TOP_PROBABILITY_GAP_THRESHOLD:
            return "leading hypotheses are insufficiently separated"
    except (TypeError, ValueError):
        pass
    if decision.get("any_driver_significant") is False:
        return "no candidate driver passed the validity gate"
    return "deterministic evidence requires clarification"


def _deterministic_rows(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    det = bundle.get("deterministic_evidence")
    rows = det.get("ranked_drivers") if isinstance(det, dict) else None
    if not isinstance(rows, list):
        return []
    clean: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        try:
            base = float(row.get("softmax_probability") or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        row["softmax_probability"] = base
        # Remove the effect of the previous Stage-4 run before re-evaluating policy.
        row["fused_probability"] = base
        clean.append(row)
    return clean


def _action_context(
    *,
    bundle: dict[str, Any],
    decision: dict[str, Any],
    ranked: list[dict[str, Any]],
) -> dict[str, Any]:
    decomp = bundle.get("decomposition")
    decomp = decomp if isinstance(decomp, dict) else {}
    source_health = decision.get("source_health")
    source_health = source_health if isinstance(source_health, dict) else {"healthy": False}

    impacts = decomp.get("component_impacts")
    impacts = impacts if isinstance(impacts, dict) else {}
    baselines = decomp.get("component_baselines")
    baselines = baselines if isinstance(baselines, dict) else {}
    incidents = decomp.get("component_incidents")
    incidents = incidents if isinstance(incidents, dict) else {}

    pct = decision.get("percent_change")
    try:
        pct = float(pct) if pct is not None else None
    except (TypeError, ValueError):
        pct = None

    return action_engine.build_action_context(
        kpi=str(decision.get("kpi") or bundle.get("kpi") or ""),
        region=str(decision.get("region") or bundle.get("region") or ""),
        decision_type=str(decision.get("decision_type") or "investigate"),
        confidence=str(decision.get("confidence_level") or decision.get("confidence") or "Low"),
        ranked_drivers=ranked,
        source_health=source_health,
        component_impacts=impacts,
        component_baselines=baselines,
        component_incidents=incidents,
        complex_interaction=(
            str(decomp.get("narrative_mode") or "") == "complex_interaction"
            or bool(decomp.get("is_volatile"))
        ),
        kpi_relative_change=pct,
    )


def _persist_bundle(
    *, kpi: str, region: str, window_start: dt.date, bundle: dict[str, Any]
) -> None:
    database.execute(
        """
        REPLACE INTO analysis.evidence_bundle (
            kpi, region, window_start, bundle_json, generated_at
        ) VALUES (%s, %s, %s, %s, %s)
        """,
        params=(
            kpi,
            region,
            window_start,
            _json_dumps(bundle),
            dt.datetime.utcnow(),
        ),
    )


def _persist_recommendation(
    *, kpi: str, region: str, window_start: dt.date, recommendation: str
) -> None:
    database.execute(
        """
        UPDATE analysis.decision_packets
        SET recommended_action = %s
        WHERE kpi = %s AND region = %s AND window_start = %s
        """,
        params=(recommendation, kpi, region, window_start),
    )


def main() -> None:
    bundles = database.query_df(
        """
        SELECT kpi, region, window_start, bundle_json
        FROM analysis.evidence_bundle
        ORDER BY region, kpi, window_start
        """
    )
    if bundles.empty:
        print("No evidence bundles found; nothing to refresh.")
        return

    _clear_stage4_tables()

    routes = {"deterministic_only": 0, "web": 0, "internal": 0, "none": 0}
    updated = 0

    for _, record in bundles.iterrows():
        kpi = str(record["kpi"])
        region = str(record["region"])
        window_start = pd.Timestamp(record["window_start"]).date()
        bundle = _obj(record.get("bundle_json"))
        ranked = _deterministic_rows(bundle)
        decision = bundle.get("decision")
        decision = dict(decision) if isinstance(decision, dict) else {}
        decision.setdefault("kpi", kpi)
        decision.setdefault("region", region)
        decision.setdefault("window_start", str(window_start))

        # Rebuild the action layer from the untouched deterministic probabilities
        # before deciding whether Stage 4 should run.
        deterministic_action = _action_context(
            bundle=bundle, decision=decision, ranked=ranked
        )
        decision["action_context"] = deterministic_action
        decision["recommended_action"] = deterministic_action.get("recommended_action")

        retrieved_payload: list[dict[str, Any]] = []
        plan_payload: dict[str, Any] | None = None
        final_probs = {
            str(row.get("driver_name") or ""): float(row.get("softmax_probability") or 0.0)
            for row in ranked
            if str(row.get("driver_name") or "")
        }

        if ranked and orchestrator.needs_clarification(decision, ranked):
            plan = orchestrator.generate_retrieval_plan(decision, ranked)
            plan_payload = plan.to_dict()
            routes[plan.retrieval_target] = routes.get(plan.retrieval_target, 0) + 1
            _persist_clarification(
                kpi=kpi,
                region=region,
                window_start=window_start,
                reason=_stage4_reason(decision, ranked),
                question=plan.clarification_question,
            )

            if plan.retrieval_target == "web" and plan.retrieval_query:
                results = retrieval.retrieve_evidence(plan.retrieval_query) or []
                candidates = orchestrator.select_retrieval_candidates(
                    ranked, limit=2, decision_packet=decision
                )
                hypotheses = [
                    str(item.get("driver_name") or "").strip()
                    for item in candidates
                    if str(item.get("driver_name") or "").strip()
                ]

                if results and hypotheses:
                    try:
                        batch = retrieval.score_relevance_batch(
                            question=plan.clarification_question,
                            retrieval_query=plan.retrieval_query,
                            hypotheses=hypotheses,
                            results=results,
                        )
                    except Exception:
                        batch = {"hypothesis_scores": {}, "source_assessments": []}
                else:
                    batch = {"hypothesis_scores": {}, "source_assessments": []}

                hypothesis_scores = batch.get("hypothesis_scores", {})
                hypothesis_scores = hypothesis_scores if isinstance(hypothesis_scores, dict) else {}
                retrieval_support = {
                    name: float(
                        (hypothesis_scores.get(name) or {}).get("effective_support", 0.0)
                    )
                    for name in hypotheses
                }
                assessments = batch.get("source_assessments", [])
                assessments = assessments if isinstance(assessments, list) else []
                assessment_map = {
                    (int(a.get("source_index", -1)), str(a.get("hypothesis") or "")): a
                    for a in assessments
                    if isinstance(a, dict)
                }

                for source_index, result in enumerate(results, start=1):
                    for name in hypotheses:
                        assessment = assessment_map.get(
                            (source_index, name),
                            {
                                "support": 0.0,
                                "confidence": 0.0,
                                "reason": "no source-level assessment",
                            },
                        )
                        support = float(assessment.get("support", 0.0) or 0.0)
                        confidence = float(assessment.get("confidence", 0.0) or 0.0)
                        retrieval.store_retrieved_context(
                            kpi=kpi,
                            region=region,
                            window_start=window_start,
                            hypothesis=name,
                            query=plan.retrieval_query,
                            result=result,
                            support=support,
                            confidence=confidence,
                        )
                        retrieved_payload.append(
                            {
                                "hypothesis": name,
                                "clarification_question": plan.clarification_question,
                                "query": plan.retrieval_query,
                                "source_title": result.get("title"),
                                "source_url": result.get("source_url"),
                                "support": support,
                                "confidence": confidence,
                                "reason": assessment.get("reason"),
                            }
                        )

                if hypotheses:
                    deterministic_scores = {
                        str(r.get("driver_name") or ""): float(r.get("normalized_score") or 0.0)
                        for r in ranked
                        if str(r.get("driver_name") or "")
                    }
                    deterministic_probabilities = {
                        str(r.get("driver_name") or ""): float(r.get("softmax_probability") or 0.0)
                        for r in ranked
                        if str(r.get("driver_name") or "")
                    }
                    final_probs = evidence_fusion.fuse_and_log(
                        kpi=kpi,
                        region=region,
                        window_start=window_start,
                        deterministic_scores=deterministic_scores,
                        deterministic_probabilities=deterministic_probabilities,
                        retrieval_support=retrieval_support,
                        retrieval_query=plan.retrieval_query,
                    )
        else:
            routes["deterministic_only"] += 1

        for row in ranked:
            name = str(row.get("driver_name") or "")
            row["fused_probability"] = float(
                final_probs.get(name, row.get("softmax_probability") or 0.0)
            )

        final_action = _action_context(bundle=bundle, decision=decision, ranked=ranked)
        decision["action_context"] = final_action
        decision["recommended_action"] = final_action.get("recommended_action")

        bundle["deterministic_evidence"] = {"ranked_drivers": ranked}
        bundle["retrieval_plan"] = plan_payload
        bundle["retrieved_evidence"] = retrieved_payload
        bundle["decision"] = decision

        narrations: dict[str, Any] = {}
        for persona in ("Executive", "Growth analyst"):
            try:
                narrations[persona] = narrator.narrate(bundle, persona)
            except Exception:
                narrations[persona] = narrator.template_narrate(bundle, persona)
        bundle["narration"] = narrations

        _persist_recommendation(
            kpi=kpi,
            region=region,
            window_start=window_start,
            recommendation=str(final_action.get("recommended_action") or ""),
        )
        _persist_bundle(
            kpi=kpi, region=region, window_start=window_start, bundle=bundle
        )
        updated += 1
        route = (
            str(plan_payload.get("retrieval_target"))
            if isinstance(plan_payload, dict)
            else "deterministic_only"
        )
        print(f"UPDATED {kpi}/{region}/{window_start} | {route}")

    print(f"\nRefreshed Stage 4 for {updated} bundle(s).")
    print("Routes:", routes)
    print("Deterministic RCA tables were not recomputed.")


if __name__ == "__main__":
    main()
