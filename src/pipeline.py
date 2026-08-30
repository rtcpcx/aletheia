"""
Aletheia — src/pipeline.py

Final execution order tying every stage together. Steps 1-7 (contract load
through decision packet) are fully deterministic and require no network
access. Step 8 (clarification/retrieval/fusion) only runs when deterministic
evidence is genuinely ambiguous, and never writes to raw.* or mart.*.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import time

import pandas as pd

from src import action_engine, database, decomposition_engine, evidence_engine
from src import evidence_fusion, guardrails_engine, narrator, orchestrator, retrieval
from src.contracts import KpiContract, detect_formula_operator, load_contracts
from src.driver_discovery import (
    assert_no_incident_leakage,
    discover_candidates,
)
from src.signal_engine import run_signal_engine_for_all

TOP_PROBABILITY_GAP_LOW = 0.10
MONITOR_CHANGE_THRESHOLD = 0.02


def _json_safe(value):
    """Normalize nested values so MySQL JSON never receives NaN/Infinity."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (dt.datetime, dt.date)):
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
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)
    return numeric if math.isfinite(numeric) else None


def _json_dumps_mysql(value) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _baseline_value(
    operator: str, a0: float, b0: float | None
) -> float | None:
    """
    Returns the KPI's pre-incident baseline value (K0) so total_change can
    be expressed as a true relative change (K1-K0)/K0, rather than an
    absolute value of mismatched scale being compared/formatted as if it
    were already a percentage.
    """
    if operator == "single":
        return a0
    if operator == "multiply":
        return a0 * b0 if b0 is not None else None
    if operator == "divide":
        if b0 is None or abs(b0) < 1e-9:
            return None
        return a0 / b0
    return None


def _component_values(
    contract: KpiContract, windows, use_incident: bool
) -> dict[str, tuple[float, float]]:
    """
    Returns {component_name: (period_start_value, period_end_value)} for
    each of the KPI's components, computed as the mean of the first vs.
    second half of the relevant window (historical baseline vs. incident).
    """
    df = windows.incident_window if use_incident else windows.model_training_window
    result: dict[str, tuple[float, float]] = {}

    for component in contract.components:
        series = df[component.name].dropna()
        if series.empty:
            result[component.name] = (0.0, 0.0)
            continue
        if use_incident:
            half = max(1, len(series) // 2)
            result[component.name] = (
                float(series.iloc[:half].mean()),
                float(series.iloc[half:].mean() if len(series) > half else series.iloc[:half].mean()),
            )
        else:
            result[component.name] = (
                float(series.iloc[: len(series) // 2].mean()) if len(series) > 1 else float(series.mean()),
                float(series.iloc[len(series) // 2:].mean()),
            )
    return result


_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _load_driver_incident_context(
    *,
    region: str,
    contract: KpiContract,
    incident_window: pd.DataFrame,
) -> pd.DataFrame:
    """Load the calendar span needed to evaluate locked driver lags.

    The normal incident window begins at the KPI changepoint. A positive driver lag
    means the relevant driver observation occurred before that date. This read adds
    only the minimum pre-incident context required to apply already-selected lags; it
    does not participate in lag selection or model fitting.
    """
    if incident_window.empty or "metric_date" not in incident_window.columns:
        return incident_window.copy()

    start = pd.to_datetime(incident_window["metric_date"], errors="coerce").min()
    end = pd.to_datetime(incident_window["metric_date"], errors="coerce").max()
    if pd.isna(start) or pd.isna(end):
        return incident_window.copy()

    max_lag = max((int(driver.max_lag) for driver in contract.root_drivers), default=0)
    context_start = start.date() - dt.timedelta(days=max_lag)
    context_end = end.date()

    select_parts = ["metric_date"]
    event_drivers: list[tuple[str, str]] = []
    seen_aliases: set[str] = set()
    for driver in contract.root_drivers:
        name = str(driver.name)
        column = str(driver.column)
        if not (_SQL_IDENTIFIER.match(name) and _SQL_IDENTIFIER.match(column)):
            continue
        if name in seen_aliases:
            continue
        seen_aliases.add(name)
        if column == "event_type":
            event_drivers.append((name, column))
        else:
            select_parts.append(f"`{column}` AS `{name}`")

    if event_drivers and "event_type" not in select_parts:
        select_parts.append("event_type")

    sql = (
        "SELECT " + ", ".join(select_parts)
        + " FROM mart.daily_kpi_evidence "
        + "WHERE region = %s AND metric_date BETWEEN %s AND %s "
        + "ORDER BY metric_date"
    )
    try:
        frame = database.query_df(sql, params=(region, context_start, context_end))
    except Exception:
        return incident_window.copy()

    if frame.empty:
        return incident_window.copy()

    for driver_name, _ in event_drivers:
        frame[driver_name] = (frame.get("event_type") == driver_name).astype(float)
    return frame


def _build_decision_packet(
    kpi: str,
    region: str,
    window_start: dt.date,
    decomposition,
    relative_change: float | None,
    ranked_drivers: list[dict],
    freshness_gap_days: int,
    source_health: dict | None = None,
    component_impacts: dict[str, float] | None = None,
) -> dict:
    # Source-health metadata is evaluated as of the incident date before this
    # function is called. Missing metadata fails closed rather than allowing a
    # potentially stale analysis to receive High confidence.
    source_health = source_health or {
        "status": "source_health_unavailable",
        "healthy": False,
        "stale_sources": [],
    }
    source_health_ok = bool(source_health.get("healthy", False))

    # relative_change is None only when the pre-incident baseline was ~0
    # (relative change is mathematically undefined there). In that case,
    # treat any non-trivial absolute movement away from zero as worth
    # investigating, since there is no meaningful percentage to threshold
    # against.
    if relative_change is not None:
        change_is_material = abs(relative_change) > MONITOR_CHANGE_THRESHOLD
    else:
        change_is_material = abs(decomposition.total_change) > 1e-9

    component_impacts = component_impacts or {}
    dominant_component = None
    finite_impacts: list[tuple[str, float]] = []
    for name, value in component_impacts.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite_impacts.append((str(name), abs(number)))
    if finite_impacts:
        dominant_component = max(finite_impacts, key=lambda item: item[1])[0]

    relevant_drivers = (
        [
            row
            for row in ranked_drivers
            if str(row.get("explains_component") or "") == dominant_component
        ]
        if dominant_component
        else list(ranked_drivers)
    )
    if not relevant_drivers:
        # Fail safely if component metadata is unavailable/incomplete rather than
        # comparing component-local weights from unrelated KPI components.
        relevant_drivers = list(ranked_drivers)
        dominant_component = None

    def _valid_driver(row: dict) -> bool:
        mode = str(row.get("evidence_mode") or "").lower()
        if mode == "structural_break":
            return True
        return bool(row.get("is_significant"))

    def _usable_driver(row: dict) -> bool:
        mode = str(row.get("evidence_mode") or "").lower()
        status = str(row.get("model_status") or "").lower()
        return mode != "insufficient_evidence" and not status.startswith("insufficient")

    if not relevant_drivers:
        confidence = "Low"
        decision = "monitor"
        top_gap = 0.0
        any_significant = False
        top2_mass = 0.0
    else:
        # Relative hypothesis weights are normalized within a KPI component.
        # Therefore confidence/ambiguity must also be computed within the
        # dominant component; global comparisons would be mathematically invalid.
        sorted_drivers = sorted(
            relevant_drivers, key=lambda d: float(d.get("softmax_probability") or 0.0), reverse=True
        )
        top_prob = float(sorted_drivers[0].get("softmax_probability") or 0.0)
        second_prob = (
            float(sorted_drivers[1].get("softmax_probability") or 0.0)
            if len(sorted_drivers) > 1
            else 0.0
        )
        top_gap = top_prob - second_prob
        any_significant = any(_valid_driver(d) for d in relevant_drivers)
        top2_mass = top_prob + second_prob
        decision = "investigate" if change_is_material else "monitor"
        model_sufficient = any(_usable_driver(d) for d in relevant_drivers)

        confidence = guardrails_engine.compute_confidence(
            decision=decision,
            top2_probability_mass=top2_mass,
            top_probability_gap=top_gap,
            top_driver_significant=_valid_driver(sorted_drivers[0]),
            model_sufficient=model_sufficient,
            freshness_gap_days=freshness_gap_days,
            source_health_ok=source_health_ok,
        )

    stale_sources = [
        str(source)
        for source in source_health.get("stale_sources", [])
        if str(source).strip()
    ]

    if stale_sources:
        freshness_caveat = (
            "Source health issue detected for: "
            + ", ".join(stale_sources)
            + ". RCA confidence has been downgraded."
        )
    elif freshness_gap_days > guardrails_engine.FRESHNESS_STALE_DAYS:
        freshness_caveat = (
            f"Underlying source data is approximately "
            f"{freshness_gap_days} day(s) stale."
        )
    else:
        freshness_caveat = None

    if not source_health_ok:
        recommended_action = (
            "Validate or restore the affected source feeds before acting on RCA."
        )
    elif decision == "investigate":
        recommended_action = "Investigate top-ranked drivers before acting."
    else:
        recommended_action = "No action required; continue monitoring."

    return {
        "kpi": kpi,
        "region": region,
        "window_start": window_start,
        "percent_change": relative_change,
        "absolute_change": decomposition.total_change,
        "confidence": confidence,
        "confidence_level": confidence,
        "top_probability_gap": top_gap,
        "top2_probability_mass": top2_mass,
        "evidence_component": dominant_component,
        "any_driver_significant": any_significant,
        "decision_type": decision,
        "recommended_action": recommended_action,
        "freshness_caveat": freshness_caveat,
        # Kept in the in-memory packet/evidence bundle for auditability.
        # _persist_decision_packet intentionally persists only its existing
        # relational columns, so no schema migration is required.
        "source_health": source_health,
    }


def _persist_driver_evidence(
    kpi: str, region: str, window_start: dt.date, evidence: list
) -> None:
    if not evidence:
        return
    sql = """
        REPLACE INTO analysis.driver_evidence (
            kpi,
            region,
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
        )
        VALUES (
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s
        )
    """
    rows = [
        (
            kpi,
            region,
            window_start,
            e.driver_name,
            e.explains_component,

            e.evidence_mode,
            e.model_status,

            e.best_lag_days,

            e.baseline_value,
            e.incident_value,

            e.driver_zscore,
            e.structural_break_score,

            e.historical_coefficient,
            e.holdout_correlation,
            e.p_value,
            e.is_significant,
            e.coefficient_stability,
            e.evidence_score,
            e.normalized_score,
            e.softmax_probability,
      )
        for e in evidence
    ]
    database.executemany(sql, rows)


def _persist_decomposition(
    kpi: str, region: str, window_start: dt.date, decomp
) -> None:
    sql = """
        REPLACE INTO analysis.pvm_decomposition (
            kpi, region, window_start, decomposition_type,
            effect_a, effect_b, interaction_effect, residual, total_change,
            is_volatile, narrative_mode
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    database.execute(
        sql,
        (
            kpi, region, window_start, decomp.decomposition_type,
            decomp.effect_a, decomp.effect_b, decomp.interaction_effect,
            decomp.residual, decomp.total_change,
            decomp.is_volatile, decomp.narrative_mode,
        ),
    )


def _persist_decision_packet(packet: dict) -> None:
    sql = """
        REPLACE INTO analysis.decision_packets (
            kpi, region, window_start, percent_change, confidence_level,
            recommended_action, freshness_caveat
        ) VALUES (%s,%s,%s,%s,%s,%s,%s)
    """
    database.execute(
        sql,
        (
            packet["kpi"], packet["region"], packet["window_start"],
            packet["percent_change"], packet["confidence_level"],
            packet["recommended_action"], packet["freshness_caveat"],
        ),
    )


def _ambiguity_reason(packet: dict, ranked_drivers: list[dict] | None = None) -> str:
    if orchestrator.requires_external_verification(packet, ranked_drivers):
        return "material external hypothesis requires independent public corroboration"
    reasons = []
    if packet.get("confidence_level") == "Low":
        reasons.append("overall confidence is Low")
    if packet.get("top_probability_gap", 1.0) < TOP_PROBABILITY_GAP_LOW:
        reasons.append(f"top hypothesis gap ({packet.get('top_probability_gap'):.3f}) is below threshold")
    if not packet.get("any_driver_significant", True):
        reasons.append("no candidate driver passed the significance/validity gate")
    return "; ".join(reasons) if reasons else "deterministic evidence was ambiguous"


def _persist_clarification_request(
    kpi: str,
    region: str,
    window_start: dt.date,
    reason: str,
    requested_clarification: str,
) -> None:
    sql = """
        INSERT INTO analysis.clarification_requests (
            kpi, region, window_start, reason, requested_clarification, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """
    database.execute(
        sql,
        (kpi, region, window_start, reason, requested_clarification, dt.datetime.utcnow()),
    )


def _persist_evidence_bundle(
    kpi: str, region: str, window_start: dt.date, bundle: dict
) -> None:
    sql = """
        REPLACE INTO analysis.evidence_bundle (
            kpi, region, window_start, bundle_json, generated_at
        ) VALUES (%s, %s, %s, %s, %s)
    """
    database.execute(
        sql,
        (kpi, region, window_start, _json_dumps_mysql(bundle), dt.datetime.utcnow()),
    )


def run_pipeline() -> None:
    """
    For each KPI and region:

    1. Load contract.
    2. Check entity history sufficiency.
    3. Detect changepoints.

    For each valid changepoint:

    4. Discover candidates and build lag-selection / historical /
       incident windows.
    5. Run decomposition.
    6. For every component: select lag on historical-only data, lock lag,
       fit longitudinal historical model, validate sample sufficiency,
       calculate inferential/bootstrapped validity, calculate coefficient
       stability, calculate incident severity, compute normalized
       evidence, calculate calibrated hypothesis weights.
    7. Evaluate incident-time source health and build deterministic decision packet.
    8. If deterministic evidence is ambiguous: generate clarification
       question, retrieve external evidence, store in
       analysis.retrieved_context, score bounded hypothesis support, fuse
       downstream, log before/after probabilities.
    9. Serialize full evidence bundle.
    10. Narrate for both personas.

    Never writes retrieval output to raw.* or mart.*.
    """
    contracts = load_contracts()

    # Stage 1 first, for every KPI/region, persisting to
    # analysis.detected_changepoints.
    run_signal_engine_for_all()

    changepoints_df = database.query_df(
        "SELECT kpi, region, changepoint_date FROM analysis.detected_changepoints"
    )

    for _, cp_row in changepoints_df.iterrows():
        kpi = cp_row["kpi"]
        region = cp_row["region"]
        changepoint_date = pd.Timestamp(cp_row["changepoint_date"]).date()

        try:
            _process_changepoint(kpi, region, changepoint_date, contracts[kpi])
        except Exception as exc:
            print(
                f"[pipeline] FAILED {kpi}/{region}/{changepoint_date}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue


def _process_changepoint(
    kpi: str, region: str, changepoint_date: dt.date, contract: KpiContract
) -> None:
    operator = detect_formula_operator(contract.formula)

    windows = discover_candidates(kpi, region, changepoint_date, contract)
    assert_no_incident_leakage(windows)

    if windows.model_training_window.empty or windows.incident_window.empty:
        return

    # --- Step 5: decomposition -----------------------------------
    component_vals = _component_values(contract, windows, use_incident=True)
    historical_component_vals = _component_values(contract, windows, use_incident=False)

    if operator == "single":
        comp_name = contract.components[0].name
        a0, a1 = historical_component_vals[comp_name][1], component_vals[comp_name][1]
        decomp = decomposition_engine.decompose("single", a0, a1)
        k0 = _baseline_value("single", a0, None)
    elif operator in ("multiply", "divide"):
        comp_a, comp_b = contract.components[0].name, contract.components[1].name
        a0 = historical_component_vals[comp_a][1]
        a1 = component_vals[comp_a][1]
        b0 = historical_component_vals[comp_b][1]
        b1 = component_vals[comp_b][1]
        decomp = decomposition_engine.decompose(operator, a0, a1, b0, b1)
        k0 = _baseline_value(operator, a0, b0)
    else:
        return

    relative_change = (
        decomp.total_change / k0 if k0 is not None and abs(k0) > 1e-9 else None
    )

    _persist_decomposition(kpi, region, changepoint_date, decomp)

    # --- Step 6: evidence per component ---------------------------
    # Include the pre-incident dates required by locked positive lags. This
    # context is interpretation-only; lag selection/model fitting still use the
    # historical training window exclusively.
    driver_incident_context = _load_driver_incident_context(
        region=region,
        contract=contract,
        incident_window=windows.incident_window,
    )

    all_evidence = []
    for component in contract.components:
        drivers = contract.drivers_for(component.name)
        if not drivers:
            continue
        evidence = evidence_engine.compute_evidence_scores(
            kpi=kpi,
            region=region,
            component=component.name,
            drivers=drivers,
            historical_data=windows.model_training_window,
            incident_data=windows.incident_window,
            incident_context_data=driver_incident_context,
        )
        all_evidence.extend(evidence)

    _persist_driver_evidence(kpi, region, changepoint_date, all_evidence)

    ranked = [
        {
            "driver_name": e.driver_name,
            "explains_component": e.explains_component,
            "evidence_mode": getattr(e, "evidence_mode", None),
            "model_status": getattr(e, "model_status", None),
            "best_lag_days": e.best_lag_days,
            "baseline_value": e.baseline_value,
            "incident_value": e.incident_value,
            "structural_break_score": e.structural_break_score,
            "historical_coefficient": e.historical_coefficient,
            "p_value": e.p_value,
            "is_significant": e.is_significant,
            "coefficient_stability": e.coefficient_stability,
            "driver_zscore": e.driver_zscore,
            "evidence_score": e.evidence_score,
            "normalized_score": e.normalized_score,
            "softmax_probability": e.softmax_probability,
            "historical_lag_correlation": getattr(e, "historical_lag_correlation", 0.0),
            "incident_severity_strength": getattr(e, "incident_severity_strength", 0.0),
            "direction_consistency": getattr(e, "direction_consistency", "not_evaluated"),
            "mechanism_weight": getattr(e, "mechanism_weight", 1.0),
            "source_cadence_days": getattr(e, "source_cadence_days", 1),
            "lag_resolution_days": getattr(e, "lag_resolution_days", 1),
            "causal_role": getattr(e, "causal_role", "direct"),
            "mediates_through": getattr(e, "mediates_through", None),
            "expected_effect_sign": getattr(e, "expected_effect_sign", "unknown"),
        }
        for e in all_evidence
    ]

    # --- Source-health guardrail ---------------------------------------
    # Evaluate source health using only records available ON OR BEFORE the
    # incident date. This avoids historical leakage from "latest" health rows
    # and checks every raw source used by the KPI or its candidate drivers.
    driver_names = [
        row["driver_name"]
        for row in ranked
        if row.get("driver_name")
    ]
    required_sources = guardrails_engine.source_dependencies_for_analysis(
        kpi=kpi,
        driver_names=driver_names,
    )
    source_health = guardrails_engine.check_source_health(
        region=region,
        as_of_date=changepoint_date,
        source_names=required_sources,
    )
    freshness_gap = int(source_health.get("freshness_gap_days", 0) or 0)

    component_baselines = {
        name: values[1] for name, values in historical_component_vals.items()
    }
    component_incidents = {
        name: values[1] for name, values in component_vals.items()
    }
    if operator == "single":
        component_impacts = {contract.components[0].name: decomp.effect_a}
    else:
        component_impacts = {
            contract.components[0].name: decomp.effect_a,
            contract.components[1].name: decomp.effect_b,
        }

    # --- Step 7: decision packet ------------------------------------
    packet = _build_decision_packet(
        kpi,
        region,
        changepoint_date,
        decomp,
        relative_change,
        ranked,
        freshness_gap,
        source_health,
        component_impacts=component_impacts,
    )
    action_context = action_engine.build_action_context(
        kpi=kpi,
        region=region,
        decision_type=packet["decision_type"],
        confidence=packet["confidence_level"],
        ranked_drivers=ranked,
        source_health=source_health,
        component_impacts=component_impacts,
        component_baselines=component_baselines,
        component_incidents=component_incidents,
        complex_interaction=(
            decomp.narrative_mode == "complex_interaction" or bool(decomp.is_volatile)
        ),
        kpi_relative_change=relative_change,
    )
    packet["action_context"] = action_context
    packet["recommended_action"] = action_context["recommended_action"]
    _persist_decision_packet(packet)

    # --- Step 8: adaptive clarification / retrieval / fusion -----------
    retrieved_evidence_payload: list[dict] = []
    retrieval_plan_payload: dict | None = None
    final_probabilities = {
        r["driver_name"]: r["softmax_probability"] for r in ranked
    }

    if orchestrator.needs_clarification(packet, ranked) and ranked:
        plan = orchestrator.generate_retrieval_plan(packet, ranked)
        retrieval_plan_payload = plan.to_dict()

        _persist_clarification_request(
            kpi=kpi,
            region=region,
            window_start=changepoint_date,
            reason=_ambiguity_reason(packet, ranked),
            requested_clarification=plan.clarification_question,
        )

        # Web search is conditional. Internal/none plans deliberately abstain
        # rather than searching synthetic or ungrounded business labels.
        if plan.retrieval_target == "web" and plan.retrieval_query:
            results = retrieval.retrieve_evidence(plan.retrieval_query)
            candidate_items = orchestrator.select_retrieval_candidates(
                ranked, limit=2, decision_packet=packet
            )
            hypotheses = [
                str(item.get("driver_name") or "").strip()
                for item in candidate_items
                if str(item.get("driver_name") or "").strip()
            ]

            scored_batch = retrieval.score_relevance_batch(
                question=plan.clarification_question,
                retrieval_query=plan.retrieval_query,
                hypotheses=hypotheses,
                results=results,
            )

            hypothesis_scores = scored_batch.get("hypothesis_scores", {})
            retrieval_support = {
                name: float(
                    hypothesis_scores.get(name, {}).get("effective_support", 0.0)
                )
                for name in hypotheses
            }

            # Persist SOURCE-LEVEL support/confidence rather than stamping one
            # aggregate score onto every retrieved document.
            assessment_map = {
                (int(a.get("source_index", -1)), str(a.get("hypothesis") or "")): a
                for a in scored_batch.get("source_assessments", [])
                if isinstance(a, dict)
            }

            for source_index, result in enumerate(results, start=1):
                for driver_name in hypotheses:
                    assessment = assessment_map.get(
                        (source_index, driver_name),
                        {
                            "support": 0.0,
                            "confidence": 0.0,
                            "reason": "no source-level assessment",
                        },
                    )
                    source_support = float(assessment.get("support", 0.0))
                    source_confidence = float(assessment.get("confidence", 0.0))

                    retrieval.store_retrieved_context(
                        kpi=kpi,
                        region=region,
                        window_start=changepoint_date,
                        hypothesis=driver_name,
                        query=plan.retrieval_query,
                        result=result,
                        support=source_support,
                        confidence=source_confidence,
                    )
                    retrieved_evidence_payload.append(
                        {
                            "hypothesis": driver_name,
                            "clarification_question": plan.clarification_question,
                            "query": plan.retrieval_query,
                            "source_title": result.get("title"),
                            "source_url": result.get("source_url"),
                            "support": source_support,
                            "confidence": source_confidence,
                            "reason": assessment.get("reason"),
                        }
                    )

            # The updated fusion implementation is an exact no-op when all
            # support is zero and leaves unassessed hypotheses untouched.
            if hypotheses:
                deterministic_scores = {
                    r["driver_name"]: r["normalized_score"] for r in ranked
                }
                deterministic_probabilities = {
                    r["driver_name"]: r["softmax_probability"] for r in ranked
                }
                final_probabilities = evidence_fusion.fuse_and_log(
                    kpi=kpi,
                    region=region,
                    window_start=changepoint_date,
                    deterministic_scores=deterministic_scores,
                    deterministic_probabilities=deterministic_probabilities,
                    retrieval_support=retrieval_support,
                    retrieval_query=plan.retrieval_query,
                )

    # --- Step 9: serialize evidence bundle ---------------------------
    for r in ranked:
        r["fused_probability"] = final_probabilities.get(
            r["driver_name"], r["softmax_probability"]
        )
    ranked.sort(key=lambda d: d["fused_probability"], reverse=True)

    # Rebuild the action recommendation after bounded retrieval fusion so the
    # business recommendation and final displayed evidence cannot disagree.
    # Selection remains component-aware: weights from different KPI components
    # are never treated as directly comparable.
    action_context = action_engine.build_action_context(
        kpi=kpi,
        region=region,
        decision_type=packet["decision_type"],
        confidence=packet["confidence_level"],
        ranked_drivers=ranked,
        source_health=source_health,
        component_impacts=component_impacts,
        component_baselines=component_baselines,
        component_incidents=component_incidents,
        complex_interaction=(
            decomp.narrative_mode == "complex_interaction" or bool(decomp.is_volatile)
        ),
        kpi_relative_change=relative_change,
    )
    packet["action_context"] = action_context
    packet["recommended_action"] = action_context["recommended_action"]
    _persist_decision_packet(packet)

    bundle = {
        "kpi": kpi,
        "region": region,
        "window_start": str(changepoint_date),
        "deterministic_evidence": {"ranked_drivers": ranked},
        "retrieved_evidence": retrieved_evidence_payload,
        "retrieval_plan": retrieval_plan_payload,
        "decomposition": {
            "decomposition_type": decomp.decomposition_type,
            "effect_a": decomp.effect_a,
            "effect_b": decomp.effect_b,
            "interaction_effect": decomp.interaction_effect,
            "residual": decomp.residual,
            "total_change": decomp.total_change,
            "is_volatile": decomp.is_volatile,
            "narrative_mode": decomp.narrative_mode,
            "component_impacts": component_impacts,
            "component_baselines": component_baselines,
            "component_incidents": component_incidents,
        },
        "model_validity": {
            "any_driver_significant": packet["any_driver_significant"],
        },
        "confidence": {"level": packet["confidence_level"]},
        "decision": packet,
    }
    _persist_evidence_bundle(kpi, region, changepoint_date, bundle)

    # --- Step 10: narrate for both personas ---------------------------
    for persona in ("Executive", "Growth analyst"):
        try:
            narration = narrator.narrate(bundle, persona)
        except Exception:
            narration = narrator.template_narrate(bundle, persona)
        bundle.setdefault("narration", {})[persona] = narration

    _persist_evidence_bundle(kpi, region, changepoint_date, bundle)


if __name__ == "__main__":
    started = time.time()
    run_pipeline()
    elapsed = time.time() - started
    print(f"Pipeline run complete in {elapsed:.1f}s")
