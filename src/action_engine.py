"""
Aletheia — src/action_engine.py

Deterministic business-action layer.

This module converts already-computed RCA evidence into a specific, business-readable
"what to do next" recommendation. It does NOT create evidence, change rankings, or
claim causality. Action language is config-driven by driver in
config/action_playbooks.yaml rather than scenario/date hardcoding.

V2 selection rules deliberately distinguish:
- a strong historical relationship,
- an actual incident-time movement,
- whether that movement is directionally consistent with the affected KPI component,
- whether evidence is decisive enough to recommend action rather than validation.

A driver that ranks highly but stayed broadly unchanged during the incident is NOT
promoted to a direct intervention. That prevents recommendations such as
"Prioritize marketing spend. Marketing spend was stable."
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "action_playbooks.yaml"
AMBIGUITY_GAP_THRESHOLD = 0.10
MATERIAL_DIRECTION_THRESHOLD = 0.02
COEFFICIENT_EPSILON = 1e-10


def _friendly(name: Any) -> str:
    text = str(name or "").strip().replace("_", " ")
    return " ".join(part.capitalize() for part in text.split()) or "Unknown"


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


@lru_cache(maxsize=4)
def _load_playbooks(path_text: str = str(DEFAULT_CONFIG_PATH)) -> dict[str, Any]:
    path = Path(path_text)
    if not path.exists():
        return {"defaults": {}, "playbooks": {}}
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    return payload if isinstance(payload, dict) else {"defaults": {}, "playbooks": {}}


def _weight(row: dict[str, Any]) -> float:
    fused = _as_float(row.get("fused_probability"))
    if fused is not None:
        return fused
    raw = _as_float(row.get("softmax_probability"))
    return raw if raw is not None else 0.0


def _usable(row: dict[str, Any]) -> bool:
    mode = str(row.get("evidence_mode") or "").lower()
    status = str(row.get("model_status") or "").lower()
    if mode == "insufficient_evidence":
        return False
    if status.startswith("insufficient"):
        return False
    return True


def _direction(baseline: Any, incident: Any) -> tuple[str, float | None]:
    base = _as_float(baseline)
    inc = _as_float(incident)
    if base is None or inc is None:
        return "unknown", None

    delta = inc - base
    if abs(base) > 1e-12:
        relative = delta / abs(base)
        if abs(relative) < MATERIAL_DIRECTION_THRESHOLD:
            return "unchanged", relative
    else:
        relative = None
        if abs(delta) < 1e-12:
            return "unchanged", relative

    return ("increased" if delta > 0 else "decreased"), relative


def _direction_phrase(direction: str) -> str:
    return {
        "increased": "increased during the incident",
        "decreased": "decreased during the incident",
        "unchanged": "was broadly unchanged during the incident",
        "unknown": "does not have a reliable incident-versus-baseline comparison",
    }.get(direction, "changed during the incident")


def _component_order(component_impacts: dict[str, Any]) -> list[str]:
    scored: list[tuple[str, float]] = []
    for name, value in component_impacts.items():
        number = _as_float(value)
        scored.append((str(name), abs(number) if number is not None else 0.0))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [name for name, _ in scored]


def _drivers_for_component(
    ranked_drivers: list[dict[str, Any]], component: str
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in ranked_drivers
        if str(row.get("explains_component") or "") == component and _usable(row)
    ]
    rows.sort(key=_weight, reverse=True)
    return rows


def _all_drivers_for_component(
    ranked_drivers: list[dict[str, Any]], component: str
) -> list[dict[str, Any]]:
    rows = [
        row
        for row in ranked_drivers
        if str(row.get("explains_component") or "") == component
    ]
    rows.sort(key=_weight, reverse=True)
    return rows


def _action_readiness(action_level: str) -> str:
    return {
        "act": "Act after operational confirmation",
        "validate": "Validate first",
        "validate_unexplained": "Investigate evidence gap",
        "data_quality_first": "Fix data first",
        "monitor": "Monitor",
    }.get(action_level, "Validate first")


def _playbook_payload(driver_name: str, direction: str) -> tuple[str, str, dict[str, str]]:
    config = _load_playbooks()
    defaults = config.get("defaults", {}) if isinstance(config.get("defaults"), dict) else {}
    playbooks = config.get("playbooks", {}) if isinstance(config.get("playbooks"), dict) else {}
    entry = playbooks.get(driver_name, {}) if isinstance(playbooks.get(driver_name), dict) else {}

    label = str(entry.get("label") or _friendly(driver_name))
    owner = str(entry.get("owner") or defaults.get("owner") or "Business owner")

    raw = entry.get(direction)
    if raw is None:
        raw = entry.get("unchanged")
    if raw is None:
        raw = defaults.get(direction) or defaults.get("unchanged") or {}

    if isinstance(raw, str):
        payload = {
            "next_check": raw,
            "action_if_confirmed": "Make the narrowest operational change supported by the check, then monitor the KPI response.",
            "hidden_check": raw,
        }
    elif isinstance(raw, dict):
        payload = {
            "next_check": str(raw.get("next_check") or "Validate the operational change behind this signal."),
            "action_if_confirmed": str(
                raw.get("action_if_confirmed")
                or "Correct the confirmed operational issue and monitor whether the KPI responds."
            ),
            "hidden_check": str(
                raw.get("hidden_check")
                or raw.get("next_check")
                or "Break the aggregate signal into operational sub-segments before changing the business lever."
            ),
        }
    else:
        payload = {
            "next_check": "Validate the operational change behind this signal.",
            "action_if_confirmed": "Correct the confirmed operational issue and monitor whether the KPI responds.",
            "hidden_check": "Break the aggregate signal into operational sub-segments before changing the business lever.",
        }
    return label, owner, payload


def _source_health_action(source_health: dict[str, Any]) -> dict[str, Any] | None:
    if bool(source_health.get("healthy", False)):
        return None

    stale = [str(v) for v in source_health.get("stale_sources", []) if str(v).strip()]
    missing = [str(v) for v in source_health.get("missing_sources", []) if str(v).strip()]
    affected = stale + [v for v in missing if v not in stale]
    label = ", ".join(affected) if affected else "required source data"
    next_check = f"Restore or validate {label}, then confirm that completeness and freshness have returned to normal."
    action_if_confirmed = "Rerun the RCA on healthy data before changing any business lever."
    recommendation = f"Do not act on the RCA yet. {next_check} {action_if_confirmed}"
    return {
        "action_level": "data_quality_first",
        "primary_driver": None,
        "primary_driver_label": None,
        "primary_component": None,
        "primary_component_label": None,
        "owner": "Data / Analytics Engineering",
        "driver_direction": None,
        "component_direction": None,
        "finding": f"The analysis depends on {label}, and that source is stale, incomplete, or missing.",
        "why_it_matters": "A business recommendation would be unsafe while a required source is unhealthy.",
        "next_check": next_check,
        "action_if_confirmed": action_if_confirmed,
        "why": f"The analysis depends on {label}, and that source is stale, incomplete, or missing.",
        "recommended_action": recommendation,
        "secondary_driver": None,
        "secondary_driver_label": None,
        "secondary_check": None,
        "is_ambiguous": True,
        "movement_status": "data_quality_blocked",
        "direction_alignment": "not_evaluated",
        "evidence_confidence": "Low",
        "action_readiness": _action_readiness("data_quality_first"),
        "leading_explanation_driver": None,
        "leading_explanation_driver_label": None,
    }


def _component_direction(
    component: str,
    component_baselines: dict[str, Any],
    component_incidents: dict[str, Any],
) -> str:
    return _direction(component_baselines.get(component), component_incidents.get(component))[0]


def _direction_alignment(row: dict[str, Any], component_direction: str) -> str:
    stored=str(row.get("direction_consistency") or "").strip().lower()
    if stored in {"aligned","contradictory","not_evaluated"}: return stored
    driver_direction=_direction(row.get("baseline_value"),row.get("incident_value"))[0]
    if driver_direction in {"unknown","unchanged"}: return "not_evaluated"
    if str(row.get("evidence_mode") or "").lower()!="historical_relationship": return "not_evaluated"
    expected_sign = str(row.get("expected_effect_sign") or "unknown").strip().lower()
    if expected_sign == "positive":
        effect_sign = 1
    elif expected_sign == "negative":
        effect_sign = -1
    else:
        slope=_as_float(row.get("historical_lag_correlation"))
        if slope is None or abs(slope)<=COEFFICIENT_EPSILON: slope=_as_float(row.get("historical_coefficient"))
        if slope is None or abs(slope)<=COEFFICIENT_EPSILON: return "not_evaluated"
        effect_sign = 1 if slope > 0 else -1
    if component_direction not in {"increased","decreased"}: return "not_evaluated"
    pred=(1 if driver_direction=="increased" else -1)*effect_sign; actual=1 if component_direction=="increased" else -1
    return "aligned" if pred==actual else "contradictory"


def _candidate_state(row: dict[str, Any], component_direction: str) -> dict[str, Any]:
    direction, relative_change = _direction(row.get("baseline_value"), row.get("incident_value"))
    alignment = _direction_alignment(row, component_direction)
    moved = direction in {"increased", "decreased"}
    unknown = direction == "unknown"
    contradictory = alignment == "contradictory"

    # A structural break is specifically an incident movement, so it is actionable
    # when the direction is observed. Historical evidence additionally must not point
    # opposite the observed component movement.
    actionable = _usable(row) and moved and not contradictory
    return {
        "row": row,
        "direction": direction,
        "relative_change": relative_change,
        "alignment": alignment,
        "moved": moved,
        "unknown": unknown,
        "contradictory": contradictory,
        "actionable": actionable,
        "weight": _weight(row),
    }


def _select_actionable_driver(rows: list[dict[str, Any]], component_direction: str) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    states=[_candidate_state(row,component_direction) for row in rows]; actionable=[x for x in states if x["actionable"]]; actionable.sort(key=lambda x:x["weight"],reverse=True)
    if not actionable: return None,states
    selected=actionable[0]; row=selected["row"]; mediator=str(row.get("mediates_through") or "").strip()
    if str(row.get("causal_role") or "").lower()=="upstream" and mediator:
        mediated=next((x for x in actionable if str(x["row"].get("driver_name") or "")==mediator),None)
        if mediated is not None: selected=mediated
    return selected,states


def _upstream_trigger_for(driver_name: str,rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates=[r for r in rows if str(r.get("causal_role") or "").lower()=="upstream" and str(r.get("mediates_through") or "")==driver_name and _usable(r)]
    candidates.sort(key=_weight,reverse=True); return candidates[0] if candidates else None


def _kpi_direction(relative_change: float | None) -> str:
    if relative_change is None:
        return "changed"
    if relative_change > 0:
        return "increased"
    if relative_change < 0:
        return "decreased"
    return "was broadly unchanged"


def build_action_context(
    *,
    kpi: str,
    region: str,
    decision_type: str,
    confidence: str,
    ranked_drivers: list[dict[str, Any]],
    source_health: dict[str, Any] | None,
    component_impacts: dict[str, Any],
    component_baselines: dict[str, Any] | None = None,
    component_incidents: dict[str, Any] | None = None,
    complex_interaction: bool = False,
    kpi_relative_change: float | None = None,
) -> dict[str, Any]:
    """Build a specific, evidence-bounded business recommendation.

    Selection policy:
    1. data quality gates business action;
    2. monitor decisions do not manufacture intervention;
    3. choose the KPI component with the largest absolute measured contribution;
    4. rank drivers only within that component;
    5. a driver must have materially moved during the incident to be a direct action target;
    6. for historical relationships, reject incident movement that points opposite the
       observed component direction when a reliable coefficient sign is available;
    7. Low/split/complex evidence produces a validation plan, not a committed intervention;
    8. if no measured candidate moved, say so explicitly and recommend targeted drill-down
       rather than pretending the highest historical relationship explains the incident.
    """
    source_health = source_health or {}
    source_action = _source_health_action(source_health)
    if source_action is not None:
        return source_action

    if str(decision_type).lower() != "investigate":
        finding = f"{_friendly(kpi)} did not cross the material-change threshold for action."
        return {
            "action_level": "monitor",
            "primary_driver": None,
            "primary_driver_label": None,
            "primary_component": None,
            "primary_component_label": None,
            "owner": None,
            "driver_direction": None,
            "component_direction": None,
            "finding": finding,
            "why_it_matters": "There is no structural incident that warrants a root-cause intervention right now.",
            "next_check": "Continue monitoring the KPI and source health for a new structural change.",
            "action_if_confirmed": "No immediate business change is required.",
            "why": finding,
            "recommended_action": "No immediate intervention is recommended. Continue monitoring the KPI and source health for a new structural change.",
            "secondary_driver": None,
            "secondary_driver_label": None,
            "secondary_check": None,
            "is_ambiguous": False,
            "movement_status": "monitor",
            "direction_alignment": "not_evaluated",
            "evidence_confidence": str(confidence).title(),
            "action_readiness": _action_readiness("monitor"),
            "leading_explanation_driver": None,
            "leading_explanation_driver_label": None,
        }

    component_baselines = component_baselines or {}
    component_incidents = component_incidents or {}

    components = _component_order(component_impacts)
    if not components:
        components = list(
            dict.fromkeys(
                str(row.get("explains_component") or "")
                for row in ranked_drivers
                if str(row.get("explains_component") or "")
            )
        )

    # The dominant measured KPI component must remain primary even if its driver
    # evidence is weak. Falling through to a less-important component would create
    # a confident recommendation about the wrong part of the KPI movement.
    primary_component: str | None = components[0] if components else None
    all_primary_rows = (
        _all_drivers_for_component(ranked_drivers, primary_component)
        if primary_component
        else []
    )
    primary_rows = (
        _drivers_for_component(ranked_drivers, primary_component)
        if primary_component
        else []
    )

    if primary_component is None or not primary_rows:
        component_label = _friendly(primary_component) if primary_component else "the dominant KPI component"
        statuses = {str(row.get("model_status") or "").lower() for row in all_primary_rows}
        modes = {str(row.get("evidence_mode") or "").lower() for row in all_primary_rows}
        insufficient_history = bool(all_primary_rows) and all(
            status.startswith("insufficient") or mode == "insufficient_evidence"
            for status, mode in zip(
                [str(row.get("model_status") or "").lower() for row in all_primary_rows],
                [str(row.get("evidence_mode") or "").lower() for row in all_primary_rows],
            )
        )
        if insufficient_history:
            finding = f"{component_label} drove the material KPI movement, but the available historical sample is not sufficient to distinguish its candidate drivers reliably."
            next_check_text = f"Verify or extend historical coverage for the drivers of {component_label}, then rerun the RCA before choosing an intervention."
            validation_reason = "insufficient_history"
        elif not all_primary_rows:
            finding = f"{component_label} drove the material KPI movement, but the current KPI contract has no driver evidence for that component."
            next_check_text = f"Review the KPI contract and add the missing operational signals that can explain {component_label}, then rerun the RCA."
            validation_reason = "driver_coverage_gap"
        else:
            finding = f"A material change was detected in {component_label}, but none of its candidate drivers has usable evidence for this incident."
            next_check_text = f"Validate source coverage and historical sufficiency for the drivers of {component_label}, then rerun the RCA."
            validation_reason = "no_usable_driver"
        return {
            "action_level": "validate",
            "primary_driver": None,
            "primary_driver_label": None,
            "primary_component": primary_component,
            "primary_component_label": component_label,
            "owner": None,
            "driver_direction": None,
            "component_direction": None,
            "finding": finding,
            "why_it_matters": "Changing a business lever without usable evidence for the dominant KPI component would turn uncertainty into guesswork.",
            "next_check": next_check_text,
            "action_if_confirmed": "Rerun the RCA after the evidence gap is repaired; choose a business intervention only if a driver is then supported.",
            "why": finding,
            "recommended_action": f"Do not change a business lever yet. {next_check_text}",
            "secondary_driver": None,
            "secondary_driver_label": None,
            "secondary_check": None,
            "is_ambiguous": True,
            "movement_status": "no_usable_driver",
            "direction_alignment": "not_evaluated",
            "validation_reason": validation_reason,
            "evidence_confidence": str(confidence).title(),
            "action_readiness": _action_readiness("validate"),
            "leading_explanation_driver": None,
            "leading_explanation_driver_label": None,
        }

    component_label = _friendly(primary_component)
    comp_direction = _component_direction(primary_component, component_baselines, component_incidents)
    explanation = primary_rows[0]
    explanation_name = str(explanation.get("driver_name") or "")
    explanation_direction = _direction(explanation.get("baseline_value"), explanation.get("incident_value"))[0]
    explanation_label = _playbook_payload(explanation_name, explanation_direction)[0] if explanation_name else None
    selected_state, states = _select_actionable_driver(primary_rows, comp_direction)

    # If no candidate in the dominant component materially moved, do not turn the
    # highest historical relationship into an operational recommendation.
    if selected_state is None:
        historical_top = primary_rows[0]
        historical_name = str(historical_top.get("driver_name") or "")
        historical_direction = _direction(
            historical_top.get("baseline_value"), historical_top.get("incident_value")
        )[0]
        historical_label, owner, playbook = _playbook_payload(historical_name, historical_direction)

        moved_but_contradictory = [state for state in states if state["moved"] and state["contradictory"]]
        unknown_states = [state for state in states if state["unknown"]]

        if moved_but_contradictory:
            reason = (
                f"The measured drivers for {component_label} either stayed stable or moved in a direction that does not match the observed {component_label} change."
            )
        elif unknown_states:
            reason = (
                f"The dominant change is in {component_label}, but the available driver baselines are incomplete or the measured drivers did not move materially."
            )
        else:
            reason = (
                f"{historical_label} is historically related to {component_label}, but it {_direction_phrase(historical_direction)}. No measured driver of {component_label} changed strongly enough to justify a direct intervention."
            )

        next_check = playbook["hidden_check"]
        action_if_confirmed = (
            "If the drill-down reveals a real sub-segment change, use the corresponding operational playbook; otherwise inspect drivers not yet represented in the KPI contract."
        )
        recommendation = (
            f"Do not prioritize {historical_label} as the operational cause yet. {reason} "
            f"{next_check} {action_if_confirmed}"
        )
        return {
            "action_level": "validate_unexplained",
            "primary_driver": historical_name or None,
            "primary_driver_label": historical_label,
            "primary_component": primary_component,
            "primary_component_label": component_label,
            "owner": owner,
            "driver_direction": historical_direction,
            "component_direction": comp_direction,
            "finding": reason,
            "why_it_matters": f"{component_label} is the dominant measured part of the {_friendly(kpi)} movement, but the current incident-time driver data do not explain it cleanly.",
            "next_check": next_check,
            "action_if_confirmed": action_if_confirmed,
            "why": reason,
            "recommended_action": recommendation,
            "secondary_driver": None,
            "secondary_driver_label": None,
            "secondary_check": None,
            "is_ambiguous": True,
            "movement_status": "no_material_driver_movement",
            "direction_alignment": "not_evaluated",
            "evidence_mode": str(historical_top.get("evidence_mode") or ""),
            "best_lag_days": historical_top.get("best_lag_days"),
            "driver_baseline_value": historical_top.get("baseline_value"),
            "driver_incident_value": historical_top.get("incident_value"),
            "evidence_confidence": str(confidence).title(),
            "action_readiness": _action_readiness("validate_unexplained"),
            "leading_explanation_driver": explanation_name or historical_name or None,
            "leading_explanation_driver_label": explanation_label or historical_label,
            "action_target_driver": None,
            "action_target_driver_label": None,
        }

    primary = selected_state["row"]
    driver_name = str(primary.get("driver_name") or "")
    driver_direction = str(selected_state["direction"])
    alignment = str(selected_state["alignment"])
    primary_weight = float(selected_state["weight"])
    driver_label, owner, playbook = _playbook_payload(driver_name, driver_direction)
    upstream_row = _upstream_trigger_for(driver_name, primary_rows)
    upstream_name = str(upstream_row.get("driver_name") or "") if upstream_row else None
    upstream_direction = _direction(upstream_row.get("baseline_value"), upstream_row.get("incident_value"))[0] if upstream_row else "unknown"
    upstream_label = _playbook_payload(upstream_name, upstream_direction)[0] if upstream_name else None
    mechanism_chain = f"{upstream_label} → {driver_label} → {component_label}" if upstream_label else None

    # Secondary comparisons must be within the same KPI component and must also
    # have materially moved. Stable historical relationships are not used as the
    # competing operational alternative.
    actionable_others = [
        state
        for state in states
        if state is not selected_state and state["actionable"]
    ]
    actionable_others.sort(key=lambda state: state["weight"], reverse=True)
    secondary_state = actionable_others[0] if actionable_others else None
    secondary = secondary_state["row"] if secondary_state else None
    secondary_weight = float(secondary_state["weight"]) if secondary_state else 0.0
    gap = primary_weight - secondary_weight

    secondary_name = str(secondary.get("driver_name") or "") if secondary else None
    secondary_direction = str(secondary_state["direction"]) if secondary_state else "unchanged"
    secondary_label = (
        _playbook_payload(secondary_name, secondary_direction)[0]
        if secondary_name
        else None
    )

    evidence_mode = str(primary.get("evidence_mode") or "")
    if evidence_mode == "structural_break":
        evidence_phrase = "it shows a new incident-time structural shift compared with its prior baseline"
    elif evidence_mode == "historical_relationship":
        evidence_phrase = "it has a supported historical relationship with this KPI component and also moved during the incident"
    else:
        evidence_phrase = "it is the strongest usable moved signal currently available for this KPI component"

    finding = (
        f"{driver_label} {_direction_phrase(driver_direction)} while {component_label} {_direction_phrase(comp_direction)}."
        if comp_direction != "unknown"
        else f"{driver_label} {_direction_phrase(driver_direction)} during the incident."
    )
    why_it_matters = (
        f"{component_label} is the KPI component with the largest measured contribution to the current {_friendly(kpi)} change, and {evidence_phrase}."
    )
    if alignment == "aligned":
        why_it_matters += " The incident direction is consistent with the historical relationship."
    if upstream_label:
        why_it_matters += f" {upstream_label} is an upstream trigger that may act through {driver_label}; treat it as mechanism context rather than a competing peer cause."
    if explanation_name and explanation_name != driver_name:
        why_it_matters += (
            f" {explanation_label} remains the strongest evidence-ranked explanation for {component_label}; "
            f"{driver_label} is the first operational signal that moved clearly enough to check directly."
        )

    low_confidence = str(confidence).lower() == "low"
    selected_validity_weak = (
        evidence_mode == "historical_relationship"
        and primary.get("is_significant") is not True
    )
    split_evidence = secondary_state is not None and gap <= AMBIGUITY_GAP_THRESHOLD + 1e-12
    ambiguous = low_confidence or selected_validity_weak or split_evidence or complex_interaction

    next_check = playbook["next_check"]
    action_if_confirmed = playbook["action_if_confirmed"]

    if ambiguous:
        if secondary_label:
            secondary_is_upstream = bool(secondary and str(secondary.get("causal_role") or "").lower() == "upstream" and str(secondary.get("mediates_through") or "") == driver_name)
            secondary_check = (f"Validate {secondary_label} in parallel as the upstream trigger that may be acting through {driver_label}." if secondary_is_upstream else f"Validate {secondary_label} in parallel because it also moved and remains a plausible explanation for {component_label}.")
            recommendation = (
                f"Validate {driver_label} first rather than changing a broad business lever immediately. "
                f"{next_check} If confirmed: {action_if_confirmed} {secondary_check}"
            )
        else:
            secondary_check = None
            recommendation = (
                f"Validate {driver_label} first rather than making a broad intervention immediately. "
                f"{next_check} If confirmed: {action_if_confirmed}"
            )
        action_level = "validate"
    else:
        secondary_is_upstream = bool(secondary and str(secondary.get("causal_role") or "").lower() == "upstream" and str(secondary.get("mediates_through") or "") == driver_name)
        secondary_check = (f"Also validate {secondary_label} as the upstream trigger that may be acting through {driver_label}." if secondary_label and secondary_is_upstream else (f"If the {driver_label} check is not confirmed, investigate {secondary_label} next." if secondary_label else None))
        recommendation = (
            f"Work on {driver_label} first. {next_check} If confirmed: {action_if_confirmed}"
        )
        if secondary_check:
            recommendation += " " + secondary_check
        action_level = "act"

    return {
        "action_level": action_level,
        "primary_driver": driver_name,
        "primary_driver_label": driver_label,
        "primary_component": primary_component,
        "primary_component_label": component_label,
        "owner": owner,
        "driver_direction": driver_direction,
        "component_direction": comp_direction,
        "finding": finding,
        "why_it_matters": why_it_matters,
        "next_check": next_check,
        "action_if_confirmed": action_if_confirmed,
        "why": why_it_matters,
        "recommended_action": recommendation,
        "secondary_driver": secondary_name,
        "secondary_driver_label": secondary_label,
        "secondary_check": secondary_check,
        "is_ambiguous": ambiguous,
        "movement_status": "material_movement",
        "direction_alignment": alignment,
        "evidence_mode": evidence_mode,
        "best_lag_days": primary.get("best_lag_days"),
        "driver_baseline_value": primary.get("baseline_value"),
        "driver_incident_value": primary.get("incident_value"),
        "kpi_direction": _kpi_direction(kpi_relative_change),
        "evidence_confidence": str(confidence).title(),
        "action_readiness": _action_readiness(action_level),
        "leading_explanation_driver": explanation_name or None,
        "leading_explanation_driver_label": explanation_label,
        "action_target_driver": driver_name or None,
        "action_target_driver_label": driver_label,
        "explanation_differs_from_action_target": bool(explanation_name and explanation_name != driver_name),
        "upstream_driver": upstream_name,
        "upstream_driver_label": upstream_label,
        "mechanism_chain": mechanism_chain,
    }
