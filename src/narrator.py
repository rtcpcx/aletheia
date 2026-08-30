"""
Aletheia — src/narrator.py

Business-language narration over the final evidence bundle.

The narrator is presentation-only:
- it cannot add evidence or change rankings;
- it cannot invent causes, numbers, confidence, owners, or actions;
- it receives a small structured business interpretation, not raw model internals;
- recommendations come from the deterministic action engine;
- technical statistics remain available in the audit UI, not in the narrative.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

try:
    import ollama
except ImportError:  # deterministic fallback remains usable without Ollama
    ollama = None


NARRATION_MODEL = os.getenv("ALETHEIA_LLM_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

Persona = Literal["Executive", "Growth analyst"]

FORBIDDEN_NARRATIVE_TERMS = (
    "softmax",
    "coefficient",
    "p-value",
    "p value",
    "z-score",
    "z score",
    "normalized score",
    "ridge",
    "regression coefficient",
    "model_status",
    "evidence_score",
    "fused probability",
    "probability_after",
    "probability_before",
)


NARRATOR_SYSTEM = """
You are the plain-business-language narration layer of Aletheia.

You are NOT an analyst discovering a cause. The deterministic action engine has
already decided what the evidence supports. Your only job is to communicate that
structured interpretation clearly to a business user.

The payload contains these concepts:
- finding: what changed in the relevant operational signal;
- why_it_matters: why that signal is relevant to the KPI component that moved;
- next_check: the exact operational check to perform now;
- action_if_confirmed: the intervention allowed only after that check confirms the issue;
- secondary_driver / secondary_check: a competing explanation that must remain open;
- action_level: act, validate, validate_unexplained, data_quality_first, or monitor;
- evidence_confidence: confidence in the analytical evidence, which can be High even when
  the correct operational recommendation is still to validate rather than act;
- action_readiness: whether the business should act, validate, investigate an evidence gap,
  fix data, or simply monitor;
- leading_explanation_driver: strongest evidence-ranked explanation for the dominant KPI component;
- action_target_driver: first operational signal that moved clearly enough to check. These may differ.
- upstream_driver / mechanism_chain: an initiating signal that acts through the selected operational driver; keep the upstream trigger distinct from the nearer operational mechanism.

HARD RULES
- Never invent a cause, number, confidence, owner, action, event, company, or date.
- Never claim that evidence proves causality. Say "strongest supported explanation",
  "first issue to validate", or "plausible explanation".
- Never say softmax, coefficient, p-value, z-score, Ridge, regression, normalized
  score, model status, or other backend/model terminology.
- Never expose statistical ranking mechanics in the normal narrative.
- Never confuse evidence confidence with action readiness. A High evidence-confidence case
  may still require validation if the incident-time operational movement is not confirmed.
- If leading_explanation_driver differs from action_target_driver, say so plainly: distinguish
  the strongest explanation from the first operational check. Do not silently turn the action
  target into the claimed root cause.
- When a mechanism_chain is supplied, describe it as an upstream trigger acting through a nearer
  operational mechanism; do not flatten both signals into competing peer causes.
- Do not turn a historically strong but incident-stable driver into a cause.
- If action_level is validate_unexplained, state plainly that the dominant KPI
  component moved but the currently measured drivers did not move enough to justify
  a direct intervention. The next step is drill-down / missing-driver investigation.
- If action_level is data_quality_first, fix the source before any business action.
- If action_level is validate, name the primary check and the secondary check when supplied.
- If action_level is act, state what should be worked on first, what to check, and what to do only if confirmed.
- If action_level is monitor, state that no immediate intervention is required.
- If multiple_components_moved_together is true, do not manufacture an exact isolated percentage split.
- External context is supporting context only, never a verified internal causal fact.

STYLE
- Use concrete nouns: platform uptime, support tickets, stock availability, marketing spend.
- Avoid phrases such as "the model indicates", "historical coefficient", "evidence weight",
  "statistical significance", or "top-ranked driver".
- Do not repeat the same recommendation twice.
- Prefer short sentences with a clear sequence: what happened → why it matters → check now → act if confirmed.

PERSONAS
- Executive: 3-5 short sentences. Outcome first, then the first operational priority and action.
- Growth analyst: 4-7 sentences. Include the affected KPI component and the secondary check where relevant.

Return ONLY JSON:
{
  "headline": "<one plain-language sentence>",
  "narrative": "<plain-language business explanation>",
  "caveats": ["<only material caveats>"]
}
"""


def _client():
    if ollama is None:
        raise RuntimeError("ollama package is not installed")
    if OLLAMA_HOST:
        return ollama.Client(host=OLLAMA_HOST)
    return ollama.Client()


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Narration response is not a JSON object")
    return parsed


def _normalise_output(parsed: dict[str, Any]) -> dict[str, Any]:
    headline = str(parsed.get("headline", "")).strip()
    narrative = str(parsed.get("narrative", "")).strip()
    caveats = parsed.get("caveats", [])
    if not headline:
        raise ValueError("Missing narration headline")
    if not narrative:
        raise ValueError("Missing narration narrative")
    if not isinstance(caveats, list):
        raise ValueError("Narration caveats must be a list")

    result = {
        "headline": headline,
        "narrative": narrative,
        "caveats": [str(item).strip() for item in caveats if str(item).strip()],
    }
    combined = " ".join([result["headline"], result["narrative"], *result["caveats"]]).lower()
    leaked = [term for term in FORBIDDEN_NARRATIVE_TERMS if term in combined]
    if leaked:
        raise ValueError(f"Narration leaked technical jargon: {', '.join(leaked)}")
    return result


def _decision(bundle: dict[str, Any]) -> dict[str, Any]:
    value = bundle.get("decision", {})
    return value if isinstance(value, dict) else {}


def _action_context(bundle: dict[str, Any]) -> dict[str, Any]:
    value = _decision(bundle).get("action_context")
    if not isinstance(value, dict):
        value = bundle.get("action_context")
    return value if isinstance(value, dict) else {}


def _confidence(bundle: dict[str, Any]) -> str:
    value = bundle.get("confidence")
    if isinstance(value, dict):
        value = value.get("confidence_level") or value.get("level")
    if value is None:
        value = _decision(bundle).get("confidence_level")
    return str(value or "Low").title()


def _kpi(bundle: dict[str, Any]) -> str:
    return str(bundle.get("kpi") or _decision(bundle).get("kpi") or "KPI")


def _region(bundle: dict[str, Any]) -> str:
    return str(bundle.get("region") or _decision(bundle).get("region") or "the selected region")


def _friendly(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ")
    return " ".join(part.capitalize() for part in text.split()) or "Unknown"


def _percent_change(bundle: dict[str, Any]) -> float | None:
    value = _decision(bundle).get("percent_change")
    if value is None:
        value = bundle.get("percent_change")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _format_change(value: float | None) -> tuple[str, str]:
    if value is None:
        return "changed from a near-zero baseline", "changed"
    if value > 0:
        return f"increased by {abs(value):.1%}", "increased"
    if value < 0:
        return f"decreased by {abs(value):.1%}", "decreased"
    return "was broadly unchanged", "was broadly unchanged"


def _complex_interaction(bundle: dict[str, Any]) -> bool:
    decomposition = bundle.get("decomposition", {})
    if not isinstance(decomposition, dict):
        return False
    return decomposition.get("narrative_mode") == "complex_interaction" or decomposition.get("is_volatile") is True


def _existing_caveats(bundle: dict[str, Any]) -> list[str]:
    caveats: list[str] = []
    decision = _decision(bundle)
    freshness = decision.get("freshness_caveat")
    if freshness:
        caveats.append(str(freshness))

    if _complex_interaction(bundle):
        caveats.append("Multiple KPI components moved together, so one isolated percentage attribution would be misleading.")

    retrieved = bundle.get("retrieved_evidence")
    if retrieved:
        caveats.append("External information is supporting context only; the business recommendation remains grounded in internal evidence.")

    action = _action_context(bundle)
    level = str(action.get("action_level") or "")
    if level == "validate_unexplained":
        caveats.append("The currently measured drivers do not explain the dominant incident movement cleanly enough for a direct intervention.")
    elif action.get("is_ambiguous") and level != "data_quality_first":
        caveats.append("The evidence is not decisive enough to treat one explanation as proven; validate the named alternatives before a broad intervention.")

    seen: set[str] = set()
    result: list[str] = []
    for item in caveats:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _business_payload(bundle: dict[str, Any], persona: Persona) -> dict[str, Any]:
    """Send only business interpretation to the LLM, never raw model statistics."""
    action = _action_context(bundle)
    change_text, _ = _format_change(_percent_change(bundle))

    external_context: list[dict[str, str]] = []
    retrieved = bundle.get("retrieved_evidence")
    if isinstance(retrieved, list):
        for item in retrieved[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("source_title") or "").strip()
            reason = str(item.get("reason") or "").strip()
            if title or reason:
                external_context.append({"source": title, "relevance": reason})

    return {
        "persona": persona,
        "kpi": _friendly(_kpi(bundle)),
        "region": _region(bundle),
        "kpi_change": change_text,
        "evidence_confidence": action.get("evidence_confidence") or _confidence(bundle),
        "action_readiness": action.get("action_readiness"),
        "business_interpretation": {
            "action_level": action.get("action_level"),
            "leading_explanation_driver": action.get("leading_explanation_driver_label"),
            "action_target_driver": action.get("action_target_driver_label") or action.get("primary_driver_label"),
            "explanation_differs_from_action_target": action.get("explanation_differs_from_action_target"),
            "upstream_driver": action.get("upstream_driver_label"),
            "mechanism_chain": action.get("mechanism_chain"),
            "finding": action.get("finding"),
            "why_it_matters": action.get("why_it_matters"),
            "primary_driver": action.get("primary_driver_label"),
            "primary_component": action.get("primary_component_label"),
            "driver_direction": action.get("driver_direction"),
            "component_direction": action.get("component_direction"),
            "next_check": action.get("next_check"),
            "action_if_confirmed": action.get("action_if_confirmed"),
            "owner": action.get("owner"),
            "secondary_driver": action.get("secondary_driver_label"),
            "secondary_check": action.get("secondary_check"),
            "movement_status": action.get("movement_status"),
            "is_ambiguous": action.get("is_ambiguous"),
        },
        "multiple_components_moved_together": _complex_interaction(bundle),
        "external_context": external_context,
        "caveats": _existing_caveats(bundle),
    }


def narrate(evidence_bundle: dict[str, Any], persona: Persona) -> dict[str, Any]:
    """Generate business language; fail safely to a deterministic template."""
    if persona not in ("Executive", "Growth analyst"):
        raise ValueError("persona must be 'Executive' or 'Growth analyst'")
    payload = _business_payload(evidence_bundle, persona)
    try:
        response = _client().chat(
            model=NARRATION_MODEL,
            messages=[
                {"role": "system", "content": NARRATOR_SYSTEM},
                {"role": "user", "content": json.dumps(payload, default=str)},
            ],
            format="json",
            options={"temperature": 0},
        )
        content = ""
        if hasattr(response, "message"):
            content = getattr(response.message, "content", "") or ""
        elif isinstance(response, dict):
            message = response.get("message", {})
            if isinstance(message, dict):
                content = str(message.get("content", "") or "")
        return _normalise_output(_extract_json_object(content))
    except Exception:
        return template_narrate(evidence_bundle, persona)


def _headline(bundle: dict[str, Any], action: dict[str, Any]) -> str:
    kpi = _friendly(_kpi(bundle))
    region = _region(bundle)
    _, change_verb = _format_change(_percent_change(bundle))
    level = str(action.get("action_level") or "").lower()
    driver = str(action.get("primary_driver_label") or "").strip()
    component = str(action.get("primary_component_label") or "").strip()

    if level == "data_quality_first":
        return f"{kpi} in {region} needs a data-quality fix before any business action."
    if level == "monitor":
        return f"{kpi} in {region} does not currently require an operational intervention."
    if level == "validate_unexplained":
        return f"{kpi} {change_verb} in {region}, but the measured drivers do not yet explain the dominant {component or 'KPI'} movement."
    if driver:
        verb = "work on" if level == "act" else "validate"
        return f"{kpi} {change_verb} in {region}; {driver} is the first issue to {verb}."
    return f"{kpi} {change_verb} in {region}; the evidence needs validation before action."


def template_narrate(evidence_bundle: dict[str, Any], persona: str) -> dict[str, Any]:
    """Deterministic plain-language fallback derived only from the action context."""
    if persona not in ("Executive", "Growth analyst"):
        raise ValueError("persona must be 'Executive' or 'Growth analyst'")

    kpi = _friendly(_kpi(evidence_bundle))
    region = _region(evidence_bundle)
    change_text, _ = _format_change(_percent_change(evidence_bundle))
    confidence = _confidence(evidence_bundle)
    action = _action_context(evidence_bundle)
    evidence_confidence = str(action.get("evidence_confidence") or confidence)
    action_readiness = str(action.get("action_readiness") or "Validate first")
    leading_explanation = str(action.get("leading_explanation_driver_label") or "").strip()
    action_target = str(action.get("action_target_driver_label") or action.get("primary_driver_label") or "").strip()

    level = str(action.get("action_level") or "validate")
    primary_driver = str(action.get("primary_driver_label") or "").strip()
    primary_component = str(action.get("primary_component_label") or "").strip()
    finding = str(action.get("finding") or "").strip()
    why_it_matters = str(action.get("why_it_matters") or "").strip()
    next_check = str(action.get("next_check") or "").strip()
    action_if_confirmed = str(action.get("action_if_confirmed") or "").strip()
    owner = str(action.get("owner") or "").strip()
    secondary = str(action.get("secondary_driver_label") or "").strip()
    secondary_check = str(action.get("secondary_check") or "").strip()
    upstream = str(action.get("upstream_driver_label") or "").strip()
    mechanism_chain = str(action.get("mechanism_chain") or "").strip()

    sentences: list[str] = [f"{kpi} {change_text} in {region}."]

    if level == "data_quality_first":
        if finding:
            sentences.append(finding)
        if next_check:
            sentences.append(next_check)
        if action_if_confirmed:
            sentences.append(action_if_confirmed)
        sentences.append(f"Evidence confidence remains {evidence_confidence}; action readiness is {action_readiness.lower()} until the source issue is resolved.")

    elif level == "monitor":
        if finding:
            sentences.append(finding)
        if next_check:
            sentences.append(next_check)

    elif level == "validate_unexplained":
        if finding:
            sentences.append(finding)
        if why_it_matters:
            sentences.append(why_it_matters)
        if next_check:
            sentences.append(f"Check next: {next_check}")
        if action_if_confirmed:
            sentences.append(action_if_confirmed)
        sentences.append(f"Evidence confidence is {evidence_confidence}, but action readiness is {action_readiness.lower()}; no direct business lever should be changed from the current aggregate evidence alone.")

    else:
        if leading_explanation and action_target and leading_explanation != action_target:
            sentences.append(
                f"The strongest current explanation is {leading_explanation}, while {action_target} is the first operational signal to check directly."
            )
        if mechanism_chain and upstream:
            sentences.append(f"The operating mechanism is {mechanism_chain}. Treat {upstream} as the upstream trigger and {primary_driver} as the nearer issue to check first.")
        if primary_component and primary_driver:
            sentences.append(f"The first operational path to {('work on' if level == 'act' else 'validate')} is {primary_driver} → {primary_component} → {kpi}.")
        elif primary_driver:
            sentences.append(f"The first issue to {('work on' if level == 'act' else 'validate')} is {primary_driver}.")

        if finding:
            sentences.append(finding)
        if why_it_matters:
            sentences.append(why_it_matters)
        if next_check:
            sentences.append(f"Check now: {next_check}")
        if action_if_confirmed:
            sentences.append(f"If confirmed: {action_if_confirmed}")

        if persona == "Growth analyst" and owner:
            sentences.append(f"Suggested owner: {owner}.")
        if secondary and secondary_check:
            sentences.append(secondary_check)
        sentences.append(f"Evidence confidence is {evidence_confidence}. Action readiness: {action_readiness}.")

    result = {
        "headline": _headline(evidence_bundle, action),
        "narrative": " ".join(sentence for sentence in sentences if sentence),
        "caveats": _existing_caveats(evidence_bundle),
    }
    return _normalise_output(result)
