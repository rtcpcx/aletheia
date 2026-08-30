"""
Aletheia — src/orchestrator.py

Stage 4: adaptive clarification and retrieval planning.

The deterministic analytical pipeline has already selected changepoints,
lags, fitted statistical models, and ranked candidate drivers before this
module is called. Stage 4 therefore MUST NOT redo statistical analysis or ask
external search engines to verify Aletheia's own model outputs.

Pipeline contract:
    deterministic decision packet
        -> ambiguity gate
        -> sanitized retrieval context
        -> one retrieval plan
             * clarification question (human-facing)
             * retrieval query (search-engine-facing, when appropriate)
             * retrieval target: web | internal | none
        -> optional downstream retrieval
        -> confidence-bounded textual support
        -> bounded downstream reweighting
        -> auditable result

This module never writes raw.* or mart.*, never chooses changepoints/lags, and
never retrains statistical models.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

import yaml

try:
    import ollama
except ImportError:  # keeps deterministic fallbacks importable
    ollama = None


PLANNER_MODEL = os.getenv("ALETHEIA_LLM_MODEL", "llama3.1:8b")
# Backwards-compatible alias used by older imports/tests.
CLARIFICATION_MODEL = PLANNER_MODEL
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

TOP_PROBABILITY_GAP_THRESHOLD = float(
    os.getenv("ALETHEIA_TOP_PROBABILITY_GAP", "0.10")
)

SUPPORT_MIN = -0.5
SUPPORT_MAX = 1.0
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0

MAX_RETRIEVAL_CANDIDATES = 2
MAX_QUERY_CHARS = 180
MAX_QUERY_WORDS = 24
MAX_QUESTION_CHARS = 420

RetrievalTarget = Literal["web", "internal", "none"]

# These labels are useful for an internal demo but are not independently
# resolvable public-world locations. A web search using only one of them would
# invite spurious grounding (e.g. generic "South region" results).
_GENERIC_REGION_LABELS = {
    "north",
    "south",
    "east",
    "west",
    "central",
    "global",
    "all",
    "unknown",
    "other",
}

# Stage 4 should never ask a search engine to verify these model-internal
# quantities. Keep the list deterministic so prompt failures cannot leak them.
_FORBIDDEN_RETRIEVAL_TERMS = (
    "coefficient",
    "z-score",
    "z score",
    "p-value",
    "p value",
    "softmax",
    "statistical significance",
    "significance level",
    "normalized score",
    "normalised score",
    "evidence score",
    "regression coefficient",
    "ridge coefficient",
    "holdout correlation",
    "model probability",
    "causal probability",
)

_RETRIEVAL_PLANNER_SYSTEM_PROMPT = r"""
You are the Stage 4 Retrieval Planner for Aletheia, a business KPI root-cause
analysis system.

Aletheia has ALREADY performed deterministic statistical analysis. Do not
repeat that analysis and do not ask external sources to verify internal model
outputs.

Your job is to identify exactly ONE independently checkable fact that would
help discriminate between the supplied competing business hypotheses.

Return three things:

1. clarification_question
   - One natural-language question for an analyst.
   - It should explain what additional real-world fact would discriminate
     between the supplied hypotheses.
   - Ask exactly one question.

2. retrieval_query
   - If public web evidence could answer the question, provide a short,
     search-engine-friendly query.
   - Prefer compact entity/date/location/event/business terms over a full
     conversational question.
   - Otherwise return null.

3. retrieval_target
   - "web" only when the supplied context contains enough real-world grounding
     for a public search to be meaningful.
   - "internal" when the required evidence belongs in company systems or
     internal operational records.
   - "none" when the supplied context cannot support a meaningful retrieval.

Hard rules:
- Use ONLY facts, names, dates, locations, components and hypotheses supplied
  in the input. Never invent companies, products, locations, events or dates.
- Never invent comparison periods such as a prior year, prior month or control
  date unless that comparison period is explicitly supplied in the input.
- Each candidate may include retrieval_scope and external_hypothesis metadata.
  Respect that metadata: internal candidates belong in company systems;
  external candidates may use public evidence; conditional_external candidates
  require their concrete supplied public anchor before web retrieval.
- Do not claim causality.
- Do not ask vague questions such as "what happened?".
- Do not simply ask whether the current leading hypothesis is correct.
- Never ask about coefficients, z-scores, p-values, statistical significance,
  regression outputs, softmax weights, probabilities, normalized scores or
  evidence scores. Those values already exist inside Aletheia.
- A web query must concern independently observable external-world evidence,
  such as a named competitor action, public outage, weather event, regulatory
  event, logistics disruption, public market event or other externally
  resolvable occurrence.
- Internal business measurements such as marketing spend, inventory records,
  CRM activity, conversion logs, order logs, support tickets or internal
  operational records generally require retrieval_target="internal" unless a
  supplied external anchor makes a public search independently useful.
- If the context contains only generic synthetic geography such as North/South
  and generic driver labels, prefer "internal" or "none" rather than pretending
  web search can identify the incident.
- If retrieval_target is "internal", retrieval_query MUST be null.
- If retrieval_target is "none", retrieval_query MUST be null.
- Only retrieval_target="web" may contain a non-null retrieval_query.

Return ONLY valid JSON, no markdown:
{
  "clarification_question": "... ?",
  "retrieval_query": "..." or null,
  "retrieval_target": "web|internal|none"
}
"""
RETRIEVAL_HYPOTHESIS_CONFIG_PATH = os.getenv(
    "ALETHEIA_RETRIEVAL_HYPOTHESES_CONFIG",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
        "retrieval_hypotheses.yaml",
    ),
)


def _load_retrieval_hypothesis_config() -> dict[str, Any]:
    """
    Load retrieval-scope metadata for deterministic Stage-4 routing.

    Unknown hypotheses default to INTERNAL so adding a new driver cannot
    accidentally make it web-searchable before its retrieval policy is defined.
    """
    fallback = {
        "defaults": {"unknown_scope": "internal"},
        "hypotheses": {},
    }

    try:
        with open(RETRIEVAL_HYPOTHESIS_CONFIG_PATH, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return fallback

    if not isinstance(loaded, dict):
        return fallback

    hypotheses = loaded.get("hypotheses")
    if not isinstance(hypotheses, dict):
        hypotheses = {}

    defaults = loaded.get("defaults")
    if not isinstance(defaults, dict):
        defaults = {"unknown_scope": "internal"}

    return {
        "defaults": defaults,
        "hypotheses": hypotheses,
    }


_RETRIEVAL_HYPOTHESIS_CONFIG = _load_retrieval_hypothesis_config()


def _hypothesis_rule(name: str) -> dict[str, Any]:
    hypotheses = _RETRIEVAL_HYPOTHESIS_CONFIG.get("hypotheses", {})
    rule = hypotheses.get(name, {}) if isinstance(hypotheses, dict) else {}
    if not isinstance(rule, dict):
        rule = {}

    defaults = _RETRIEVAL_HYPOTHESIS_CONFIG.get("defaults", {})
    unknown_scope = (
        defaults.get("unknown_scope", "internal")
        if isinstance(defaults, dict)
        else "internal"
    )

    scope = str(rule.get("retrieval_scope", unknown_scope)).strip().lower()
    if scope not in {"internal", "external", "conditional_external"}:
        scope = "internal"

    return {
        **rule,
        "retrieval_scope": scope,
    }


@dataclass(frozen=True)
class RetrievalPlan:
    clarification_question: str
    retrieval_query: str | None
    retrieval_target: RetrievalTarget
    planner_status: str = "generated"
    planner_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ollama_client():
    """Return an Ollama client using the configured local endpoint."""
    if ollama is None:
        raise RuntimeError("ollama package is not installed")
    if OLLAMA_HOST:
        return ollama.Client(host=OLLAMA_HOST)
    return ollama.Client()


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object even if a local model accidentally adds fences/text."""
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
        raise ValueError("LLM response was not a JSON object")
    return parsed


def _nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get_confidence(packet: dict[str, Any]) -> str:
    """Accept both decision-packet and evidence-bundle confidence shapes."""
    value = packet.get("confidence")
    if isinstance(value, dict):
        value = value.get("confidence_level") or value.get("level")
    if value is None:
        value = packet.get("confidence_level")
    if value is None:
        value = _nested_dict(packet.get("decision")).get("confidence_level")
    return str(value or "").strip().lower()


def _get_top_gap(packet: dict[str, Any]) -> float:
    candidates = [
        packet.get("top_probability_gap"),
        _nested_dict(packet.get("decision")).get("top_probability_gap"),
        _nested_dict(packet.get("confidence")).get("top_probability_gap"),
    ]
    for value in candidates:
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 1.0


def _has_significant_driver(packet: dict[str, Any]) -> bool:
    candidates = [
        packet.get("any_driver_significant"),
        packet.get("top_driver_significant"),
        _nested_dict(packet.get("decision")).get("any_driver_significant"),
        _nested_dict(packet.get("model_validity")).get("any_driver_significant"),
    ]
    for value in candidates:
        if value is not None:
            return bool(value)
    # Missing validity metadata should not itself manufacture ambiguity.
    return True


def _retrieval_policy_default(name: str, fallback: Any) -> Any:
    defaults = _RETRIEVAL_HYPOTHESIS_CONFIG.get("defaults", {})
    if isinstance(defaults, dict) and name in defaults:
        return defaults[name]
    return fallback


def _policy_float(rule: dict[str, Any], name: str, fallback: float) -> float:
    value = rule.get(name, _retrieval_policy_default(name, fallback))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _policy_bool(rule: dict[str, Any], name: str, fallback: bool) -> bool:
    value = rule.get(name, _retrieval_policy_default(name, fallback))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value) if value is not None else fallback


def _candidate_has_usable_evidence(item: dict[str, Any]) -> bool:
    """Fail closed when Stage 3 explicitly abstained or produced no evidence."""
    mode = str(item.get("evidence_mode") or "").strip().lower()
    status = str(item.get("model_status") or "").strip().lower()
    if mode == "insufficient_evidence":
        return False
    if status.startswith("insufficient"):
        return False
    score = _safe_float(item.get("evidence_score"))
    if score is not None and score <= 0.0:
        return False
    if str(item.get("direction_consistency") or "").strip().lower() == "contradictory":
        return False
    return True


def _component_pool(
    decision_packet: dict[str, Any] | None,
    driver_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred = _preferred_component(decision_packet)
    pool = list(driver_evidence)
    if preferred:
        rows = [
            row
            for row in pool
            if str(row.get("explains_component") or "").strip() == preferred
        ]
        if rows:
            pool = rows
    return pool


def _public_web_candidate_eligible(
    item: dict[str, Any],
    pool: list[dict[str, Any]],
    decision_packet: dict[str, Any] | None,
) -> bool:
    """Whether public evidence could materially discriminate this candidate.

    Public search is intentionally stricter than merely being *searchable*.
    The candidate must already have usable deterministic evidence, meaningful
    within-component weight, meaningful absolute evidence, and (by default) a
    second plausible deterministic hypothesis to discriminate against.
    Conditional-external hypotheses additionally require their concrete named
    public anchor.
    """
    name = _candidate_name(item)
    if not name or not _candidate_scope_allows_web(name, decision_packet):
        return False
    if not _candidate_has_usable_evidence(item):
        return False

    rule = _hypothesis_rule(name)
    min_weight = _policy_float(rule, "public_candidate_min_weight", 0.15)
    min_score = _policy_float(rule, "public_candidate_min_evidence_score", 0.05)
    competing_min_weight = _policy_float(rule, "public_competing_min_weight", 0.15)
    require_competing = _policy_bool(
        rule, "public_require_competing_hypothesis", True
    )

    if _rank_value(item) < min_weight:
        return False
    evidence_score = abs(_safe_float(item.get("evidence_score")) or 0.0)
    if evidence_score < min_score:
        return False

    if require_competing:
        name = _candidate_name(item)
        has_competitor = any(
            _candidate_name(other) != name
            and _candidate_has_usable_evidence(other)
            and _rank_value(other) >= competing_min_weight
            for other in pool
        )
        if not has_competitor:
            return False

    return True


def requires_external_verification(
    decision_packet: dict[str, Any],
    driver_evidence: list[dict[str, Any]] | None,
) -> bool:
    """Check whether a decision-relevant external hypothesis merits corroboration."""
    if not driver_evidence:
        return False
    pool = _component_pool(decision_packet, list(driver_evidence))
    for row in pool:
        if not _public_web_candidate_eligible(row, pool, decision_packet):
            continue
        name = _candidate_name(row)
        rule = _hypothesis_rule(name)
        if str(rule.get("verification_mode") or "").strip().lower() != "corroborate_if_material":
            continue
        zscore = abs(_safe_float(row.get("driver_zscore")) or 0.0)
        structural = abs(_safe_float(row.get("structural_break_score")) or 0.0)
        min_z = float(rule.get("verification_min_abs_zscore", 3.0) or 3.0)
        min_break = float(rule.get("verification_min_structural_break", 0.15) or 0.15)
        if zscore >= min_z or structural >= min_break:
            return True
    return False

def needs_clarification(
    decision_packet: dict[str, Any],
    driver_evidence: list[dict[str, Any]] | None = None,
    threshold: float = TOP_PROBABILITY_GAP_THRESHOLD,
) -> bool:
    """Stage-4 gate for unresolved ambiguity OR material external verification."""
    decision_type = str(
        decision_packet.get("decision_type")
        or _nested_dict(decision_packet.get("decision")).get("decision_type")
        or ""
    ).strip().lower()
    if decision_type and decision_type != "investigate":
        return False
    action_context = _nested_dict(decision_packet.get("action_context"))
    action_level = str(action_context.get("action_level") or "").strip().lower()
    if action_level in {"monitor", "data_quality_first"}:
        return False
    source_health = _nested_dict(decision_packet.get("source_health"))
    if source_health and source_health.get("healthy") is False:
        return False
    if requires_external_verification(decision_packet, driver_evidence):
        return True
    if action_context.get("is_ambiguous") is True:
        return True
    confidence = _get_confidence(decision_packet)
    top_gap = _get_top_gap(decision_packet)
    any_significant = _has_significant_driver(decision_packet)
    return confidence == "low" or top_gap < threshold or not any_significant


def _candidate_name(item: dict[str, Any]) -> str:
    return str(
        item.get("driver_name")
        or item.get("hypothesis")
        or item.get("name")
        or ""
    ).strip()


def _rank_value(item: dict[str, Any]) -> float:
    for key in (
        "softmax_probability",
        "probability",
        "normalized_score",
        "evidence_score",
    ):
        try:
            value = item.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _preferred_component(decision_packet: dict[str, Any] | None) -> str | None:
    if not isinstance(decision_packet, dict):
        return None
    action = _nested_dict(decision_packet.get("action_context"))
    value = action.get("primary_component")
    return str(value).strip() if value not in (None, "") else None


def _candidate_scope_allows_web(name: str, decision_packet: dict[str, Any] | None) -> bool:
    rule = _hypothesis_rule(name)
    scope = str(rule.get("retrieval_scope") or "internal").strip().lower()
    if scope == "external":
        return True
    if scope != "conditional_external":
        return False

    required = [
        str(key)
        for key in (rule.get("required_anchor_any") or [])
        if str(key).strip()
    ]
    if not required:
        return True
    packet = decision_packet if isinstance(decision_packet, dict) else {}
    anchors = _external_anchors(packet)
    return any(str(anchors.get(key) or "").strip() for key in required)


def select_retrieval_candidates(
    driver_evidence: list[dict[str, Any]],
    limit: int = MAX_RETRIEVAL_CANDIDATES,
    decision_packet: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return a small, component-consistent hypothesis set for Stage 4.

    The strongest deterministic candidate is always preserved. A public-search
    candidate gets a reserved slot only when it is not merely searchable but
    *decision-relevant*: usable evidence, material weight/score, directionally
    plausible, and (by default) able to discriminate against another plausible
    hypothesis in the same KPI component.
    """
    if limit <= 0:
        return []

    pool = _component_pool(decision_packet, list(driver_evidence))
    ranked = sorted(pool, key=_rank_value, reverse=True)

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in ranked:
        name = _candidate_name(item)
        if not name or name in seen:
            continue
        seen.add(name)
        unique.append(item)

    if not unique:
        return []

    selected: list[dict[str, Any]] = [unique[0]]
    if limit == 1:
        return selected

    selected_names = {_candidate_name(item) for item in selected}

    # Backwards compatibility for callers that do not provide a decision packet:
    # preserve the old config-only candidate behavior. Production Stage 4 always
    # supplies the packet and therefore uses the stricter precision policy.
    def web_candidate(item: dict[str, Any]) -> bool:
        if decision_packet is None:
            return (
                _rank_value(item) > 0.0
                and _candidate_scope_allows_web(_candidate_name(item), None)
            )
        return _public_web_candidate_eligible(item, pool, decision_packet)

    if not any(web_candidate(item) for item in selected):
        external = next((item for item in unique[1:] if web_candidate(item)), None)
        if external is not None:
            selected.append(external)
            selected_names.add(_candidate_name(external))

    for item in unique[1:]:
        if len(selected) >= limit:
            break
        name = _candidate_name(item)
        if name in selected_names:
            continue
        selected.append(item)
        selected_names.add(name)

    return selected[:limit]

def _top_two_hypotheses(driver_evidence: list[dict[str, Any]]) -> list[str]:
    """Backwards-compatible helper retained for tests/older imports."""
    return [
        _candidate_name(item)
        for item in select_retrieval_candidates(driver_evidence, limit=2)
        if _candidate_name(item)
    ]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _direction_from_packet(packet: dict[str, Any]) -> str:
    change = _safe_float(packet.get("percent_change"))
    if change is None:
        change = _safe_float(packet.get("absolute_change"))
    if change is None or abs(change) < 1e-12:
        return "flat_or_unknown"
    return "up" if change > 0 else "down"


def _external_anchors(packet: dict[str, Any]) -> dict[str, str]:
    """
    Preserve optional externally resolvable identifiers if future datasets add
    them to the decision packet. No key is invented and empty values are
    discarded.
    """
    keys = (
        "company",
        "brand",
        "product",
        "product_id",
        "competitor",
        "competitor_name",
        "market",
        "city",
        "state",
        "country",
        "location",
        "geography",
        "event_name",
    )
    anchors: dict[str, str] = {}
    for key in keys:
        value = packet.get(key)
        if value not in (None, ""):
            anchors[key] = str(value).strip()
    return anchors


def build_retrieval_context(
    decision_packet: dict[str, Any],
    driver_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the sanitized Stage-4 planner payload.

    Model-internal magnitudes remain hidden. The only new V4.1 signal exposed
    to the planner is an *effective retrieval scope*: an externally searchable
    driver is downgraded to internal unless deterministic evidence says public
    corroboration could materially discriminate the current decision.
    """
    pool = _component_pool(decision_packet, list(driver_evidence))
    candidates = []
    for item in select_retrieval_candidates(
        driver_evidence, decision_packet=decision_packet
    ):
        name = _candidate_name(item)
        rule = _hypothesis_rule(name)
        configured_scope = str(rule.get("retrieval_scope", "internal"))
        public_eligible = _public_web_candidate_eligible(
            item, pool, decision_packet
        )
        effective_scope = configured_scope if public_eligible else "internal"
        candidates.append(
            {
                "name": name,
                "explains_component": item.get("explains_component"),
                "evidence_mode": item.get("evidence_mode"),
                "model_status": item.get("model_status"),
                "retrieval_scope": effective_scope,
                "configured_retrieval_scope": configured_scope,
                "external_hypothesis": (
                    rule.get("external_hypothesis") if public_eligible else None
                ),
                "public_web_eligible": public_eligible,
            }
        )

    incident = decision_packet.get("window_start")
    if isinstance(incident, (dt.datetime, dt.date)):
        incident = incident.isoformat()
    elif incident is not None:
        incident = str(incident)

    return {
        "kpi": decision_packet.get("kpi"),
        "region": decision_packet.get("region"),
        "incident_date": incident,
        "direction": _direction_from_packet(decision_packet),
        "confidence": _get_confidence(decision_packet) or "unknown",
        "decision_type": decision_packet.get("decision_type"),
        "candidate_hypotheses": candidates,
        "external_anchors": _external_anchors(decision_packet),
    }

def _contains_forbidden_terms(text: str | None) -> bool:
    value = (text or "").lower()
    return any(term in value for term in _FORBIDDEN_RETRIEVAL_TERMS)


def _clean_query(value: Any) -> str | None:
    if value is None:
        return None
    query = re.sub(r"\s+", " ", str(value)).strip().strip('"\'')
    return query or None


def _normalise_words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
        if len(token) >= 3
    }


def _extract_years(text: str | None) -> set[str]:
    return set(re.findall(r"\b(?:19|20)\d{2}\b", text or ""))


def _extract_iso_dates(text: str | None) -> set[str]:
    return set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text or ""))


def _supplied_grounding_text(context: dict[str, Any]) -> str:
    """Flatten only deterministic planner inputs used for grounding."""
    parts: list[str] = []

    for key in ("kpi", "region", "incident_date"):
        value = context.get(key)
        if value:
            parts.append(str(value))

    for candidate in context.get("candidate_hypotheses", []):
        if not isinstance(candidate, dict):
            continue
        for key in ("name", "explains_component", "external_hypothesis"):
            value = candidate.get(key)
            if value:
                parts.append(str(value))

    anchors = context.get("external_anchors", {})
    if isinstance(anchors, dict):
        parts.extend(str(v) for v in anchors.values() if v)

    return " ".join(parts)


def _query_is_grounded(query: str, context: dict[str, Any]) -> bool:
    """Require lexical grounding and reject invented explicit dates/years."""
    supplied_text = _supplied_grounding_text(context)

    supplied_tokens = _normalise_words(supplied_text)
    query_tokens = _normalise_words(query)
    if not (supplied_tokens & query_tokens):
        return False

    query_dates = _extract_iso_dates(query)
    supplied_dates = _extract_iso_dates(supplied_text)
    if query_dates - supplied_dates:
        return False

    query_years = _extract_years(query)
    supplied_years = _extract_years(supplied_text)
    if query_years - supplied_years:
        return False

    return True


def _query_contains_resolvable_anchor(query: str, context: dict[str, Any]) -> bool:
    """Require the web query itself to contain a supplied real-world anchor."""
    candidate_anchors: list[str] = []

    region = str(context.get("region") or "").strip()
    if region and region.lower() not in _GENERIC_REGION_LABELS:
        candidate_anchors.append(region)

    anchors = context.get("external_anchors", {})
    if isinstance(anchors, dict):
        candidate_anchors.extend(
            str(value).strip()
            for value in anchors.values()
            if str(value or "").strip()
        )

    query_tokens = _normalise_words(query)
    for anchor in candidate_anchors:
        anchor_tokens = _normalise_words(anchor)
        if anchor_tokens and anchor_tokens.issubset(query_tokens):
            return True

    return False


def _query_matches_external_hypothesis(query: str, context: dict[str, Any]) -> bool:
    """Require a configured search term for an externally-searchable candidate."""
    allowed_terms: set[str] = set()

    for candidate in context.get("candidate_hypotheses", []):
        if not isinstance(candidate, dict):
            continue

        name = str(candidate.get("name") or "").strip()
        rule = _hypothesis_rule(name)
        scope = str(candidate.get("retrieval_scope") or "internal").strip().lower()
        if scope not in {"external", "conditional_external"}:
            continue

        for value in rule.get("query_terms", []) or []:
            allowed_terms |= _normalise_words(str(value))

        allowed_terms |= _normalise_words(name)
        external_label = rule.get("external_hypothesis")
        if external_label:
            allowed_terms |= _normalise_words(str(external_label))

    if not allowed_terms:
        return False

    return bool(_normalise_words(query) & allowed_terms)


def _has_resolvable_web_anchor(context: dict[str, Any]) -> bool:
    anchors = context.get("external_anchors", {})
    if isinstance(anchors, dict) and any(str(v).strip() for v in anchors.values()):
        return True

    region = str(context.get("region") or "").strip()
    if region and region.lower() not in _GENERIC_REGION_LABELS:
        return True

    # With only generic demo geography and generic metric/driver labels, web
    # retrieval is more likely to fabricate relevance than add evidence.
    return False


def _configured_external_fallback_plan(
    context: dict[str, Any],
    *,
    reason: str,
) -> RetrievalPlan | None:
    """Build a deterministic, grounded web plan for an explicit external hypothesis.

    This is not scenario logic. The route is driven only by
    config/retrieval_hypotheses.yaml plus the supplied region/date. It exists so
    a local planner failure cannot silently suppress externally verifiable
    evidence such as weather or a public outage.
    """
    if not _has_resolvable_web_anchor(context):
        return None

    external_candidate: dict[str, Any] | None = None
    for candidate in context.get("candidate_hypotheses", []):
        if not isinstance(candidate, dict):
            continue
        name = str(candidate.get("name") or "").strip()
        if not name:
            continue
        rule = _hypothesis_rule(name)
        scope = str(candidate.get("retrieval_scope") or "internal").strip().lower()
        if scope == "external":
            external_candidate = candidate
            break
        if scope == "conditional_external":
            required = [
                str(key)
                for key in (rule.get("required_anchor_any") or [])
                if str(key).strip()
            ]
            anchors = context.get("external_anchors", {})
            if isinstance(anchors, dict) and (
                not required
                or any(str(anchors.get(key) or "").strip() for key in required)
            ):
                external_candidate = candidate
                break

    if external_candidate is None:
        return None

    name = str(external_candidate.get("name") or "").strip()
    rule = _hypothesis_rule(name)
    scope = str(external_candidate.get("retrieval_scope") or "internal").strip().lower()
    external_label = str(rule.get("external_hypothesis") or name).strip()
    query_terms = [str(v).strip() for v in (rule.get("query_terms") or []) if str(v).strip()]
    search_terms = query_terms[:4] if query_terms else [external_label.replace("_", " ")]

    region = str(context.get("region") or "").strip()
    incident = str(context.get("incident_date") or "").strip()

    concrete_anchor = ""
    if scope == "conditional_external":
        anchors = context.get("external_anchors", {})
        required = [
            str(key)
            for key in (rule.get("required_anchor_any") or [])
            if str(key).strip()
        ]
        if isinstance(anchors, dict):
            for key in required:
                value = str(anchors.get(key) or "").strip()
                if value:
                    concrete_anchor = value
                    break

    query_parts = [concrete_anchor, region, incident, *search_terms]
    # Preserve order while avoiding duplicate anchor tokens such as
    # duplicate location tokens in a fallback query.
    deduped_parts: list[str] = []
    seen_parts: set[str] = set()
    for part in query_parts:
        key = part.lower().strip()
        if not key or key in seen_parts:
            continue
        seen_parts.add(key)
        deduped_parts.append(part.strip())
    query = " ".join(deduped_parts).strip()

    names = [
        str(item.get("name") or "").strip()
        for item in context.get("candidate_hypotheses", [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    if len(names) >= 2:
        question = (
            f"What public evidence around the incident could distinguish {names[0]} "
            f"from {names[1]} in {region}?"
        )
    else:
        question = (
            f"What public evidence around the incident could verify whether "
            f"{external_label.replace('_', ' ')} was present in {region}?"
        )

    plan = RetrievalPlan(
        clarification_question=question,
        retrieval_query=query or None,
        retrieval_target="web",
        planner_status="policy_fallback",
        planner_reason=reason,
    )
    valid, _ = _validate_plan(plan, context)
    return plan if valid else None


def _normalize_retrieval_plan(
    plan: RetrievalPlan,
    context: dict[str, Any],
) -> RetrievalPlan:
    """Canonicalize planner output against deterministic retrieval policy.

    Explicit ``external`` hypotheses are policy-level declarations that their
    missing evidence belongs on the public web. When Stage 4 is already active
    and such a candidate has a real-world anchor, a local-model ``internal`` or
    ``none`` answer is upgraded to a grounded web plan rather than silently
    suppressing retrieval. Conditional external hypotheses still require their
    configured concrete anchor.
    """
    target = str(plan.retrieval_target or "").strip().lower()
    question = str(plan.clarification_question or "").strip()

    external_policy_plan = _configured_external_fallback_plan(
        context, reason="config-approved external hypothesis requires public verification"
    )

    if target == "web" and (
        not _has_resolvable_web_anchor(context)
        or not _web_retrieval_is_semantically_allowed(context)
    ):
        target = "internal"
        query = None
        reason = (
            "web plan downgraded to internal: "
            "candidate hypotheses are not externally verifiable"
        )
    elif target in {"internal", "none"} and external_policy_plan is not None:
        return RetrievalPlan(
            clarification_question=question or external_policy_plan.clarification_question,
            retrieval_query=external_policy_plan.retrieval_query,
            retrieval_target="web",
            planner_status=plan.planner_status,
            planner_reason="planner route normalized by external retrieval policy",
        )
    elif target in {"internal", "none"}:
        query = None
        reason = plan.planner_reason
    else:
        query = _clean_query(plan.retrieval_query)
        reason = plan.planner_reason

    return RetrievalPlan(
        clarification_question=question,
        retrieval_query=query,
        retrieval_target=target,  # type: ignore[arg-type]
        planner_status=plan.planner_status,
        planner_reason=reason,
    )


def _validate_plan(plan: RetrievalPlan, context: dict[str, Any]) -> tuple[bool, str]:
    question = plan.clarification_question.strip()
    query = _clean_query(plan.retrieval_query)

    if not question or question.count("?") != 1:
        return False, "clarification question must contain exactly one question"
    if len(question) > MAX_QUESTION_CHARS:
        return False, "clarification question is too long"
    if _contains_forbidden_terms(question):
        return False, "clarification question references model-internal statistics"
    if plan.retrieval_target not in {"web", "internal", "none"}:
        return False, "invalid retrieval target"
    

    if plan.retrieval_target == "web":
        if not query:
            return False, "web retrieval requires a query"
        if "?" in query:
            return False, "web query must be search-oriented rather than a question"
        if len(query) > MAX_QUERY_CHARS or len(query.split()) > MAX_QUERY_WORDS:
            return False, "web query is too long"
        if _contains_forbidden_terms(query):
            return False, "web query references model-internal statistics"
        if not _query_is_grounded(query, context):
            return False, "web query is not grounded in supplied context"
        if not _has_resolvable_web_anchor(context):
            return False, "no externally resolvable anchor is available"
        if not _query_contains_resolvable_anchor(query, context):
            return False, "web query does not contain a supplied external anchor"
        if not _web_retrieval_is_semantically_allowed(context):
            return False, "candidate hypotheses are not externally verifiable"
        if not _query_matches_external_hypothesis(query, context):
            return False, "web query does not match an externally searchable hypothesis"
    else:
        # Internal/none plans must not accidentally leak a query into web code.
        if query:
            return False, "non-web retrieval target must use a null query"

    return True, "valid"


def _safe_fallback_plan(
    context: dict[str, Any],
    reason: str = "planner unavailable or invalid",
) -> RetrievalPlan:
    external = _configured_external_fallback_plan(context, reason=reason)
    if external is not None:
        return external

    names = [
        str(item.get("name") or "").strip()
        for item in context.get("candidate_hypotheses", [])
        if isinstance(item, dict)
    ]
    names = [name for name in names if name]

    if len(names) >= 2:
        question = (
            f"What independently verifiable record could distinguish {names[0]} "
            f"from {names[1]} during the incident window?"
        )
    elif len(names) == 1:
        question = (
            f"What independently verifiable record could confirm whether "
            f"{names[0]} was active during the incident window?"
        )
    else:
        question = (
            "What independently verifiable record could distinguish the leading "
            "explanations during the incident window?"
        )

    return RetrievalPlan(
        clarification_question=question,
        retrieval_query=None,
        retrieval_target="none",
        planner_status="safe_fallback",
        planner_reason=reason,
    )


def generate_retrieval_plan(
    decision_packet: dict[str, Any],
    driver_evidence: list[dict[str, Any]],
) -> RetrievalPlan:
    """
    Generate a validated Stage-4 retrieval plan from sanitized business context.

    Any malformed, ungrounded, model-internal or externally unresolvable web
    plan is converted to a safe abstention. This is intentionally stricter than
    merely trusting temperature=0 output.
    """
    context = build_retrieval_context(decision_packet, driver_evidence)

    if not context["candidate_hypotheses"]:
        return _safe_fallback_plan(context, "no candidate hypotheses supplied")

    try:
        response = _ollama_client().chat(
            model=PLANNER_MODEL,
            messages=[
                {"role": "system", "content": _RETRIEVAL_PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(context, default=str)},
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

        parsed = _extract_json_object(content)
        target = str(parsed.get("retrieval_target", "")).strip().lower()
        if target not in {"web", "internal", "none"}:
            raise ValueError("invalid retrieval_target")

        plan = RetrievalPlan(
            clarification_question=str(
                parsed.get("clarification_question", "")
            ).strip(),
            retrieval_query=_clean_query(parsed.get("retrieval_query")),
            retrieval_target=target,  # type: ignore[arg-type]
            planner_status="generated",
            planner_reason=None,
        )

        # Normalize benign formatting/shape inconsistencies before applying
        # strict semantic validation. In particular, internal/none targets
        # must never carry a web query downstream.
        plan = _normalize_retrieval_plan(plan, context)

        valid, reason = _validate_plan(plan, context)
        if not valid:
            return _safe_fallback_plan(context, f"planner output rejected: {reason}")
        return plan

    except Exception as exc:
        return _safe_fallback_plan(
            context,
            f"planner unavailable or invalid: {type(exc).__name__}",
        )


def _deterministic_question(driver_evidence: list[dict[str, Any]]) -> str:
    """Backwards-compatible deterministic question helper."""
    context = {
        "candidate_hypotheses": [
            {"name": name} for name in _top_two_hypotheses(driver_evidence)
        ]
    }
    return _safe_fallback_plan(context).clarification_question


def generate_clarifying_question(
    decision_packet: dict[str, Any],
    driver_evidence: list[dict[str, Any]],
) -> str:
    """
    Backwards-compatible wrapper.

    New code should call generate_retrieval_plan() so the human clarification
    question is not accidentally reused as the web search query.
    """
    return generate_retrieval_plan(
        decision_packet, driver_evidence
    ).clarification_question


def run_stage4(
    decision_packet: dict[str, Any],
    driver_evidence: list[dict[str, Any]],
    *,
    retrieve_evidence: Callable[[str], list[dict[str, Any]]] | None = None,
    score_relevance: Callable[[str, str, list[dict[str, Any]]], Any] | None = None,
    score_relevance_batch: Callable[..., dict[str, Any]] | None = None,
    reweight_and_renormalize: Callable[
        [dict[str, float], dict[str, float]], dict[str, float]
    ] | None = None,
) -> dict[str, Any]:
    """
    Standalone Stage-4 execution helper.

    The project pipeline may perform persistence itself, but this function is
    useful for unit/integration testing and injected offline tests.
    """
    probabilities_before = _probabilities_from_evidence(driver_evidence)
    result: dict[str, Any] = {
        "clarification_needed": needs_clarification(decision_packet, driver_evidence),
        "clarifying_question": None,
        "retrieval_plan": None,
        "retrieved_evidence": [],
        "source_assessments": [],
        "retrieval_support": {},
        "probabilities_before": probabilities_before,
        "probabilities_after": dict(probabilities_before),
        "status": "deterministic_only",
    }

    if not result["clarification_needed"]:
        return result

    plan = generate_retrieval_plan(decision_packet, driver_evidence)
    result["clarifying_question"] = plan.clarification_question
    result["retrieval_plan"] = plan.to_dict()

    if plan.retrieval_target != "web" or not plan.retrieval_query:
        result["status"] = (
            "clarification_internal"
            if plan.retrieval_target == "internal"
            else "clarification_abstained"
        )
        return result

    if retrieve_evidence is None:
        from .retrieval import retrieve_evidence as _retrieve_evidence

        retrieve_evidence = _retrieve_evidence

    if score_relevance_batch is None:
        try:
            from .retrieval import score_relevance_batch as _score_relevance_batch

            score_relevance_batch = _score_relevance_batch
        except ImportError:
            score_relevance_batch = None

    if score_relevance is None:
        from .retrieval import score_relevance as _score_relevance

        score_relevance = _score_relevance

    if reweight_and_renormalize is None:
        from .evidence_fusion import (
            reweight_and_renormalize as _reweight_and_renormalize,
        )

        reweight_and_renormalize = _reweight_and_renormalize

    try:
        retrieved = retrieve_evidence(plan.retrieval_query) or []
    except Exception:
        retrieved = []
    result["retrieved_evidence"] = retrieved

    candidates = select_retrieval_candidates(
        driver_evidence, decision_packet=decision_packet
    )
    hypotheses = [_candidate_name(item) for item in candidates if _candidate_name(item)]
    support: dict[str, float] = {name: 0.0 for name in hypotheses}

    if retrieved and hypotheses and score_relevance_batch is not None:
        try:
            batch = score_relevance_batch(
                question=plan.clarification_question,
                retrieval_query=plan.retrieval_query,
                hypotheses=hypotheses,
                results=retrieved,
            )
            result["source_assessments"] = batch.get("source_assessments", [])
            for name in hypotheses:
                score = _nested_dict(batch.get("hypothesis_scores")).get(name, {})
                value = _safe_float(_nested_dict(score).get("effective_support"))
                if value is not None:
                    support[name] = max(SUPPORT_MIN, min(SUPPORT_MAX, value))
        except Exception:
            pass
    elif retrieved and hypotheses and score_relevance is not None:
        # Compatibility path for callers injecting only the legacy scorer.
        for name in hypotheses:
            try:
                raw = score_relevance(plan.clarification_question, name, retrieved)
                if isinstance(raw, dict):
                    raw_support = raw.get("support", 0.0)
                    raw_confidence = raw.get("confidence", 0.0)
                    value = float(raw_support) * float(raw_confidence)
                else:
                    value = float(raw)
                support[name] = max(SUPPORT_MIN, min(SUPPORT_MAX, value))
            except Exception:
                support[name] = 0.0

    result["retrieval_support"] = support

    if not probabilities_before:
        result["status"] = "ambiguous_no_deterministic_probabilities"
        return result

    # Only assessed top hypotheses receive support entries. Updated fusion code
    # preserves all unassessed hypotheses exactly and preserves the assessed
    # hypotheses' total pre-retrieval mass.
    if any(abs(value) > 1e-12 for value in support.values()):
        try:
            probabilities_after = reweight_and_renormalize(
                probabilities_before,
                support,
            )
            result["probabilities_after"] = _safe_float_dict(
                probabilities_after,
                normalize=False,
            )
            result["status"] = "retrieval_fused"
        except Exception:
            result["status"] = "retrieval_scored_fusion_failed"
    else:
        result["status"] = "retrieval_no_effective_support"

    return result


def _deterministic_scores(
    driver_evidence: list[dict[str, Any]],
) -> dict[str, float]:
    """Retained for backwards compatibility; prefer normalized_score first."""
    scores: dict[str, float] = {}
    for item in driver_evidence:
        name = _candidate_name(item)
        if not name:
            continue
        for key in ("normalized_score", "evidence_score", "softmax_probability"):
            try:
                value = float(item.get(key))
                if value >= 0:
                    scores[name] = value
                    break
            except (TypeError, ValueError):
                continue
    return scores


def _probabilities_from_evidence(
    driver_evidence: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Return stored relative hypothesis weights without inventing a global
    normalization across KPI components.

    Evidence-engine softmax is component-relative. Renormalizing all drivers
    across components here would silently change its semantics.
    """
    values: dict[str, float] = {}
    for item in driver_evidence:
        name = _candidate_name(item)
        if not name:
            continue
        try:
            probability = float(
                item.get("softmax_probability", item.get("probability"))
            )
        except (TypeError, ValueError):
            continue
        if probability >= 0:
            # If duplicate driver rows occur, preserve the strongest occurrence.
            values[name] = max(values.get(name, 0.0), probability)
    return values


def _safe_float_dict(
    value: Any,
    *,
    normalize: bool = False,
) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}

    result: dict[str, float] = {}
    for key, item in value.items():
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            result[str(key)] = number

    if normalize:
        total = sum(result.values())
        if total > 0:
            return {key: number / total for key, number in result.items()}
    return result
def _candidate_names(context: dict[str, Any]) -> list[str]:
    names = []
    for item in context.get("candidate_hypotheses", []):
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names


def _required_anchor_available(context: dict[str, Any], required_keys: list[str]) -> bool:
    anchors = context.get("external_anchors", {})
    if not isinstance(anchors, dict):
        return False
    return any(str(anchors.get(key) or "").strip() for key in required_keys)


def _web_retrieval_is_semantically_allowed(context: dict[str, Any]) -> bool:
    """Use the *effective* candidate scope produced by the precision gate."""
    candidates = context.get("candidate_hypotheses", [])
    if not isinstance(candidates, list) or not candidates:
        return False

    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        scope = str(item.get("retrieval_scope") or "internal").strip().lower()
        if scope == "external":
            return True
        if scope == "conditional_external":
            rule = _hypothesis_rule(name)
            required = [
                str(key)
                for key in (rule.get("required_anchor_any") or [])
                if str(key).strip()
            ]
            if not required or _required_anchor_available(context, required):
                return True
    return False

