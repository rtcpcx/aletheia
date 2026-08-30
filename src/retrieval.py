"""
Aletheia — src/retrieval.py

External retrieval is downstream support only. This module has NO write path to
raw.* or mart.*; its only persistence target is analysis.retrieved_context.

Stack:
    - ddgs            web search (DuckDuckGo)
    - requests + bs4  best-effort page extraction
    - ollama          local structured relevance assessment

Design goals:
    * the search query is supplied by the Stage-4 retrieval planner, not by
      blindly reusing a human-facing clarification question;
    * duplicate/empty results are removed before scoring;
    * one batched LLM call scores each source against each target hypothesis;
    * source-level assessments remain auditable;
    * aggregated retrieval support is confidence-weighted and bounded;
    * failures safely degrade to zero support rather than fabricated evidence.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

try:
    import ollama
except ImportError:
    ollama = None

try:
    from ddgs import DDGS
except ImportError:  # pragma: no cover
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

from src import database

RELEVANCE_MODEL = os.getenv("ALETHEIA_LLM_MODEL", "llama3.1:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

MAX_RETRIEVAL_RESULTS = int(os.getenv("ALETHEIA_MAX_RETRIEVAL_RESULTS", "5"))
FETCH_TIMEOUT_SECONDS = float(os.getenv("ALETHEIA_FETCH_TIMEOUT_SECONDS", "8"))
MAX_FETCHED_CHARS = int(os.getenv("ALETHEIA_MAX_FETCHED_CHARS", "3000"))
MAX_SCORING_CHARS_PER_SOURCE = int(
    os.getenv("ALETHEIA_MAX_SCORING_CHARS_PER_SOURCE", "1200")
)
USER_AGENT = "AletheiaResearchBot/2.0 (+internal decision-intelligence tool)"

SUPPORT_MIN = -0.5
SUPPORT_MAX = 1.0
CONFIDENCE_MIN = 0.0
CONFIDENCE_MAX = 1.0


# ---------------------------------------------------------------------------
# Search + fetch (no LLM involved)
# ---------------------------------------------------------------------------

def _ddgs_text_search(query: str, max_results: int) -> list[dict[str, Any]]:
    if DDGS is None:
        raise RuntimeError("ddgs (or duckduckgo_search) package is not installed")
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def _canonical_url(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parts = urlsplit(url.strip())
        if not parts.scheme or not parts.netloc:
            return url.strip()
        # Remove fragments and common trailing slash noise. Query parameters are
        # retained because some publishers use them for article identity.
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))
    except Exception:
        return url.strip()


def _fetch_page_text(url: str) -> tuple[str | None, str]:
    """Return (text, fetch_status); never raise network/parser failures."""
    try:
        response = requests.get(
            url,
            timeout=FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None, "fetch_failed"

    content_type = response.headers.get("Content-Type", "").lower()
    if "html" not in content_type and "text" not in content_type:
        return None, "unsupported_content_type"

    try:
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return None, "parse_failed"

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None, "empty_page"
    return text[:MAX_FETCHED_CHARS], "fetched"


def retrieve_evidence(query: str) -> list[dict[str, Any]]:
    """
    Retrieve candidate public evidence using a search-optimized query.

    Returns deduplicated results with search_rank and fetch_status. Search or
    network failure returns [] and is never interpreted as negative evidence.
    """
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        return []

    try:
        # Ask for a few extras because URL/title deduplication can remove rows.
        raw_results = _ddgs_text_search(query, max(MAX_RETRIEVAL_RESULTS * 2, 5))
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()

    for item in raw_results:
        title = str(item.get("title") or "").strip() or None
        raw_url = item.get("href") or item.get("url") or item.get("link")
        url = _canonical_url(str(raw_url)) if raw_url else None
        snippet = str(item.get("body") or item.get("description") or "").strip() or None

        title_key = re.sub(r"\W+", " ", (title or "").lower()).strip()
        if url and url in seen_urls:
            continue
        if title_key and title_key in seen_titles:
            continue

        if url:
            seen_urls.add(url)
        if title_key:
            seen_titles.add(title_key)

        text = None
        fetch_status = "snippet_only"
        if url:
            text, fetch_status = _fetch_page_text(url)
        if not text:
            text = snippet

        if not (title or url or text):
            continue

        results.append(
            {
                "search_rank": len(results) + 1,
                "title": title,
                "source_url": url,
                "text": text,
                "snippet": snippet,
                "published_at": item.get("date"),
                "fetch_status": fetch_status,
            }
        )
        if len(results) >= MAX_RETRIEVAL_RESULTS:
            break

    return results


# ---------------------------------------------------------------------------
# Batched source × hypothesis relevance scoring (one local-LLM call)
# ---------------------------------------------------------------------------

_BATCH_RELEVANCE_SYSTEM_PROMPT = r"""
You are the external-evidence assessment layer of Aletheia.

You receive:
- one clarification question,
- the exact web retrieval query,
- a short list of EXISTING business hypotheses,
- retrieved source snippets/text.

For EACH source and EACH supplied hypothesis, score only what the supplied text
itself supports or contradicts. Do not use outside knowledge and do not infer a
fact that is not present in the text.

support must be in [-0.5, 1.0]:
  -0.5 = the source directly contradicts the hypothesis
   0.0 = irrelevant, generic, ambiguous, or provides no useful evidence
   1.0 = strongly and specifically corroborates the hypothesis

confidence must be in [0, 1] and means DIRECTNESS of textual evidence:
   0.0 = unrelated/no usable evidence
   1.0 = directly and specifically addresses the hypothesis

Important rules:
- Retrieved evidence is not deterministic truth and is not proof of causality.
- A source that merely contains similar words but is about a different place,
  company, date, event or research topic is irrelevant: support=0,
  confidence=0.
- Do not reward generic academic discussion of a concept when the retrieval
  question concerns a concrete business incident.
- Do not score a hypothesis more highly simply because it was named in the
  question.
- Return every requested source_index × hypothesis combination exactly once.

Return ONLY valid JSON:
{
  "assessments": [
    {
      "source_index": 1,
      "hypothesis": "exact supplied hypothesis name",
      "support": 0.0,
      "confidence": 0.0,
      "reason": "one concise sentence"
    }
  ]
}
"""

# Legacy scorer prompt retained for compatibility with callers/tests that use
# score_relevance() directly.
_RELEVANCE_SYSTEM_PROMPT = """
You score how much retrieved text supports or contradicts a specific business
KPI hypothesis. Use only the supplied retrieved text. Irrelevant or generic
material must receive support=0 and confidence=0. Retrieved evidence is not
proof of causality.

Return ONLY JSON:
{"support": <float in [-0.5, 1.0]>, "confidence": <float in [0, 1]>, "reason": "<one sentence>"}
"""


def _ollama_client():
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
        raise ValueError("Relevance response is not a JSON object")
    return parsed


def _bounded_float(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _zero_batch(hypotheses: list[str], results: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    source_assessments = []
    for index, _ in enumerate(results, start=1):
        for hypothesis in hypotheses:
            source_assessments.append(
                {
                    "source_index": index,
                    "hypothesis": hypothesis,
                    "support": 0.0,
                    "confidence": 0.0,
                    "reason": reason,
                }
            )
    return {
        "source_assessments": source_assessments,
        "hypothesis_scores": {
            hypothesis: {
                "weighted_support": 0.0,
                "retrieval_confidence": 0.0,
                "effective_support": 0.0,
                "n_sources_scored": len(results),
            }
            for hypothesis in hypotheses
        },
        "status": "no_support",
    }


def _aggregate_assessments(
    hypotheses: list[str],
    results: list[dict[str, Any]],
    assessments: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Confidence-weighted aggregation.

    weighted_support = sum(support_i * confidence_i) / sum(confidence_i)

    retrieval_confidence requires roughly two high-confidence source-equivalents
    for full confidence when >=2 sources were retrieved. This prevents one
    isolated weak hit from receiving the same influence as corroboration while
    still allowing a single-result search to reach full confidence.

    effective_support = weighted_support * retrieval_confidence
    """
    output: dict[str, dict[str, Any]] = {}
    n_results = len(results)
    confidence_target = float(max(1, min(2, n_results)))

    for hypothesis in hypotheses:
        rows = [a for a in assessments if a.get("hypothesis") == hypothesis]
        confidence_sum = sum(float(a.get("confidence", 0.0)) for a in rows)
        weighted_numerator = sum(
            float(a.get("support", 0.0)) * float(a.get("confidence", 0.0))
            for a in rows
        )
        weighted_support = (
            weighted_numerator / confidence_sum if confidence_sum > 1e-12 else 0.0
        )
        retrieval_confidence = min(1.0, confidence_sum / confidence_target)
        effective_support = weighted_support * retrieval_confidence

        output[hypothesis] = {
            "weighted_support": max(SUPPORT_MIN, min(SUPPORT_MAX, weighted_support)),
            "retrieval_confidence": max(
                CONFIDENCE_MIN, min(CONFIDENCE_MAX, retrieval_confidence)
            ),
            "effective_support": max(
                SUPPORT_MIN, min(SUPPORT_MAX, effective_support)
            ),
            "n_sources_scored": len(rows),
        }

    return output


def score_relevance_batch(
    *,
    question: str,
    retrieval_query: str,
    hypotheses: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Score all source × hypothesis pairs in ONE Ollama call.

    The returned source_assessments are suitable for source-level audit rows;
    hypothesis_scores contains confidence-weighted effective_support for the
    bounded fusion layer.
    """
    hypotheses = [str(h).strip() for h in hypotheses if str(h).strip()]
    hypotheses = list(dict.fromkeys(hypotheses))
    if not hypotheses or not results:
        return _zero_batch(hypotheses, results, "no hypotheses or retrieval results")

    sources = []
    for index, result in enumerate(results, start=1):
        sources.append(
            {
                "source_index": index,
                "title": result.get("title"),
                "source_url": result.get("source_url"),
                "published_at": result.get("published_at"),
                "text": str(result.get("text") or "")[:MAX_SCORING_CHARS_PER_SOURCE],
            }
        )

    payload = {
        "clarification_question": question,
        "retrieval_query": retrieval_query,
        "hypotheses": hypotheses,
        "sources": sources,
    }

    try:
        response = _ollama_client().chat(
            model=RELEVANCE_MODEL,
            messages=[
                {"role": "system", "content": _BATCH_RELEVANCE_SYSTEM_PROMPT},
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
        parsed = _extract_json_object(content)
        raw_assessments = parsed.get("assessments", [])
        if not isinstance(raw_assessments, list):
            raise ValueError("assessments must be a list")
    except Exception:
        return _zero_batch(
            hypotheses,
            results,
            "local LLM unavailable or returned invalid batch output",
        )

    allowed_hypotheses = set(hypotheses)
    n_sources = len(results)
    parsed_map: dict[tuple[int, str], dict[str, Any]] = {}

    for item in raw_assessments:
        if not isinstance(item, dict):
            continue
        try:
            source_index = int(item.get("source_index"))
        except (TypeError, ValueError):
            continue
        hypothesis = str(item.get("hypothesis") or "").strip()
        if not (1 <= source_index <= n_sources) or hypothesis not in allowed_hypotheses:
            continue

        parsed_map[(source_index, hypothesis)] = {
            "source_index": source_index,
            "hypothesis": hypothesis,
            "support": _bounded_float(item.get("support"), SUPPORT_MIN, SUPPORT_MAX),
            "confidence": _bounded_float(
                item.get("confidence"), CONFIDENCE_MIN, CONFIDENCE_MAX
            ),
            "reason": str(item.get("reason") or "").strip(),
        }

    # Complete the Cartesian product deterministically. Missing model rows are
    # zero evidence, never silently dropped.
    assessments: list[dict[str, Any]] = []
    for source_index in range(1, n_sources + 1):
        for hypothesis in hypotheses:
            assessments.append(
                parsed_map.get(
                    (source_index, hypothesis),
                    {
                        "source_index": source_index,
                        "hypothesis": hypothesis,
                        "support": 0.0,
                        "confidence": 0.0,
                        "reason": "assessment missing from model output",
                    },
                )
            )

    hypothesis_scores = _aggregate_assessments(hypotheses, results, assessments)
    has_effective_support = any(
        abs(float(score.get("effective_support", 0.0))) > 1e-12
        for score in hypothesis_scores.values()
    )

    return {
        "source_assessments": assessments,
        "hypothesis_scores": hypothesis_scores,
        "status": "scored" if has_effective_support else "no_effective_support",
    }


def score_relevance(
    question: str,
    hypothesis: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Backwards-compatible single-hypothesis wrapper.

    New Stage-4 code should prefer score_relevance_batch() to reduce LLM calls
    and preserve source-level auditability.
    """
    batch = score_relevance_batch(
        question=question,
        retrieval_query=question,
        hypotheses=[hypothesis],
        results=results,
    )
    score = batch.get("hypothesis_scores", {}).get(hypothesis, {})
    assessments = [
        a for a in batch.get("source_assessments", []) if a.get("hypothesis") == hypothesis
    ]
    reasons = [str(a.get("reason") or "").strip() for a in assessments if a.get("reason")]
    return {
        "support": float(score.get("effective_support", 0.0)),
        "confidence": float(score.get("retrieval_confidence", 0.0)),
        "reason": reasons[0] if reasons else "no useful retrieved evidence",
    }


# ---------------------------------------------------------------------------
# Persistence — the ONLY write path this module has
# ---------------------------------------------------------------------------

def store_retrieved_context(
    kpi: str,
    region: str,
    window_start: dt.date,
    hypothesis: str,
    query: str,
    result: dict[str, Any],
    support: float,
    confidence: float,
) -> None:
    """Persist one source-level external evidence assessment."""
    sql = """
        INSERT INTO analysis.retrieved_context (
            kpi, region, window_start, hypothesis, retrieval_query,
            source_title, source_url, retrieved_text,
            retrieval_support, retrieval_confidence, retrieved_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    database.execute(
        sql,
        (
            kpi,
            region,
            window_start,
            hypothesis,
            query,
            result.get("title"),
            result.get("source_url"),
            result.get("text"),
            max(SUPPORT_MIN, min(SUPPORT_MAX, float(support))),
            max(CONFIDENCE_MIN, min(CONFIDENCE_MAX, float(confidence))),
            dt.datetime.utcnow(),
        ),
    )
