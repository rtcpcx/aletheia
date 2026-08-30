"""
Aletheia — eval/test_stage4_integration.py

Integration/regression tests for Stage 4 without requiring live web access.

What this validates:
1. Clear deterministic evidence does NOT invoke Stage 4 retrieval.
2. Ambiguous evidence routed to "internal" does NOT invoke the web.
3. Ambiguous evidence routed to "web" executes:
      retrieval plan -> retrieval -> batched source scoring -> bounded fusion.
4. Only the top two hypotheses receive retrieval support.
5. Unassessed hypotheses remain exactly unchanged.
6. The assessed hypotheses preserve their total pre-retrieval mass.
7. Zero effective retrieval support is an exact no-op.

The default tests are deterministic and offline. They intentionally mock the
planner output and external evidence so CI/debugging does not depend on Ollama,
DuckDuckGo, network state, or current search rankings.

Run from the repository root:

    python -m unittest eval.test_stage4_integration -v

or:

    python eval/test_stage4_integration.py

This file does not write to MySQL and does not modify raw.*, mart.*, analysis.*,
or app.*.
"""

from __future__ import annotations

import math
import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from src import orchestrator
from src.evidence_fusion import reweight_and_renormalize


EPS = 1e-10


def _ambiguous_packet(**overrides: Any) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "kpi": "revenue",
        "region": "Mumbai",
        "window_start": "2026-08-12",
        "percent_change": -0.14,
        "absolute_change": -1400.0,
        "confidence": "Low",
        "confidence_level": "Low",
        "top_probability_gap": 0.04,
        "any_driver_significant": False,
        "decision_type": "investigate",
        # These are explicit external anchors supplied to Stage 4.
        "city": "Mumbai",
        "country": "India",
    }
    packet.update(overrides)
    return packet


def _unambiguous_packet(**overrides: Any) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "kpi": "revenue",
        "region": "North",
        "window_start": "2026-08-05",
        "percent_change": -0.12,
        "absolute_change": -1200.0,
        "confidence": "High",
        "confidence_level": "High",
        "top_probability_gap": 0.42,
        "any_driver_significant": True,
        "decision_type": "investigate",
    }
    packet.update(overrides)
    return packet


def _driver_evidence() -> list[dict[str, Any]]:
    """
    Three existing deterministic hypotheses.

    Stage 4 should assess only the two highest-ranked unique candidates:
      1. stock_availability
      2. competitor_price_index

    marketing_spend must remain untouched by retrieval.
    """
    return [
        {
            "driver_name": "stock_availability",
            "explains_component": "units_sold",
            "evidence_mode": "structural_break",
            "model_status": "historical_variance_unavailable",
            "normalized_score": 1.0,
            "softmax_probability": 0.52,
        },
        {
            "driver_name": "competitor_price_index",
            "explains_component": "average_selling_price",
            "evidence_mode": "structural_break",
            "model_status": "historical_variance_unavailable",
            "normalized_score": 0.96,
            "softmax_probability": 0.48,
        },
        {
            "driver_name": "marketing_spend",
            "explains_component": "units_sold",
            "evidence_mode": "historical_relationship",
            "model_status": "fitted",
            "normalized_score": 0.25,
            "softmax_probability": 0.20,
        },
    ]


def _web_plan() -> orchestrator.RetrievalPlan:
    return orchestrator.RetrievalPlan(
        clarification_question=(
            "Do public incident records around 2026-08-12 distinguish "
            "stock availability pressure from competitor pricing pressure?"
        ),
        retrieval_query=(
            "Mumbai India August 2026 supply disruption competitor pricing"
        ),
        retrieval_target="web",
        planner_status="generated",
        planner_reason=None,
    )


def _internal_plan() -> orchestrator.RetrievalPlan:
    return orchestrator.RetrievalPlan(
        clarification_question=(
            "Do internal inventory and pricing records distinguish "
            "stock availability pressure from competitor pricing pressure "
            "during the incident window?"
        ),
        retrieval_query=None,
        retrieval_target="internal",
        planner_status="generated",
        planner_reason="internal records are the appropriate evidence source",
    )


def _mock_results() -> list[dict[str, Any]]:
    return [
        {
            "title": "Regional logistics interruption",
            "source_url": "https://example.test/logistics",
            "text": (
                "A regional logistics interruption constrained product "
                "availability during the incident period."
            ),
            "published_at": "2026-08-12",
            "fetch_status": "mock",
        },
        {
            "title": "Competitor pricing bulletin",
            "source_url": "https://example.test/pricing",
            "text": (
                "A competitor announced a price reduction during the same "
                "period, creating additional pricing pressure."
            ),
            "published_at": "2026-08-12",
            "fetch_status": "mock",
        },
    ]


def _positive_batch_scorer(
    *,
    question: str,
    retrieval_query: str,
    hypotheses: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Deterministic mock of retrieval.score_relevance_batch().

    Inventory receives stronger external support than competitor pricing.
    """
    assert question
    assert retrieval_query
    assert hypotheses == ["stock_availability", "competitor_price_index"]
    assert len(results) == 2

    assessments = [
        {
            "source_index": 1,
            "hypothesis": "stock_availability",
            "support": 0.90,
            "confidence": 0.90,
            "reason": "The source directly describes constrained availability.",
        },
        {
            "source_index": 1,
            "hypothesis": "competitor_price_index",
            "support": 0.00,
            "confidence": 0.20,
            "reason": "The logistics source does not address competitor pricing.",
        },
        {
            "source_index": 2,
            "hypothesis": "stock_availability",
            "support": 0.00,
            "confidence": 0.20,
            "reason": "The pricing source does not establish inventory pressure.",
        },
        {
            "source_index": 2,
            "hypothesis": "competitor_price_index",
            "support": 0.55,
            "confidence": 0.80,
            "reason": "The source directly reports competitor price reduction.",
        },
    ]

    return {
        "source_assessments": assessments,
        "hypothesis_scores": {
            "stock_availability": {
                "weighted_support": 0.81,
                "retrieval_confidence": 0.90,
                "effective_support": 0.729,
                "n_sources_scored": 2,
            },
            "competitor_price_index": {
                "weighted_support": 0.44,
                "retrieval_confidence": 0.80,
                "effective_support": 0.352,
                "n_sources_scored": 2,
            },
        },
        "status": "scored",
    }


def _zero_batch_scorer(
    *,
    question: str,
    retrieval_query: str,
    hypotheses: list[str],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    assessments: list[dict[str, Any]] = []
    for source_index in range(1, len(results) + 1):
        for hypothesis in hypotheses:
            assessments.append(
                {
                    "source_index": source_index,
                    "hypothesis": hypothesis,
                    "support": 0.0,
                    "confidence": 0.0,
                    "reason": "Retrieved source is irrelevant to this hypothesis.",
                }
            )

    return {
        "source_assessments": assessments,
        "hypothesis_scores": {
            hypothesis: {
                "weighted_support": 0.0,
                "retrieval_confidence": 0.0,
                "effective_support": 0.0,
                "n_sources_scored": len(results),
            }
            for hypothesis in hypotheses
        },
        "status": "no_effective_support",
    }


class Stage4IntegrationTests(unittest.TestCase):
    def assertAlmostSame(self, a: float, b: float, msg: str = "") -> None:
        self.assertTrue(
            math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=EPS),
            msg or f"{a!r} != {b!r}",
        )

    def test_unambiguous_packet_skips_retrieval_completely(self) -> None:
        calls = {"retrieve": 0, "score": 0}

        def forbidden_retrieve(query: str) -> list[dict[str, Any]]:
            calls["retrieve"] += 1
            raise AssertionError("retrieval must not run for clear deterministic evidence")

        def forbidden_score(**kwargs: Any) -> dict[str, Any]:
            calls["score"] += 1
            raise AssertionError("relevance scoring must not run")

        result = orchestrator.run_stage4(
            _unambiguous_packet(),
            _driver_evidence(),
            retrieve_evidence=forbidden_retrieve,
            score_relevance_batch=forbidden_score,
            reweight_and_renormalize=reweight_and_renormalize,
        )

        self.assertFalse(result["clarification_needed"])
        self.assertEqual(result["status"], "deterministic_only")
        self.assertEqual(calls["retrieve"], 0)
        self.assertEqual(calls["score"], 0)
        self.assertEqual(
            result["probabilities_before"],
            result["probabilities_after"],
        )

    def test_internal_route_never_calls_web(self) -> None:
        calls = {"retrieve": 0, "score": 0}

        def forbidden_retrieve(query: str) -> list[dict[str, Any]]:
            calls["retrieve"] += 1
            raise AssertionError("internal clarification must not call web retrieval")

        def forbidden_score(**kwargs: Any) -> dict[str, Any]:
            calls["score"] += 1
            raise AssertionError("internal clarification must not score web sources")

        with patch.object(
            orchestrator,
            "generate_retrieval_plan",
            return_value=_internal_plan(),
        ):
            result = orchestrator.run_stage4(
                _ambiguous_packet(region="South", city=None, country=None),
                _driver_evidence(),
                retrieve_evidence=forbidden_retrieve,
                score_relevance_batch=forbidden_score,
                reweight_and_renormalize=reweight_and_renormalize,
            )

        self.assertTrue(result["clarification_needed"])
        self.assertEqual(result["status"], "clarification_internal")
        self.assertEqual(result["retrieval_plan"]["retrieval_target"], "internal")
        self.assertIsNone(result["retrieval_plan"]["retrieval_query"])
        self.assertEqual(calls["retrieve"], 0)
        self.assertEqual(calls["score"], 0)
        self.assertEqual(
            result["probabilities_before"],
            result["probabilities_after"],
        )

    def test_web_route_retrieves_scores_top2_and_fuses_bounded_support(self) -> None:
        seen_queries: list[str] = []

        def mock_retrieve(query: str) -> list[dict[str, Any]]:
            seen_queries.append(query)
            return _mock_results()

        with patch.object(
            orchestrator,
            "generate_retrieval_plan",
            return_value=_web_plan(),
        ):
            result = orchestrator.run_stage4(
                _ambiguous_packet(),
                _driver_evidence(),
                retrieve_evidence=mock_retrieve,
                score_relevance_batch=_positive_batch_scorer,
                reweight_and_renormalize=reweight_and_renormalize,
            )

        self.assertEqual(result["status"], "retrieval_fused")
        self.assertEqual(seen_queries, [_web_plan().retrieval_query])
        self.assertEqual(len(result["retrieved_evidence"]), 2)
        self.assertEqual(len(result["source_assessments"]), 4)

        # Only the top two deterministic hypotheses are externally assessed.
        self.assertEqual(
            set(result["retrieval_support"]),
            {"stock_availability", "competitor_price_index"},
        )
        self.assertNotIn("marketing_spend", result["retrieval_support"])

        before = result["probabilities_before"]
        after = result["probabilities_after"]

        # Stronger inventory support should move its weight upward.
        self.assertGreater(after["stock_availability"], before["stock_availability"])

        # Competitor remains plausible, but loses relative mass within top-two.
        self.assertLess(
            after["competitor_price_index"],
            before["competitor_price_index"],
        )

        # Unassessed hypothesis is an exact no-op.
        self.assertAlmostSame(
            after["marketing_spend"],
            before["marketing_spend"],
            "unassessed hypothesis changed during retrieval fusion",
        )

        # Retrieval redistributes existing mass; it does not manufacture mass.
        before_top2 = (
            before["stock_availability"]
            + before["competitor_price_index"]
        )
        after_top2 = (
            after["stock_availability"]
            + after["competitor_price_index"]
        )
        self.assertAlmostSame(
            before_top2,
            after_top2,
            "assessed hypotheses did not preserve their pre-retrieval mass",
        )

        # Bounded influence: no assessed weight can explode relative to its
        # deterministic starting point under lambda <= 0.25.
        self.assertLessEqual(
            after["stock_availability"],
            before_top2 + EPS,
        )
        self.assertGreaterEqual(after["stock_availability"], 0.0)
        self.assertGreaterEqual(after["competitor_price_index"], 0.0)

    def test_zero_effective_support_is_exact_noop(self) -> None:
        def mock_retrieve(query: str) -> list[dict[str, Any]]:
            return _mock_results()

        with patch.object(
            orchestrator,
            "generate_retrieval_plan",
            return_value=_web_plan(),
        ):
            result = orchestrator.run_stage4(
                _ambiguous_packet(),
                _driver_evidence(),
                retrieve_evidence=mock_retrieve,
                score_relevance_batch=_zero_batch_scorer,
                reweight_and_renormalize=reweight_and_renormalize,
            )

        self.assertEqual(result["status"], "retrieval_no_effective_support")
        self.assertEqual(
            result["retrieval_support"],
            {
                "stock_availability": 0.0,
                "competitor_price_index": 0.0,
            },
        )

        before = result["probabilities_before"]
        after = result["probabilities_after"]

        self.assertEqual(set(before), set(after))
        for name in before:
            self.assertAlmostSame(
                before[name],
                after[name],
                f"{name} moved even though retrieval support was zero",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
