from __future__ import annotations

import unittest

from src import orchestrator


class V41RetrievalPrecisionTests(unittest.TestCase):
    def packet(self, **updates):
        packet = {
            "kpi": "revenue",
            "region": "Chennai",
            "window_start": "2024-12-01",
            "percent_change": -0.13,
            "confidence_level": "High",
            "decision_type": "investigate",
            "top_probability_gap": 0.30,
            "any_driver_significant": True,
            "source_health": {"healthy": True},
            "action_context": {
                "action_level": "act",
                "is_ambiguous": False,
                "primary_component": "units_sold",
            },
        }
        packet.update(updates)
        return packet

    @staticmethod
    def row(name, weight, score, *, mode="historical_relationship", direction="aligned", z=0.0, brk=0.0):
        return {
            "driver_name": name,
            "explains_component": "units_sold",
            "softmax_probability": weight,
            "evidence_score": score,
            "driver_zscore": z,
            "structural_break_score": brk,
            "direction_consistency": direction,
            "evidence_mode": mode,
            "model_status": "fitted" if mode != "insufficient_evidence" else "insufficient_history",
        }

    def test_insufficient_external_evidence_never_opens_web(self):
        rows = [
            self.row("stock_availability", 0.34, 0.0, mode="insufficient_evidence"),
            self.row("weather_index", 0.33, 0.0, mode="insufficient_evidence", z=8.0),
            self.row("marketing_spend", 0.33, 0.0, mode="insufficient_evidence"),
        ]
        self.assertFalse(orchestrator.requires_external_verification(self.packet(), rows))
        context = orchestrator.build_retrieval_context(self.packet(), rows)
        weather = next(c for c in context["candidate_hypotheses"] if c["name"] == "weather_index")
        self.assertEqual(weather["retrieval_scope"], "internal")
        self.assertFalse(weather["public_web_eligible"])

    def test_clear_external_leader_without_plausible_competitor_does_not_auto_search(self):
        rows = [
            self.row("weather_index", 0.89, 0.18, z=5.0),
            self.row("marketing_spend", 0.06, 0.01),
            self.row("stock_availability", 0.05, 0.01),
        ]
        self.assertFalse(orchestrator.requires_external_verification(self.packet(), rows))
        context = orchestrator.build_retrieval_context(self.packet(), rows)
        weather = next(c for c in context["candidate_hypotheses"] if c["name"] == "weather_index")
        self.assertEqual(weather["retrieval_scope"], "internal")

    def test_competitive_weather_hypothesis_is_web_eligible(self):
        rows = [
            self.row("stock_availability", 0.65, 0.19, mode="structural_break", brk=0.19),
            self.row("weather_index", 0.31, 0.14, z=5.6),
            self.row("marketing_spend", 0.04, 0.01, direction="contradictory"),
        ]
        self.assertTrue(orchestrator.requires_external_verification(self.packet(), rows))
        context = orchestrator.build_retrieval_context(self.packet(), rows)
        weather = next(c for c in context["candidate_hypotheses"] if c["name"] == "weather_index")
        self.assertEqual(weather["retrieval_scope"], "external")
        self.assertTrue(weather["public_web_eligible"])

    def test_conditional_competitor_query_cannot_piggyback_on_weather_eligibility(self):
        context = {
            "kpi": "revenue",
            "region": "Chennai",
            "incident_date": "2024-12-01",
            "candidate_hypotheses": [
                {
                    "name": "weather_index",
                    "explains_component": "units_sold",
                    "retrieval_scope": "external",
                    "external_hypothesis": "weather_disruption",
                    "public_web_eligible": True,
                },
                {
                    "name": "competitor_price_index",
                    "explains_component": "units_sold",
                    "retrieval_scope": "internal",
                    "configured_retrieval_scope": "conditional_external",
                    "external_hypothesis": None,
                    "public_web_eligible": False,
                },
            ],
            "external_anchors": {},
        }
        plan = orchestrator.RetrievalPlan(
            clarification_question="What public evidence could distinguish the competing explanations?",
            retrieval_query="competitor price change Chennai 2024-12-01",
            retrieval_target="web",
        )
        valid, reason = orchestrator._validate_plan(plan, context)
        self.assertFalse(valid)
        self.assertIn("externally searchable hypothesis", reason)

    def test_named_competitor_anchor_can_enable_conditional_external(self):
        packet = self.packet(competitor_name="ExampleCo")
        rows = [
            self.row("competitor_price_index", 0.45, 0.12, z=4.0),
            self.row("stock_availability", 0.40, 0.15, mode="structural_break", brk=0.2),
            self.row("marketing_spend", 0.15, 0.03),
        ]
        # Conditional-external competitor signals do not auto-open the web just
        # because an anchor exists; they become *eligible* when Stage 4 is needed.
        context = orchestrator.build_retrieval_context(packet, rows)
        competitor = next(c for c in context["candidate_hypotheses"] if c["name"] == "competitor_price_index")
        self.assertEqual(competitor["retrieval_scope"], "conditional_external")
        fallback = orchestrator._configured_external_fallback_plan(context, reason="test")
        self.assertIsNotNone(fallback)
        self.assertIn("ExampleCo", fallback.retrieval_query or "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
