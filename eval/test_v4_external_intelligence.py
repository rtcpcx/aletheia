from __future__ import annotations

import unittest
from unittest.mock import patch

from src import orchestrator


class V4ExternalIntelligenceTests(unittest.TestCase):
    def _packet(self):
        return {
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
                "action_level": "validate",
                "is_ambiguous": False,
                "primary_component": "units_sold",
            },
        }

    def _evidence(self):
        return [
            {
                "driver_name": "stock_availability",
                "explains_component": "units_sold",
                "softmax_probability": 0.65,
                "driver_zscore": 0.0,
                "structural_break_score": 0.19,
                "evidence_score": 0.19,
                "direction_consistency": "aligned",
                "evidence_mode": "structural_break",
            },
            {
                "driver_name": "weather_index",
                "explains_component": "units_sold",
                "softmax_probability": 0.31,
                "driver_zscore": -5.6,
                "structural_break_score": 0.0,
                "evidence_score": 0.14,
                "direction_consistency": "aligned",
                "evidence_mode": "historical_relationship",
            },
            {
                "driver_name": "marketing_spend",
                "explains_component": "units_sold",
                "softmax_probability": 0.04,
                "driver_zscore": 4.6,
                "structural_break_score": 0.0,
                "evidence_score": 0.01,
                "direction_consistency": "contradictory",
                "evidence_mode": "historical_relationship",
            },
        ]

    def test_material_external_candidate_can_trigger_stage4_even_when_not_ambiguous(self) -> None:
        packet = self._packet()
        evidence = self._evidence()
        self.assertTrue(orchestrator.requires_external_verification(packet, evidence))
        self.assertTrue(orchestrator.needs_clarification(packet, evidence))

    def test_policy_fallback_is_grounded_and_not_scenario_hardcoded(self) -> None:
        with patch.object(orchestrator, "_ollama_client", side_effect=RuntimeError("offline")):
            plan = orchestrator.generate_retrieval_plan(self._packet(), self._evidence())
        self.assertEqual(plan.retrieval_target, "web")
        query = plan.retrieval_query or ""
        self.assertIn("Chennai", query)
        self.assertIn("2024-12-01", query)
        self.assertTrue(any(term in query.lower() for term in ("weather", "rain", "storm")))
        self.assertNotIn("fengal", query.lower())

    def test_internal_only_material_signal_does_not_trigger_external_verification(self) -> None:
        evidence = [
            {
                "driver_name": "marketing_spend",
                "explains_component": "units_sold",
                "softmax_probability": 0.9,
                "driver_zscore": 7.0,
                "structural_break_score": 0.0,
                "direction_consistency": "aligned",
            }
        ]
        self.assertFalse(orchestrator.requires_external_verification(self._packet(), evidence))


if __name__ == "__main__":
    unittest.main(verbosity=2)
