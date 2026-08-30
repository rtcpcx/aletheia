from __future__ import annotations

import unittest
from unittest.mock import patch

from src import orchestrator


class Stage4PolicyV3Tests(unittest.TestCase):
    def _packet(self, **overrides):
        packet = {
            "kpi": "revenue",
            "region": "Chennai",
            "window_start": "2024-12-01",
            "percent_change": -0.137,
            "confidence_level": "Low",
            "decision_type": "investigate",
            "top_probability_gap": 0.04,
            "any_driver_significant": True,
            "source_health": {"healthy": True},
            "action_context": {
                "action_level": "validate",
                "is_ambiguous": True,
                "primary_component": "units_sold",
            },
        }
        packet.update(overrides)
        return packet

    def _evidence(self):
        return [
            {
                "driver_name": "competitor_price_index",
                "explains_component": "average_selling_price",
                "softmax_probability": 0.92,
                "evidence_score": 0.8,
                "evidence_mode": "historical_relationship",
            },
            {
                "driver_name": "marketing_spend",
                "explains_component": "units_sold",
                "softmax_probability": 0.52,
                "evidence_score": 0.5,
                "evidence_mode": "historical_relationship",
            },
            {
                "driver_name": "stock_availability",
                "explains_component": "units_sold",
                "softmax_probability": 0.31,
                "evidence_score": 0.3,
                "evidence_mode": "historical_relationship",
            },
            {
                "driver_name": "weather_index",
                "explains_component": "units_sold",
                "softmax_probability": 0.17,
                "evidence_score": 0.2,
                "evidence_mode": "structural_break",
            },
        ]

    def test_monitor_never_triggers_stage4(self):
        packet = self._packet(
            decision_type="monitor",
            confidence_level="Low",
            action_context={"action_level": "monitor", "is_ambiguous": False},
        )
        self.assertFalse(orchestrator.needs_clarification(packet))

    def test_unhealthy_source_stops_before_retrieval(self):
        packet = self._packet(
            source_health={"healthy": False, "stale_sources": ["marketing"]},
            action_context={"action_level": "data_quality_first", "is_ambiguous": True},
        )
        self.assertFalse(orchestrator.needs_clarification(packet))

    def test_candidates_are_component_local_and_reserve_external_slot(self):
        selected = orchestrator.select_retrieval_candidates(
            self._evidence(), limit=2, decision_packet=self._packet()
        )
        names = [row["driver_name"] for row in selected]
        self.assertEqual(names[0], "marketing_spend")
        self.assertIn("weather_index", names)
        self.assertNotIn("competitor_price_index", names)

    def test_external_weather_falls_back_to_grounded_web_plan_without_llm(self):
        with patch.object(orchestrator, "_ollama_client", side_effect=RuntimeError("offline")):
            plan = orchestrator.generate_retrieval_plan(self._packet(), self._evidence())
        self.assertEqual(plan.retrieval_target, "web")
        self.assertIsNotNone(plan.retrieval_query)
        query = plan.retrieval_query or ""
        self.assertIn("Chennai", query)
        self.assertIn("2024", query)
        self.assertTrue(any(term in query.lower() for term in ("weather", "rain", "storm", "cyclone")))

    def test_internal_only_ambiguity_does_not_invent_web(self):
        evidence = [
            {
                "driver_name": "marketing_spend",
                "explains_component": "units_sold",
                "softmax_probability": 0.52,
                "evidence_score": 0.5,
            },
            {
                "driver_name": "stock_availability",
                "explains_component": "units_sold",
                "softmax_probability": 0.48,
                "evidence_score": 0.45,
            },
        ]
        with patch.object(orchestrator, "_ollama_client", side_effect=RuntimeError("offline")):
            plan = orchestrator.generate_retrieval_plan(self._packet(), evidence)
        self.assertNotEqual(plan.retrieval_target, "web")
        self.assertIsNone(plan.retrieval_query)

    def test_conditional_competitor_needs_named_anchor(self):
        packet = self._packet(
            region="Mumbai",
            action_context={
                "action_level": "validate",
                "is_ambiguous": True,
                "primary_component": "average_selling_price",
            },
        )
        evidence = [
            {
                "driver_name": "competitor_price_index",
                "explains_component": "average_selling_price",
                "softmax_probability": 0.55,
                "evidence_score": 0.5,
            },
            {
                "driver_name": "average_discount_pct",
                "explains_component": "average_selling_price",
                "softmax_probability": 0.45,
                "evidence_score": 0.4,
            },
        ]
        with patch.object(orchestrator, "_ollama_client", side_effect=RuntimeError("offline")):
            plan = orchestrator.generate_retrieval_plan(packet, evidence)
        self.assertNotEqual(plan.retrieval_target, "web")

        anchored = dict(packet)
        anchored["competitor_name"] = "Example Competitor"
        with patch.object(orchestrator, "_ollama_client", side_effect=RuntimeError("offline")):
            plan2 = orchestrator.generate_retrieval_plan(anchored, evidence)
        self.assertEqual(plan2.retrieval_target, "web")
        self.assertIn("Example Competitor", plan2.retrieval_query or "")


if __name__ == "__main__":
    unittest.main()
