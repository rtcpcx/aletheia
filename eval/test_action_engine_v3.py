from __future__ import annotations

import unittest

from src import action_engine


class ActionEngineV3Tests(unittest.TestCase):
    def test_dominant_component_without_usable_evidence_does_not_fall_through(self) -> None:
        rows = [
            {
                "driver_name": "stock_availability",
                "explains_component": "units_sold",
                "evidence_mode": "insufficient_evidence",
                "model_status": "insufficient_history",
                "softmax_probability": 0.0,
                "baseline_value": 0.8,
                "incident_value": 0.5,
            },
            {
                "driver_name": "average_discount_pct",
                "explains_component": "average_selling_price",
                "evidence_mode": "historical_relationship",
                "model_status": "fitted",
                "softmax_probability": 0.95,
                "baseline_value": 0.03,
                "incident_value": 0.06,
                "historical_coefficient": -1.0,
                "is_significant": True,
            },
        ]
        result = action_engine.build_action_context(
            kpi="revenue",
            region="X",
            decision_type="investigate",
            confidence="High",
            ranked_drivers=rows,
            source_health={"healthy": True},
            component_impacts={"units_sold": -100.0, "average_selling_price": -10.0},
            component_baselines={"units_sold": 100.0, "average_selling_price": 10.0},
            component_incidents={"units_sold": 80.0, "average_selling_price": 9.0},
        )
        self.assertEqual(result["action_level"], "validate")
        self.assertEqual(result["primary_component"], "units_sold")
        self.assertEqual(result["validation_reason"], "insufficient_history")

    def test_explanation_and_action_target_are_distinct_when_needed(self) -> None:
        rows = [
            {
                "driver_name": "competitor_price_index",
                "explains_component": "average_selling_price",
                "evidence_mode": "historical_relationship",
                "model_status": "fitted",
                "softmax_probability": 0.8,
                "baseline_value": 1.0,
                "incident_value": 1.0,
                "historical_coefficient": 1.0,
                "is_significant": True,
            },
            {
                "driver_name": "average_discount_pct",
                "explains_component": "average_selling_price",
                "evidence_mode": "historical_relationship",
                "model_status": "fitted",
                "softmax_probability": 0.2,
                "baseline_value": 0.03,
                "incident_value": 0.06,
                "historical_coefficient": -1.0,
                "is_significant": True,
            },
        ]
        result = action_engine.build_action_context(
            kpi="revenue",
            region="X",
            decision_type="investigate",
            confidence="High",
            ranked_drivers=rows,
            source_health={"healthy": True},
            component_impacts={"average_selling_price": -20.0},
            component_baselines={"average_selling_price": 10.0},
            component_incidents={"average_selling_price": 8.0},
        )
        self.assertEqual(result["leading_explanation_driver"], "competitor_price_index")
        self.assertEqual(result["action_target_driver"], "average_discount_pct")
        self.assertTrue(result["explanation_differs_from_action_target"])
        self.assertIn("strongest evidence-ranked explanation", result["why_it_matters"])


if __name__ == "__main__":
    unittest.main()
