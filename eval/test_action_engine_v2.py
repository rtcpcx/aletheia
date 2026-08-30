from __future__ import annotations

import unittest

from src import action_engine, narrator


def driver(
    name: str,
    component: str,
    weight: float,
    baseline: float,
    incident: float,
    *,
    coefficient: float = 1.0,
    mode: str = "historical_relationship",
    significant: bool = True,
) -> dict:
    return {
        "driver_name": name,
        "explains_component": component,
        "fused_probability": weight,
        "softmax_probability": weight,
        "baseline_value": baseline,
        "incident_value": incident,
        "historical_coefficient": coefficient,
        "evidence_mode": mode,
        "model_status": "fitted" if mode == "historical_relationship" else "historical_variance_unavailable",
        "is_significant": significant,
        "best_lag_days": 0,
    }


class ActionEngineV2Tests(unittest.TestCase):
    def healthy(self) -> dict:
        return {"healthy": True, "stale_sources": [], "missing_sources": []}

    def test_stable_top_historical_driver_is_not_prioritized(self) -> None:
        rows = [
            driver("marketing_spend", "units_sold", 0.85, 100.0, 101.0),  # +1%, stable
            driver("stock_availability", "units_sold", 0.15, 0.95, 0.55, mode="structural_break"),
        ]
        result = action_engine.build_action_context(
            kpi="revenue",
            region="Bengaluru",
            decision_type="investigate",
            confidence="High",
            ranked_drivers=rows,
            source_health=self.healthy(),
            component_impacts={"units_sold": -100.0, "average_selling_price": -5.0},
            component_baselines={"units_sold": 100.0},
            component_incidents={"units_sold": 70.0},
            kpi_relative_change=-0.20,
        )
        self.assertEqual(result["primary_driver"], "stock_availability")
        self.assertNotIn("Work on Marketing spend", result["recommended_action"])

    def test_all_stable_drivers_produce_no_direct_intervention(self) -> None:
        rows = [
            driver("marketing_spend", "units_sold", 0.8, 100.0, 100.5),
            driver("stock_availability", "units_sold", 0.2, 0.90, 0.91),
        ]
        result = action_engine.build_action_context(
            kpi="revenue",
            region="X",
            decision_type="investigate",
            confidence="High",
            ranked_drivers=rows,
            source_health=self.healthy(),
            component_impacts={"units_sold": -50.0},
            component_baselines={"units_sold": 100.0},
            component_incidents={"units_sold": 70.0},
            kpi_relative_change=-0.15,
        )
        self.assertEqual(result["action_level"], "validate_unexplained")
        self.assertIn("Do not prioritize", result["recommended_action"])
        self.assertNotIn("Work on", result["recommended_action"])

    def test_directionally_contradictory_historical_driver_is_skipped(self) -> None:
        rows = [
            # Positive relationship + driver increased, but component decreased: contradictory.
            driver("competitor_price_index", "average_selling_price", 0.8, 1.0, 1.3, coefficient=1.0),
            # Positive relationship + discount decreased + component decreased: aligned.
            driver("average_discount_pct", "average_selling_price", 0.2, 0.20, 0.10, coefficient=1.0),
        ]
        result = action_engine.build_action_context(
            kpi="revenue",
            region="X",
            decision_type="investigate",
            confidence="High",
            ranked_drivers=rows,
            source_health=self.healthy(),
            component_impacts={"average_selling_price": -20.0},
            component_baselines={"average_selling_price": 100.0},
            component_incidents={"average_selling_price": 90.0},
            kpi_relative_change=-0.10,
        )
        self.assertEqual(result["primary_driver"], "average_discount_pct")
        self.assertEqual(result["direction_alignment"], "aligned")

    def test_low_confidence_keeps_moved_secondary_driver_open(self) -> None:
        rows = [
            driver("support_ticket_volume", "churned_customers", 0.55, 100, 150, coefficient=1.0),
            driver("platform_uptime_pct", "churned_customers", 0.45, 99.9, 96.0, coefficient=-1.0),
        ]
        result = action_engine.build_action_context(
            kpi="churn_rate",
            region="Delhi",
            decision_type="investigate",
            confidence="Low",
            ranked_drivers=rows,
            source_health=self.healthy(),
            component_impacts={"churned_customers": 20.0},
            component_baselines={"churned_customers": 10.0},
            component_incidents={"churned_customers": 15.0},
            kpi_relative_change=0.20,
        )
        self.assertEqual(result["action_level"], "validate")
        self.assertEqual(result["secondary_driver"], "platform_uptime_pct")
        self.assertIn("in parallel", result["recommended_action"])

    def test_source_health_blocks_business_action(self) -> None:
        result = action_engine.build_action_context(
            kpi="conversion_rate",
            region="Hyderabad",
            decision_type="investigate",
            confidence="Low",
            ranked_drivers=[],
            source_health={"healthy": False, "stale_sources": ["marketing"], "missing_sources": []},
            component_impacts={},
            kpi_relative_change=-0.10,
        )
        self.assertEqual(result["action_level"], "data_quality_first")
        self.assertIn("Restore or validate marketing", result["recommended_action"])

    def test_component_impact_precedes_cross_component_weight(self) -> None:
        rows = [
            driver("average_discount_pct", "average_selling_price", 0.99, 0.10, 0.20, coefficient=-1.0),
            driver("stock_availability", "units_sold", 0.60, 0.95, 0.60, mode="structural_break"),
        ]
        result = action_engine.build_action_context(
            kpi="revenue",
            region="X",
            decision_type="investigate",
            confidence="High",
            ranked_drivers=rows,
            source_health=self.healthy(),
            component_impacts={"units_sold": -100.0, "average_selling_price": -10.0},
            component_baselines={"units_sold": 100.0, "average_selling_price": 100.0},
            component_incidents={"units_sold": 60.0, "average_selling_price": 95.0},
            kpi_relative_change=-0.25,
        )
        self.assertEqual(result["primary_component"], "units_sold")
        self.assertEqual(result["primary_driver"], "stock_availability")

    def test_narrator_fallback_uses_business_interpretation_not_model_jargon(self) -> None:
        action = {
            "action_level": "act",
            "primary_driver_label": "Stock availability",
            "primary_component_label": "Units sold",
            "finding": "Stock availability decreased during the incident while Units sold decreased during the incident.",
            "why_it_matters": "Units sold is the dominant part of the revenue movement and stock availability also moved during the incident.",
            "next_check": "Check SKU-level stock-outs and replenishment delays.",
            "action_if_confirmed": "Rebalance inventory before increasing demand-generation activity.",
            "owner": "Supply / Inventory",
            "secondary_driver_label": None,
            "secondary_check": None,
            "is_ambiguous": False,
        }
        bundle = {
            "kpi": "revenue",
            "region": "Bengaluru",
            "confidence": {"level": "High"},
            "decision": {
                "percent_change": -0.20,
                "confidence_level": "High",
                "action_context": action,
            },
            "decomposition": {"narrative_mode": "standard", "is_volatile": False},
        }
        result = narrator.template_narrate(bundle, "Executive")
        combined = (result["headline"] + " " + result["narrative"]).lower()
        for term in narrator.FORBIDDEN_NARRATIVE_TERMS:
            self.assertNotIn(term, combined)
        self.assertIn("Stock availability", result["narrative"])
        self.assertIn("Check now", result["narrative"])


if __name__ == "__main__":
    unittest.main()
