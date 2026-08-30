from __future__ import annotations

import unittest

from src import action_engine, evidence_engine
from types import SimpleNamespace


class V4MechanismSemanticsTests(unittest.TestCase):
    def test_contract_sign_overrides_noisy_historical_sign(self) -> None:
        status, weight = evidence_engine._direction_consistency(
            driver_baseline=34.0,
            driver_incident=50.0,
            component_baseline=20.0,
            component_incident=30.0,
            lag_correlation=-0.30,  # noisy mediator sign
            expected_effect_sign="positive",
        )
        self.assertEqual(status, "aligned")
        self.assertEqual(weight, 1.0)

    def test_upstream_driver_is_discounted_when_measured_mediator_exists(self) -> None:
        support = SimpleNamespace(name="support_ticket_volume", explains="churned_customers", causal_role="direct", mediates_through=None)
        uptime = SimpleNamespace(name="platform_uptime_pct", explains="churned_customers", causal_role="upstream", mediates_through="support_ticket_volume")
        self.assertEqual(evidence_engine._mechanism_weight(support, [support, uptime]), 1.0)
        self.assertLess(evidence_engine._mechanism_weight(uptime, [support, uptime]), 1.0)

    def test_action_targets_mediator_and_keeps_upstream_trigger(self) -> None:
        rows = [
            {
                "driver_name": "platform_uptime_pct",
                "explains_component": "churned_customers",
                "evidence_mode": "historical_relationship",
                "model_status": "fitted",
                "softmax_probability": 0.55,
                "baseline_value": 0.998,
                "incident_value": 0.988,
                "direction_consistency": "aligned",
                "causal_role": "upstream",
                "mediates_through": "support_ticket_volume",
                "is_significant": True,
            },
            {
                "driver_name": "support_ticket_volume",
                "explains_component": "churned_customers",
                "evidence_mode": "historical_relationship",
                "model_status": "fitted",
                "softmax_probability": 0.45,
                "baseline_value": 34.0,
                "incident_value": 50.0,
                "direction_consistency": "aligned",
                "causal_role": "direct",
                "is_significant": True,
            },
        ]
        result = action_engine.build_action_context(
            kpi="churn_rate",
            region="X",
            decision_type="investigate",
            confidence="High",
            ranked_drivers=rows,
            source_health={"healthy": True},
            component_impacts={"churned_customers": 100.0},
            component_baselines={"churned_customers": 20.0},
            component_incidents={"churned_customers": 30.0},
        )
        self.assertEqual(result["primary_driver"], "support_ticket_volume")
        self.assertEqual(result["upstream_driver"], "platform_uptime_pct")
        self.assertIn("Platform uptime", result["mechanism_chain"])
        self.assertIn("Support-ticket volume", result["mechanism_chain"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
