from __future__ import annotations

import sys
import types
import unittest

import pandas as pd

# Lightweight stubs so this pure alignment helper can be tested in isolation.
contracts = types.ModuleType("src.contracts")
class RootDriver:
    def __init__(self, name: str = "x", max_lag: int = 0, explains: str = "component") -> None:
        self.name = name
        self.max_lag = max_lag
        self.explains = explains
contracts.RootDriver = RootDriver
sys.modules.setdefault("src.contracts", contracts)

# Lightweight stubs so this pure alignment helper can be tested without a DB driver.
guardrails = types.ModuleType("src.guardrails_engine")
guardrails.MIN_ABSOLUTE_OBSERVATIONS = 30
guardrails.MIN_OBSERVATIONS_PER_FEATURE = 10
guardrails.check_model_sufficiency = lambda n_observations, n_features: {"sufficient": True}
sys.modules.setdefault("src.guardrails_engine", guardrails)

stationarity = types.ModuleType("src.stationarity")
stationarity.is_usable_series = lambda s: True
stationarity.make_stationary = lambda s: (s, False)
sys.modules.setdefault("src.stationarity", stationarity)

from src.evidence_engine import _lag_aligned_incident_series


class LagAlignedIncidentTests(unittest.TestCase):
    def test_positive_lag_reads_preincident_driver_dates(self) -> None:
        incident = pd.DataFrame({
            "metric_date": pd.date_range("2024-08-19", periods=3, freq="D"),
            "support_ticket_volume": [10.0, 10.0, 10.0],
        })
        context = pd.DataFrame({
            "metric_date": pd.date_range("2024-08-10", periods=12, freq="D"),
            "support_ticket_volume": [10, 10, 50, 55, 60, 10, 10, 10, 10, 10, 10, 10],
        })
        aligned = _lag_aligned_incident_series(
            incident_data=incident,
            incident_context_data=context,
            driver_name="support_ticket_volume",
            lag_days=7,
        )
        self.assertEqual(aligned.tolist(), [50, 55, 60])

    def test_zero_lag_uses_same_dates(self) -> None:
        incident = pd.DataFrame({
            "metric_date": pd.date_range("2024-06-10", periods=3, freq="D"),
            "stock_availability": [0.4, 0.5, 0.6],
        })
        aligned = _lag_aligned_incident_series(
            incident_data=incident,
            incident_context_data=incident,
            driver_name="stock_availability",
            lag_days=0,
        )
        self.assertEqual(aligned.tolist(), [0.4, 0.5, 0.6])

    def test_missing_context_falls_back_safely(self) -> None:
        incident = pd.DataFrame({
            "metric_date": pd.date_range("2024-01-01", periods=2, freq="D"),
            "x": [1.0, 2.0],
        })
        aligned = _lag_aligned_incident_series(
            incident_data=incident,
            incident_context_data=None,
            driver_name="x",
            lag_days=3,
        )
        self.assertEqual(aligned.tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
