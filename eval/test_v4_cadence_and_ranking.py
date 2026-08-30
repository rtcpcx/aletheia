from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from types import SimpleNamespace
from src import evidence_engine


class V4CadenceAndRankingTests(unittest.TestCase):
    def test_weekly_source_searches_only_weekly_lag_steps(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2024-01-01", periods=140, freq="D")
        week_values = rng.normal(size=20)
        driver = pd.Series(np.repeat(week_values, 7), index=dates)
        component = driver + pd.Series(rng.normal(0, 0.02, len(dates)), index=dates)

        lag, _ = evidence_engine.lag_search(
            driver.reset_index(drop=True),
            component.reset_index(drop=True),
            14,
            historical_dates=pd.Series(dates),
            source_cadence_days=7,
        )
        self.assertIn(lag, {0, 7, 14})
        self.assertEqual(lag % 7, 0)

    def test_directionally_contradictory_historical_driver_loses_to_structural_break(self) -> None:
        n = 90
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        historical = pd.DataFrame(
            {
                "metric_date": dates,
                "units_sold": np.linspace(100, 110, n),
                "marketing_spend": np.linspace(90, 110, n),
                "stock_availability": np.ones(n),
            }
        )
        incident_dates = pd.date_range("2024-04-01", periods=11, freq="D")
        incident = pd.DataFrame(
            {
                "metric_date": incident_dates,
                "units_sold": np.full(11, 60.0),
                "marketing_spend": np.full(11, 150.0),
                "stock_availability": np.full(11, 0.1),
            }
        )
        drivers = [
            SimpleNamespace(name="marketing_spend", table="mart.x", column="marketing_spend", explains="units_sold", max_lag=7, source_cadence_days=1, causal_role="direct", mediates_through=None, expected_effect_sign="positive"),
            SimpleNamespace(name="stock_availability", table="mart.x", column="stock_availability", explains="units_sold", max_lag=3, source_cadence_days=1, causal_role="direct", mediates_through=None, expected_effect_sign="positive"),
        ]

        fake_fit = evidence_engine.RidgeFitResult(
            status="fitted",
            coefficients={"marketing_spend": 1.0, "stock_availability": 0.0},
            n_observations=80,
            holdout_r2=0.5,
            coefficient_stability={"marketing_spend": 1.0, "stock_availability": 0.0},
        )

        def fake_lag(driver, component, max_lag, **kwargs):
            # Contract-positive marketing relationship; stock is historically flat.
            if float(pd.Series(driver).std(ddof=0)) < 1e-9:
                return 0, 0.0
            return 0, 0.8

        with patch.object(evidence_engine, "lag_search", side_effect=fake_lag), \
             patch.object(evidence_engine, "fit_ridge_group", return_value=fake_fit), \
             patch.object(
                 evidence_engine,
                 "inferential_validity",
                 return_value={"is_significant": True, "p_value": 0.01},
             ):
            result = evidence_engine.compute_evidence_scores(
                "revenue", "X", "units_sold", drivers, historical, incident
            )

        ranked = sorted(result, key=lambda row: row.softmax_probability, reverse=True)
        self.assertEqual(ranked[0].driver_name, "stock_availability")
        marketing = next(row for row in result if row.driver_name == "marketing_spend")
        self.assertEqual(marketing.direction_consistency, "contradictory")
        self.assertLess(marketing.evidence_score, ranked[0].evidence_score)

    def test_component_without_declared_drivers_is_safe(self) -> None:
        df = pd.DataFrame({"metric_date": pd.date_range("2024-01-01", periods=40), "x": range(40)})
        self.assertEqual(
            evidence_engine.compute_evidence_scores("kpi", "X", "x", [], df, df),
            [],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
