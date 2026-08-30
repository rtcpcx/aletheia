from __future__ import annotations

import datetime as dt
import os
import unittest
from unittest.mock import patch

import pandas as pd

from src import feedback_engine


class FeedbackIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cutoff = dt.datetime(2026, 8, 30, 12, 0, 0)
        self.context = {
            "action_level": "act",
            "primary_driver": "stock_availability",
            "recommended_action": "Prioritize Stock Availability.",
            "why": "Deterministic evidence text.",
        }

    @patch("src.feedback_engine.database.query_df")
    def test_no_feedback_is_exact_noop_on_readiness(self, query_df):
        query_df.return_value = pd.DataFrame()
        result = feedback_engine.apply_feedback_to_action_context(
            kpi="revenue",
            region="Bengaluru",
            window_start=dt.date(2024, 6, 10),
            action_context=self.context,
            analysis_started_at=self.cutoff,
        )
        cal = result["feedback_calibration"]
        self.assertFalse(cal["applied"])
        self.assertEqual(cal["adjustment"], 0.0)
        self.assertEqual(cal["base_readiness_score"], cal["adjusted_readiness_score"])

    @patch("src.feedback_engine.database.query_df")
    def test_helpful_feedback_increases_readiness(self, query_df):
        query_df.return_value = pd.DataFrame(
            [{"disposition": "Helpful", "region": "Bengaluru"} for _ in range(8)]
        )
        result = feedback_engine.apply_feedback_to_action_context(
            kpi="revenue", region="Bengaluru", window_start=dt.date(2024, 6, 10),
            action_context=self.context, analysis_started_at=self.cutoff,
        )
        cal = result["feedback_calibration"]
        self.assertGreater(cal["adjustment"], 0.0)
        self.assertGreater(cal["adjusted_readiness_score"], cal["base_readiness_score"])

    @patch("src.feedback_engine.database.query_df")
    def test_negative_feedback_decreases_readiness(self, query_df):
        query_df.return_value = pd.DataFrame(
            [{"disposition": "Not helpful", "region": "Bengaluru"} for _ in range(8)]
        )
        result = feedback_engine.apply_feedback_to_action_context(
            kpi="revenue", region="Bengaluru", window_start=dt.date(2024, 6, 10),
            action_context=self.context, analysis_started_at=self.cutoff,
        )
        cal = result["feedback_calibration"]
        self.assertLess(cal["adjustment"], 0.0)
        self.assertLess(cal["adjusted_readiness_score"], cal["base_readiness_score"])

    @patch("src.feedback_engine.database.query_df")
    def test_adjustment_is_bounded(self, query_df):
        query_df.return_value = pd.DataFrame(
            [{"disposition": "Helpful", "region": "Bengaluru"} for _ in range(10000)]
        )
        result = feedback_engine.apply_feedback_to_action_context(
            kpi="revenue", region="Bengaluru", window_start=dt.date(2024, 6, 10),
            action_context=self.context, analysis_started_at=self.cutoff,
        )
        self.assertLessEqual(abs(result["feedback_calibration"]["adjustment"]), 0.10)

    @patch("src.feedback_engine.database.query_df")
    def test_feedback_never_changes_deterministic_action_identity(self, query_df):
        query_df.return_value = pd.DataFrame(
            [{"disposition": "Not helpful", "region": "Bengaluru"} for _ in range(20)]
        )
        result = feedback_engine.apply_feedback_to_action_context(
            kpi="revenue", region="Bengaluru", window_start=dt.date(2024, 6, 10),
            action_context=self.context, analysis_started_at=self.cutoff,
        )
        for key in ("action_level", "primary_driver", "recommended_action", "why"):
            self.assertEqual(result[key], self.context[key])

    @patch.dict(os.environ, {"ALETHEIA_FEEDBACK_INCLUDE_DEMO": "0"}, clear=False)
    @patch("src.feedback_engine.database.query_df")
    def test_demo_feedback_is_excluded_by_default(self, query_df):
        query_df.return_value = pd.DataFrame()
        feedback_engine.load_feedback_signal(
            kpi="revenue",
            region="Bengaluru",
            primary_driver="stock_availability",
            action_level="act",
            analysis_started_at=self.cutoff,
        )
        sql = query_df.call_args.args[0]
        self.assertIn("COALESCE(is_demo, 0) = 0", sql)

    @patch("src.feedback_engine.database.query_df", side_effect=RuntimeError("db down"))
    def test_feedback_store_failure_is_exact_noop(self, query_df):
        result = feedback_engine.apply_feedback_to_action_context(
            kpi="revenue", region="Bengaluru", window_start=dt.date(2024, 6, 10),
            action_context=self.context, analysis_started_at=self.cutoff,
        )
        cal = result["feedback_calibration"]
        self.assertFalse(cal["applied"])
        self.assertEqual(cal["adjustment"], 0.0)
        self.assertEqual(cal["base_readiness_score"], cal["adjusted_readiness_score"])
        self.assertEqual(result["primary_driver"], self.context["primary_driver"])

    @patch("src.feedback_engine.database.query_df")
    def test_analysis_cutoff_uses_absolute_epoch_for_mysql_timestamp(self, query_df):
        query_df.return_value = pd.DataFrame()
        feedback_engine.load_feedback_signal(
            kpi="revenue",
            region="Bengaluru",
            primary_driver="stock_availability",
            action_level="act",
            analysis_started_at=self.cutoff,
        )
        sql = query_df.call_args.args[0]
        params = query_df.call_args.kwargs["params"]
        expected_epoch = self.cutoff.replace(tzinfo=dt.timezone.utc).timestamp()
        self.assertIn("created_at < FROM_UNIXTIME(%s)", sql)
        self.assertEqual(params[-1], expected_epoch)

    def test_analysis_cutoff_epoch_handles_aware_datetime(self):
        aware = dt.datetime(
            2026, 8, 30, 17, 30, 0,
            tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)),
        )
        expected = dt.datetime(
            2026, 8, 30, 12, 0, 0, tzinfo=dt.timezone.utc
        ).timestamp()
        self.assertEqual(feedback_engine._analysis_cutoff_epoch(aware), expected)

    @patch("src.feedback_engine.database.execute")
    def test_record_feedback_is_parameterized_and_contextual(self, execute):
        feedback_engine.record_feedback(
            persona="Executive",
            region="Bengaluru",
            kpi="revenue",
            window_start="2024-06-10",
            primary_driver="stock_availability",
            action_level="act",
            disposition="Helpful",
            comment="O'Brien approved",
        )
        sql, params = execute.call_args.args
        self.assertIn("VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", sql)
        self.assertIn("O'Brien approved", params)


if __name__ == "__main__":
    unittest.main()
