import unittest

from src.orchestrator import (
    RetrievalPlan,
    _query_is_grounded,
    _query_contains_resolvable_anchor,
    _query_matches_external_hypothesis,
    _web_retrieval_is_semantically_allowed,
    _validate_plan,
)


class Stage4GroundingTests(unittest.TestCase):

    def test_internal_hypotheses_cannot_use_web(self):
        context = {
            "kpi": "stock_availability",
            "region": "Chennai",
            "incident_date": "2024-12-01",
            "candidate_hypotheses": [
                {
                    "name": "units_sold",
                    "retrieval_scope": "internal",
                },
                {
                    "name": "marketing_spend",
                    "retrieval_scope": "internal",
                },
            ],
            "external_anchors": {},
        }

        self.assertFalse(
            _web_retrieval_is_semantically_allowed(context)
        )


    def test_weather_chennai_is_web_eligible(self):
        context = {
            "kpi": "revenue",
            "region": "Chennai",
            "incident_date": "2024-12-01",
            "candidate_hypotheses": [
                {
                    "name": "weather_index",
                    "retrieval_scope": "external",
                    "external_hypothesis": "weather_disruption",
                },
                {
                    "name": "marketing_spend",
                    "retrieval_scope": "internal",
                },
            ],
            "external_anchors": {},
        }

        self.assertTrue(
            _web_retrieval_is_semantically_allowed(context)
        )

        query = "Chennai weather disruption 2024-12-01"

        self.assertTrue(
            _query_is_grounded(query, context)
        )

        self.assertTrue(
            _query_contains_resolvable_anchor(
                query,
                context,
            )
        )

        self.assertTrue(
            _query_matches_external_hypothesis(
                query,
                context,
            )
        )


    def test_invented_year_is_rejected(self):
        context = {
            "kpi": "revenue",
            "region": "Chennai",
            "incident_date": "2024-12-01",
            "candidate_hypotheses": [
                {
                    "name": "weather_index",
                    "retrieval_scope": "external",
                    "external_hypothesis": "weather_disruption",
                }
            ],
            "external_anchors": {},
        }

        query = (
            "Chennai weather disruption "
            "2024-12-01 vs 2023-12-01"
        )

        self.assertFalse(
            _query_is_grounded(query, context)
        )


    def test_competitor_without_name_cannot_use_web(self):
        context = {
            "kpi": "revenue",
            "region": "Mumbai",
            "incident_date": "2024-04-15",
            "candidate_hypotheses": [
                {
                    "name": "competitor_price_index",
                    "retrieval_scope": "conditional_external",
                    "external_hypothesis": "competitor_action",
                }
            ],
            "external_anchors": {},
        }

        self.assertFalse(
            _web_retrieval_is_semantically_allowed(context)
        )


    def test_competitor_with_name_can_use_web(self):
        context = {
            "kpi": "revenue",
            "region": "Mumbai",
            "incident_date": "2024-04-15",
            "candidate_hypotheses": [
                {
                    "name": "competitor_price_index",
                    "retrieval_scope": "conditional_external",
                    "external_hypothesis": "competitor_action",
                }
            ],
            "external_anchors": {
                "competitor_name": "ExampleCompetitor"
            },
        }

        self.assertTrue(
            _web_retrieval_is_semantically_allowed(context)
        )

        query = (
            "ExampleCompetitor pricing Mumbai 2024-04-15"
        )

        self.assertTrue(
            _query_is_grounded(query, context)
        )

        self.assertTrue(
            _query_contains_resolvable_anchor(
                query,
                context,
            )
        )

        self.assertTrue(
            _query_matches_external_hypothesis(
                query,
                context,
            )
        )


    def test_valid_weather_web_plan_passes_full_validation(self):
        context = {
            "kpi": "revenue",
            "region": "Chennai",
            "incident_date": "2024-12-01",
            "candidate_hypotheses": [
                {
                    "name": "weather_index",
                    "retrieval_scope": "external",
                    "external_hypothesis": "weather_disruption",
                },
                {
                    "name": "marketing_spend",
                    "retrieval_scope": "internal",
                },
            ],
            "external_anchors": {},
        }

        plan = RetrievalPlan(
            clarification_question=(
                "Was there a weather disruption "
                "in Chennai around the incident date?"
            ),
            retrieval_query=(
                "Chennai weather disruption 2024-12-01"
            ),
            retrieval_target="web",
        )

        valid, reason = _validate_plan(
            plan,
            context,
        )

        self.assertTrue(valid, reason)


if __name__ == "__main__":
    unittest.main()