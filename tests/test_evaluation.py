import unittest
from unittest.mock import MagicMock

from source.evaluate_agent import (
    REQUIRED_CATEGORIES,
    evaluate_classification,
    evaluate_free_text_response,
    load_eval_dataset,
    validate_no_prompt_leakage,
)
from source.models import ExtractedConstraints


class TestEvaluationSuite(unittest.TestCase):
    def test_load_dataset_covers_all_categories(self):
        dataset = load_eval_dataset("evals/eval_dataset.json")
        cases = dataset["classification_cases"]
        self.assertGreaterEqual(len(cases), 12)

        present_categories = {c["category"] for c in cases}
        for required in REQUIRED_CATEGORIES:
            self.assertIn(required, present_categories)

    def test_no_prompt_leakage_passes_on_project_dataset(self):
        dataset = load_eval_dataset("evals/eval_dataset.json")
        validate_no_prompt_leakage(dataset, "source/dialogue_agent.py")

    def test_no_prompt_leakage_raises_on_overlap(self):
        with open("source/dialogue_agent.py", "r", encoding="utf-8-sig") as f:
            source_lines = [line.strip() for line in f if len(line.strip()) > 15]
        self.assertTrue(len(source_lines) > 0)

        leaked_dataset = {
            "classification_cases": [
                {
                    "id": "leak_test",
                    "category": "regression",
                    "candidate_reply": source_lines[0],
                    "expected_status": "CONFIRMED",
                    "expected_arrival_time": None,
                }
            ]
        }
        with self.assertRaises(ValueError):
            validate_no_prompt_leakage(leaked_dataset, "source/dialogue_agent.py")

    def test_evaluate_classification_metrics_and_failures(self):
        mock_agent = MagicMock()
        mock_agent.get_usage_summary.return_value = {"total_calls": 3, "total_tokens": 150}
        mock_agent.estimate_cost_usd.return_value = 0.00012

        mock_agent.process_reply.side_effect = [
            ExtractedConstraints(willingness="CONFIRMED", arrival_time=None, notes="מגיעה", confidence=0.95),
            ExtractedConstraints(willingness="CONFIRMED", arrival_time="08:30", notes="מאחרת", confidence=0.70),
            ExtractedConstraints(willingness="DECLINED", arrival_time=None, notes="לא יכולה", confidence=0.92),
        ]

        sample_cases = [
            {
                "id": "c1",
                "category": "common",
                "candidate_reply": "באה בזמן",
                "expected_status": "CONFIRMED",
                "expected_arrival_time": None,
            },
            {
                "id": "c2",
                "category": "misleading",
                "candidate_reply": "אבוא ב-08:30",
                "expected_status": "CONDITIONAL",
                "expected_arrival_time": "08:30",
            },
            {
                "id": "c3",
                "category": "common",
                "candidate_reply": "לא אוכל מחר",
                "expected_status": "DECLINED",
                "expected_arrival_time": None,
            },
        ]

        report = evaluate_classification(mock_agent, sample_cases)
        self.assertEqual(report["total_cases"], 3)
        self.assertEqual(report["status_correct"], 2)
        self.assertEqual(report["exact_match_count"], 2)
        self.assertAlmostEqual(report["status_accuracy"], 0.6667, places=4)
        self.assertEqual(report["low_confidence_count"], 1)
        self.assertEqual(len(report["failures"]), 1)
        self.assertEqual(report["failures"][0]["id"], "c2")

    def test_evaluate_free_text_response(self):
        good_msg = "היי חן! נשמח לעזרתך מחר בגן שקד בין השעות 08:00-14:00. תוכלי להגיע?"
        good_report = evaluate_free_text_response(
            good_msg,
            required_terms=["גן שקד", "08:00"],
            forbidden_terms=["חניה חינם", "בונוס"],
        )
        self.assertTrue(good_report["passed"])

        bad_msg = "היי! יש חניה חינם ליד הגן, נשמח לראותך."
        bad_report = evaluate_free_text_response(
            bad_msg,
            required_terms=["גן שקד", "08:00"],
            forbidden_terms=["חניה חינם"],
        )
        self.assertFalse(bad_report["passed"])
        self.assertIn("גן שקד", bad_report["missing_required_terms"])
        self.assertIn("חניה חינם", bad_report["found_forbidden_terms"])


if __name__ == "__main__":
    unittest.main()