"""Unit tests for dialogue_agent and logprobs confidence scoring."""

import math
import unittest
from types import SimpleNamespace
from typing import Optional, List, Tuple

from source.models import (
    CandidateMatch,
    DialogueContext,
    ExtractedConstraints,
)
from source.dialogue_agent import (
    DialogueAgent,
    _clean_model_output,
    _extract_status_alternatives,
    _extract_status_confidence,
    _parse_llm_constraints,
    confirm_arrival_dialogue,
    generate_outreach_message,
)


class TestDialogueAgent(unittest.TestCase):
    """Test suite for dialogue agent LLM parameters, logprobs, and constraints extraction."""

    def setUp(self):
        self.candidate = CandidateMatch(
            substitute_id=1,
            first_name="חן",
            last_name="כהן",
            city="ירושלים",
            neighborhood="רחביה",
            experience_years=5,
            certifications=["עזרה ראשונה"],
            distance_km=2.5,
            phone_number="050-1111111",
        )
        self.context = DialogueContext(
            candidate_name="חן כהן",
            garden_name="Gan Shaked",
            shift_date="2026-11-22",
            shift_hours="08:00-14:00",
        )

    def test_stub_mode_generate_outreach(self):
        """Test outreach message generation in stub mode."""
        msg = generate_outreach_message(
            candidate=self.candidate,
            garden_name="Gan Shaked",
            shift_date="2026-11-22",
            shift_hours="08:00-14:00",
            stub_mode=True,
        )
        self.assertIn("חן", msg)
        self.assertIn("Gan Shaked", msg)
        self.assertIn("2026-11-22", msg)

    def test_stub_mode_confirm_arrival(self):
        """Test structured arrival confirmation in stub mode."""
        res = confirm_arrival_dialogue(
            context=self.context,
            candidate_reply="אני אגיע בזמן",
            stub_mode=True,
        )
        self.assertEqual(res.willingness, "CONFIRMED")
        self.assertIsNone(res.arrival_time)
        self.assertIsNone(res.confidence)
        self.assertIsNone(res.top_alternatives)

    def test_clean_model_output(self):
        """Test stripping of <think>...</think> blocks."""
        raw = "<think>Analyzing candidate reply...\nShe is asking for late arrival.</think>STATUS: CONDITIONAL\nARRIVAL: 08:30"
        cleaned = _clean_model_output(raw)
        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("Analyzing", cleaned)
        self.assertTrue(cleaned.startswith("STATUS: CONDITIONAL"))

    def test_extract_status_confidence_none_cases(self):
        """Test graceful handling when logprobs or choice is None."""
        self.assertIsNone(_extract_status_confidence(None))

        choice_no_logprobs = SimpleNamespace(logprobs=None)
        self.assertIsNone(_extract_status_confidence(choice_no_logprobs))

        choice_empty_content = SimpleNamespace(
            logprobs=SimpleNamespace(content=[])
        )
        self.assertIsNone(_extract_status_confidence(choice_empty_content))

    def test_extract_status_confidence_high(self):
        """Test high confidence extraction (>= 0.85)."""
        # logprob -0.05 -> exp(-0.05) ~= 0.9512
        token_items = [
            SimpleNamespace(token="STATUS", logprob=-0.01),
            SimpleNamespace(token=":", logprob=-0.001),
            SimpleNamespace(token=" ", logprob=-0.001),
            SimpleNamespace(
                token="CONFIRMED",
                logprob=-0.0513,
                top_logprobs=[
                    SimpleNamespace(token="CONFIRMED", logprob=-0.0513),
                    SimpleNamespace(token="CONDITIONAL", logprob=-3.0),
                ],
            ),
            SimpleNamespace(token="\n", logprob=-0.001),
        ]
        choice = SimpleNamespace(
            logprobs=SimpleNamespace(content=token_items)
        )
        conf = _extract_status_confidence(choice)
        self.assertIsNotNone(conf)
        self.assertAlmostEqual(conf, 0.9500, places=3)
        self.assertGreaterEqual(conf, 0.85)

        alts = _extract_status_alternatives(choice)
        self.assertEqual(len(alts), 2)
        self.assertEqual(alts[0][0], "CONFIRMED")

    def test_extract_status_confidence_low_with_warning(self):
        """Test low confidence extraction (< 0.85) triggers coordinator warning note."""
        # logprob -0.3228 -> exp(-0.3228) ~= 0.7241
        token_items = [
            SimpleNamespace(token="STATUS", logprob=-0.01),
            SimpleNamespace(token=": ", logprob=-0.001),
            SimpleNamespace(
                token=" COND",
                logprob=-0.3228,
                top_logprobs=[
                    SimpleNamespace(token=" COND", logprob=-0.3228),
                    SimpleNamespace(token=" CONF", logprob=-1.45),
                    SimpleNamespace(token=" DECL", logprob=-3.20),
                ],
            ),
            SimpleNamespace(token="ITIONAL", logprob=-0.001),
        ]
        choice = SimpleNamespace(
            logprobs=SimpleNamespace(content=token_items)
        )
        conf = _extract_status_confidence(choice)
        self.assertIsNotNone(conf)
        self.assertAlmostEqual(conf, 0.7241, places=3)
        self.assertLess(conf, 0.85)

        alts = _extract_status_alternatives(choice)
        self.assertEqual(len(alts), 3)

        raw_output = "STATUS: CONDITIONAL\nARRIVAL: 08:30\nNOTES: אגיע ב-08:30"
        parsed = _parse_llm_constraints(
            raw_output=raw_output,
            confidence=conf,
            top_alternatives=alts,
        )
        self.assertEqual(parsed.willingness, "CONDITIONAL")
        self.assertEqual(parsed.arrival_time, "08:30")
        self.assertIn("[דורש בדיקת רכזת - ביטחון סיווג נמוך]", parsed.notes)
        self.assertEqual(parsed.confidence, conf)
        self.assertEqual(parsed.top_alternatives, alts)

    def test_high_confidence_no_warning(self):
        """Verify that high confidence (>= 0.85) does NOT append coordinator warning."""
        raw_output = "STATUS: CONFIRMED\nARRIVAL: NONE\nNOTES: מגיעה בזמן"
        parsed = _parse_llm_constraints(
            raw_output=raw_output,
            confidence=0.96,
        )
        self.assertEqual(parsed.willingness, "CONFIRMED")
        self.assertNotIn("[דורש בדיקת רכזת - ביטחון סיווג נמוך]", parsed.notes)

    def test_none_confidence_no_warning(self):
        """Verify that when confidence is None, warning is not appended."""
        raw_output = "STATUS: CONFIRMED\nARRIVAL: NONE\nNOTES: מגיעה בזמן"
        parsed = _parse_llm_constraints(
            raw_output=raw_output,
            confidence=None,
        )
        self.assertNotIn("[דורש בדיקת רכזת - ביטחון סיווג נמוך]", parsed.notes)

    def test_extracted_constraints_backward_compatibility(self):
        """Verify ExtractedConstraints default initialization without confidence."""
        ec = ExtractedConstraints()
        self.assertEqual(ec.willingness, "CONFIRMED")
        self.assertIsNone(ec.arrival_time)
        self.assertEqual(ec.notes, "Stub confirmation")
        self.assertIsNone(ec.confidence)
        self.assertIsNone(ec.top_alternatives)

    def test_dialogue_agent_class(self):
        """Test DialogueAgent interface with stub_mode."""
        agent = DialogueAgent(stub_mode=True)
        opening = agent.create_opening_message(
            self.candidate, "Gan Shaked", "2026-11-22", "08:00-14:00"
        )
        self.assertIn("Gan Shaked", opening)

        reply_res = agent.process_reply(self.context, "אגיע")
        self.assertEqual(reply_res.willingness, "CONFIRMED")


if __name__ == "__main__":
    unittest.main()
