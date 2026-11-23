"""Unit tests for dialogue_agent, logprobs confidence scoring, context engineering, and LLM economics."""

import json
import math
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from source.models import (
    CandidateMatch,
    DialogueContext,
    ExtractedConstraints,
    ReplyExtractionSchema,
)
from source.dialogue_agent import (
    DialogueAgent,
    _clean_model_output,
    _enforce_deterministic_constraints,
    _extract_status_alternatives,
    _extract_status_confidence,
    _extract_token_usage,
    _parse_llm_constraints,
    _parse_pydantic_from_raw,
    _resolve_classifier_model,
    _resolve_outreach_model,
    confirm_arrival_dialogue,
    continue_conversation,
    generate_outreach_message,
)


class TestDialogueAgent(unittest.TestCase):
    """Test suite for dialogue agent LLM parameters, logprobs, constraints extraction, and economics."""

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

    @patch("source.dialogue_agent._get_llm_client")
    def test_context_engineering_outreach_message(self, mock_get_client):
        """Verify generate_outreach_message applies role separation, one-shot example, and JSON input."""
        mock_response = MagicMock()
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="היי חן, התפנה שיבוץ בגן שקד. נשמח לעזרתך!"
                )
            )
        ]
        mock_response.usage = SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=35,
            prompt_tokens_details=SimpleNamespace(cached_tokens=50),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_outreach_message(
            candidate=self.candidate,
            garden_name="Gan Shaked",
            shift_date="2026-11-22",
            shift_hours="08:00-14:00",
            stub_mode=False,
        )

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]

        self.assertEqual(call_kwargs["temperature"], 0.7)
        self.assertEqual(call_kwargs["max_tokens"], 150)

        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 4)

        # 1. System message
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("וואטסאפ", messages[0]["content"])
        self.assertIn("אל תמציא", messages[0]["content"])

        # 2. One-shot user example
        self.assertEqual(messages[1]["role"], "user")
        example_input = json.loads(messages[1]["content"])
        self.assertEqual(example_input["candidate_name"], "מיכל לוי")

        # 3. One-shot assistant example
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("מיכל", messages[2]["content"])

        # 4. Final user input with structured JSON and /no_think
        self.assertEqual(messages[3]["role"], "user")
        self.assertTrue(messages[3]["content"].endswith("/no_think"))
        json_payload = messages[3]["content"].replace("/no_think", "").strip()
        parsed_input = json.loads(json_payload)
        self.assertEqual(parsed_input["candidate_name"], "חן כהן")
        self.assertEqual(parsed_input["garden_name"], "Gan Shaked")
        self.assertEqual(parsed_input["shift_date"], "2026-11-22")
        self.assertEqual(parsed_input["shift_hours"], "08:00-14:00")

        self.assertEqual(result, "היי חן, התפנה שיבוץ בגן שקד. נשמח לעזרתך!")

    @patch("source.dialogue_agent._get_llm_client")
    def test_context_engineering_confirm_arrival(self, mock_get_client):
        """Verify confirm_arrival_dialogue uses client.chat.completions.parse with ReplyExtractionSchema."""
        mock_response = MagicMock()
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=ReplyExtractionSchema(
                        status="CONDITIONAL",
                        arrival_time="08:30",
                        notes="תגיע ב-08:30",
                    ),
                    content='{"status": "CONDITIONAL", "arrival_time": "08:30", "notes": "תגיע ב-08:30"}',
                ),
                logprobs=None,
            )
        ]
        mock_response.usage = SimpleNamespace(
            prompt_tokens=85,
            completion_tokens=22,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.parse.return_value = mock_response
        mock_get_client.return_value = mock_client

        candidate_reply = "אולי אצליח להגיע ב-08:30 אם אמצא סידור לילד"
        result = confirm_arrival_dialogue(
            context=self.context,
            candidate_reply=candidate_reply,
            stub_mode=False,
        )

        mock_client.chat.completions.parse.assert_called_once()
        call_kwargs = mock_client.chat.completions.parse.call_args[1]

        self.assertEqual(call_kwargs["response_format"], ReplyExtractionSchema)
        self.assertEqual(call_kwargs["temperature"], 0.0)
        self.assertEqual(call_kwargs["seed"], 42)
        self.assertEqual(call_kwargs["max_tokens"], 80)
        self.assertTrue(call_kwargs["logprobs"])
        self.assertEqual(call_kwargs["top_logprobs"], 3)

        messages = call_kwargs["messages"]
        # System + 3 few-shot pairs (6 messages) + 1 runtime user message = 8 messages
        self.assertEqual(len(messages), 8)

        # 1. System message checks
        self.assertEqual(messages[0]["role"], "system")
        system_content = messages[0]["content"]
        expected_negative_constraint = (
            "Under no circumstances classify a reply as CONFIRMED if the candidate mentions arriving late, "
            "leaving early, or any condition/uncertainty (such as 'if', 'maybe', 'only at...'). "
            "Even if the candidate sounds enthusiastic, any time constraint or condition MUST be classified as CONDITIONAL."
        )
        self.assertIn(expected_negative_constraint, system_content)
        self.assertIn("עד 10 מילים", system_content)

        # 2. Few-shot pairs check (assistant replies must match ReplyExtractionSchema JSON)
        ex1_parsed = ReplyExtractionSchema.model_validate_json(messages[2]["content"])
        self.assertEqual(ex1_parsed.status, "CONFIRMED")
        self.assertIsNone(ex1_parsed.arrival_time)

        ex2_parsed = ReplyExtractionSchema.model_validate_json(messages[4]["content"])
        self.assertEqual(ex2_parsed.status, "CONDITIONAL")
        self.assertEqual(ex2_parsed.arrival_time, "08:30")

        ex3_parsed = ReplyExtractionSchema.model_validate_json(messages[6]["content"])
        self.assertEqual(ex3_parsed.status, "DECLINED")
        self.assertIsNone(ex3_parsed.arrival_time)

        # 3. Final runtime user prompt check
        self.assertEqual(messages[7]["role"], "user")
        self.assertTrue(messages[7]["content"].endswith("/no_think"))
        json_payload = messages[7]["content"].replace("/no_think", "").strip()
        runtime_data = json.loads(json_payload)
        self.assertEqual(runtime_data["candidate_name"], "חן כהן")
        self.assertEqual(runtime_data["garden_name"], "Gan Shaked")
        self.assertEqual(runtime_data["shift_date"], "2026-11-22")
        self.assertEqual(runtime_data["shift_hours"], "08:00-14:00")
        self.assertEqual(runtime_data["candidate_reply"], candidate_reply)

        self.assertEqual(result.willingness, "CONDITIONAL")
        self.assertEqual(result.arrival_time, "08:30")

    def test_extract_token_usage_full(self):
        """Test token usage extraction including cached tokens."""
        mock_resp = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=150,
                completion_tokens=40,
                prompt_tokens_details=SimpleNamespace(cached_tokens=50),
            )
        )
        usage = _extract_token_usage(mock_resp)
        self.assertEqual(usage["prompt_tokens"], 150)
        self.assertEqual(usage["completion_tokens"], 40)
        self.assertEqual(usage["cached_tokens"], 50)
        self.assertEqual(usage["total_tokens"], 190)

    def test_extract_token_usage_none_and_missing_details(self):
        """Test token usage extraction handles missing usage or details gracefully."""
        # 1. response is None
        self.assertEqual(_extract_token_usage(None)["prompt_tokens"], 0)

        # 2. usage is None
        resp_no_usage = SimpleNamespace(usage=None)
        self.assertEqual(_extract_token_usage(resp_no_usage)["total_tokens"], 0)

        # 3. prompt_tokens_details is None (typical for local Ollama)
        resp_ollama = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=25,
                prompt_tokens_details=None,
            )
        )
        usage = _extract_token_usage(resp_ollama)
        self.assertEqual(usage["prompt_tokens"], 100)
        self.assertEqual(usage["completion_tokens"], 25)
        self.assertEqual(usage["cached_tokens"], 0)
        self.assertEqual(usage["total_tokens"], 125)

    @patch("source.dialogue_agent._get_llm_client")
    def test_dialogue_agent_usage_tracking_and_cost(self, mock_get_client):
        """Verify DialogueAgent tracks cumulative tokens, calls, and calculates estimated cost."""
        # First call: outreach
        resp1 = MagicMock()
        resp1.choices = [SimpleNamespace(message=SimpleNamespace(content="הודעה 1"))]
        resp1.usage = SimpleNamespace(
            prompt_tokens=140,
            completion_tokens=30,
            prompt_tokens_details=SimpleNamespace(cached_tokens=60),
        )

        # Second call: classification via parse
        resp2 = MagicMock()
        resp2.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=ReplyExtractionSchema(
                        status="CONFIRMED",
                        arrival_time=None,
                        notes="מגיעה בזמן",
                    ),
                    content='{"status": "CONFIRMED", "arrival_time": null, "notes": "מגיעה בזמן"}',
                ),
                logprobs=None,
            )
        ]
        resp2.usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = resp1
        mock_client.chat.completions.parse.return_value = resp2
        mock_get_client.return_value = mock_client

        agent = DialogueAgent(stub_mode=False)
        self.assertEqual(agent.total_calls, 0)
        self.assertEqual(agent.estimate_cost_usd(), 0.0)

        # Call 1
        agent.create_opening_message(self.candidate, "גן שקד", "2026-11-22", "08:00-14:00")
        self.assertEqual(agent.total_calls, 1)
        self.assertEqual(agent.total_prompt_tokens, 140)
        self.assertEqual(agent.total_completion_tokens, 30)
        self.assertEqual(agent.total_cached_tokens, 60)
        self.assertEqual(agent.last_call_usage["prompt_tokens"], 140)

        # Call 2
        agent.process_reply(self.context, "אגיע בזמן")
        self.assertEqual(agent.total_calls, 2)
        self.assertEqual(agent.total_prompt_tokens, 240)
        self.assertEqual(agent.total_completion_tokens, 50)
        self.assertEqual(agent.total_cached_tokens, 100)
        self.assertEqual(agent.last_call_usage["prompt_tokens"], 100)

        summary = agent.get_usage_summary()
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["total_prompt_tokens"], 240)
        self.assertEqual(summary["total_completion_tokens"], 50)
        self.assertEqual(summary["total_cached_tokens"], 100)
        self.assertEqual(summary["total_tokens"], 290)

        # Cost calculation check:
        # non_cached_prompt = 240 - 100 = 140
        # input_cost = (140 / 1e6) * 0.15 = 0.000021
        # cached_cost = (100 / 1e6) * 0.075 = 0.0000075
        # output_cost = (50 / 1e6) * 0.60 = 0.000030
        expected_cost = round(
            (140 / 1e6) * 0.15 + (100 / 1e6) * 0.075 + (50 / 1e6) * 0.60, 6
        )
        self.assertEqual(agent.estimate_cost_usd(), expected_cost)

    def test_dialogue_agent_stub_mode_usage(self):
        """Verify that stub_mode tracks calls and zero tokens without cost."""
        agent = DialogueAgent(stub_mode=True)
        agent.create_opening_message(self.candidate, "גן שקד", "2026-11-22", "08:00-14:00")
        agent.process_reply(self.context, "אגיע")

        summary = agent.get_usage_summary()
        self.assertEqual(summary["total_calls"], 2)
        self.assertEqual(summary["total_prompt_tokens"], 0)
        self.assertEqual(summary["total_completion_tokens"], 0)
        self.assertEqual(summary["total_cached_tokens"], 0)
        self.assertEqual(agent.estimate_cost_usd(), 0.0)

    @patch("source.dialogue_agent._get_llm_client")
    def test_task_specific_model_tiering(self, mock_get_client):
        """Verify model tiering environment variables and fallbacks."""
        mock_response = MagicMock()
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    parsed=ReplyExtractionSchema(status="CONFIRMED", arrival_time=None, notes="אישור"),
                    content='{"status": "CONFIRMED", "arrival_time": null, "notes": "אישור"}',
                ),
                logprobs=None,
            )
        ]
        mock_response.usage = None
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_client.chat.completions.parse.return_value = mock_response
        mock_get_client.return_value = mock_client

        # 1. Fallback to base model when tiering env vars are not set
        with patch.dict(os.environ, {"OPENAI_MODEL": "base-model-test"}, clear=False):
            if "OPENAI_OUTREACH_MODEL" in os.environ:
                del os.environ["OPENAI_OUTREACH_MODEL"]
            if "OPENAI_CLASSIFIER_MODEL" in os.environ:
                del os.environ["OPENAI_CLASSIFIER_MODEL"]

            self.assertEqual(_resolve_outreach_model(), "base-model-test")
            self.assertEqual(_resolve_classifier_model(), "base-model-test")

            generate_outreach_message(self.candidate, "גן שקד", "2026-11-22", "08:00-14:00")
            call1_model = mock_client.chat.completions.create.call_args[1]["model"]
            self.assertEqual(call1_model, "base-model-test")

        # 2. Distinct model tiering for outreach vs classification
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "base-model-test",
                "OPENAI_OUTREACH_MODEL": "strong-outreach-model",
                "OPENAI_CLASSIFIER_MODEL": "fast-classifier-model",
            },
            clear=False,
        ):
            self.assertEqual(_resolve_outreach_model(), "strong-outreach-model")
            self.assertEqual(_resolve_classifier_model(), "fast-classifier-model")

            generate_outreach_message(self.candidate, "גן שקד", "2026-11-22", "08:00-14:00")
            call2_model = mock_client.chat.completions.create.call_args[1]["model"]
            self.assertEqual(call2_model, "strong-outreach-model")

            confirm_arrival_dialogue(self.context, "אגיע בזמן")
            call3_model = mock_client.chat.completions.parse.call_args[1]["model"]
            self.assertEqual(call3_model, "fast-classifier-model")

    def test_pydantic_schema_validation(self):
        """Verify ReplyExtractionSchema validates allowed categories and rejects invalid ones."""
        # Allowed statuses
        for valid_status in ("CONFIRMED", "DECLINED", "CONDITIONAL"):
            schema = ReplyExtractionSchema(
                status=valid_status,
                arrival_time="08:30" if valid_status == "CONDITIONAL" else None,
                notes="תשובה תקינה",
            )
            self.assertEqual(schema.status, valid_status)

        # Invalid status should raise ValidationError
        with self.assertRaises(ValidationError):
            ReplyExtractionSchema(
                status="MAYBE",  # Not in Literal
                arrival_time=None,
                notes="אולי אגיע",
            )

    def test_logprobs_confidence_in_json(self):
        """Verify _extract_status_confidence and _extract_status_alternatives handle JSON-wrapped tokens."""
        # Tokens as generated in JSON structured output
        token_items = [
            SimpleNamespace(token='{\n  "status": "', logprob=-0.01),
            SimpleNamespace(
                token='CONFIRMED',
                logprob=-0.08,
                top_logprobs=[
                    SimpleNamespace(token='"CONFIRMED"', logprob=-0.08),
                    SimpleNamespace(token='"CONDITIONAL"', logprob=-2.7),
                    SimpleNamespace(token='"DECLINED"', logprob=-4.5),
                ],
            ),
            SimpleNamespace(token='",\n  "arrival_time": null\n}', logprob=-0.001),
        ]
        choice = SimpleNamespace(logprobs=SimpleNamespace(content=token_items))

        conf = _extract_status_confidence(choice)
        self.assertIsNotNone(conf)
        self.assertAlmostEqual(conf, round(math.exp(-0.08), 4), places=4)

        alts = _extract_status_alternatives(choice)
        self.assertEqual(len(alts), 3)
        self.assertEqual(alts[0][0], "CONFIRMED")
        self.assertEqual(alts[1][0], "CONDITIONAL")
        self.assertEqual(alts[2][0], "DECLINED")

    @patch("source.dialogue_agent._get_llm_client")
    def test_confirm_arrival_fallback_to_create(self, mock_get_client):
        """Verify graceful fallback to client.chat.completions.create if .parse raises an error."""
        mock_response = MagicMock()
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"status": "CONDITIONAL", "arrival_time": "08:30", "notes": "הסייעת תגיע ב-08:30"}'
                ),
                logprobs=None,
            )
        ]
        mock_response.usage = SimpleNamespace(
            prompt_tokens=90,
            completion_tokens=25,
            prompt_tokens_details=None,
        )

        mock_client = MagicMock()
        # Simulate local Ollama or unsupported server raising on parse
        mock_client.chat.completions.parse.side_effect = Exception("Ollama endpoint does not support structured outputs")
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = confirm_arrival_dialogue(
            context=self.context,
            candidate_reply="אגיע ב-08:30",
            stub_mode=False,
        )

        # Both parse and create should have been invoked (attempted parse, fell back to create)
        mock_client.chat.completions.parse.assert_called_once()
        mock_client.chat.completions.create.assert_called_once()

        self.assertEqual(result.willingness, "CONDITIONAL")
        self.assertEqual(result.arrival_time, "08:30")
        self.assertIn("08:30", result.notes)

    def test_enforce_deterministic_constraints_override_confirmed_with_arrival_time(self):
        """Verify _enforce_deterministic_constraints overrides CONFIRMED + '8:30' into CONDITIONAL + '08:30'."""
        # Case 1: CONFIRMED with late arrival time "8:30" must be overridden to CONDITIONAL and "08:30"
        c1 = ExtractedConstraints(
            willingness="CONFIRMED",
            arrival_time="8:30",
            notes="הסייעת אישרה אך מגיעה ב-8:30",
        )
        res1 = _enforce_deterministic_constraints(c1)
        self.assertEqual(res1.willingness, "CONDITIONAL")
        self.assertEqual(res1.arrival_time, "08:30")

        # Case 2: CONFIRMED with normalized late arrival time "09:15"
        c2 = ExtractedConstraints(
            willingness="CONFIRMED",
            arrival_time="09:15",
            notes="מגיעה ב-09:15",
        )
        res2 = _enforce_deterministic_constraints(c2)
        self.assertEqual(res2.willingness, "CONDITIONAL")
        self.assertEqual(res2.arrival_time, "09:15")

        # Case 3: CONFIRMED without arrival_time remains CONFIRMED
        c3 = ExtractedConstraints(
            willingness="CONFIRMED",
            arrival_time=None,
            notes="אישרה הגעה מלאה",
        )
        res3 = _enforce_deterministic_constraints(c3)
        self.assertEqual(res3.willingness, "CONFIRMED")
        self.assertIsNone(res3.arrival_time)

    def test_enforce_deterministic_constraints_sanitize_invalid_arrival_time(self):
        """Verify invalid/out-of-range arrival_time strings are sanitized to None."""
        invalid_cases = ["NONE", "invalid", "25:00", "08:65", "noon", "8-30", ""]
        for bad_time in invalid_cases:
            c = ExtractedConstraints(
                willingness="CONFIRMED",
                arrival_time=bad_time,
                notes="תשובה עם שעה לא תקינה",
            )
            res = _enforce_deterministic_constraints(c)
            self.assertIsNone(
                res.arrival_time,
                f"Expected arrival_time to be None for input '{bad_time}'",
            )
            # Since arrival_time is None, CONFIRMED is preserved
            self.assertEqual(res.willingness, "CONFIRMED")

    @patch("source.dialogue_agent._get_llm_client")
    def test_continue_conversation_grounding_constraint(self, mock_get_client):
        """Verify continue_conversation injects anti-hallucination grounding constraint in system prompt."""
        mock_response = MagicMock()
        mock_response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="היי חן, לגבי חניה אבדוק מול הרכזת ואחזור אלייך."
                )
            )
        ]
        mock_response.usage = SimpleNamespace(
            prompt_tokens=110,
            completion_tokens=20,
            prompt_tokens_details=None,
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        reply = continue_conversation(
            context=self.context,
            candidate_message="יש חניה צמודה לגן?",
            stub_mode=False,
        )

        self.assertIn("חניה", reply)
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]

        system_message = messages[0]
        self.assertEqual(system_message["role"], "system")
        system_content = system_message["content"]

        expected_grounding = (
            "Answer ONLY using facts explicitly provided in the shift context. "
            "Never invent or guess missing details (such as parking, pay rates, or gate codes). "
            "If the substitute asks about something not in the context, politely state in Hebrew "
            "that you will check with the coordinator and get back to her."
        )
        self.assertIn(expected_grounding, system_content)

    def test_dialogue_agent_continue_conversation_stub_mode(self):
        """Verify DialogueAgent.continue_conversation in stub mode operates deterministically and records usage."""
        agent = DialogueAgent(stub_mode=True)
        self.assertEqual(agent.total_calls, 0)

        reply = agent.continue_conversation(
            context=self.context,
            candidate_message="מה הכתובת המדויקת של הגן?",
        )

        self.assertIn("חן כהן", reply)
        self.assertIn("Gan Shaked", reply)
        self.assertIn("אבדוק עם הרכזת", reply)
        self.assertEqual(agent.total_calls, 1)
        self.assertEqual(agent.get_usage_summary()["total_calls"], 1)


if __name__ == "__main__":
    unittest.main()
