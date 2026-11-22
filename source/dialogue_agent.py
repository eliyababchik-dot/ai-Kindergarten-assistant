"""Conversational Dialogue Agent Module.

Manages arrival confirmation conversations with substitute candidates over WhatsApp,
generates personalized outreach messages, and extracts schedule constraints using
the standard Chat Completions API (configured for local Ollama or OpenAI).
Applies Context Engineering best practices: role separation, structured JSON inputs,
unambiguous constraints, anti-alignment bias guards, and token-efficient few-shot examples.
Includes LLM economics tracking: token usage monitoring, prompt cache accounting,
and model tiering.
"""

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# Ensure project root and source dir are in sys.path for direct script execution
_current_dir = Path(__file__).resolve().parent
_project_root = _current_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    from source.models import CandidateMatch, DialogueContext, ExtractedConstraints
except (ImportError, ModuleNotFoundError):
    try:
        from .models import CandidateMatch, DialogueContext, ExtractedConstraints
    except (ImportError, ModuleNotFoundError):
        from models import CandidateMatch, DialogueContext, ExtractedConstraints

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "qwen3:4b")
OUTREACH_MODEL = os.getenv("OPENAI_OUTREACH_MODEL", MODEL)
CLASSIFIER_MODEL = os.getenv("OPENAI_CLASSIFIER_MODEL", MODEL)
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")


def _resolve_outreach_model(override: Optional[str] = None) -> str:
    """Resolve model for outreach messages with tiering fallback."""
    if override:
        return override
    base_model = os.getenv("OPENAI_MODEL", "qwen3:4b")
    return os.getenv("OPENAI_OUTREACH_MODEL", base_model)


def _resolve_classifier_model(override: Optional[str] = None) -> str:
    """Resolve model for reply classification with tiering fallback."""
    if override:
        return override
    base_model = os.getenv("OPENAI_MODEL", "qwen3:4b")
    return os.getenv("OPENAI_CLASSIFIER_MODEL", base_model)


def _get_llm_client() -> OpenAI:
    """Initialize and return the OpenAI-compatible Chat Completions client."""
    return OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
    )


def _clean_model_output(text: str) -> str:
    """Strip optional reasoning blocks (<think>...</think>) returned by local models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def _extract_token_usage(response: Any) -> Dict[str, int]:
    """Safely extract token usage metrics from an LLM completion response.

    Extracts prompt_tokens, completion_tokens, and cached_tokens (if reported
    via prompt_tokens_details.cached_tokens). Handles missing usage or missing
    cached token details gracefully for local Ollama and non-OpenAI endpoints.

    Returns:
        Dict with keys 'prompt_tokens', 'completion_tokens', 'cached_tokens', 'total_tokens'.
    """
    if response is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }

    prompt_tokens = getattr(usage, "prompt_tokens", 0)
    if prompt_tokens is None and isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens", 0)
    prompt_tokens = int(prompt_tokens or 0)

    completion_tokens = getattr(usage, "completion_tokens", 0)
    if completion_tokens is None and isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens", 0)
    completion_tokens = int(completion_tokens or 0)

    cached_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")

    if details is not None:
        cached = getattr(details, "cached_tokens", 0)
        if cached is None and isinstance(details, dict):
            cached = details.get("cached_tokens", 0)
        cached_tokens = int(cached or 0)

    total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "total_tokens": total_tokens,
    }


def _find_status_token_item(choice: Any) -> Optional[Any]:
    """Find the token logprob entry corresponding to the classification status decision.

    Inspects choice.logprobs.content to locate the token representing the status
    decision (CONFIRMED, DECLINED, or CONDITIONAL).
    """
    if choice is None:
        return None

    logprobs = getattr(choice, "logprobs", None)
    if logprobs is None and isinstance(choice, dict):
        logprobs = choice.get("logprobs")
    if logprobs is None:
        return None

    content = getattr(logprobs, "content", None)
    if content is None and isinstance(logprobs, dict):
        content = logprobs.get("content")
    if not content:
        return None

    status_values = ("CONFIRMED", "DECLINED", "CONDITIONAL")
    status_prefixes = ("CONF", "DECL", "COND")

    # Pass 1: Scan for the first token after the 'STATUS:' header
    seen_status_keyword = False
    for item in content:
        raw_token = getattr(item, "token", None)
        if raw_token is None and isinstance(item, dict):
            raw_token = item.get("token", "")
        if not raw_token:
            continue

        token_upper = raw_token.strip().upper()

        if "STATUS" in token_upper:
            seen_status_keyword = True
            # Case where token contains both header and value (e.g. "STATUS: CONFIRMED")
            for status in status_values:
                if status in token_upper:
                    return item
            continue

        if seen_status_keyword:
            if token_upper in {":", ""}:
                continue
            if any(token_upper.startswith(p) for p in status_prefixes) or any(
                s.startswith(token_upper) for s in status_values if len(token_upper) >= 3
            ):
                return item
            if token_upper in status_values:
                return item
            if "ARRIVAL" in token_upper or "NOTES" in token_upper:
                break

    # Pass 2: Fallback scan across all tokens for direct status matches or prefixes
    for item in content:
        raw_token = getattr(item, "token", None)
        if raw_token is None and isinstance(item, dict):
            raw_token = item.get("token", "")
        if not raw_token:
            continue
        token_upper = raw_token.strip().upper()
        if token_upper in status_values or any(token_upper == p for p in status_prefixes):
            return item

    return None


def _extract_status_confidence(choice: Any) -> Optional[float]:
    """Inspect choice.logprobs.content and compute linear probability (0.0 to 1.0) using math.exp.

    Returns None if choice.logprobs is None or if the status token is not found.
    """
    token_item = _find_status_token_item(choice)
    if token_item is None:
        return None

    logprob = getattr(token_item, "logprob", None)
    if logprob is None and isinstance(token_item, dict):
        logprob = token_item.get("logprob")

    if logprob is None:
        return None

    try:
        linear_prob = math.exp(float(logprob))
        return round(max(0.0, min(1.0, linear_prob)), 4)
    except (ValueError, OverflowError):
        return None


def _extract_status_alternatives(choice: Any) -> List[Tuple[str, float]]:
    """Inspect top_logprobs on the status token to get alternative candidates and linear probabilities."""
    token_item = _find_status_token_item(choice)
    if token_item is None:
        return []

    top_logprobs = getattr(token_item, "top_logprobs", None)
    if top_logprobs is None and isinstance(token_item, dict):
        top_logprobs = token_item.get("top_logprobs")

    if not top_logprobs:
        return []

    alternatives = []
    for item in top_logprobs:
        t_str = getattr(item, "token", None)
        if t_str is None and isinstance(item, dict):
            t_str = item.get("token", "")
        t_lp = getattr(item, "logprob", None)
        if t_lp is None and isinstance(item, dict):
            t_lp = item.get("logprob")

        if t_str is not None and t_lp is not None:
            try:
                prob = round(max(0.0, min(1.0, math.exp(float(t_lp)))), 4)
                alternatives.append((t_str.strip(), prob))
            except (ValueError, OverflowError):
                continue

    return alternatives


def generate_outreach_message(
    candidate: CandidateMatch,
    garden_name: str,
    shift_date: str,
    shift_hours: str,
    stub_mode: bool = False,
    model: Optional[str] = None,
    usage_out: Optional[Dict[str, int]] = None,
) -> str:
    """Generate a personalized WhatsApp outreach message for the top candidate.

    Prompt Caching Design:
        Static persona instructions and one-shot examples are placed at the beginning
        (the prefix) of the messages list. Only the dynamic runtime JSON payload is placed
        at the end (the suffix). This maximizes cloud provider prompt caching efficiency
        and reduces time-to-first-token (TTFT).
    """
    if stub_mode:
        if usage_out is not None:
            usage_out.update({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
            })
        return (
            f"היי {candidate.first_name}, התפנה שיבוץ ב{garden_name} "
            f"בתאריך {shift_date} ({shift_hours}). האם תוכלי להגיע?"
        )

    target_model = _resolve_outreach_model(model)

    system_prompt = (
        "אתה עוזר גיוס ושיבוץ של רשת גני ילדים.\n"
        "תפקידך לכתוב הודעת וואטסאפ קצרה, חמה ומקצועית בעברית לסייעת מחליפה מובילה עבור שיבוץ פנוי.\n"
        "כללים והנחיות:\n"
        "- פנה לסייעת בשמה הפרטי.\n"
        "- ציין את שם הגן, תאריך המשמרת ושעות המשמרת במדויק מתוך נתוני הקלט.\n"
        "- בקש ממנה לאשר הגעה או לעדכן אם יש לה אילוץ שעות.\n"
        "- שמור על נוסח תמציתי ומזמין (עד 3-4 משפטים, מותאם למגבלת 150 טוקנים).\n"
        "- הנחיה שלילית: אל תמציא פרטי משמרת, שכר או מידע שלא סופק בקלט.\n"
        "- פלוט אך ורק את נוסח הודעת הוואטסאפ עצמה, ללא הקדמות, מרכאות או פניות מערכת."
    )

    one_shot_input = {
        "candidate_name": "מיכל לוי",
        "experience_years": 4,
        "certifications": ["עזרה ראשונה"],
        "distance_km": 1.8,
        "garden_name": "גן תמר",
        "shift_date": "2026-11-23",
        "shift_hours": "08:00-14:00",
    }
    one_shot_output = (
        "היי מיכל, מה שלומך? התפנה לנו שיבוץ בגן תמר בתאריך 2026-11-23 בין השעות 08:00-14:00. "
        "נשמח מאוד אם תוכלי להגיע להחליף! האם את פנויה, או שיש לך אילוץ שעות כלשהו? תודה ויום מקסים!"
    )

    runtime_input = {
        "candidate_name": f"{candidate.first_name} {candidate.last_name}",
        "experience_years": candidate.experience_years,
        "certifications": candidate.certifications if candidate.certifications else [],
        "distance_km": candidate.distance_km,
        "garden_name": garden_name,
        "shift_date": shift_date,
        "shift_hours": shift_hours,
    }

    final_user_content = json.dumps(runtime_input, ensure_ascii=False) + "\n/no_think"

    # Invariant prefix (system + one-shot) followed by dynamic suffix for optimal prompt caching
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(one_shot_input, ensure_ascii=False)},
        {"role": "assistant", "content": one_shot_output},
        {"role": "user", "content": final_user_content},
    ]

    client = _get_llm_client()
    response = client.chat.completions.create(
        model=target_model,
        messages=messages,
        temperature=0.7,
        max_tokens=150,
    )
    if usage_out is not None:
        usage_out.update(_extract_token_usage(response))
    return _clean_model_output(response.choices[0].message.content)


def confirm_arrival_dialogue(
    context: DialogueContext,
    candidate_reply: str,
    stub_mode: bool = False,
    model: Optional[str] = None,
    usage_out: Optional[Dict[str, int]] = None,
) -> ExtractedConstraints:
    """Send candidate reply to the LLM and extract structured schedule constraints.

    Prompt Caching Design:
        Static system instructions, classification rules, negative constraints, and
        the 3 few-shot examples are placed as the invariant prefix of the message history.
        Only the dynamic candidate reply JSON payload is sent as the final user message.
        This allows cloud prompt caches to hit 100% of the instruction/example tokens.
    """
    if stub_mode:
        if usage_out is not None:
            usage_out.update({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_tokens": 0,
                "total_tokens": 0,
            })
        return ExtractedConstraints(
            willingness="CONFIRMED",
            arrival_time=None,
            notes="Stub confirmation",
            confidence=None,
            top_alternatives=None,
        )

    target_model = _resolve_classifier_model(model)

    system_prompt = (
        "אתה עוזר לרכזת גני ילדים לניתוח תשובות וואטסאפ של סייעות מחליפות וחילוץ אילוצי שיבוץ.\n"
        "עליך לנתח את תוכן ההודעה ששלחה הסייעת ולהחזיר פלט מובנה בפורמט המדויק הבא (3 שורות בלבד, ללא שום תוספת):\n"
        "STATUS: CONFIRMED או DECLINED או CONDITIONAL\n"
        "ARRIVAL: שעת הגעה בפורמט HH:MM אם ציינה איחור/הגעה חלקית, אחרת NONE\n"
        "NOTES: משפט קצר ותמציתי בעברית של עד 10 מילים המסכם את תשובת הסייעת\n\n"
        "כללי סיווג:\n"
        "- CONFIRMED: אישור הגעה מלא בזמן מתחילת המשמרת ללא שום תנאי או הסתייגות.\n"
        "- DECLINED: סירוב מפורש להגיע למשמרת (למשל מחלה, חוסר פניות, סירוב).\n"
        "- CONDITIONAL: כל מקרה של הגעה מאוחרת, הגעה חלקית, או תנאי לקיום המשמרת.\n\n"
        "הנחיה שלילית קריטית (מניעת הטיית ריצוי / Alignment Bias):\n"
        "Under no circumstances classify a reply as CONFIRMED if the candidate mentions arriving late, "
        "leaving early, or any condition/uncertainty (such as 'if', 'maybe', 'only at...'). "
        "Even if the candidate sounds enthusiastic, any time constraint or condition MUST be classified as CONDITIONAL."
    )

    # Few-shot 1: Clear confirmation
    ex1_input = {
        "candidate_name": "שרה לוי",
        "garden_name": "גן שקד",
        "shift_date": "2026-11-22",
        "shift_hours": "08:00-14:00",
        "candidate_reply": "בטח, מגיעה ב-08:00 בדיוק",
    }
    ex1_output = (
        "STATUS: CONFIRMED\n"
        "ARRIVAL: NONE\n"
        "NOTES: הסייעת אישרה הגעה בזמן"
    )

    # Few-shot 2: Enthusiastic reply with late arrival
    ex2_input = {
        "candidate_name": "רחל כהן",
        "garden_name": "גן שקד",
        "shift_date": "2026-11-22",
        "shift_hours": "08:00-14:00",
        "candidate_reply": "אשמח להגיע, רק אוכל להגיע ב-08:30",
    }
    ex2_output = (
        "STATUS: CONDITIONAL\n"
        "ARRIVAL: 08:30\n"
        "NOTES: הסייעת תשמח להגיע אך באיחור ל-08:30"
    )

    # Few-shot 3: Clear decline
    ex3_input = {
        "candidate_name": "נועה ישראלי",
        "garden_name": "גן שקד",
        "shift_date": "2026-11-22",
        "shift_hours": "08:00-14:00",
        "candidate_reply": "סליחה, אני חולה היום ולא אוכל להגיע",
    }
    ex3_output = (
        "STATUS: DECLINED\n"
        "ARRIVAL: NONE\n"
        "NOTES: הסייעת חולה ולא תוכל להגיע למשמרת"
    )

    runtime_input = {
        "candidate_name": context.candidate_name,
        "garden_name": context.garden_name,
        "shift_date": context.shift_date,
        "shift_hours": context.shift_hours,
        "candidate_reply": candidate_reply,
    }

    final_user_content = json.dumps(runtime_input, ensure_ascii=False) + "\n/no_think"

    # Invariant prefix (system + 3 few-shot turns) followed by dynamic user suffix for optimal prompt caching
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(ex1_input, ensure_ascii=False)},
        {"role": "assistant", "content": ex1_output},
        {"role": "user", "content": json.dumps(ex2_input, ensure_ascii=False)},
        {"role": "assistant", "content": ex2_output},
        {"role": "user", "content": json.dumps(ex3_input, ensure_ascii=False)},
        {"role": "assistant", "content": ex3_output},
        {"role": "user", "content": final_user_content},
    ]

    client = _get_llm_client()
    response = client.chat.completions.create(
        model=target_model,
        messages=messages,
        temperature=0.0,
        seed=42,
        max_tokens=80,
        logprobs=True,
        top_logprobs=3,
    )

    if usage_out is not None:
        usage_out.update(_extract_token_usage(response))

    choice = response.choices[0]
    raw_output = _clean_model_output(choice.message.content)
    confidence = _extract_status_confidence(choice)
    top_alternatives = _extract_status_alternatives(choice)

    return _parse_llm_constraints(
        raw_output=raw_output,
        confidence=confidence,
        top_alternatives=top_alternatives,
    )


def _parse_llm_constraints(
    raw_output: str,
    confidence: Optional[float] = None,
    top_alternatives: Optional[List[Tuple[str, float]]] = None,
) -> ExtractedConstraints:
    """Parse the 3-line LLM output into an ExtractedConstraints object."""
    willingness = "CONDITIONAL"
    arrival_time: Optional[str] = None
    notes = raw_output

    for line in raw_output.splitlines():
        clean_line = line.strip()
        if clean_line.startswith("STATUS:"):
            val = clean_line.replace("STATUS:", "").strip().upper()
            if val in {"CONFIRMED", "DECLINED", "CONDITIONAL"}:
                willingness = val
        elif clean_line.startswith("ARRIVAL:"):
            val = clean_line.replace("ARRIVAL:", "").strip()
            if val.upper() != "NONE" and val:
                arrival_time = val
        elif clean_line.startswith("NOTES:"):
            notes = clean_line.replace("NOTES:", "").strip()

    if confidence is not None and confidence < 0.85:
        warning = "[דורש בדיקת רכזת - ביטחון סיווג נמוך]"
        if warning not in notes:
            notes = f"{notes} {warning}".strip()

    return ExtractedConstraints(
        willingness=willingness,
        arrival_time=arrival_time,
        notes=notes,
        confidence=confidence,
        top_alternatives=top_alternatives,
    )


class DialogueAgent:
    """Agent interface for managing conversations with substitutes and monitoring LLM token economics."""

    def __init__(
        self,
        stub_mode: bool = False,
        outreach_model: Optional[str] = None,
        classifier_model: Optional[str] = None,
    ) -> None:
        self.stub_mode = stub_mode
        self.outreach_model = outreach_model
        self.classifier_model = classifier_model
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_cached_tokens: int = 0
        self.total_calls: int = 0
        self.last_call_usage: Optional[Dict[str, int]] = None

    def _record_call_usage(self, usage: Dict[str, int]) -> None:
        """Internal helper to aggregate session token usage metrics."""
        self.total_calls += 1
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.total_cached_tokens += usage.get("cached_tokens", 0)
        self.last_call_usage = usage

    def get_usage_summary(self) -> Dict[str, int]:
        """Return cumulative token usage and call metrics for the current session."""
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }

    def estimate_cost_usd(
        self,
        input_price_per_1m: float = 0.15,
        output_price_per_1m: float = 0.60,
        cached_input_price_per_1m: float = 0.075,
    ) -> float:
        """Compute estimated API cost in USD based on tracked token usage.

        Accounts for non-cached prompt tokens, cached prompt tokens (discounted),
        and completion tokens according to per-million-tokens pricing.
        """
        non_cached_prompt_tokens = max(0, self.total_prompt_tokens - self.total_cached_tokens)
        cost = (
            (non_cached_prompt_tokens / 1_000_000.0) * input_price_per_1m
            + (self.total_cached_tokens / 1_000_000.0) * cached_input_price_per_1m
            + (self.total_completion_tokens / 1_000_000.0) * output_price_per_1m
        )
        return round(cost, 6)

    def create_opening_message(
        self,
        candidate: CandidateMatch,
        garden_name: str,
        shift_date: str,
        shift_hours: str,
    ) -> str:
        """Generate the initial outreach message to a candidate and track usage."""
        call_usage: Dict[str, int] = {}
        msg = generate_outreach_message(
            candidate=candidate,
            garden_name=garden_name,
            shift_date=shift_date,
            shift_hours=shift_hours,
            stub_mode=self.stub_mode,
            model=self.outreach_model,
            usage_out=call_usage,
        )
        self._record_call_usage(call_usage)
        return msg

    def process_reply(
        self, context: DialogueContext, candidate_reply: str
    ) -> ExtractedConstraints:
        """Process candidate reply, extract constraints, and track usage."""
        call_usage: Dict[str, int] = {}
        constraints = confirm_arrival_dialogue(
            context=context,
            candidate_reply=candidate_reply,
            stub_mode=self.stub_mode,
            model=self.classifier_model,
            usage_out=call_usage,
        )
        self._record_call_usage(call_usage)
        return constraints


if __name__ == "__main__":
    sample_candidate = CandidateMatch(
        substitute_id=1,
        first_name="חן",
        last_name="כהן",
        city="ירושלים",
        neighborhood="רחביה",
        experience_years=5,
        certifications=["עזרה ראשונה", "חינוך מיוחד"],
        distance_km=2.5,
        phone_number="050-1111111",
    )

    garden_name = "Gan Shaked"
    shift_date = "2026-11-22"
    shift_hours = "08:00-14:00"
    ambiguous_reply = "אולי אצליח להגיע ב-08:30 אם אמצא סידור לילד"

    agent = DialogueAgent(stub_mode=False)

    print("==================================================")
    print("1. יצירת הודעת פתיחה (temperature=0.7, max_tokens=150)")
    print("==================================================")
    try:
        opening = agent.create_opening_message(
            sample_candidate, garden_name, shift_date, shift_hours
        )
        print("נוסח ההודעה שנשלחה:")
        print(opening)
    except Exception as exc:
        print(f"[שים לב: שרת ה-LLM המקומי אינו זמין כרגע: {exc}]")
        print("מדגים נוסח במצב stub_mode=True:")
        agent = DialogueAgent(stub_mode=True)
        opening = agent.create_opening_message(
            sample_candidate, garden_name, shift_date, shift_hours
        )
        print(opening)

    print("\n==================================================")
    print("2. ניתוח תשובה עמומה (temperature=0.0, seed=42, max_tokens=80, logprobs=True)")
    print("==================================================")
    ctx = DialogueContext(
        candidate_name="חן כהן",
        garden_name=garden_name,
        shift_date=shift_date,
        shift_hours=shift_hours,
    )
    print(f"תשובת המועמדת: \"{ambiguous_reply}\"\n")

    try:
        result = agent.process_reply(ctx, ambiguous_reply)
        print("תוצאת חילוץ אילוצים (ExtractedConstraints):")
        print(f"  - סטטוס זמינות (willingness): {result.willingness}")
        print(f"  - שעת הגעה (arrival_time):    {result.arrival_time}")
        print(f"  - הערות (notes):              {result.notes}")
        print(f"  - ציון ביטחון (confidence):   {result.confidence}")
        if result.top_alternatives:
            print("  - חלופות טוקנים מובילות (top token alternatives):")
            for token_str, prob in result.top_alternatives:
                print(f"      * '{token_str}': {prob:.4f} ({prob * 100:.2f}%)")
        elif result.confidence is None:
            print("  - לא הוחזרו logprobs מהמודל המקומי.")
    except Exception as exc:
        print(f"[שים לב: שרת ה-LLM המקומי אינו זמין כרגע: {exc}]")
        print("מדגים ניתוח במצב stub_mode=True:")
        result = agent.process_reply(ctx, ambiguous_reply)
        print(result)

        print("\n[הדגמת מנגנון חישוב ציון ביטחון (logprobs confidence) ודגל בדיקת רכזת]:")
        simulated_raw = (
            "STATUS: CONDITIONAL\n"
            "ARRIVAL: 08:30\n"
            "NOTES: הסייעת תגיע ב-08:30 אם תמצא סידור לילד"
        )
        simulated_confidence = 0.7241
        simulated_alternatives = [
            ("CONDITIONAL", 0.7241),
            ("CONFIRMED", 0.2346),
            ("DECLINED", 0.0408),
        ]
        simulated_result = _parse_llm_constraints(
            raw_output=simulated_raw,
            confidence=simulated_confidence,
            top_alternatives=simulated_alternatives,
        )
        print(f"  - סטטוס: {simulated_result.willingness}")
        print(f"  - שעת הגעה: {simulated_result.arrival_time}")
        print(f"  - הערות (עם דגל רכזת): {simulated_result.notes}")
        print(f"  - ציון ביטחון: {simulated_result.confidence}")
        print("  - חלופות טוקנים מובילות:")
        for t_name, t_p in simulated_result.top_alternatives:
            print(f"      * '{t_name}': {t_p:.4f} ({t_p * 100:.2f}%)")

    print("\n==================================================")
    print("3. מעקב שימוש בטוקנים ועלויות (Token Usage & Economics)")
    print("==================================================")
    summary = agent.get_usage_summary()
    print("סיכום שימוש מצטבר (Usage Summary):")
    for key, val in summary.items():
        print(f"  - {key}: {val}")

    estimated_cost = agent.estimate_cost_usd()
    print(f"\nעלות משוערכת בדולרים (Estimated Cost): ${estimated_cost:.6f}")
    if agent.last_call_usage:
        print(f"שימוש בקריאה האחרונה (Last Call Usage): {agent.last_call_usage}")

    if agent.stub_mode:
        print("\n[הדגמת חישוב עלויות עבור נתוני שימוש לדוגמה עם Prompt Caching]:")
        sim_agent = DialogueAgent()
        sim_agent.total_calls = 2
        sim_agent.total_prompt_tokens = 350
        sim_agent.total_cached_tokens = 220
        sim_agent.total_completion_tokens = 65
        print(f"  - מדדים מדומים: {sim_agent.get_usage_summary()}")
        print(f"  - עלות משוערכת: ${sim_agent.estimate_cost_usd():.6f}")
