"""Conversational Dialogue Agent Module.

Manages arrival confirmation conversations with substitute candidates over WhatsApp,
generates personalized outreach messages, and extracts schedule constraints using
the standard Chat Completions API (configured for local Ollama or OpenAI).
"""

import math
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1/")
API_KEY = os.getenv("OPENAI_API_KEY", "ollama")


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
) -> str:
    """Generate a personalized WhatsApp outreach message for the top candidate."""
    if stub_mode:
        return (
            f"היי {candidate.first_name}, התפנה שיבוץ ב{garden_name} "
            f"בתאריך {shift_date} ({shift_hours}). האם תוכלי להגיע?"
        )

    certs_text = ", ".join(candidate.certifications) if candidate.certifications else "ללא תעודות מיוחדות"
    prompt = (
        f"כתוב הודעת וואטסאפ קצרה, חמה ומקצועית בעברית לסייעת מחליפה.\n"
        f"פרטי הסייעת: {candidate.first_name} {candidate.last_name}, "
        f"ניסיון: {candidate.experience_years} שנים, הכשרות: {certs_text}, "
        f"מרחק מהגן: {candidate.distance_km} ק\"מ.\n"
        f"פרטי השיבוץ: גן '{garden_name}', תאריך {shift_date}, שעות {shift_hours}.\n"
        f"בקש ממנה לאשר אם היא פנויה להגיע או לעדכן אם יש לה אילוץ שעות. "
        f"החזר רק את נוסח ההודעה עצמה ללא הקדמות.\n"
        f"/no_think"
    )

    client = _get_llm_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.7,
        max_tokens=150,
    )
    return _clean_model_output(response.choices[0].message.content)


def confirm_arrival_dialogue(
    context: DialogueContext,
    candidate_reply: str,
    stub_mode: bool = False,
) -> ExtractedConstraints:
    """Send candidate reply to the LLM and extract structured schedule constraints."""
    if stub_mode:
        return ExtractedConstraints(
            willingness="CONFIRMED",
            arrival_time=None,
            notes="Stub confirmation",
            confidence=None,
            top_alternatives=None,
        )

    prompt = (
        f"אתה עוזר לרכזת גני ילדים לנתח תשובת וואטסאפ מסייעת מחליפה.\n"
        f"שם הסייעת: {context.candidate_name}\n"
        f"גן: {context.garden_name} | תאריך: {context.shift_date} | שעות משמרת: {context.shift_hours}\n"
        f"תשובת הסייעת: \"{candidate_reply}\"\n\n"
        f"נתח את התשובה והחזר תשובה בפורמט המדויק הבא (3 שורות בלבד):\n"
        f"STATUS: CONFIRMED או DECLINED או CONDITIONAL\n"
        f"ARRIVAL: שעת הגעה בפורמט HH:MM אם ציינה איחור/הגעה חלקית, אחרת NONE\n"
        f"NOTES: סיכום קצר בעברית של תשובת הסייעת\n"
        f"/no_think"
    )

    client = _get_llm_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        seed=42,
        max_tokens=80,
        logprobs=True,
        top_logprobs=3,
    )

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
    """Agent interface for managing conversations with substitutes."""

    def __init__(self, stub_mode: bool = False) -> None:
        self.stub_mode = stub_mode

    def create_opening_message(
        self,
        candidate: CandidateMatch,
        garden_name: str,
        shift_date: str,
        shift_hours: str,
    ) -> str:
        """Generate the initial outreach message to a candidate."""
        return generate_outreach_message(
            candidate=candidate,
            garden_name=garden_name,
            shift_date=shift_date,
            shift_hours=shift_hours,
            stub_mode=self.stub_mode,
        )

    def process_reply(
        self, context: DialogueContext, candidate_reply: str
    ) -> ExtractedConstraints:
        """Process candidate reply and return extracted constraints."""
        return confirm_arrival_dialogue(
            context=context,
            candidate_reply=candidate_reply,
            stub_mode=self.stub_mode,
        )


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

    print("==================================================")
    print("1. יצירת הודעת פתיחה (temperature=0.7, max_tokens=150)")
    print("==================================================")
    try:
        agent = DialogueAgent(stub_mode=False)
        opening = agent.create_opening_message(
            sample_candidate, garden_name, shift_date, shift_hours
        )
        print("נוסח ההודעה שנשלחה:")
        print(opening)
    except Exception as exc:
        print(f"[שים לב: שרת ה-LLM המקומי אינו זמין כרגע: {exc}]")
        print("מדגים נוסח במצב stub_mode=True:")
        stub_agent = DialogueAgent(stub_mode=True)
        opening = stub_agent.create_opening_message(
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
        stub_agent = DialogueAgent(stub_mode=True)
        result = stub_agent.process_reply(ctx, ambiguous_reply)
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
