"""Conversational Dialogue Agent Module.

Manages arrival confirmation conversations with substitute candidates over WhatsApp,
generates personalized outreach messages, and extracts schedule constraints using
the standard Chat Completions API (configured for local Ollama or OpenAI).
Applies Context Engineering best practices: role separation, structured JSON inputs,
unambiguous constraints, anti-alignment bias guards, and token-efficient few-shot examples.
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
) -> str:
    """Generate a personalized WhatsApp outreach message for the top candidate."""
    if stub_mode:
        return (
            f"היי {candidate.first_name}, התפנה שיבוץ ב{garden_name} "
            f"בתאריך {shift_date} ({shift_hours}). האם תוכלי להגיע?"
        )

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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(one_shot_input, ensure_ascii=False)},
        {"role": "assistant", "content": one_shot_output},
        {"role": "user", "content": final_user_content},
    ]

    client = _get_llm_client()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
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
        model=MODEL,
        messages=messages,
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
