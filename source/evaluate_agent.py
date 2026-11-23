"""
Automated LLM Evaluation & Regression Runner (Module 3 Part H).

Evaluates classification accuracy against a labeled Hebrew WhatsApp dataset,
verifies zero prompt-to-eval data leakage, tracks regression failures, and
provides deterministic + optional LLM-as-a-judge scoring for free-text replies.
"""

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Dict, List, Optional

from source.dialogue_agent import DialogueAgent
from source.models import DialogueContext


REQUIRED_CATEGORIES = (
    "common",
    "hard",
    "edge",
    "misleading",
    "varied_style",
    "regression",
)


def load_eval_dataset(path: str = "evals/eval_dataset.json") -> Dict[str, Any]:
    """Load the evaluation dataset from JSON (supports UTF-8 with or without BOM)."""
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if "classification_cases" not in data or not isinstance(data["classification_cases"], list):
        raise ValueError("Evaluation dataset must contain a 'classification_cases' list.")
    return data


def validate_no_prompt_leakage(
    dataset: Dict[str, Any],
    dialogue_agent_source_path: str = "source/dialogue_agent.py",
) -> None:
    """
    Ensure evaluation inputs do not duplicate few-shot examples embedded in prompts.
    Raises ValueError if verbatim overlap is detected.
    """
    if not os.path.exists(dialogue_agent_source_path):
        return

    with open(dialogue_agent_source_path, "r", encoding="utf-8-sig") as f:
        source_text = f.read()

    for case in dataset.get("classification_cases", []):
        reply = (case.get("candidate_reply") or "").strip()
        if len(reply) >= 5 and reply in source_text:
            raise ValueError(
                f"Prompt leakage detected: eval case '{case.get('id')}' appears verbatim in {dialogue_agent_source_path}"
            )


def _build_eval_context(case: Dict[str, Any]) -> DialogueContext:
    """Safely construct a DialogueContext compatible with dataclass or Pydantic definitions."""
    defaults = {
        "candidate_id": case.get("id", "eval_cand"),
        "candidate_name": "סייעת לבדיקה",
        "garden_name": "גן שקד",
        "shift_date": "2026-11-24",
        "shift_hours": "08:00-14:00",
    }
    if dataclasses.is_dataclass(DialogueContext):
        valid_fields = {f.name for f in dataclasses.fields(DialogueContext)}
    elif hasattr(DialogueContext, "model_fields"):
        valid_fields = set(DialogueContext.model_fields.keys())
    elif hasattr(DialogueContext, "__fields__"):
        valid_fields = set(DialogueContext.__fields__.keys())
    else:
        valid_fields = {"candidate_name", "garden_name", "shift_date", "shift_hours"}

    filtered_kwargs = {k: v for k, v in defaults.items() if k in valid_fields}
    return DialogueContext(**filtered_kwargs)


def evaluate_classification(
    agent: DialogueAgent,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run deterministic classification evaluation over all cases in the eval set.
    """
    total_cases = len(cases)
    status_correct = 0
    arrival_correct = 0
    exact_match_count = 0
    low_confidence_count = 0
    per_category: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []

    for case in cases:
        case_id = case.get("id", "unknown")
        category = case.get("category", "uncategorized")
        reply = case["candidate_reply"]
        expected_status = case["expected_status"]
        expected_arrival = case.get("expected_arrival_time")

        if category not in per_category:
            per_category[category] = {
                "total": 0,
                "status_correct": 0,
                "exact_match": 0,
                "status_accuracy": 0.0,
                "exact_match_accuracy": 0.0,
            }

        ctx = _build_eval_context(case)
        result = agent.process_reply(ctx, reply)

        actual_status = getattr(result, "willingness", None)
        actual_arrival = getattr(result, "arrival_time", None)
        confidence = getattr(result, "confidence", None)

        if confidence is not None and confidence < 0.85:
            low_confidence_count += 1

        is_status_ok = actual_status == expected_status
        is_arrival_ok = actual_arrival == expected_arrival
        is_exact_ok = is_status_ok and is_arrival_ok

        per_category[category]["total"] += 1
        if is_status_ok:
            status_correct += 1
            per_category[category]["status_correct"] += 1
        if is_arrival_ok:
            arrival_correct += 1
        if is_exact_ok:
            exact_match_count += 1
            per_category[category]["exact_match"] += 1
        else:
            failures.append(
                {
                    "id": case_id,
                    "category": category,
                    "candidate_reply": reply,
                    "expected_status": expected_status,
                    "actual_status": actual_status,
                    "expected_arrival_time": expected_arrival,
                    "actual_arrival_time": actual_arrival,
                    "notes": case.get("notes", ""),
                }
            )

    for cat_stats in per_category.values():
        cat_total = cat_stats["total"]
        if cat_total > 0:
            cat_stats["status_accuracy"] = round(cat_stats["status_correct"] / cat_total, 4)
            cat_stats["exact_match_accuracy"] = round(cat_stats["exact_match"] / cat_total, 4)

    status_accuracy = round(status_correct / total_cases, 4) if total_cases > 0 else 0.0
    exact_match_accuracy = round(exact_match_count / total_cases, 4) if total_cases > 0 else 0.0

    usage_summary = agent.get_usage_summary() if hasattr(agent, "get_usage_summary") else {}
    estimated_cost = agent.estimate_cost_usd() if hasattr(agent, "estimate_cost_usd") else 0.0

    return {
        "total_cases": total_cases,
        "status_correct": status_correct,
        "arrival_correct": arrival_correct,
        "exact_match_count": exact_match_count,
        "status_accuracy": status_accuracy,
        "exact_match_accuracy": exact_match_accuracy,
        "low_confidence_count": low_confidence_count,
        "per_category": per_category,
        "failures": failures,
        "usage_summary": usage_summary,
        "estimated_cost_usd": estimated_cost,
    }


def evaluate_free_text_response(
    response_text: str,
    required_terms: List[str],
    forbidden_terms: Optional[List[str]] = None,
    max_chars: int = 350,
    use_llm_judge: bool = False,
    agent: Optional[DialogueAgent] = None,
) -> Dict[str, Any]:
    """
    Evaluate a free-text Hebrew WhatsApp message using deterministic rules,
    with optional probabilistic LLM-as-a-judge rubric scoring (1-5).
    """
    forbidden_terms = forbidden_terms or []
    cleaned = (response_text or "").strip()

    has_hebrew = any("\u0590" <= ch <= "\u05FF" for ch in cleaned)
    missing_terms = [term for term in required_terms if term not in cleaned]
    found_forbidden = [term for term in forbidden_terms if term in cleaned]
    within_length = 0 < len(cleaned) <= max_chars

    passed = (
        has_hebrew
        and len(missing_terms) == 0
        and len(found_forbidden) == 0
        and within_length
    )

    report: Dict[str, Any] = {
        "passed": passed,
        "has_hebrew": has_hebrew,
        "missing_required_terms": missing_terms,
        "found_forbidden_terms": found_forbidden,
        "char_length": len(cleaned),
        "within_length": within_length,
        "judge_score": None,
        "judge_reason": None,
    }

    if use_llm_judge and agent is not None and not getattr(agent, "stub_mode", True):
        client = getattr(agent, "client", None)
        if client is not None:
            judge_prompt = (
                "Rate this Hebrew WhatsApp message for a kindergarten substitute teacher from 1 to 5 "
                "on clarity, politeness, and conciseness. Reply with JSON: "
                '{"score": <int 1-5>, "reason": "<short English explanation>"}.\n\n'
                f"Message: {cleaned}"
            )
            resp = client.chat.completions.create(
                model=getattr(agent, "outreach_model", getattr(agent, "model", "qwen3:4b")),
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            raw_judge = (resp.choices[0].message.content or "").strip()
            try:
                parsed_judge = json.loads(raw_judge)
                report["judge_score"] = parsed_judge.get("score")
                report["judge_reason"] = parsed_judge.get("reason")
            except Exception:
                report["judge_reason"] = raw_judge

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 3 Part H LLM Evaluation Suite")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Run offline using DialogueAgent(stub_mode=True) without calling an LLM server",
    )
    parser.add_argument(
        "--dataset",
        default="evals/eval_dataset.json",
        help="Path to the evaluation JSON dataset",
    )
    args = parser.parse_args()

    dataset = load_eval_dataset(args.dataset)
    validate_no_prompt_leakage(dataset)

    agent = DialogueAgent(stub_mode=args.stub)
    results = evaluate_classification(agent, dataset["classification_cases"])

    mode_label = "STUB MODE (Offline)" if args.stub else "LIVE MODEL MODE"
    print(f"=== LLM Evaluation Report [{mode_label}] ===")
    print(f"Total Cases:          {results['total_cases']}")
    print(f"Status Accuracy:      {results['status_accuracy'] * 100:.1f}% ({results['status_correct']}/{results['total_cases']})")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy'] * 100:.1f}% ({results['exact_match_count']}/{results['total_cases']})")
    print(f"Low Confidence (<0.85): {results['low_confidence_count']}")
    print(f"Usage Summary:        {results['usage_summary']}")
    print(f"Estimated Cost (USD): ${results['estimated_cost_usd']:.6f}")
    print("\n--- Breakdown by Category ---")
    for cat, stats in results["per_category"].items():
        print(
            f"  {cat:<14} | Status: {stats['status_accuracy'] * 100:5.1f}% "
            f"| Exact: {stats['exact_match_accuracy'] * 100:5.1f}% ({stats['exact_match']}/{stats['total']})"
        )

    if results["failures"]:
        print("\n--- Regression / Mismatch Failures ---")
        for fail in results["failures"]:
            print(
                f"  [{fail['id']} | {fail['category']}] Reply: {fail['candidate_reply']!r}\n"
                f"     Expected: ({fail['expected_status']}, {fail['expected_arrival_time']}) "
                f"!= Actual: ({fail['actual_status']}, {fail['actual_arrival_time']})"
            )


if __name__ == "__main__":
    main()