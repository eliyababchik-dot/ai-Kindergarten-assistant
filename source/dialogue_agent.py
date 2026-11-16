"""Conversational Dialogue Agent Module (AI Stub).

This module manages arrival confirmation conversations with candidates over WhatsApp
and extracts schedule constraints (e.g. late arrival, partial shift availability).

In this MVP stage, it is implemented strictly as a typed stub without live LLM calls,
returning a deterministic ExtractedConstraints result.
"""

from typing import Optional

try:
    from source.models import DialogueContext, ExtractedConstraints
except ImportError:
    from .models import DialogueContext, ExtractedConstraints


def confirm_arrival_dialogue(
    context: DialogueContext, candidate_reply: str
) -> ExtractedConstraints:
    """Manage arrival confirmation dialogue and extract schedule constraints.

    This is a stub: it returns the same result for every candidate response.

    Args:
        context: DialogueContext containing the candidate details, kindergarten,
            shift hours, and message history.
        candidate_reply: The text message received from the candidate.

    Returns:
        ExtractedConstraints indicating willingness and arrival details.
    """
    return ExtractedConstraints(
        willingness="CONFIRMED",
        arrival_time=None,
        notes="Stub confirmation",
    )


class DialogueAgent:
    """Agent interface for managing multi-turn conversations with substitutes."""

    def __init__(self, stub_mode: bool = True) -> None:
        """Initialize the dialogue agent.

        Args:
            stub_mode: Flag indicating whether to use deterministic stub responses.
        """
        self.stub_mode = stub_mode

    def process_reply(
        self, context: DialogueContext, candidate_reply: str
    ) -> ExtractedConstraints:
        """Process candidate reply and return extracted constraints."""
        return confirm_arrival_dialogue(context, candidate_reply)
