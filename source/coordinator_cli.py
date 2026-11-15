"""Coordinator CLI and User Interface Module.

This module provides the user interface for kindergarten coordinators, handling
console input capture, input validation, and rendering formatted recommendations.

Scenario 1 Role:
    Captures absence details (garden, date, shift hours) from the coordinator and
    displays the resulting ranked candidate recommendations.
MVP Context:
    Fulfills feature 2 ("Reporting sudden absences and opening a request") and
    provides coordinator visibility into placement recommendations and statuses.
"""

from typing import List
from source.models import AbsenceRequest, RankedCandidate
from source.orchestrator import AbsenceOrchestrator


class CoordinatorCLI:
    """Command-line interface controller for kindergarten coordinators."""

    def __init__(self, orchestrator: AbsenceOrchestrator) -> None:
        """Initialize the coordinator CLI with a workflow orchestrator dependency.

        Args:
            orchestrator: AbsenceOrchestrator instance used to execute business operations.
        """
        pass

    def capture_absence_input(self) -> AbsenceRequest:
        """Prompt coordinator for absence details and validate the input fields.

        Collects kindergarten name, absence date (YYYY-MM-DD), shift hours (start-end),
        and any optional special requirements.

        Returns:
            A validated AbsenceRequest object ready for orchestrator processing.
        """
        pass

    def display_ranked_substitutes(
        self, garden_name: str, ranked_candidates: List[RankedCandidate]
    ) -> None:
        """Format and print ranked candidate recommendations to the console.

        Displays a clean table showing rank, assistant name, heuristic score,
        commute distance, and years of experience.

        Args:
            garden_name: Name of the kindergarten where the absence occurred.
            ranked_candidates: Sorted list of RankedCandidate objects to display.
        """
        pass

    def run_absence_flow(self) -> None:
        """Execute the interactive absence intake and recommendation workflow.

        Captures coordinator input, dispatches the request to the orchestrator,
        and renders the top ranked candidate matches to the user.
        """
        pass
