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

from typing import List, Optional

try:
    from source.models import AbsenceRequest, RankedCandidate
    from source.orchestrator import AbsenceOrchestrator
except ImportError:
    from .models import AbsenceRequest, RankedCandidate
    from .orchestrator import AbsenceOrchestrator


class CoordinatorCLI:
    """Command-line interface controller for kindergarten coordinators."""

    def __init__(self, orchestrator: Optional[AbsenceOrchestrator] = None) -> None:
        """Initialize the coordinator CLI with a workflow orchestrator dependency.

        Args:
            orchestrator: AbsenceOrchestrator instance used to execute business operations.
        """
        self.orchestrator = orchestrator or AbsenceOrchestrator()

    def capture_absence_input(
        self,
        garden_name: Optional[str] = None,
        date: Optional[str] = None,
        hours: Optional[str] = None,
    ) -> AbsenceRequest:
        """Prompt coordinator for absence details or accept provided parameters.

        Collects kindergarten name, absence date (YYYY-MM-DD), shift hours (start-end),
        and any optional special requirements.

        Returns:
            A validated AbsenceRequest object ready for orchestrator processing.
        """
        target_garden = garden_name or input("Enter kindergarten name: ").strip()
        target_date = date or input("Enter absence date (YYYY-MM-DD): ").strip()
        target_hours = hours or input("Enter shift hours (e.g. 08:00-14:00): ").strip()

        start_time, end_time = "08:00", "14:00"
        if "-" in target_hours:
            parts = target_hours.split("-")
            start_time, end_time = parts[0].strip(), parts[1].strip()

        return AbsenceRequest(
            garden_name=target_garden,
            date=target_date,
            start_time=start_time,
            end_time=end_time,
        )

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
        print(f"\n--- Ranked Substitutes for Absence at {garden_name} ---")
        if not ranked_candidates:
            print("No available substitutes found.")
            return

        for cand in ranked_candidates:
            sub = cand.candidate
            print(
                f"{cand.rank}. {sub.first_name} {sub.last_name} (Score: {cand.score}) | "
                f"Distance: {sub.distance_km}km, Exp: {sub.experience_years} yrs"
            )

    def run_absence_flow(
        self,
        garden_name: Optional[str] = None,
        date: Optional[str] = None,
        hours: Optional[str] = None,
    ) -> List[RankedCandidate]:
        """Execute the interactive absence intake and recommendation workflow.

        Captures coordinator input, dispatches the request to the orchestrator,
        and renders the top ranked candidate matches to the user.

        Args:
            garden_name: Optional garden name for non-interactive execution.
            date: Optional date for non-interactive execution.
            hours: Optional hours for non-interactive execution.

        Returns:
            The list of RankedCandidate objects produced.
        """
        request = self.capture_absence_input(garden_name, date, hours)
        results = self.orchestrator.handle_absence_report(request)
        self.display_ranked_substitutes(request.garden_name, results)
        return results