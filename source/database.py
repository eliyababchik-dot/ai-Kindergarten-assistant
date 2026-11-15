"""Data Storage and Database Gateway Module.

This module provides data access and persistence abstraction for SQLite,
managing table creation, candidate filtering queries, and atomic status updates.

Scenario 1 Role:
    Executes relational SQL queries joining substitute profiles with distance
    tables for a specified kindergarten and absence date.
MVP Context:
    Fulfills feature 3 ("Initial availability filtering via SQL queries") and
    feature 5 ("Closing the assignment via SQL UPDATE").
"""

from typing import List, Optional
from source.models import AbsenceRequest, CandidateMatch, GardenDistance, Substitute


class DatabaseGateway:
    """Gateway for executing SQL operations against the SQLite database."""

    def __init__(self, db_path: str = "substitutes.db") -> None:
        """Initialize the database gateway with a target SQLite database file.

        Args:
            db_path: Path to the SQLite database file.
        """
        pass

    def initialize_schema(self) -> None:
        """Create necessary database tables if they do not already exist.

        Initializes tables for substitutes, substitute_distances, absences,
        and finalized assignments.
        """
        pass

    def save_substitute(self, substitute: Substitute) -> int:
        """Insert a single substitute assistant record into the database.

        Args:
            substitute: Substitute model to insert.

        Returns:
            The generated integer primary key identifier of the inserted record.
        """
        pass

    def save_distances(self, distances: List[GardenDistance]) -> None:
        """Batch insert distance records mapping substitutes to kindergartens.

        Args:
            distances: Collection of GardenDistance records to insert.
        """
        pass

    def create_absence(self, absence: AbsenceRequest) -> int:
        """Record a new absence request in the database with status 'OPEN'.

        Args:
            absence: AbsenceRequest details to record.

        Returns:
            The generated absence_id for tracking throughout the placement lifecycle.
        """
        pass

    def get_candidates_for_garden(
        self, garden_name: str, target_date: str
    ) -> List[CandidateMatch]:
        """Query substitutes with their calculated distance to the specified garden.

        Executes a SQL JOIN between substitutes and substitute_distances, filtering
        out assistants who already have a confirmed assignment on target_date.

        Args:
            garden_name: Exact name of the kindergarten with the absence.
            target_date: Date string (YYYY-MM-DD) to check availability for.

        Returns:
            A list of eligible CandidateMatch objects with distance and experience metrics.
        """
        pass

    def update_assignment(
        self, absence_id: int, substitute_id: int, status: str = "FILLED"
    ) -> bool:
        """Close an absence assignment and lock the substitute schedule via SQL UPDATE.

        Args:
            absence_id: Identifier of the absence request being resolved.
            substitute_id: Identifier of the confirmed substitute assistant.
            status: New terminal status for the absence (defaults to 'FILLED').

        Returns:
            True if the database record was updated successfully, False otherwise.
        """
        pass
