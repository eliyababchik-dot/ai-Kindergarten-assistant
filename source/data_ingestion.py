"""Data Ingestion Module.

This module is responsible for loading and validating pre-prepared files containing
substitute profiles and kindergarten commute distance matrices, seeding them into
persistent storage.

Scenario 1 Role:
    Loads initial assistant profiles and distance matrices from static files into
    the SQLite database prior to querying.
MVP Context:
    Fulfills feature 1 of the MVP ("Collecting and storing assistant profile data
    via pre-prepared files").
"""

from typing import List, Optional
from source.models import Substitute, GardenDistance


class DataIngestionService:
    """Service for parsing and loading pre-prepared files into the database."""

    def __init__(self, db_gateway: Optional[object] = None) -> None:
        """Initialize the data ingestion service with a database gateway dependency.

        Args:
            db_gateway: DatabaseGateway instance used to persist ingested records.
        """
        pass

    def load_substitutes(self, file_path: str) -> List[Substitute]:
        """Parse and validate assistant profiles from a pre-prepared file (CSV/JSON).

        Args:
            file_path: Absolute or relative file path to the profiles dataset.

        Returns:
            A list of validated Substitute entities ready for storage.
        """
        pass

    def load_distances(self, file_path: str) -> List[GardenDistance]:
        """Parse and validate commute distance mappings from a pre-prepared file.

        Args:
            file_path: Absolute or relative file path to the distances dataset.

        Returns:
            A list of validated GardenDistance entities.
        """
        pass

    def seed_database(self, substitutes_file: str, distances_file: str) -> int:
        """Execute complete database seeding from both profiles and distances datasets.

        Args:
            substitutes_file: Path to the assistant profiles file.
            distances_file: Path to the kindergarten distances file.

        Returns:
            Total count of records successfully persisted into the database.
        """
        pass
