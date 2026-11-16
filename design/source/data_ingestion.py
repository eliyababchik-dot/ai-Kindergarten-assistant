"""Data Ingestion Module.

This module is responsible for loading and validating pre-prepared files containing
substitute profiles and kindergarten commute distance matrices, seeding them into
persistent storage.

Scenario 1 Role:
    Loads initial assistant profiles and distance matrices from static files or default
    seeds into the SQLite database prior to querying.
MVP Context:
    Fulfills feature 1 of the MVP ("Collecting and storing assistant profile data
    via pre-prepared files").
"""

import csv
import json
import os
from typing import List, Optional

try:
    from source.database import DatabaseGateway
    from source.models import GardenDistance, Substitute
except ImportError:
    from .database import DatabaseGateway
    from .models import GardenDistance, Substitute


class DataIngestionService:
    """Service for parsing and loading pre-prepared files into the database."""

    def __init__(self, db_gateway: Optional[DatabaseGateway] = None) -> None:
        """Initialize the data ingestion service with a database gateway dependency.

        Args:
            db_gateway: DatabaseGateway instance used to persist ingested records.
        """
        self.db_gateway = db_gateway or DatabaseGateway()

    def load_substitutes(self, file_path: str) -> List[Substitute]:
        """Parse and validate assistant profiles from a pre-prepared file (CSV/JSON).

        Args:
            file_path: Absolute or relative file path to the profiles dataset.

        Returns:
            A list of validated Substitute entities ready for storage.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Substitutes file not found: {file_path}")

        substitutes = []
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    substitutes.append(
                        Substitute(
                            name=item["name"],
                            experience_years=int(item["experience_years"]),
                            phone_number=item.get("phone_number", ""),
                        )
                    )
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    substitutes.append(
                        Substitute(
                            name=row["name"],
                            experience_years=int(row["experience_years"]),
                            phone_number=row.get("phone_number", ""),
                        )
                    )
        return substitutes

    def load_distances(self, file_path: str) -> List[GardenDistance]:
        """Parse and validate commute distance mappings from a pre-prepared file.

        Args:
            file_path: Absolute or relative file path to the distances dataset.

        Returns:
            A list of validated GardenDistance entities.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Distances file not found: {file_path}")

        distances = []
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    distances.append(
                        GardenDistance(
                            substitute_id=int(item["substitute_id"]),
                            garden_name=item["garden_name"],
                            distance_km=float(item["distance_km"]),
                        )
                    )
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    distances.append(
                        GardenDistance(
                            substitute_id=int(row["substitute_id"]),
                            garden_name=row["garden_name"],
                            distance_km=float(row["distance_km"]),
                        )
                    )
        return distances

    def seed_database(self, substitutes_file: str, distances_file: str) -> int:
        """Execute complete database seeding from both profiles and distances datasets.

        Args:
            substitutes_file: Path to the assistant profiles file.
            distances_file: Path to the kindergarten distances file.

        Returns:
            Total count of records successfully persisted into the database.
        """
        subs = self.load_substitutes(substitutes_file)
        for sub in subs:
            self.db_gateway.save_substitute(sub)

        dists = self.load_distances(distances_file)
        self.db_gateway.save_distances(dists)

        return len(subs) + len(dists)

    def seed_default_data(self) -> int:
        """Seed the database with default sample data from setup_db.py and scenarios.md.

        Includes candidates and distances for both English ('Gan Shaked', 'Gan Rimon')
        and Hebrew ('גן שקד', 'גן רימון') representations, covering all test cases.

        Returns:
            Total count of records seeded.
        """
        conn = self.db_gateway.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM substitutes")
        count = cursor.fetchone()[0]
        if self.db_gateway.db_path != ":memory:":
            conn.close()

        if count > 0:
            return 0

        # Sample substitutes covering setup_db.py and scenarios.md
        substitutes_data = [
            Substitute(name="Sarah", experience_years=3, phone_number="050-1111111"),
            Substitute(name="Leah", experience_years=1, phone_number="050-2222222"),
            Substitute(name="Rachel", experience_years=5, phone_number="050-3333333"),
            Substitute(name="Rivka", experience_years=2, phone_number="050-4444444"),
        ]

        sub_ids = []
        for sub in substitutes_data:
            sub_id = self.db_gateway.save_substitute(sub)
            sub_ids.append(sub_id)

        sarah_id, leah_id, rachel_id, rivka_id = sub_ids

        # Commute distances covering Scenario 1 test cases
        distances_data = [
            # Gan Shaked (English)
            GardenDistance(substitute_id=sarah_id, garden_name="Gan Shaked", distance_km=1.0),
            GardenDistance(substitute_id=leah_id, garden_name="Gan Shaked", distance_km=5.0),
            GardenDistance(substitute_id=rachel_id, garden_name="Gan Shaked", distance_km=7.0),
            GardenDistance(substitute_id=rivka_id, garden_name="Gan Shaked", distance_km=9.0),

            # Gan Rimon (English)
            GardenDistance(substitute_id=sarah_id, garden_name="Gan Rimon", distance_km=8.0),
            GardenDistance(substitute_id=leah_id, garden_name="Gan Rimon", distance_km=4.5),
            GardenDistance(substitute_id=rachel_id, garden_name="Gan Rimon", distance_km=2.0),
            GardenDistance(substitute_id=rivka_id, garden_name="Gan Rimon", distance_km=8.0),

            # גן שקד (Hebrew from setup_db.py)
            GardenDistance(substitute_id=sarah_id, garden_name="גן שקד", distance_km=1.0),
            GardenDistance(substitute_id=leah_id, garden_name="גן שקד", distance_km=4.0),
            GardenDistance(substitute_id=rachel_id, garden_name="גן שקד", distance_km=7.0),
            GardenDistance(substitute_id=rivka_id, garden_name="גן שקד", distance_km=9.0),

            # גן רימון (Hebrew from setup_db.py)
            GardenDistance(substitute_id=sarah_id, garden_name="גן רימון", distance_km=8.0),
            GardenDistance(substitute_id=leah_id, garden_name="גן רימון", distance_km=4.5),
            GardenDistance(substitute_id=rachel_id, garden_name="גן רימון", distance_km=1.5),
            GardenDistance(substitute_id=rivka_id, garden_name="גן רימון", distance_km=8.0),
        ]

        self.db_gateway.save_distances(distances_data)

        # Date availability matching Scenario 1 test cases:
        # Case 1 (2026-11-18): Gan Shaked -> only Sarah & Leah available
        self.db_gateway.set_availability(rachel_id, "2026-11-18", is_available=False)
        self.db_gateway.set_availability(rivka_id, "2026-11-18", is_available=False)

        # Case 2 (2026-11-19): Gan Rimon -> only Rachel & Rivka available
        self.db_gateway.set_availability(sarah_id, "2026-11-19", is_available=False)
        self.db_gateway.set_availability(leah_id, "2026-11-19", is_available=False)

        return len(substitutes_data) + len(distances_data)
