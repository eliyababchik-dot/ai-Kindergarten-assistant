"""Data Ingestion Module.

Loads and validates assistant profiles and kindergarten commute distance matrices,
seeding them into persistent storage in full alignment with weekly availability.
"""

import csv
import datetime
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
    """Service for parsing and loading pre-prepared files and default seeds."""

    def __init__(self, db_gateway: Optional[DatabaseGateway] = None) -> None:
        self.db_gateway = db_gateway or DatabaseGateway()

    def load_substitutes(self, file_path: str) -> List[Substitute]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Substitutes file not found: {file_path}")

        substitutes = []
        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    substitutes.append(
                        Substitute(
                            first_name=item["first_name"],
                            last_name=item["last_name"],
                            city=item.get("city", ""),
                            neighborhood=item.get("neighborhood", ""),
                            age=int(item.get("age", 0)),
                            experience_years=int(item["experience_years"]),
                            certifications=item.get("certifications", []),
                            phone_number=item.get("phone_number", ""),
                            available_days=item.get("available_days", []),
                        )
                    )
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    certs = [c.strip() for c in row.get("certifications", "").split(",") if c.strip()]
                    days = [d.strip() for d in row.get("available_days", "").split(",") if d.strip()]
                    substitutes.append(
                        Substitute(
                            first_name=row["first_name"],
                            last_name=row["last_name"],
                            city=row.get("city", ""),
                            neighborhood=row.get("neighborhood", ""),
                            age=int(row.get("age", 0)),
                            experience_years=int(row["experience_years"]),
                            certifications=certs,
                            phone_number=row.get("phone_number", ""),
                            available_days=days,
                        )
                    )
        return substitutes

    def load_distances(self, file_path: str) -> List[GardenDistance]:
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
        subs = self.load_substitutes(substitutes_file)
        for sub in subs:
            self.db_gateway.save_substitute(sub)

        dists = self.load_distances(distances_file)
        self.db_gateway.save_distances(dists)

        return len(subs) + len(dists)

    def seed_default_data(self) -> int:
        """Seed default substitutes, distances, and weekday-aligned availability."""
        conn = self.db_gateway.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM substitutes")
        count = cursor.fetchone()[0]
        if self.db_gateway.db_path != ":memory:":
            conn.close()

        if count > 0:
            return 0

        substitutes_data = [
            Substitute(
                first_name="חן",
                last_name="כהן",
                city="ירושלים",
                neighborhood="רחביה",
                age=28,
                experience_years=5,
                certifications=["עזרה ראשונה", "חינוך מיוחד"],
                phone_number="050-1111111",
                available_days=["ראשון", "שני", "שלישי", "רביעי", "חמישי"],
            ),
            Substitute(
                first_name="מיכל",
                last_name="לוי",
                city="ירושלים",
                neighborhood="תלפיות",
                age=35,
                experience_years=10,
                certifications=["הדרכת גיל הרך"],
                phone_number="052-2222222",
                available_days=["שני", "שלישי", "רביעי"],
            ),
            Substitute(
                first_name="שרה",
                last_name="ישראלי",
                city="מבשרת ציון",
                neighborhood="רכס חלילים",
                age=24,
                experience_years=2,
                certifications=["עזרה ראשונה"],
                phone_number="054-3333333",
                available_days=["ראשון", "שלישי", "חמישי"],
            ),
        ]

        for sub in substitutes_data:
            self.db_gateway.save_substitute(sub)

        chen, michal, sarah = substitutes_data

        distances_data = [
            # חן (רחביה)
            GardenDistance(substitute_id=chen.id, garden_name="Gan Shaked", distance_km=2.5),
            GardenDistance(substitute_id=chen.id, garden_name="Gan Rimon", distance_km=5.0),
            # מיכל (תלפיות)
            GardenDistance(substitute_id=michal.id, garden_name="Gan Shaked", distance_km=6.0),
            GardenDistance(substitute_id=michal.id, garden_name="Gan Rimon", distance_km=2.0),
            # שרה (מבשרת ציון)
            GardenDistance(substitute_id=sarah.id, garden_name="Gan Shaked", distance_km=9.0),
            GardenDistance(substitute_id=sarah.id, garden_name="Gan Rimon", distance_km=14.0),
        ]

        self.db_gateway.save_distances(distances_data)

        # סנכרון זמינות לפי ימי השבוע:
        # 2026-11-22 = יום ראשון (חן ושרה פנויות, מיכל לא עובדת בימי ראשון)
        # 2026-11-23 = יום שני (חן ומיכל פנויות, שרה לא עובדת בימי שני)
        test_dates = ["2026-11-22", "2026-11-23"]
        for date_str in test_dates:
            weekday_name = DatabaseGateway.HEBREW_WEEKDAYS[
                datetime.date.fromisoformat(date_str).weekday()
            ]
            for sub in substitutes_data:
                is_free = weekday_name in sub.available_days
                self.db_gateway.set_availability(sub.id, date_str, is_available=is_free)

        return len(substitutes_data) + len(distances_data)


if __name__ == "__main__":
    db = DatabaseGateway("substitutes.db")
    service = DataIngestionService(db)
    inserted = service.seed_default_data()
    print(f"Database seeded successfully ({inserted} records).")