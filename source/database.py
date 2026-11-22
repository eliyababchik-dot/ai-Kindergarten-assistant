"""Data Storage and Database Gateway Module.

This module provides data access and persistence abstraction for SQLite,
managing table creation, candidate filtering queries, and atomic status updates.
"""

import datetime
import json
import sqlite3
from typing import List, Optional

try:
    from source.models import (
        AbsenceRequest,
        CandidateMatch,
        GardenDistance,
        Substitute,
    )
except ImportError:
    from .models import (
        AbsenceRequest,
        CandidateMatch,
        GardenDistance,
        Substitute,
    )


class DatabaseGateway:
    """Gateway for executing SQL operations against the SQLite database."""

    GARDEN_ALIASES = {
        "gan shaked": ["Gan Shaked", "גן שקד"],
        "גן שקד": ["Gan Shaked", "גן שקד"],
        "gan rimon": ["Gan Rimon", "גן רימון"],
        "גן רימון": ["Gan Rimon", "גן רימון"],
    }

    HEBREW_WEEKDAYS = {
        6: "ראשון",
        0: "שני",
        1: "שלישי",
        2: "רביעי",
        3: "חמישי",
        4: "שישי",
        5: "שבת",
    }

    def __init__(self, db_path: str = "substitutes.db") -> None:
        self.db_path = db_path
        self._memory_conn: Optional[sqlite3.Connection] = None
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:")
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS substitutes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            city TEXT,
            neighborhood TEXT,
            age INTEGER,
            experience_years INTEGER NOT NULL,
            certifications TEXT,
            phone_number TEXT DEFAULT '',
            available_days TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS substitute_distances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            substitute_id INTEGER NOT NULL,
            garden_name TEXT NOT NULL,
            distance_km REAL NOT NULL,
            FOREIGN KEY (substitute_id) REFERENCES substitutes (id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS absences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            garden_name TEXT NOT NULL,
            date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            status TEXT DEFAULT 'OPEN',
            assigned_substitute_id INTEGER,
            requirements TEXT DEFAULT '',
            FOREIGN KEY (assigned_substitute_id) REFERENCES substitutes (id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS substitute_availability (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            substitute_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            is_available INTEGER DEFAULT 1,
            FOREIGN KEY (substitute_id) REFERENCES substitutes (id)
        )
        """)
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()

    def save_substitute(self, substitute: Substitute) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()

        certs_json = json.dumps(substitute.certifications, ensure_ascii=False)
        days_json = json.dumps(substitute.available_days, ensure_ascii=False)

        cursor.execute(
            """
            INSERT INTO substitutes (
                first_name, last_name, city, neighborhood, age,
                experience_years, certifications, phone_number, available_days
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                substitute.first_name,
                substitute.last_name,
                substitute.city,
                substitute.neighborhood,
                substitute.age,
                substitute.experience_years,
                certs_json,
                substitute.phone_number,
                days_json,
            ),
        )
        sub_id = cursor.lastrowid
        substitute.id = sub_id
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()
        return sub_id

    def save_distances(self, distances: List[GardenDistance]) -> None:
        records = [
            (dist.substitute_id, dist.garden_name, dist.distance_km)
            for dist in distances
        ]
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO substitute_distances (substitute_id, garden_name, distance_km)
            VALUES (?, ?, ?)
            """,
            records,
        )
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()

    def set_availability(self, substitute_id: int, date: str, is_available: bool) -> None:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO substitute_availability (substitute_id, date, is_available)
            VALUES (?, ?, ?)
            """,
            (substitute_id, date, 1 if is_available else 0),
        )
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()

    def create_absence(self, absence: AbsenceRequest) -> int:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO absences (garden_name, date, start_time, end_time, status, requirements)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                absence.garden_name,
                absence.date,
                absence.start_time,
                absence.end_time,
                absence.status,
                ",".join(absence.requirements),
            ),
        )
        absence_id = cursor.lastrowid
        absence.absence_id = absence_id
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()
        return absence_id

    def get_candidates_for_garden(
        self, garden_name: str, target_date: str
    ) -> List[CandidateMatch]:
        normalized_name = garden_name.strip().lower()
        search_names = self.GARDEN_ALIASES.get(normalized_name, [garden_name])

        target_weekday = None
        try:
            parsed_date = datetime.date.fromisoformat(target_date)
            target_weekday = self.HEBREW_WEEKDAYS[parsed_date.weekday()]
        except ValueError:
            target_weekday = None

        query = f"""
        SELECT
            s.id, s.first_name, s.last_name, s.city, s.neighborhood,
            s.experience_years, s.certifications, s.phone_number, s.available_days,
            MIN(d.distance_km) AS distance_km
        FROM substitutes s
        JOIN substitute_distances d ON s.id = d.substitute_id
        WHERE d.garden_name IN ({','.join(['?'] * len(search_names))})
          AND s.id NOT IN (
              SELECT assigned_substitute_id FROM absences
              WHERE date = ? AND status = 'FILLED' AND assigned_substitute_id IS NOT NULL
          )
          AND s.id NOT IN (
              SELECT substitute_id FROM substitute_availability
              WHERE date = ? AND is_available = 0
          )
        GROUP BY s.id
        """
        params = list(search_names) + [target_date, target_date]

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            days = []
            if row["available_days"]:
                try:
                    days = json.loads(row["available_days"])
                except json.JSONDecodeError:
                    days = []

            if target_weekday and days and target_weekday not in days:
                continue

            certs = []
            if row["certifications"]:
                try:
                    certs = json.loads(row["certifications"])
                except json.JSONDecodeError:
                    certs = []

            results.append(
                CandidateMatch(
                    substitute_id=row["id"],
                    first_name=row["first_name"],
                    last_name=row["last_name"],
                    city=row["city"] or "",
                    neighborhood=row["neighborhood"] or "",
                    experience_years=row["experience_years"],
                    certifications=certs,
                    distance_km=float(row["distance_km"]),
                    phone_number=row["phone_number"] or "",
                )
            )

        if self.db_path != ":memory:":
            conn.close()
        return results

    def update_assignment(
        self, absence_id: int, substitute_id: int, status: str = "FILLED"
    ) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE absences
            SET status = ?, assigned_substitute_id = ?
            WHERE id = ?
            """,
            (status, substitute_id, absence_id),
        )
        affected = cursor.rowcount
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()
        return affected > 0