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

    # Garden name aliases to ensure compatibility with English and Hebrew queries
    GARDEN_ALIASES = {
        "gan shaked": ["Gan Shaked", "גן שקד"],
        "גן שקד": ["Gan Shaked", "גן שקד"],
        "gan rimon": ["Gan Rimon", "גן רימון"],
        "גן רימון": ["Gan Rimon", "גן רימון"],
    }

    def __init__(self, db_path: str = "substitutes.db") -> None:
        """Initialize the database gateway with a target SQLite database file.

        Args:
            db_path: Path to the SQLite database file (or ':memory:' for tests).
        """
        self.db_path = db_path
        self._memory_conn: Optional[sqlite3.Connection] = None
        self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Open or return a connection to the SQLite database."""
        if self.db_path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:")
                self._memory_conn.row_factory = sqlite3.Row
            return self._memory_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_schema(self) -> None:
        """Create necessary database tables if they do not already exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        # Substitutes table (extended with phone_number for future WhatsApp dialogue)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS substitutes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            experience_years INTEGER NOT NULL,
            phone_number TEXT DEFAULT ''
        )
        """)

        # Distances between substitutes and specific kindergartens
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS substitute_distances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            substitute_id INTEGER NOT NULL,
            garden_name TEXT NOT NULL,
            distance_km REAL NOT NULL,
            FOREIGN KEY (substitute_id) REFERENCES substitutes (id)
        )
        """)

        # Absence requests table
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

        # Substitute availability calendar
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
        """Insert a single substitute assistant record into the database.

        Args:
            substitute: Substitute model to insert.

        Returns:
            The generated integer primary key identifier of the inserted record.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO substitutes (name, experience_years, phone_number)
            VALUES (?, ?, ?)
            """,
            (substitute.name, substitute.experience_years, substitute.phone_number),
        )
        sub_id = cursor.lastrowid
        substitute.id = sub_id
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()
        return sub_id

    def save_distances(self, distances: List[GardenDistance]) -> None:
        """Batch insert distance records mapping substitutes to kindergartens.

        Args:
            distances: Collection of GardenDistance records to insert.
        """
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
        """Record availability or absence of a substitute on a given date."""
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
        """Record a new absence request in the database with status 'OPEN'.

        Args:
            absence: AbsenceRequest details to record.

        Returns:
            The generated absence_id for tracking throughout the placement lifecycle.
        """
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
        """Query substitutes with their calculated distance to the specified garden.

        Executes a SQL JOIN between substitutes and substitute_distances, filtering
        out assistants who already have a confirmed assignment on target_date.

        Args:
            garden_name: Exact name of the kindergarten with the absence.
            target_date: Date string (YYYY-MM-DD) to check availability for.

        Returns:
            A list of eligible CandidateMatch objects with distance and experience metrics.
        """
        normalized_name = garden_name.strip().lower()
        search_names = self.GARDEN_ALIASES.get(normalized_name, [garden_name])

        query = f"""
        SELECT s.id, s.name, s.experience_years, MIN(d.distance_km) AS distance_km, s.phone_number
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

        results = [
            CandidateMatch(
                substitute_id=row["id"],
                name=row["name"],
                experience_years=row["experience_years"],
                distance_km=float(row["distance_km"]),
                phone_number=row["phone_number"] or "",
            )
            for row in rows
        ]
        if self.db_path != ":memory:":
            conn.close()
        return results

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
