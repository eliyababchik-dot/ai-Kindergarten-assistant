"""Workflow Orchestrator Module.

This module acts as the core application service, coordinating data flow between
storage, ranking, user interface, and external communication services.

Scenario 1 Role:
    Directs the workflow for an incoming absence: records the request, fetches
    candidates from database storage, invokes the heuristic engine for ranking,
    and returns the sorted candidate recommendations.
MVP Context:
    Serves as the central coordinator across the full MVP lifecycle, connecting
    initial absence intake to dialogue outreach and assignment closure.
"""

from typing import List, Optional

try:
    from source.data_ingestion import DataIngestionService
    from source.database import DatabaseGateway
    from source.models import AbsenceRequest, RankedCandidate
    from source.ranking_engine import HeuristicRankingEngine
except ImportError:
    from .data_ingestion import DataIngestionService
    from .database import DatabaseGateway
    from .models import AbsenceRequest, RankedCandidate
    from .ranking_engine import HeuristicRankingEngine


class AbsenceOrchestrator:
    """Coordinates business workflows across domain services for staff placements."""

    def __init__(
        self,
        db_gateway: Optional[DatabaseGateway] = None,
        ranking_engine: Optional[HeuristicRankingEngine] = None,
        ingestion_service: Optional[DataIngestionService] = None,
    ) -> None:
        """Initialize the orchestrator with required domain dependencies.

        Args:
            db_gateway: DatabaseGateway for database reads and writes.
            ranking_engine: HeuristicRankingEngine for scoring candidate matches.
            ingestion_service: Optional service for pre-prepared file bootstrapping.
        """
        self.db_gateway = db_gateway or DatabaseGateway()
        self.ranking_engine = ranking_engine or HeuristicRankingEngine()
        self.ingestion_service = ingestion_service or DataIngestionService(self.db_gateway)

    def bootstrap_system(
        self, substitutes_file: str = "", distances_file: str = ""
    ) -> None:
        """Initialize the database schema and seed initial pre-prepared data.

        Args:
            substitutes_file: Optional path to pre-prepared assistant profiles dataset.
            distances_file: Optional path to pre-prepared kindergarten distances dataset.
        """
        self.db_gateway.initialize_schema()
        if substitutes_file and distances_file:
            self.ingestion_service.seed_database(substitutes_file, distances_file)
        else:
            self.ingestion_service.seed_default_data()

    def handle_absence_report(
        self, request: AbsenceRequest
    ) -> List[RankedCandidate]:
        """Execute the Scenario 1 placement pipeline for a reported absence.

        Performs the following sequence:
        1. Records the absence request in persistent storage.
        2. Queries eligible substitute candidates for the specified kindergarten and date.
        3. Invokes the heuristic ranking engine to score and order the candidates.
        4. Returns the sorted list of ranked candidate recommendations.

        Args:
            request: AbsenceRequest containing kindergarten name, date, and hours.

        Returns:
            A list of RankedCandidate objects sorted by descending heuristic score.
        """
        # Ensure default data is seeded if database is fresh
        self.ingestion_service.seed_default_data()

        # 1. Record the absence request in storage
        self.db_gateway.create_absence(request)

        # 2. Query available candidate substitutes for the target garden and date
        candidates = self.db_gateway.get_candidates_for_garden(
            request.garden_name, request.date
        )

        # 3. Compute heuristic scores and order by suitability
        ranked_candidates = self.ranking_engine.rank_candidates(candidates, request)

        return ranked_candidates
