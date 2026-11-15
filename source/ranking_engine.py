"""Heuristic Ranking and Dynamic Re-sorting Engine Module.

This module encapsulates the business rules and mathematical heuristics for scoring
and sorting candidate substitutes based on distance, experience, and operational constraints.

Scenario 1 Role:
    Evaluates candidate distance and experience, producing an initial sorted list
    of top matches (e.g., Sarah: 92, Leah: 65).
MVP Context:
    Fulfills feature 3 ("Initial availability filtering and heuristic ranking") and
    feature 5 ("Re-sorting heuristically based on dialogue-extracted constraints").
"""

from typing import Dict, List, Optional
from source.models import (
    AbsenceRequest,
    CandidateConstraint,
    CandidateMatch,
    RankedCandidate,
    ScoringWeights,
)


class HeuristicRankingEngine:
    """Computes heuristic suitability scores and orders candidates for placement."""

    def __init__(self, weights: Optional[ScoringWeights] = None) -> None:
        """Initialize the ranking engine with customizable scoring weights.

        Args:
            weights: ScoringWeights instance defining coefficients for experience,
                distance penalty, and delay penalty. If None, default weights are used.
        """
        pass

    def calculate_score(
        self,
        candidate: CandidateMatch,
        constraint: Optional[CandidateConstraint] = None,
    ) -> float:
        """Calculate the numerical heuristic score for a single candidate.

        Combines positive contributions from experience with penalties for commute
        distance, as well as any operational delays extracted during dialogue.

        Formula:
            Score = (experience * W_exp) - (distance_km * W_dist) - (delay_min * W_delay)

        Args:
            candidate: CandidateMatch containing experience and distance attributes.
            constraint: Optional CandidateConstraint extracted from WhatsApp dialogue.

        Returns:
            Computed numerical score (higher is better).
        """
        pass

    def rank_candidates(
        self,
        candidates: List[CandidateMatch],
        request: AbsenceRequest,
        constraints: Optional[Dict[int, CandidateConstraint]] = None,
    ) -> List[RankedCandidate]:
        """Filter and rank candidate substitutes in descending order of suitability.

        Applies hard filters (e.g., checking requirements or hard unavailability),
        computes heuristic scores for all eligible candidates, and returns them
        sorted from highest to lowest score with rank indices assigned.

        Args:
            candidates: Unsorted list of candidates retrieved from storage.
            request: The specific AbsenceRequest containing garden and shift parameters.
            constraints: Optional mapping of substitute_id to CandidateConstraint
                used when performing dynamic re-sorting after dialogue confirmation.

        Returns:
            Sorted list of RankedCandidate objects with assigned rank numbers (1 = top match).
        """
        pass
