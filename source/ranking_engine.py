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

try:
    from source.models import (
        AbsenceRequest,
        CandidateConstraint,
        CandidateMatch,
        RankedCandidate,
        ScoringWeights,
    )
except ImportError:
    from .models import (
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
                distance penalty, and delay penalty. If None, default weights from setup_db.py
                (experience * 10 - distance * 2) are used.
        """
        self.weights = weights or ScoringWeights()

    def calculate_score(
        self,
        candidate: CandidateMatch,
        constraint: Optional[CandidateConstraint] = None,
    ) -> float:
        """Calculate the numerical heuristic score for a single candidate.

        Combines positive contributions from experience with penalties for commute
        distance, as well as any operational delays extracted during dialogue.

        Formula (from setup_db.py):
            Score = (experience * W_exp) - (distance_km * W_dist) - (delay_min * W_delay)

        Args:
            candidate: CandidateMatch containing experience and distance attributes.
            constraint: Optional CandidateConstraint extracted from WhatsApp dialogue.

        Returns:
            Computed numerical score (higher is better).
        """
        exp_points = candidate.experience_years * self.weights.experience_weight
        dist_penalty = candidate.distance_km * self.weights.distance_penalty_weight
        delay_penalty = 0.0

        if constraint and constraint.delay_minutes > 0:
            delay_penalty = constraint.delay_minutes * self.weights.delay_penalty_per_minute

        raw_score = (exp_points - dist_penalty - delay_penalty) + self.weights.base_score
        return round(raw_score, 2)

    def rank_candidates(
        self,
        candidates: List[CandidateMatch],
        request: AbsenceRequest,
        constraints: Optional[Dict[int, CandidateConstraint]] = None,
    ) -> List[RankedCandidate]:
        """Filter and rank candidate substitutes in descending order of suitability.

        Applies hard filters (e.g. checking requirements or hard unavailability),
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
        scored_candidates = []
        constraints_map = constraints or {}

        for candidate in candidates:
            # Check if candidate has reported hard unavailability in dialogue
            cand_constraint = constraints_map.get(candidate.substitute_id)
            if cand_constraint and not cand_constraint.is_available:
                continue

            score = self.calculate_score(candidate, cand_constraint)
            breakdown = {
                "experience_points": candidate.experience_years * self.weights.experience_weight,
                "distance_penalty": candidate.distance_km * self.weights.distance_penalty_weight,
                "delay_penalty": (
                    cand_constraint.delay_minutes * self.weights.delay_penalty_per_minute
                    if cand_constraint and cand_constraint.delay_minutes > 0
                    else 0.0
                ),
            }
            scored_candidates.append((candidate, score, breakdown))

        # Sort descending by score; tie-breaker: closer distance, then higher experience
        scored_candidates.sort(
            key=lambda item: (item[1], -item[0].distance_km, item[0].experience_years),
            reverse=True,
        )

        ranked = []
        for rank_idx, (candidate, score, breakdown) in enumerate(scored_candidates, start=1):
            ranked.append(
                RankedCandidate(
                    candidate=candidate,
                    score=score,
                    rank=rank_idx,
                    breakdown=breakdown,
                )
            )

        return ranked
