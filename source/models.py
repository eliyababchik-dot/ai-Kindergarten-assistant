"""Domain models and data transfer objects for the Kindergarten Staff Scheduling Assistant.

This module defines the core data contracts shared across modules. It includes
all data structures needed for Scenario 1 (filtering and ranking candidates),
while also defining the extensible fields required for future MVP phases
(such as candidate phone numbers for WhatsApp messaging, absence lifecycle
statuses for assignment closure, and dynamic constraint models for re-ranking).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Substitute:
    """Represents a kindergarten assistant / substitute staff member.

    Attributes:
        id: Unique database identifier for the substitute (None if not yet persisted).
        first_name: First name of the substitute assistant.
        last_name: Last name of the substitute assistant.
        city: City of residence.
        neighborhood: Neighborhood of residence.
        age: Age of the substitute assistant.
        experience_years: Total years of early childhood care experience.
        certifications: List of certifications and courses (e.g., First Aid, Special Ed).
        phone_number: Contact telephone number (used for WhatsApp outreach in MVP).
        available_days: List of days the substitute is generally available to work.
    """

    first_name: str
    last_name: str
    city: str
    neighborhood: str
    age: int
    experience_years: int
    certifications: List[str]
    phone_number: str
    available_days: List[str]
    id: Optional[int] = None


@dataclass
class GardenDistance:
    """Represents the pre-calculated travel distance from a substitute to a kindergarten.

    Attributes:
        substitute_id: Database identifier of the substitute assistant.
        garden_name: Name of the kindergarten.
        distance_km: Geographic or commute distance in kilometers.
    """

    substitute_id: int
    garden_name: str
    distance_km: float


@dataclass
class AbsenceRequest:
    """Represents a coordinator's report of a sudden staff absence at a kindergarten.

    Attributes:
        garden_name: Name of the kindergarten experiencing a staffing shortage.
        date: Date of the absence (formatted as YYYY-MM-DD).
        start_time: Shift start time (formatted as HH:MM).
        end_time: Shift end time (formatted as HH:MM).
        absence_id: Unique database identifier for this request (None if new).
        status: Current lifecycle status ('OPEN', 'IN_PROGRESS', 'FILLED', 'CANCELLED').
        assigned_substitute_id: Identifier of the assigned substitute once confirmed.
        requirements: Special tags or mandatory qualifications required for the shift.
    """

    garden_name: str
    date: str
    start_time: str
    end_time: str
    absence_id: Optional[int] = None
    status: str = "OPEN"
    assigned_substitute_id: Optional[int] = None
    requirements: List[str] = field(default_factory=list)


@dataclass
class CandidateMatch:
    """Represents a substitute eligible for an absence, populated with garden metrics.

    Attributes:
        substitute_id: Database identifier of the candidate.
        first_name: First name of the candidate.
        last_name: Last name of the candidate.
        city: City of residence.
        neighborhood: Neighborhood of residence.
        experience_years: Total years of childcare experience.
        certifications: List of relevant certifications.
        distance_km: Commute distance in kilometers to the target kindergarten.
        phone_number: Contact phone number for communication.
    """

    substitute_id: int
    first_name: str
    last_name: str
    city: str
    neighborhood: str
    experience_years: int
    certifications: List[str]
    distance_km: float
    phone_number: str = ""


@dataclass
class ScoringWeights:
    """Heuristic scoring parameters used to evaluate candidate suitability.

    Attributes:
        experience_weight: Multiplier applied per year of assistant experience.
        distance_penalty_weight: Penalty subtracted per kilometer of commute distance.
        delay_penalty_per_minute: Penalty subtracted per minute of late arrival (for dynamic re-ranking).
        base_score: Optional baseline score offset.
    """

    experience_weight: float = 10.0
    distance_penalty_weight: float = 2.0
    delay_penalty_per_minute: float = 0.5
    base_score: float = 0.0


@dataclass
class CandidateConstraint:
    """Constraints and availability feedback extracted from candidate conversations.

    Attributes:
        substitute_id: Database identifier of the candidate reporting constraints.
        delay_minutes: Number of minutes the candidate will arrive after shift start.
        is_available: Boolean flag indicating overall willingness to take the shift.
        notes: Additional qualitative constraints extracted from candidate dialogue.
    """

    substitute_id: int
    delay_minutes: int = 0
    is_available: bool = True
    notes: str = ""


@dataclass
class RankedCandidate:
    """A scored substitute candidate sorted for presentation or placement.

    Attributes:
        candidate: CandidateMatch record containing substitute profile and distance.
        score: Computed heuristic score (higher score indicates better match).
        rank: Relative position in the sorted recommendation list (1 = top match).
        breakdown: Granular scoring components explaining the total score.
    """

    candidate: CandidateMatch
    score: float
    rank: int
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class DialogueContext:
    """Context information for managing arrival confirmation dialogue.

    Attributes:
        candidate_name: Full name of the candidate being contacted.
        garden_name: Name of the kindergarten with the absence.
        shift_date: Date of the shift (YYYY-MM-DD).
        shift_hours: Expected shift hours (e.g., '08:00-14:00').
        candidate_phone: Contact telephone number.
        chat_history: List of conversation messages exchanged so far.
    """

    candidate_name: str
    garden_name: str
    shift_date: str
    shift_hours: str
    candidate_phone: str = ""
    chat_history: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExtractedConstraints:
    """Structured availability status and constraints extracted from dialogue.

    Attributes:
        willingness: Confirmation status ('CONFIRMED', 'DECLINED', 'CONDITIONAL').
        arrival_time: Explicit arrival time if candidate stated a constraint (e.g. '08:30'), or None.
        notes: Summary commentary or explanation from the dialogue.
    """

    willingness: str = "CONFIRMED"
    arrival_time: Optional[str] = None
    notes: str = "Stub confirmation"