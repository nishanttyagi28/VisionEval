"""Runtime datatypes for deterministic classification evaluation."""

from dataclasses import dataclass
from enum import Enum


class SelectionReason(str, Enum):
    PREVIOUS_FAILURE = "previous_failure"
    HIGH_RISK = "high_risk"
    LOW_CONFIDENCE = "low_confidence"
    RANDOM_COVERAGE = "random_coverage"


@dataclass(frozen=True)
class ClassificationSample:
    sample_id: str
    label: str
    confidence: float
    tags: frozenset[str] = frozenset()
    previous_failure: bool = False
    image_path: str | None = None


@dataclass(frozen=True)
class SelectedSample:
    sample: ClassificationSample
    reason: SelectionReason
    attention_score: float = 0.0
    risk_bucket: str = "unspecified"


@dataclass(frozen=True)
class ClassificationPrediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class EvaluationRecord:
    sample_id: str
    expected_label: str
    predicted_label: str
    confidence: float
    correct: bool
    selection_reason: SelectionReason
    attention_score: float = 0.0
    risk_bucket: str = "unspecified"
    cache_hit: bool = False