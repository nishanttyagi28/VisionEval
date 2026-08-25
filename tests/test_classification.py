"""Tests for explicit classification evaluation outcomes."""

from visioneval.classification.scorer import evaluate
from visioneval.core.types import (
    ClassificationPrediction,
    ClassificationSample,
    SelectedSample,
    SelectionReason,
)


def test_evaluate_scores_predictions_and_preserves_selection_reason() -> None:
    """Evaluator returns both regression-relevant accuracy and sample evidence."""
    selected = [
        SelectedSample(
            ClassificationSample("correct", "cat", 0.8), SelectionReason.PREVIOUS_FAILURE
        ),
        SelectedSample(
            ClassificationSample("incorrect", "dog", 0.8), SelectionReason.HIGH_RISK
        ),
    ]

    def adapter(sample: ClassificationSample) -> ClassificationPrediction:
        label = "cat" if sample.sample_id == "correct" else "cat"
        return ClassificationPrediction(label=label, confidence=0.9)

    summary = evaluate(selected, adapter)

    assert summary.accuracy == 0.5
    assert summary.records[0].correct is True
    assert summary.records[1].correct is False
    assert summary.records[1].selection_reason is SelectionReason.HIGH_RISK