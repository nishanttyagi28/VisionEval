"""Materialize budget-analyzer recommendations as SelectedSample rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from visioneval.core.types import ClassificationSample, SelectedSample, SelectionReason

if TYPE_CHECKING:
    from visioneval.core.budget import BudgetAnalysis


def select_budget_samples(
    samples: Sequence[ClassificationSample],
    analysis: BudgetAnalysis,
) -> list[SelectedSample]:
    """Materialize the budget analyzer recommendation as SelectedSample rows.

    Highest matching risk factor wins the selection reason (previous failure,
    then high-risk tag, low confidence, else random/novelty coverage).
    Order follows ``recommended_sample_ids`` (deterministic).
    """
    by_id = {sample.sample_id: sample for sample in samples}
    scored = {item.sample_id: item for item in analysis.samples}
    selected: list[SelectedSample] = []
    for sample_id in analysis.recommended_sample_ids:
        sample = by_id.get(sample_id)
        if sample is None:
            continue
        item = scored.get(sample_id)
        if item is not None and item.previous_failure:
            reason = SelectionReason.PREVIOUS_FAILURE
            score = 1.0
        elif item is not None and item.high_risk:
            reason = SelectionReason.HIGH_RISK
            score = 0.75
        elif item is not None and item.low_confidence:
            reason = SelectionReason.LOW_CONFIDENCE
            score = 0.5
        else:
            reason = SelectionReason.RANDOM_COVERAGE
            score = 0.25
        selected.append(SelectedSample(sample, reason, score, reason.value))
    return selected
