"""Deterministic attention-based sample selection."""

from collections.abc import Iterable
from math import floor
from random import Random

from visioneval.core.suite import AttentionConfig
from visioneval.core.types import ClassificationSample, SelectedSample, SelectionReason


_REASONS = (SelectionReason.PREVIOUS_FAILURE, SelectionReason.HIGH_RISK, SelectionReason.LOW_CONFIDENCE)
_SCORES = {SelectionReason.PREVIOUS_FAILURE: 1.0, SelectionReason.HIGH_RISK: 0.75, SelectionReason.LOW_CONFIDENCE: 0.5, SelectionReason.RANDOM_COVERAGE: 0.25}


def select_samples(samples: Iterable[ClassificationSample], config: AttentionConfig) -> list[SelectedSample]:
    """Select a seed-stable budget with explicit attention provenance."""
    remaining = sorted(samples, key=lambda sample: sample.sample_id)
    if len({sample.sample_id for sample in remaining}) != len(remaining):
        raise ValueError("sample_id values must be unique")
    budget = min(config.budget, len(remaining))
    quotas = _allocate_quotas(budget, config)
    generator = Random(config.seed)
    selected: list[SelectedSample] = []
    predicates = {
        SelectionReason.PREVIOUS_FAILURE: lambda item: config.previous_failures and item.previous_failure,
        SelectionReason.HIGH_RISK: lambda item: bool(set(item.tags) & set(config.high_risk_tags)),
        SelectionReason.LOW_CONFIDENCE: lambda item: config.low_confidence and item.confidence <= config.low_confidence_threshold,
    }
    for reason in _REASONS:
        chosen = _choose([item for item in remaining if predicates[reason](item)], quotas[reason], generator)
        selected.extend(_trace(item, reason) for item in chosen)
        chosen_ids = {item.sample_id for item in chosen}
        remaining = [item for item in remaining if item.sample_id not in chosen_ids]
    chosen = _choose(remaining, budget - len(selected), generator)
    selected.extend(_trace(item, SelectionReason.RANDOM_COVERAGE) for item in chosen)
    return selected


def _trace(sample: ClassificationSample, reason: SelectionReason) -> SelectedSample:
    return SelectedSample(sample, reason, _SCORES[reason], reason.value)


def _allocate_quotas(budget: int, config: AttentionConfig) -> dict[SelectionReason, int]:
    fractions = {SelectionReason.PREVIOUS_FAILURE: config.previous_failures_fraction, SelectionReason.HIGH_RISK: config.high_risk_fraction, SelectionReason.LOW_CONFIDENCE: config.low_confidence_fraction, SelectionReason.RANDOM_COVERAGE: config.random_coverage_fraction}
    quotas = {reason: floor(budget * fraction) for reason, fraction in fractions.items()}
    for reason in sorted(fractions, key=lambda item: (-fractions[item], item.value))[: budget - sum(quotas.values())]:
        quotas[reason] += 1
    return quotas


def _choose(candidates: list[ClassificationSample], count: int, generator: Random) -> list[ClassificationSample]:
    return generator.sample(candidates, k=min(count, len(candidates))) if count and candidates else []