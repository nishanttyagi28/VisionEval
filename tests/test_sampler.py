"""Tests for deterministic, traceable attention sampling."""

from visioneval.core.sampler import select_samples
from visioneval.core.suite import AttentionConfig
from visioneval.core.types import ClassificationSample, SelectionReason


def test_attention_sampler_uses_target_allocation_and_reasons() -> None:
    """Each target bucket receives its configured share when available."""
    samples = [
        *(ClassificationSample(f"failure-{index}", "cat", 0.9, previous_failure=True) for index in range(40)),
        *(ClassificationSample(f"risk-{index}", "cat", 0.9, frozenset({"critical"})) for index in range(30)),
        *(ClassificationSample(f"low-{index}", "cat", 0.1) for index in range(15)),
        *(ClassificationSample(f"random-{index}", "cat", 0.9) for index in range(15)),
    ]
    config = AttentionConfig(budget=100, high_risk_tags=["critical"], seed=17)

    selected = select_samples(reversed(samples), config)
    reasons = [item.reason for item in selected]

    assert len(selected) == 100
    assert reasons.count(SelectionReason.PREVIOUS_FAILURE) == 40
    assert reasons.count(SelectionReason.HIGH_RISK) == 30
    assert reasons.count(SelectionReason.LOW_CONFIDENCE) == 15
    assert reasons.count(SelectionReason.RANDOM_COVERAGE) == 15
    assert len({item.sample.sample_id for item in selected}) == 100


def test_attention_sampler_is_seeded_and_order_independent() -> None:
    """The same inputs and seed always produce the same trace."""
    samples = [ClassificationSample(f"sample-{index}", "cat", 0.9) for index in range(20)]
    config = AttentionConfig(budget=5, seed=23)

    first = select_samples(samples, config)
    second = select_samples(reversed(samples), config)

    assert first == second
    assert {item.reason for item in first} == {SelectionReason.RANDOM_COVERAGE}


def test_unused_attention_quota_cascades_before_random() -> None:
    """Empty higher buckets donate leftover budget to the next attention bucket."""
    samples = [
        *(ClassificationSample(f"risk-{index}", "cat", 0.9, frozenset({"critical"})) for index in range(70)),
        *(ClassificationSample(f"low-{index}", "cat", 0.1) for index in range(15)),
        *(ClassificationSample(f"random-{index}", "cat", 0.9) for index in range(15)),
    ]
    config = AttentionConfig(budget=100, high_risk_tags=["critical"], seed=17)

    reasons = [item.reason for item in select_samples(samples, config)]

    assert reasons.count(SelectionReason.PREVIOUS_FAILURE) == 0
    assert reasons.count(SelectionReason.HIGH_RISK) == 70
    assert reasons.count(SelectionReason.LOW_CONFIDENCE) == 15
    assert reasons.count(SelectionReason.RANDOM_COVERAGE) == 15