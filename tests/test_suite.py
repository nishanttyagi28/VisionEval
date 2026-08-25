"""Tests for strict suite validation and YAML loading."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from visioneval.core.suite import AttentionConfig, load_suite


def test_load_suite_validates_example() -> None:
    """The example suite remains a valid Phase 1 contract."""
    suite = load_suite(Path("examples/classification_suite/suite.yaml"))

    assert suite.task == "image_classification"
    assert suite.attention.budget == 100


def test_attention_fractions_must_sum_to_one() -> None:
    """Allocation validation prevents ambiguous evaluation budgets."""
    with pytest.raises(ValidationError, match="sum to 1.0"):
        AttentionConfig(random_coverage_fraction=0.10)