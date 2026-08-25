"""Tests for reproducible baseline comparison."""

from visioneval.core.baseline import Baseline, compare_baseline, load_baseline, save_baseline
from visioneval.core.types import EvaluationRecord, SelectionReason


def test_baseline_round_trip_and_regression_decision(tmp_path) -> None:
    """Baseline JSON persists stably and catches an excessive accuracy drop."""
    path = tmp_path / "baseline.json"
    baseline = Baseline(suite_name="cats", accuracy=0.95, sample_count=100)

    save_baseline(path, baseline)
    comparison = compare_baseline(load_baseline(path), 0.92, allowed_accuracy_drop=0.02)

    assert comparison.accuracy_drop == 0.03
    assert comparison.is_regression is True


def test_comparison_uses_baseline_current_overlap_only() -> None:
    """Attention-only samples do not change accuracy drop or new-failure calls."""
    baseline = Baseline("suite", 1.0, 2, {"locked-pass": True, "locked-fail": False})
    records = (
        EvaluationRecord("locked-pass", "cat", "cat", 0.9, True, SelectionReason.HIGH_RISK),
        EvaluationRecord("attention-only", "cat", "dog", 0.9, False, SelectionReason.PREVIOUS_FAILURE),
    )

    result = compare_baseline(baseline, 0.5, 0.0, records)

    assert result.baseline_accuracy == 1.0
    assert result.candidate_accuracy == 1.0
    assert result.accuracy_drop == 0.0
    assert result.new_failures == ()
    assert result.is_regression is False


def test_disjoint_selection_is_regression() -> None:
    """A run that never re-evaluates locked samples cannot pass the gate."""
    baseline = Baseline("suite", 1.0, 1, {"locked": True})
    records = (EvaluationRecord("other", "cat", "cat", 0.9, True, SelectionReason.RANDOM_COVERAGE),)

    result = compare_baseline(baseline, 1.0, 0.0, records)

    assert result.new_failures == ()
    assert result.is_regression is True