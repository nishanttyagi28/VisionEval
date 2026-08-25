"""Tests for reproducible baseline comparison."""

from visioneval.core.baseline import Baseline, compare_baseline, load_baseline, save_baseline


def test_baseline_round_trip_and_regression_decision(tmp_path) -> None:
    """Baseline JSON persists stably and catches an excessive accuracy drop."""
    path = tmp_path / "baseline.json"
    baseline = Baseline(suite_name="cats", accuracy=0.95, sample_count=100)

    save_baseline(path, baseline)
    comparison = compare_baseline(load_baseline(path), 0.92, allowed_accuracy_drop=0.02)

    assert comparison.accuracy_drop == 0.03
    assert comparison.is_regression is True