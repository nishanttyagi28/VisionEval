"""Tests for P0 cache integration and per-sample regression evidence."""

from pathlib import Path

from visioneval.classification.scorer import evaluate
from visioneval.core.baseline import Baseline, compare_baseline
from visioneval.core.cache import SQLiteCache
from visioneval.core.types import ClassificationPrediction, ClassificationSample, SelectedSample, SelectionReason


def test_prediction_cache_reuses_content_addressable_result(tmp_path: Path) -> None:
    """Second evaluation avoids adapter work for unchanged image/model inputs."""
    image = tmp_path / "image.bin"
    image.write_bytes(b"stable-image-content")
    selected = [SelectedSample(ClassificationSample("one", "cat", 0.8, image_path=str(image)), SelectionReason.RANDOM_COVERAGE, 0.25, "random_coverage")]
    calls = 0

    def adapter(sample: ClassificationSample) -> ClassificationPrediction:
        nonlocal calls
        calls += 1
        return ClassificationPrediction("cat", 0.9)

    cache = SQLiteCache(tmp_path / "cache.sqlite")
    first = evaluate(selected, adapter, cache, "model-v1", "preprocess-v1")
    second = evaluate(selected, adapter, cache, "model-v1", "preprocess-v1")

    assert (first.cache_hits, first.cache_misses) == (0, 1)
    assert (second.cache_hits, second.cache_misses) == (1, 0)
    assert calls == 1


def test_per_sample_comparison_identifies_new_and_fixed_failures() -> None:
    """Regression output distinguishes harmful failures from recovered samples."""
    baseline = Baseline("suite", 0.5, 2, {"new": True, "fixed": False})
    records = (
        _record("new", False),
        _record("fixed", True),
    )

    result = compare_baseline(baseline, 0.5, 0.0, records)

    assert result.new_failures == ("new",)
    assert result.fixed_failures == ("fixed",)
    assert result.is_regression is True


def _record(sample_id: str, correct: bool):
    return __import__("visioneval.core.types", fromlist=["EvaluationRecord"]).EvaluationRecord(sample_id, "cat", "cat" if correct else "dog", 0.9, correct, SelectionReason.HIGH_RISK, 0.75, "high_risk")