"""Tests for P0 cache integration and per-sample regression evidence."""

from pathlib import Path

from visioneval.classification.scorer import evaluate
from visioneval.core.baseline import Baseline, compare_baseline
from visioneval.core.cache import SQLiteCache, cache_identities
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


def test_prediction_cache_misses_when_model_identity_changes(tmp_path: Path) -> None:
    """A different model hash does not reuse the previous prediction."""
    image = tmp_path / "image.bin"
    image.write_bytes(b"stable-image-content")
    selected = [SelectedSample(ClassificationSample("one", "cat", 0.8, image_path=str(image)), SelectionReason.RANDOM_COVERAGE, 0.25, "random_coverage")]
    calls = 0

    def adapter(sample: ClassificationSample) -> ClassificationPrediction:
        nonlocal calls
        calls += 1
        return ClassificationPrediction("cat", 0.9)

    cache = SQLiteCache(tmp_path / "cache.sqlite")
    evaluate(selected, adapter, cache, "model-v1", "preprocess-v1")
    changed = evaluate(selected, adapter, cache, "model-v2", "preprocess-v1")

    assert (changed.cache_hits, changed.cache_misses) == (0, 1)
    assert calls == 2


def test_cache_identities_change_with_weights_and_preprocess(tmp_path: Path) -> None:
    """Weight bytes and preprocess fields are part of cache identity."""
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"model-a")

    class Adapter:
        def __init__(self, path: str, input_size: int) -> None:
            self.model_path = path
            self.input_size = input_size

    first = cache_identities("pkg:predict", Adapter(str(weights), 224))
    weights.write_bytes(b"model-b")
    second = cache_identities("pkg:predict", Adapter(str(weights), 224))
    third = cache_identities("pkg:predict", Adapter(str(weights), 256))
    other_spec = cache_identities("pkg:other", Adapter(str(weights), 256))

    assert first[0] != second[0]
    assert second[1] != third[1]
    assert third[0] != other_spec[0]


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