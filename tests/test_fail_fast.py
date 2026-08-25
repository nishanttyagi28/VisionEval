"""Integration tests for deterministic sequential fail-fast execution."""

import json
from pathlib import Path

import yaml

from visioneval.core.runner import run_suite
from visioneval.core.types import ClassificationPrediction, ClassificationSample


def test_fail_fast_stops_on_new_failure_and_writes_partial_report(tmp_path: Path) -> None:
    """A definite new failure ends execution after one deterministic sample."""
    suite_path = _write_suite(tmp_path)

    run_suite(suite_path, update_baseline=True, adapter=_predict("cat"))
    calls: list[str] = []

    def failing_adapter(sample: ClassificationSample) -> ClassificationPrediction:
        calls.append(sample.sample_id)
        return ClassificationPrediction("dog", 0.9)

    result = run_suite(suite_path, adapter=failing_adapter)
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    assert result.regression is not None and result.regression.is_regression
    assert result.partial is not None
    assert result.partial["evaluated_count"] == 1
    assert result.partial["remaining_count"] == 2
    assert len(calls) == 1
    assert report["partial"] == result.partial
    assert report["partial"]["failing_sample"]["selection_reason"] == "random_coverage"
    assert report["cache"] == {"hits": 0, "misses": 1}


def test_fail_fast_selection_and_partial_evidence_are_repeatable(tmp_path: Path) -> None:
    """Identical suite state and seed produce identical termination evidence."""
    suite_path = _write_suite(tmp_path)
    run_suite(suite_path, update_baseline=True, adapter=_predict("cat"))

    first = run_suite(suite_path, adapter=_predict("dog"))
    (tmp_path / "cache.sqlite").unlink()
    second = run_suite(suite_path, adapter=_predict("dog"))

    assert first.partial == second.partial
    assert first.summary.records == second.summary.records


def test_fail_fast_runs_all_samples_when_no_new_failure(tmp_path: Path) -> None:
    """Fail-fast does not truncate a passing candidate evaluation."""
    suite_path = _write_suite(tmp_path)
    run_suite(suite_path, update_baseline=True, adapter=_predict("cat"))

    result = run_suite(suite_path, adapter=_predict("cat"))

    assert result.partial is None
    assert len(result.summary.records) == 3
    assert result.regression is not None and not result.regression.is_regression


def _write_suite(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"samples": [{"id": sample_id, "label": "cat", "confidence": 0.9} for sample_id in ("one", "two", "three")]}), encoding="utf-8")
    suite = {"name": "fail-fast", "task": "image_classification", "model": {"adapter": "unused:predict"}, "dataset": {"manifest": str(manifest)}, "attention": {"budget": 3, "seed": 7}, "baseline": {"path": str(tmp_path / "baseline.json")}, "cache": {"path": str(tmp_path / "cache.sqlite")}, "report": {"path": str(tmp_path / "report.json")}, "execution": {"fail_fast": True}}
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(suite), encoding="utf-8")
    return path


def _predict(label: str):
    def adapter(sample: ClassificationSample) -> ClassificationPrediction:
        return ClassificationPrediction(label, 0.9)
    return adapter