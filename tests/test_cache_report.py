"""Tests for failure memory and CI-friendly report outputs."""

from pathlib import Path

from visioneval.classification.scorer import EvaluationSummary
from visioneval.core.cache import SQLiteCache
from visioneval.core.report import write_reports
from visioneval.core.types import EvaluationRecord, SelectionReason


def test_cache_remembers_latest_failure_and_reports_are_written(tmp_path: Path) -> None:
    """A failed sample becomes attention input for the following run."""
    record = EvaluationRecord("sample-1", "cat", "dog", 0.8, False, SelectionReason.HIGH_RISK)
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    cache.record((record,))
    summary = EvaluationSummary(records=(record,), accuracy=0.0)

    write_reports(tmp_path / "report.json", tmp_path / "report.md", "suite", summary, None)

    assert cache.previous_failure_ids() == {"sample-1"}
    assert '"sample_id": "sample-1"' in (tmp_path / "report.json").read_text(encoding="utf-8")
    assert "BASELINE UPDATED" in (tmp_path / "report.md").read_text(encoding="utf-8")