"""Tests for failure memory and CI-friendly report outputs."""

from pathlib import Path

from visioneval.classification.scorer import EvaluationSummary
from visioneval.core.baseline import RegressionResult
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


def _record(sample_id: str, correct: bool) -> EvaluationRecord:
    return EvaluationRecord(sample_id, "cat", "cat" if correct else "dog", 0.8, correct, SelectionReason.HIGH_RISK)


def test_failure_memory_keeps_intermittent_failures_in_attention(tmp_path: Path) -> None:
    """One recovery does not drop a sample; two consecutive passes do."""
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    cache.record((_record("sample-1", False),))
    cache.record((_record("sample-1", True),))

    assert cache.previous_failure_ids() == {"sample-1"}

    cache.record((_record("sample-1", True),))

    assert cache.previous_failure_ids() == set()


def test_failure_memory_counts_repeated_failures(tmp_path: Path) -> None:
    """Fail count grows across runs and a later fail resets recovery."""
    cache = SQLiteCache(tmp_path / "cache.sqlite")
    cache.record((_record("sample-1", False),))
    cache.record((_record("sample-1", False),))
    cache.record((_record("sample-1", True),))
    cache.record((_record("sample-1", False),))

    assert cache.previous_failure_ids() == {"sample-1"}
    with cache._connect() as connection:
        fail_count, consecutive_passes, correct = connection.execute("SELECT fail_count, consecutive_passes, correct FROM sample_outcomes WHERE sample_id = ?", ("sample-1",)).fetchone()
    assert (fail_count, consecutive_passes, correct) == (3, 0, 0)


def test_markdown_report_lists_failure_ids_and_attention_buckets(tmp_path: Path) -> None:
    """CI markdown includes sample ids and bucket counts without opening JSON."""
    summary = EvaluationSummary(
        records=(
            EvaluationRecord("new-1", "cat", "dog", 0.8, False, SelectionReason.HIGH_RISK),
            EvaluationRecord("fixed-1", "cat", "cat", 0.8, True, SelectionReason.PREVIOUS_FAILURE),
            EvaluationRecord("ok", "cat", "cat", 0.8, True, SelectionReason.RANDOM_COVERAGE),
        ),
        accuracy=2 / 3,
    )
    regression = RegressionResult(1.0, 2 / 3, 1 / 3, True, ("new-1",), ("fixed-1",))

    write_reports(tmp_path / "report.json", tmp_path / "report.md", "suite", summary, regression)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "- New failures: `1` (`new-1`)" in markdown
    assert "- Fixed failures: `1` (`fixed-1`)" in markdown
    assert "- Attention buckets: `previous_failure` `1`, `high_risk` `1`, `random_coverage` `1`" in markdown