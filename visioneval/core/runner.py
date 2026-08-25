"""Explicit end-to-end orchestration for Phase 1 classification suites."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from visioneval.classification.adapter import ClassificationAdapter, load_adapter
from visioneval.classification.scorer import EvaluationSummary, evaluate
from visioneval.core.baseline import Baseline, RegressionResult, compare_baseline, load_baseline, save_baseline
from visioneval.core.cache import SQLiteCache
from visioneval.core.report import partial_execution, write_reports
from visioneval.core.sampler import select_samples
from visioneval.core.suite import SuiteConfig, load_suite
from visioneval.core.types import ClassificationSample, EvaluationRecord


class ManifestSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str = Field(alias="id", min_length=1)
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    image_path: str | None = None


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    samples: list[ManifestSample]


@dataclass(frozen=True)
class RunResult:
    suite: SuiteConfig
    summary: EvaluationSummary
    regression: RegressionResult | None
    partial: dict[str, object] | None = None


def run_suite(suite_path: Path, update_baseline: bool = False, adapter: ClassificationAdapter | None = None) -> RunResult:
    """Run one suite and emit evidence before any fail-fast return."""
    suite = load_suite(suite_path)
    cache = SQLiteCache(Path(suite.cache.path))
    selected = select_samples(_load_samples(Path(suite.dataset.manifest), cache.previous_failure_ids()), suite.attention)
    model_hash = hashlib.sha256(suite.model.adapter.encode("utf-8")).hexdigest()
    active_adapter = adapter or load_adapter(suite.model.adapter)
    baseline = None if update_baseline else load_baseline(Path(suite.baseline.path))
    partial = None

    if suite.execution.fail_fast and baseline is not None:
        summary, regression, partial = _evaluate_fail_fast(selected, active_adapter, cache, model_hash, baseline, suite.baseline.allowed_accuracy_drop)
    else:
        summary = evaluate(selected, active_adapter, cache, model_hash, "phase1")
        regression = None if baseline is None else compare_baseline(baseline, summary.accuracy, suite.baseline.allowed_accuracy_drop, summary.records)

    cache.record(summary.records)
    if update_baseline:
        save_baseline(Path(suite.baseline.path), Baseline(suite.name, summary.accuracy, len(summary.records), {record.sample_id: record.correct for record in summary.records}))
    write_reports(Path(suite.report.path), Path(suite.report.markdown_path) if suite.report.markdown_path else None, suite.name, summary, regression, partial)
    return RunResult(suite, summary, regression, partial)


def _evaluate_fail_fast(selected, adapter: ClassificationAdapter, cache: SQLiteCache, model_hash: str, baseline: Baseline, allowed_accuracy_drop: float) -> tuple[EvaluationSummary, RegressionResult, dict[str, object] | None]:
    records: list[EvaluationRecord] = []
    hits = misses = 0
    for index, item in enumerate(selected):
        one = evaluate((item,), adapter, cache, model_hash, "phase1")
        records.extend(one.records)
        hits += one.cache_hits
        misses += one.cache_misses
        summary = _summary(records, hits, misses)
        regression = compare_baseline(baseline, summary.accuracy, allowed_accuracy_drop, summary.records)
        if one.records[0].sample_id in regression.new_failures:
            return summary, regression, partial_execution(one.records[0], len(records), len(selected) - index - 1)
    summary = _summary(records, hits, misses)
    return summary, compare_baseline(baseline, summary.accuracy, allowed_accuracy_drop, summary.records), None


def _summary(records: list[EvaluationRecord], hits: int, misses: int) -> EvaluationSummary:
    return EvaluationSummary(tuple(records), sum(record.correct for record in records) / len(records) if records else 0.0, hits, misses)


def _load_samples(path: Path, previous_failures: set[str]) -> list[ClassificationSample]:
    manifest = DatasetManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return [ClassificationSample(item.sample_id, item.label, item.confidence, frozenset(item.tags), item.sample_id in previous_failures, item.image_path) for item in manifest.samples]