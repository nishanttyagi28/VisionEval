"""Explicit end-to-end orchestration for Phase 1 classification suites."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from visioneval.classification.adapter import ClassificationAdapter, load_adapter
from visioneval.classification.scorer import EvaluationSummary, evaluate
from visioneval.core.baseline import Baseline, RegressionResult, compare_baseline, load_baseline, save_baseline, suite_hash
from visioneval.core.cache import SQLiteCache, cache_identities
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
    used_budget: bool = False
    recommended_sample_ids: tuple[str, ...] = ()


def run_suite(
    suite_path: Path,
    update_baseline: bool = False,
    adapter: ClassificationAdapter | None = None,
    *,
    use_budget: bool | None = None,
    traps_db: Path | None = None,
) -> RunResult:
    """Run one suite and emit evidence before any fail-fast return.

    When ``use_budget`` is true (CLI ``--use-budget`` or suite
    ``attention.use_budget``), selection comes from the adaptive budget
    analyzer instead of the attention-fraction sampler. Default behavior is
    unchanged when the flag/option is off.
    """
    suite = load_suite(suite_path)
    cache = SQLiteCache(Path(suite.cache.path))
    catalog = _load_samples(Path(suite.dataset.manifest), cache.previous_failure_ids())
    effective_use_budget = suite.attention.use_budget if use_budget is None else use_budget
    recommended: tuple[str, ...] = ()
    if effective_use_budget:
        from visioneval.core.budget import analyze_suite, default_traps_db, select_budget_samples

        traps_path = traps_db if traps_db is not None else default_traps_db()
        analysis = analyze_suite(suite_path, traps_db=traps_path)
        selected = select_budget_samples(catalog, analysis)
        recommended = analysis.recommended_sample_ids
    else:
        selected = select_samples(catalog, suite.attention)
    identity = (suite_hash(suite_path), suite.model.adapter, suite.attention.seed, suite.attention.budget)
    active_adapter = adapter or load_adapter(suite.model.adapter)
    model_hash, preprocess_hash = cache_identities(suite.model.adapter, active_adapter)
    baseline = None if update_baseline else load_baseline(Path(suite.baseline.path))
    partial = None

    if suite.execution.fail_fast and baseline is not None:
        summary, regression, partial = _evaluate_fail_fast(
            selected, active_adapter, cache, model_hash, preprocess_hash, baseline, suite.baseline.allowed_accuracy_drop, identity
        )
    else:
        summary = evaluate(selected, active_adapter, cache, model_hash, preprocess_hash)
        regression = None if baseline is None else compare_baseline(
            baseline, summary.accuracy, suite.baseline.allowed_accuracy_drop, summary.records, identity
        )

    cache.record(summary.records)
    if update_baseline:
        save_baseline(
            Path(suite.baseline.path),
            Baseline(
                suite.name,
                summary.accuracy,
                len(summary.records),
                {record.sample_id: record.correct for record in summary.records},
                identity[0],
                identity[1],
                identity[2],
                identity[3],
                tuple(sorted(record.sample_id for record in summary.records)),
            ),
        )
    write_reports(
        Path(suite.report.path),
        Path(suite.report.markdown_path) if suite.report.markdown_path else None,
        suite.name,
        summary,
        regression,
        partial,
    )
    return RunResult(suite, summary, regression, partial, effective_use_budget, recommended)


def _evaluate_fail_fast(
    selected, adapter: ClassificationAdapter, cache: SQLiteCache, model_hash: str, preprocess_hash: str, baseline: Baseline, allowed_accuracy_drop: float, identity: tuple[str, str, int, int]
) -> tuple[EvaluationSummary, RegressionResult, dict[str, object] | None]:
    records: list[EvaluationRecord] = []
    hits = misses = 0
    for index, item in enumerate(selected):
        one = evaluate((item,), adapter, cache, model_hash, preprocess_hash)
        records.extend(one.records)
        hits += one.cache_hits
        misses += one.cache_misses
        summary = _summary(records, hits, misses)
        regression = compare_baseline(baseline, summary.accuracy, allowed_accuracy_drop, summary.records, identity)
        if one.records[0].sample_id in regression.new_failures:
            return summary, regression, partial_execution(one.records[0], len(records), len(selected) - index - 1)
    summary = _summary(records, hits, misses)
    return summary, compare_baseline(baseline, summary.accuracy, allowed_accuracy_drop, summary.records, identity), None


def _summary(records: list[EvaluationRecord], hits: int, misses: int) -> EvaluationSummary:
    return EvaluationSummary(tuple(records), sum(record.correct for record in records) / len(records) if records else 0.0, hits, misses)


def _load_samples(path: Path, previous_failures: set[str]) -> list[ClassificationSample]:
    manifest = DatasetManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return [
        ClassificationSample(
            item.sample_id,
            item.label,
            item.confidence,
            frozenset(item.tags),
            item.sample_id in previous_failures,
            item.image_path,
        )
        for item in manifest.samples
    ]
