"""Tests for the adaptive evaluation budget analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from visioneval.cli import app
from visioneval.core.budget import (
    BudgetPolicy,
    RiskFactors,
    RiskWeights,
    analyze_samples,
    analyze_suite,
    format_human,
    score_sample,
    to_json,
)
from visioneval.core.cache import SQLiteCache
from visioneval.core.types import ClassificationSample, EvaluationRecord, SelectionReason
from visioneval.traps.store import TrapStore
from visioneval.traps.types import TrapRecord, utc_now


def _sample(
    sample_id: str,
    *,
    confidence: float = 0.9,
    tags: frozenset[str] = frozenset(),
    label: str = "cat",
) -> ClassificationSample:
    return ClassificationSample(sample_id, label, confidence, tags)


def test_risk_score_uses_documented_default_weights() -> None:
    """All four flags on: 0.5 + 0.2 + 0.2 + 0.1 = 1.0."""
    factors = RiskFactors(1, 1, 1, 1)
    assert score_sample(factors) == pytest.approx(1.0)
    assert score_sample(RiskFactors(1, 0, 0, 0)) == pytest.approx(0.5)
    assert score_sample(RiskFactors(0, 1, 0, 0)) == pytest.approx(0.2)
    assert score_sample(RiskFactors(0, 0, 1, 0)) == pytest.approx(0.2)
    assert score_sample(RiskFactors(0, 0, 0, 1)) == pytest.approx(0.1)
    assert score_sample(RiskFactors(0, 0, 0, 0)) == pytest.approx(0.0)


def test_risk_score_weights_are_configurable() -> None:
    weights = RiskWeights(previous_failure=1.0, high_risk_tag=0.0, low_confidence=0.0, novelty=0.0)
    assert score_sample(RiskFactors(1, 1, 1, 1), weights) == pytest.approx(1.0)


def test_rank_samples_is_deterministic_and_ignores_input_order() -> None:
    samples = [
        _sample("z-novel", confidence=0.9),
        _sample("a-fail", confidence=0.9),
        _sample("m-risk", confidence=0.9, tags=frozenset({"critical"})),
        _sample("b-low", confidence=0.1),
        _sample("a-fail-twin", confidence=0.9),
    ]
    first = analyze_samples(
        samples,
        suite_name="t",
        high_risk_tags=["critical"],
        low_confidence_threshold=0.5,
        previous_failure_ids={"a-fail", "a-fail-twin"},
        known_sample_ids={"a-fail", "a-fail-twin", "m-risk", "b-low"},
    )
    second = analyze_samples(
        list(reversed(samples)),
        suite_name="t",
        high_risk_tags=["critical"],
        low_confidence_threshold=0.5,
        previous_failure_ids={"a-fail", "a-fail-twin"},
        known_sample_ids={"a-fail", "a-fail-twin", "m-risk", "b-low"},
    )
    assert [item.sample_id for item in first.ranked_samples()] == [
        item.sample_id for item in second.ranked_samples()
    ]
    ranked_ids = [item.sample_id for item in first.ranked_samples()]
    # previous_failure 0.5 outranks high-risk 0.2 and low-confidence 0.2;
    # equal scores break ties by sample_id.
    assert ranked_ids[0] == "a-fail"
    assert ranked_ids[1] == "a-fail-twin"
    assert first.ranked_samples()[0].risk_score >= first.ranked_samples()[-1].risk_score


def test_budget_includes_all_previous_failures_then_coverage_floor() -> None:
    samples = [
        _sample("fail-1"),
        _sample("fail-2"),
        _sample("risk-1", tags=frozenset({"safety_critical"})),
        _sample("risk-2", tags=frozenset({"safety_critical"})),
        _sample("low-1", confidence=0.2),
        *(_sample(f"ok-{index}") for index in range(6)),
    ]
    analysis = analyze_samples(
        samples,
        suite_name="floor",
        high_risk_tags=["safety_critical"],
        low_confidence_threshold=0.5,
        previous_failure_ids={"fail-1", "fail-2"},
        known_sample_ids={"fail-1", "fail-2"},
        policy=BudgetPolicy(novelty_fraction=0.15, min_novelty_count=1, max_budget=100),
    )
    recommended = set(analysis.recommended_sample_ids)
    assert {"fail-1", "fail-2"} <= recommended
    assert recommended & {"risk-1", "risk-2"}
    assert analysis.recommended_budget < analysis.total_sample_count
    assert analysis.estimated_reduction_pct > 0
    # Previous failures come first in selection order, then highest remaining risk.
    assert analysis.recommended_sample_ids[:2] == ("fail-1", "fail-2")


def test_recommend_budget_respects_max_budget_after_failures() -> None:
    samples = [
        _sample("fail-1"),
        _sample("risk-1", tags=frozenset({"safety_critical"})),
        _sample("novel-1"),
    ]
    scored = analyze_samples(
        samples,
        suite_name="cap",
        high_risk_tags=["safety_critical"],
        low_confidence_threshold=0.5,
        previous_failure_ids={"fail-1"},
        known_sample_ids={"fail-1"},
        policy=BudgetPolicy(max_budget=1, min_novelty_count=1, novelty_fraction=0.5),
    )
    # Previous failure always fits even when it saturates max_budget.
    assert scored.recommended_sample_ids == ("fail-1",)


def test_cli_human_and_json_on_demo_suite() -> None:
    runner = CliRunner()
    human = runner.invoke(app, ["budget-analyze", "demo_suite.yaml"])
    assert human.exit_code == 0, human.stdout
    assert "VisionEval budget analysis: phase1-demo" in human.stdout
    assert "Total samples:" in human.stdout
    assert "Previous failures:" in human.stdout
    assert "High-risk:" in human.stdout
    assert "Low-confidence:" in human.stdout
    assert "Recommended budget:" in human.stdout
    assert "Estimated reduction:" in human.stdout
    assert "Top-risk samples:" in human.stdout

    encoded = runner.invoke(app, ["budget-analyze", "demo_suite.yaml", "--json"])
    assert encoded.exit_code == 0, encoded.stdout
    payload = json.loads(encoded.stdout)
    assert payload["suite"] == "phase1-demo"
    assert payload["total_sample_count"] == 3
    assert payload["previous_failure_count"] == 0
    assert payload["high_risk_count"] == 1
    assert payload["low_confidence_count"] == 1
    assert payload["recommended_budget"] >= 1
    assert "estimated_reduction_pct" in payload
    assert payload["top_risk_samples"]
    assert payload["weights"]["previous_failure"] == 0.5


def test_cli_json_matches_formatters() -> None:
    analysis = analyze_suite(Path("demo_suite.yaml"))
    assert to_json(analysis) == json.dumps(analysis.as_dict(), indent=2, sort_keys=True) + "\n"
    text = format_human(analysis)
    assert str(analysis.total_sample_count) in text
    assert f"{analysis.estimated_reduction_pct:.1f}%" in text


def test_analyze_suite_uses_cache_and_open_traps(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {"id": "alpha", "label": "cat", "confidence": 0.9, "tags": ["safety_critical"]},
                    {"id": "beta", "label": "cat", "confidence": 0.2},
                    {"id": "gamma", "label": "cat", "confidence": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    cache_path = tmp_path / "cache.sqlite3"
    cache = SQLiteCache(cache_path)
    cache.record(
        (
            EvaluationRecord(
                "alpha", "cat", "dog", 0.8, False, SelectionReason.PREVIOUS_FAILURE
            ),
        )
    )
    traps_path = tmp_path / "traps.sqlite3"
    store = TrapStore(traps_path)
    now = utc_now()
    store.upsert_failure(
        TrapRecord(
            trap_id="pope:fake:gamma:q",
            model="fake",
            sample_id="gamma",
            image_hash="",
            prompt="",
            expected_objects=(),
            absent_objects=(),
            probe_type="pope",
            last_outcome="fail",
            fail_count=1,
            consecutive_passes=0,
            created_at=now,
            updated_at=now,
            retired=False,
        )
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "memory",
                "task": "image_classification",
                "model": {"adapter": "unused:predict"},
                "dataset": {"manifest": str(manifest)},
                "attention": {"budget": 10, "seed": 1, "high_risk_tags": ["safety_critical"]},
                "baseline": {"path": str(tmp_path / "base.json")},
                "cache": {"path": str(cache_path)},
                "report": {"path": str(tmp_path / "report.json")},
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_suite(suite_path, traps_db=traps_path)
    by_id = {item.sample_id: item for item in analysis.samples}
    assert by_id["alpha"].previous_failure is True
    assert by_id["alpha"].novelty is False
    assert by_id["gamma"].previous_failure is True  # open trap
    assert by_id["gamma"].novelty is True  # not in classification cache
    assert by_id["beta"].low_confidence is True
    assert "alpha" in analysis.recommended_sample_ids
    assert "gamma" in analysis.recommended_sample_ids


def test_example_classification_suite_is_analyzable() -> None:
    analysis = analyze_suite(Path("examples/classification_suite/suite.yaml"))
    assert analysis.total_sample_count == 12
    assert analysis.high_risk_count == 2
    assert analysis.low_confidence_count == 2
    assert analysis.recommended_budget < analysis.total_sample_count
    runner = CliRunner()
    result = runner.invoke(app, ["budget-analyze", "examples/classification_suite/suite.yaml", "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["total_sample_count"] == 12


def test_existing_run_command_still_scaffolds() -> None:
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code == 0
    assert "scaffold is ready" in result.stdout
