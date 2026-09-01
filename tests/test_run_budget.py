"""Wire adaptive budget selection into visioneval run."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from typer.testing import CliRunner

from visioneval.cli import app
from visioneval.core.budget import (
    BudgetAnalysis,
    BudgetPolicy,
    RiskFactors,
    RiskWeights,
    ScoredSample,
)
from visioneval.core.budget_select import select_budget_samples
from visioneval.core.runner import run_suite
from visioneval.core.types import ClassificationPrediction, ClassificationSample, SelectionReason

def _cli_text(result) -> str:
    raw = getattr(result, "stdout", None) or result.output or ""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)


def _sample(sample_id: str, *, confidence: float = 0.9, tags: frozenset[str] = frozenset()) -> ClassificationSample:
    return ClassificationSample(sample_id, "cat", confidence, tags)


def test_select_budget_samples_preserves_order_and_reasons() -> None:
    samples = [
        _sample("z-novel"),
        _sample("a-fail"),
        _sample("m-risk", tags=frozenset({"critical"})),
        _sample("b-low", confidence=0.1),
    ]
    scored = (
        ScoredSample("a-fail", "cat", 0.9, (), RiskFactors(1, 0, 0, 0), 0.5),
        ScoredSample("m-risk", "cat", 0.9, ("critical",), RiskFactors(0, 1, 0, 0), 0.2),
        ScoredSample("b-low", "cat", 0.1, (), RiskFactors(0, 0, 1, 0), 0.2),
        ScoredSample("z-novel", "cat", 0.9, (), RiskFactors(0, 0, 0, 1), 0.1),
    )
    analysis = BudgetAnalysis(
        suite_name="t",
        samples=scored,
        recommended_sample_ids=("a-fail", "m-risk", "z-novel"),
        weights=RiskWeights(),
        policy=BudgetPolicy(),
    )
    selected = select_budget_samples(samples, analysis)
    assert [item.sample.sample_id for item in selected] == ["a-fail", "m-risk", "z-novel"]
    assert selected[0].reason == SelectionReason.PREVIOUS_FAILURE
    assert selected[1].reason == SelectionReason.HIGH_RISK
    assert selected[2].reason == SelectionReason.RANDOM_COVERAGE


def _write_suite(tmp_path: Path, *, use_budget: bool = False, budget: int = 2) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "samples": [
                    {"id": "one", "label": "cat", "confidence": 0.9, "tags": ["safety_critical"]},
                    {"id": "two", "label": "cat", "confidence": 0.2},
                    {"id": "three", "label": "cat", "confidence": 0.9},
                    {"id": "four", "label": "cat", "confidence": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    attention = {
        "budget": budget,
        "seed": 3,
        "high_risk_tags": ["safety_critical"],
        "use_budget": use_budget,
    }
    suite_path.write_text(
        yaml.safe_dump(
            {
                "name": "budget-run",
                "task": "image_classification",
                "model": {"adapter": "unused:predict"},
                "dataset": {"manifest": str(manifest)},
                "attention": attention,
                "baseline": {"path": str(tmp_path / "baseline.json")},
                "cache": {"path": str(tmp_path / "cache.sqlite")},
                "report": {"path": str(tmp_path / "report.json")},
            }
        ),
        encoding="utf-8",
    )
    return suite_path


def test_run_suite_use_budget_selects_recommended_set(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path, use_budget=False, budget=2)

    def adapter(sample: ClassificationSample) -> ClassificationPrediction:
        return ClassificationPrediction(sample.label, 0.9)

    default = run_suite(suite_path, update_baseline=True, adapter=adapter, use_budget=False)
    assert default.used_budget is False
    assert len(default.summary.records) == 2

    budgeted = run_suite(suite_path, update_baseline=True, adapter=adapter, use_budget=True)
    assert budgeted.used_budget is True
    assert budgeted.recommended_sample_ids
    assert len(budgeted.summary.records) == len(budgeted.recommended_sample_ids)
    assert {record.sample_id for record in budgeted.summary.records} == set(budgeted.recommended_sample_ids)


def test_suite_yaml_use_budget_option(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path, use_budget=True, budget=2)

    def adapter(sample: ClassificationSample) -> ClassificationPrediction:
        return ClassificationPrediction(sample.label, 0.9)

    result = run_suite(suite_path, update_baseline=True, adapter=adapter)
    assert result.used_budget is True
    assert result.recommended_sample_ids


def test_run_cli_use_budget_flag(tmp_path: Path) -> None:
    suite_path = _write_suite(tmp_path, use_budget=False, budget=2)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "from visioneval.core.types import ClassificationPrediction\n"
        "def predict(sample):\n"
        "    return ClassificationPrediction(sample.label, 0.9)\n",
        encoding="utf-8",
    )
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    payload["model"]["adapter"] = f"{adapter.stem}:predict"
    payload["model"]["adapter"] = "demo_adapter:predict"
    suite_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    help_text = _cli_text(result)
    assert "--use-budget" in help_text
    assert "--traps-db" in help_text
