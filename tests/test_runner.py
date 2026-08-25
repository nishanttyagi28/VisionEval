"""Integration test for the concrete local evaluation loop."""

import json
from pathlib import Path

import yaml

from visioneval.core.baseline import suite_hash
from visioneval.core.runner import run_suite
from visioneval.core.types import ClassificationPrediction, ClassificationSample


def test_runner_writes_baseline_reports_and_compares_candidate(tmp_path: Path) -> None:
    """A suite runs locally from YAML through baseline comparison."""
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"samples": [{"id": "one", "label": "cat", "confidence": 0.9}]}), encoding="utf-8")
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(yaml.safe_dump({"name": "test", "task": "image_classification", "model": {"adapter": "unused:predict"}, "dataset": {"manifest": str(manifest)}, "attention": {"budget": 1, "seed": 3}, "baseline": {"path": str(tmp_path / "baseline.json")}, "cache": {"path": str(tmp_path / "cache.sqlite")}, "report": {"path": str(tmp_path / "report.json")}}), encoding="utf-8")

    def adapter(sample: ClassificationSample) -> ClassificationPrediction:
        return ClassificationPrediction("cat", 0.9)

    run_suite(suite_path, update_baseline=True, adapter=adapter)
    result = run_suite(suite_path, adapter=adapter)
    baseline = json.loads((tmp_path / "baseline.json").read_text(encoding="utf-8"))

    assert result.regression is not None
    assert result.regression.is_regression is False
    assert (tmp_path / "report.json").exists()
    assert baseline["suite_hash"] == suite_hash(suite_path)
    assert baseline["model_id"] == "unused:predict"
    assert baseline["attention_seed"] == 3
    assert baseline["budget"] == 1
    assert baseline["selected_sample_ids"] == ["one"]