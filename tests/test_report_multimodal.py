"""Markdown / JSON report serializers for the multimodal layer."""

import json
from pathlib import Path

from visioneval.report.serializers import report_to_json, report_to_markdown, write_multimodal_reports


def _payload() -> dict:
    return {
        "name": "unit-demo",
        "models": [{"name": "fake-a", "kind": "fake"}],
        "samples": [
            {
                "sample_id": "red_square",
                "model": "fake-a",
                "corruption": None,
                "severity": 0.0,
                "response": "A red square on a white background.",
                "profile": {"ttft_ms": 1.5, "total_ms": 2.0, "vram_mb": None, "throughput_tps": 10.0},
                "metrics": {"clip_score": {"name": "clip_score", "value": 1.25}},
                "pope": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
                "judge": {"detail_richness": 0.7, "factual_consistency": 0.8, "spatial_accuracy": 0.6},
            }
        ],
        "degradation": [
            {"metric": "clip_score", "corruption": "gaussian_noise", "clean_score": 1.25, "resilience": 0.9}
        ],
    }


def test_report_to_json_is_sorted_and_round_trippable() -> None:
    blob = report_to_json(_payload())
    loaded = json.loads(blob)
    assert loaded["name"] == "unit-demo"
    assert blob.endswith("\n")
    assert blob.index('"degradation"') < blob.index('"name"')  # sort_keys


def test_report_to_markdown_lists_metrics_pope_and_degradation() -> None:
    md = report_to_markdown(_payload())
    assert "# VisionEval multimodal: unit-demo" in md
    assert "`fake-a`" in md
    assert "POPE" in md
    assert "Judge" in md
    assert "clip_score" in md
    assert "gaussian_noise" in md
    assert "Phase 1 classification CI harness" in md


def test_write_multimodal_reports_creates_files(tmp_path: Path) -> None:
    json_path = tmp_path / "out" / "eval.json"
    md_path = tmp_path / "out" / "eval.md"
    write_multimodal_reports(_payload(), json_path, md_path)
    assert json_path.is_file()
    assert md_path.is_file()
    assert "unit-demo" in md_path.read_text(encoding="utf-8")
