"""Black-box hallucination maps: report + traps DB aggregation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from visioneval.cli import app
from visioneval.maps.hallucination import build_map, events_from_report, to_json
from visioneval.traps.harvest import harvest_report
from visioneval.traps.store import TrapStore


def _cli_text(result) -> str:
    raw = getattr(result, "stdout", None) or result.output or ""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)


def _pope_row(*, correct_present: bool = False) -> dict:
    return {
        "sample_id": "red_square",
        "model": "fake-hallucinator",
        "corruption": None,
        "severity": 0.0,
        "prompt": "Describe the image in detail.",
        "response": "A shape.",
        "color": "red_square",
        "objects": ["square"],
        "absent_objects": ["cat"],
        "pope": {
            "accuracy": 0.5,
            "f1": 0.0,
            "total": 2,
            "probes": [
                {
                    "object": "square",
                    "expected_present": True,
                    "prompt": "Is there a square in the image?",
                    "answer": "No." if not correct_present else "Yes.",
                    "predicted": correct_present,
                    "correct": correct_present,
                },
                {
                    "object": "cat",
                    "expected_present": False,
                    "prompt": "Is there a cat in the image?",
                    "answer": "No.",
                    "predicted": False,
                    "correct": True,
                },
            ],
        },
        "judge": {
            "detail_richness": 0.1,
            "factual_consistency": 0.2,
            "spatial_accuracy": 0.3,
            "flags": ["too_short", "missing_objects:square"],
        },
    }


def test_events_from_report_covers_pope_judge_caption() -> None:
    events = events_from_report({"samples": [_pope_row()]})
    metrics = {event.metric for event in events}
    assert "pope_miss" in metrics
    assert "judge_flag" in metrics
    assert "caption_mismatch" in metrics
    assert any(event.object_name == "square" for event in events)


def test_build_map_aggregates_deterministically(tmp_path: Path) -> None:
    report = {"samples": [_pope_row()]}
    first = build_map(report)
    second = build_map(report)
    assert first.as_dict() == second.as_dict()
    assert first.by_metric["pope_miss"] >= 1
    assert first.by_sample_id["red_square"] >= 1
    assert first.by_probe_type["pope"] >= 1
    assert "square" in first.by_object


def test_build_map_merges_traps_db(tmp_path: Path) -> None:
    db = tmp_path / "traps.sqlite3"
    harvest_report({"samples": [_pope_row()]}, TrapStore(db))
    merged = build_map({"samples": [_pope_row()]}, traps_db=db)
    assert "report" in merged.sources
    assert "traps_db" in merged.sources
    assert merged.total >= 3


def test_skips_corrupted_rows() -> None:
    dirty = _pope_row()
    dirty["corruption"] = "gaussian_noise"
    events = events_from_report({"samples": [dirty]})
    assert events == []


def test_maps_cli_json(tmp_path: Path) -> None:
    report = tmp_path / "mm.json"
    report.write_text(json.dumps({"samples": [_pope_row()]}), encoding="utf-8")
    result = CliRunner().invoke(app, ["maps", str(report), "--json"])
    assert result.exit_code == 0, _cli_text(result)
    payload = json.loads(_cli_text(result))
    assert payload["total"] >= 1
    assert "by_metric" in payload
    assert "pope_miss" in payload["by_metric"]


def test_maps_cli_requires_input() -> None:
    result = CliRunner().invoke(app, ["maps"])
    assert result.exit_code != 0


def test_to_json_is_sorted_and_stable() -> None:
    first = to_json(build_map({"samples": [_pope_row()]}))
    second = to_json(build_map({"samples": [_pope_row()]}))
    assert first == second
