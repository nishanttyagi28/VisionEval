"""Trap gate: lockfile CI blocker with machine-actionable regressions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from visioneval.cli import app
from visioneval.traps.baseline import (
    compare_trap_baseline,
    save_trap_baseline,
)
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


def test_new_open_is_regression(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    lock = tmp_path / "traps.json"
    save_trap_baseline(lock, store)
    harvest_report({"samples": [_pope_row()]}, store)
    regression = compare_trap_baseline(json.loads(lock.read_text(encoding="utf-8")), store)
    assert regression.is_regression
    assert regression.new_open
    assert regression.still_open
    assert "new_open" in regression.as_dict()


def test_worse_outcome_is_regression(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store)
    lock = tmp_path / "traps.json"
    save_trap_baseline(lock, store)
    pope = next(trap for trap in store.list_open() if trap.probe_type == "pope")
    store.record_outcome(pope.trap_id, False, retire_after=2)
    regression = compare_trap_baseline(json.loads(lock.read_text(encoding="utf-8")), store)
    assert regression.is_regression
    assert pope.trap_id in regression.worse


def test_reappeared_retired_is_regression(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store)
    lock = tmp_path / "traps.json"
    pope = next(trap for trap in store.list_open() if trap.probe_type == "pope")
    store.record_outcome(pope.trap_id, True, retire_after=2)
    store.record_outcome(pope.trap_id, True, retire_after=2)
    assert store.get(pope.trap_id).retired is True
    save_trap_baseline(lock, store)
    store.upsert_failure(store.get(pope.trap_id))
    regression = compare_trap_baseline(json.loads(lock.read_text(encoding="utf-8")), store)
    assert regression.is_regression
    assert pope.trap_id in regression.reappeared


def test_matching_lockfile_passes(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store)
    lock = tmp_path / "traps.json"
    save_trap_baseline(lock, store)
    regression = compare_trap_baseline(json.loads(lock.read_text(encoding="utf-8")), store)
    assert regression.is_regression is False
    assert regression.new_open == ()
    assert regression.worse == ()
    assert regression.reappeared == ()


def test_traps_gate_cli_exit_codes(tmp_path: Path) -> None:
    db = tmp_path / "traps.sqlite3"
    lock = tmp_path / "traps.json"
    store = TrapStore(db)
    save_trap_baseline(lock, store)
    ok = CliRunner().invoke(app, ["traps", "gate", "--db", str(db), "--lockfile", str(lock), "--json"])
    assert ok.exit_code == 0, _cli_text(ok)
    payload = json.loads(_cli_text(ok))
    assert payload["is_regression"] is False

    harvest_report({"samples": [_pope_row()]}, store)
    bad = CliRunner().invoke(app, ["traps", "gate", "--db", str(db), "--lockfile", str(lock), "--json"])
    assert bad.exit_code == 1, _cli_text(bad)
    body = json.loads(_cli_text(bad))
    assert body["is_regression"] is True
    assert body["new_open"]


def test_traps_run_check_baseline_emits_gate(tmp_path: Path) -> None:
    db = tmp_path / "traps.sqlite3"
    lock = tmp_path / "traps.json"
    store = TrapStore(db)
    harvest_report({"samples": [_pope_row()]}, store)
    save_trap_baseline(lock, store)
    result = CliRunner().invoke(
        app,
        ["traps", "run", "--db", str(db), "--budget", "8", "--check-baseline", str(lock), "--json"],
    )
    assert "Traps run:" in _cli_text(result)
    assert "is_regression" in _cli_text(result)
