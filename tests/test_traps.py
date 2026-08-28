"""Living VLM traps: harvest, consecutive-pass retirement, hard-negatives, CLI."""

from __future__ import annotations

import json
from pathlib import Path

from visioneval.core.cache import SQLiteCache
from visioneval.core.types import EvaluationRecord, SelectionReason
from visioneval.models.fake import FakeVLM
from visioneval.traps.baseline import compare_trap_baseline, save_trap_baseline
from visioneval.traps.generator import generate_hard_negative
from visioneval.traps.harvest import harvest_report
from visioneval.traps.runner import run_open_traps
from visioneval.traps.store import TrapStore
from visioneval.traps.types import TrapRecord, make_trap_id, utc_now


def _record(sample_id: str, correct: bool) -> EvaluationRecord:
    return EvaluationRecord(sample_id, "cat", "cat" if correct else "dog", 0.8, correct, SelectionReason.HIGH_RISK)


def _pope_row(*, correct_present: bool = False, correct_absent: bool = True) -> dict:
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
                    "answer": "No." if correct_absent else "Yes.",
                    "predicted": (not correct_absent),
                    "correct": correct_absent,
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


def test_harvest_pope_judge_and_caption(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    report = {"name": "demo", "samples": [_pope_row()]}
    summary = harvest_report(report, store)
    types = {trap.probe_type for trap in store.list_open()}
    assert "pope" in types
    assert "judge" in types
    assert "caption" in types
    assert summary.created >= 3
    pope = next(trap for trap in store.list_open() if trap.probe_type == "pope")
    assert pope.sample_id == "red_square"
    assert pope.extra.get("object_name") == "square"
    assert pope.retired is False


def test_still_fail_keeps_trap_open(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store)
    pope_id = next(trap.trap_id for trap in store.list_open() if trap.probe_type == "pope")
    failing = FakeVLM(name="fake-hallucinator", object_map={})  # POPE always No
    first = run_open_traps(store, failing, budget=32, retire_after=2)
    assert first.failed >= 1
    trap = store.get(pope_id)
    assert trap is not None and trap.retired is False
    assert trap.consecutive_passes == 0
    assert trap.fail_count >= 2  # harvest + replay


def test_two_consecutive_passes_retire(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store)
    pope = next(trap for trap in store.list_open() if trap.probe_type == "pope")
    passing = FakeVLM(name="fake-good", object_map={"red_square": ["square"]})
    run_open_traps(store, passing, budget=8, retire_after=2)
    mid = store.get(pope.trap_id)
    assert mid is not None
    assert mid.retired is False
    assert mid.consecutive_passes == 1
    run_open_traps(store, passing, budget=8, retire_after=2)
    done = store.get(pope.trap_id)
    assert done is not None
    assert done.retired is True
    assert done.consecutive_passes >= 2
    assert pope.trap_id not in {trap.trap_id for trap in store.list_open()}


def test_seeded_hard_negatives_are_deterministic(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store, generate_hard_negatives=True, seed=7)
    first = [trap for trap in store.list_open() if trap.extra.get("hard_negative")]
    assert first
    parent = next(trap for trap in store.list_open() if trap.probe_type == "pope" and not trap.extra.get("hard_negative"))
    a = generate_hard_negative(parent, seed=7)
    b = generate_hard_negative(parent, seed=7)
    c = generate_hard_negative(parent, seed=99)
    assert a.trap_id == b.trap_id
    assert a.prompt == b.prompt
    assert a.sample_id == parent.sample_id
    assert a.probe_type == "pope"
    assert c.trap_id != a.trap_id or c.prompt != a.prompt


def test_open_traps_consume_budget_before_hard_negatives(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store)
    open_before = store.count_open()
    failing = FakeVLM(name="fake-hallucinator", object_map={})
    result = run_open_traps(
        store, failing, budget=open_before, generate_hard_negatives=True, seed=0
    )
    assert result.evaluated == open_before
    # leftover budget is 0, so no hard-negatives were injected into the run
    assert result.evaluated == open_before


def test_traps_tables_do_not_clobber_classification_memory(tmp_path: Path) -> None:
    db = tmp_path / "shared.sqlite"
    cache = SQLiteCache(db)
    cache.record((_record("sample-1", False),))
    store = TrapStore(db)
    now = utc_now()
    store.upsert_failure(
        TrapRecord(
            trap_id=make_trap_id("fake", "red_square", "pope", "cat"),
            model="fake",
            sample_id="red_square",
            image_hash="abc",
            prompt="Is there a cat in the image?",
            expected_objects=("square",),
            absent_objects=("cat",),
            probe_type="pope",
            last_outcome="fail",
            fail_count=1,
            consecutive_passes=0,
            created_at=now,
            updated_at=now,
            retired=False,
            extra={"expected_present": False, "object_name": "cat"},
        )
    )
    assert cache.previous_failure_ids() == {"sample-1"}
    assert store.count_open() == 1
    with cache._connect() as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "sample_outcomes" in tables
    assert "predictions" in tables
    assert "vlm_traps" in tables


def test_trap_baseline_flags_reappearance_and_worse(tmp_path: Path) -> None:
    store = TrapStore(tmp_path / "traps.sqlite3")
    harvest_report({"samples": [_pope_row()]}, store)
    lock = tmp_path / "traps.json"
    save_trap_baseline(lock, store)
    pope = next(trap for trap in store.list_open() if trap.probe_type == "pope")
    store.record_outcome(pope.trap_id, True, retire_after=2)
    store.record_outcome(pope.trap_id, True, retire_after=2)
    assert store.get(pope.trap_id).retired is True
    save_trap_baseline(lock, store)
    store.upsert_failure(store.get(pope.trap_id))
    regression = compare_trap_baseline(json.loads(lock.read_text(encoding="utf-8")), store)
    assert regression.is_regression
    assert pope.trap_id in regression.reappeared


def test_traps_list_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from visioneval.cli import app

    db = tmp_path / "traps.sqlite3"
    store = TrapStore(db)
    harvest_report({"samples": [_pope_row()]}, store)
    result = CliRunner().invoke(app, ["traps", "list", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "open:" in result.output
    assert "pope:" in result.output


def test_traps_harvest_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from visioneval.cli import app

    report = tmp_path / "mm.json"
    report.write_text(json.dumps({"samples": [_pope_row()]}), encoding="utf-8")
    db = tmp_path / "traps.sqlite3"
    result = CliRunner().invoke(app, ["traps", "harvest", str(report), "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert "Harvested" in result.output
    assert TrapStore(db).count_open() >= 1
