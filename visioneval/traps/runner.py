"""Replay open traps against a VLM. Open traps consume budget first."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from visioneval.metrics.llm_judge import LLMJudge
from visioneval.metrics.pope import parse_yes_no
from visioneval.models.base import BaseVLM
from visioneval.models.fake import FakeVLM
from visioneval.multimodal.config import SampleConfig
from visioneval.multimodal.fixtures import load_sample_image
from visioneval.traps.baseline import TrapRegression, compare_trap_baseline, load_trap_baseline
from visioneval.traps.generator import mint_hard_negatives
from visioneval.traps.store import TrapStore
from visioneval.traps.types import TrapRecord

DEFAULT_FACTUAL_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrapRunResult:
    evaluated: int
    passed: int
    failed: int
    retired: int
    still_open: int
    records: tuple[dict[str, Any], ...]
    regression: TrapRegression | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "evaluated": self.evaluated,
            "passed": self.passed,
            "failed": self.failed,
            "retired": self.retired,
            "still_open": self.still_open,
            "records": list(self.records),
        }
        if self.regression is not None:
            payload["regression"] = self.regression.as_dict()
        return payload


def prioritize_open_traps(
    open_traps: Sequence[TrapRecord],
    extras: Sequence[TrapRecord],
    budget: int,
) -> list[TrapRecord]:
    """Open (non-retired) traps consume budget before newly minted variants."""
    selected: list[TrapRecord] = []
    seen: set[str] = set()
    for trap in list(open_traps) + list(extras):
        if trap.trap_id in seen:
            continue
        selected.append(trap)
        seen.add(trap.trap_id)
        if len(selected) >= budget:
            break
    return selected


def evaluate_trap(trap: TrapRecord, model: BaseVLM, image: Image.Image) -> bool:
    """Return True when the model beats this trap on this replay."""
    if trap.probe_type == "pope":
        answer = model.generate(image, trap.prompt, sample_id=trap.sample_id).text
        predicted = parse_yes_no(answer)
        if predicted is None:
            return False
        expected = bool(trap.extra.get("expected_present"))
        return predicted is expected
    if trap.probe_type == "caption":
        text = model.generate(image, trap.prompt, sample_id=trap.sample_id).text.lower()
        mentioned = all(obj.lower() in text for obj in trap.expected_objects)
        leaked = any(obj.lower() in text for obj in trap.absent_objects)
        return bool(trap.expected_objects) and mentioned and not leaked
    if trap.probe_type == "judge":
        text = model.generate(image, trap.prompt, sample_id=trap.sample_id).text
        _, verdict = LLMJudge().score(
            image,
            text,
            caption=str(trap.extra.get("caption") or ""),
            objects=trap.expected_objects,
            spatial_notes=str(trap.extra.get("spatial_notes") or ""),
        )
        threshold = float(trap.extra.get("factual_threshold") or DEFAULT_FACTUAL_THRESHOLD)
        return (not verdict.flags) and verdict.factual_consistency >= threshold
    raise ValueError(f"unknown probe_type {trap.probe_type!r}")


def resolve_image(
    trap: TrapRecord,
    samples: Mapping[str, SampleConfig] | None = None,
    *,
    root: str | Path | None = None,
) -> Image.Image:
    sample = (samples or {}).get(trap.sample_id)
    if sample is not None:
        return load_sample_image(sample.image, sample.color, str(root) if root else None)
    color = trap.extra.get("color") or trap.sample_id
    return load_sample_image(None, str(color) if color else None)


def run_open_traps(
    store: TrapStore,
    model: BaseVLM,
    *,
    budget: int = 32,
    retire_after: int = 2,
    generate_hard_negatives: bool = False,
    seed: int = 0,
    samples: Mapping[str, SampleConfig] | None = None,
    root: str | Path | None = None,
    check_baseline: Path | None = None,
) -> TrapRunResult:
    """Replay open traps. Hard-negatives fill leftover budget only."""
    open_traps = store.list_open(model=None)
    extras: list[TrapRecord] = []
    remaining = max(budget - len(open_traps), 0)
    if generate_hard_negatives and remaining > 0:
        minted = mint_hard_negatives(open_traps[:budget], seed=seed)[:remaining]
        for trap in minted:
            if store.get(trap.trap_id) is None:
                store.upsert_failure(trap)
            extras.append(store.get(trap.trap_id) or trap)
    selected = prioritize_open_traps(open_traps, extras, budget)

    records: list[dict[str, Any]] = []
    passed = failed = 0
    for trap in selected:
        image = resolve_image(trap, samples, root=root)
        beat = evaluate_trap(trap, model, image)
        updated = store.record_outcome(trap.trap_id, beat, retire_after=retire_after)
        if beat:
            passed += 1
        else:
            failed += 1
        records.append(
            {
                "trap_id": updated.trap_id,
                "passed": beat,
                "retired": updated.retired,
                "fail_count": updated.fail_count,
                "consecutive_passes": updated.consecutive_passes,
                "last_outcome": updated.last_outcome,
            }
        )

    still_open = store.list_open()
    regression = None
    if check_baseline is not None:
        regression = compare_trap_baseline(load_trap_baseline(check_baseline), store)
    return TrapRunResult(
        evaluated=len(selected),
        passed=passed,
        failed=failed,
        retired=sum(1 for row in records if row["retired"]),
        still_open=len(still_open),
        records=tuple(records),
        regression=regression,
    )


def default_fake_for_traps(traps: Sequence[TrapRecord], *, name: str = "fake") -> FakeVLM:
    """FakeVLM whose object map matches trap expected objects (POPE can pass)."""
    object_map: dict[str, list[str]] = {}
    for trap in traps:
        object_map.setdefault(trap.sample_id, [])
        for item in trap.expected_objects:
            if item not in object_map[trap.sample_id]:
                object_map[trap.sample_id].append(item)
    return FakeVLM(name=name, object_map=object_map)
