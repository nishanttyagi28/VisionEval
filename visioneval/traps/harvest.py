"""Turn multimodal JSON results into durable traps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from visioneval.traps.generator import mint_hard_negatives
from visioneval.traps.store import TrapStore
from visioneval.traps.types import TrapRecord, image_identity, make_trap_id, utc_now

DEFAULT_FACTUAL_THRESHOLD = 0.5


@dataclass(frozen=True)
class HarvestSummary:
    created: int
    updated: int
    hard_negatives: int
    trap_ids: tuple[str, ...]
    failure_ids: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "updated": self.updated,
            "hard_negatives": self.hard_negatives,
            "trap_ids": list(self.trap_ids),
            "failure_ids": list(self.failure_ids),
        }


def load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"samples": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object or array")
    return payload


def harvest_report(
    report: Mapping[str, Any] | Path | str,
    store: TrapStore,
    *,
    generate_hard_negatives: bool = False,
    seed: int = 0,
    factual_threshold: float = DEFAULT_FACTUAL_THRESHOLD,
) -> HarvestSummary:
    """Harvest POPE misses, judge flags / low factual scores, and caption mismatches.

    Corrupted rows are skipped so a noise sweep does not explode the trap set.
    """
    if isinstance(report, (Path, str)):
        report = load_report(Path(report))
    samples = report.get("samples") or []
    created = updated = 0
    harvested: list[TrapRecord] = []
    seen: set[str] = set()
    for row in samples:
        if not isinstance(row, dict):
            continue
        if not _is_clean(row):
            continue
        for trap in _traps_from_row(row, factual_threshold=factual_threshold):
            if trap.trap_id in seen:
                continue
            seen.add(trap.trap_id)
            existed = store.get(trap.trap_id) is not None
            stored = store.upsert_failure(trap)
            harvested.append(stored)
            if existed:
                updated += 1
            else:
                created += 1
    failure_ids = tuple(item.trap_id for item in harvested)
    hard_negatives = 0
    if generate_hard_negatives:
        minted = mint_hard_negatives(harvested, seed=seed)
        for trap in minted:
            if store.get(trap.trap_id) is not None:
                continue
            store.upsert_failure(trap)
            hard_negatives += 1
            harvested.append(trap)
    return HarvestSummary(
        created=created,
        updated=updated,
        hard_negatives=hard_negatives,
        trap_ids=tuple(item.trap_id for item in harvested),
        failure_ids=failure_ids,
    )


def _is_clean(row: Mapping[str, Any]) -> bool:
    corruption = row.get("corruption")
    return corruption in (None, "", "clean", "none")


def _traps_from_row(row: Mapping[str, Any], *, factual_threshold: float) -> list[TrapRecord]:
    traps: list[TrapRecord] = []
    traps.extend(_pope_traps(row))
    judge_trap = _judge_trap(row, factual_threshold=factual_threshold)
    if judge_trap is not None:
        traps.append(judge_trap)
    caption_trap = _caption_trap(row)
    if caption_trap is not None:
        traps.append(caption_trap)
    return traps


def _base(row: Mapping[str, Any], probe_type: str, discriminator: str, prompt: str) -> TrapRecord:
    sample_id = str(row.get("sample_id") or row.get("id") or "sample")
    model = str(row.get("model") or "unknown")
    objects = tuple(str(item) for item in (row.get("objects") or ()))
    absent = tuple(str(item) for item in (row.get("absent_objects") or ()))
    color = row.get("color")
    image_hash = str(row.get("image_hash") or image_identity(sample_id, row.get("image"), color))
    extra = {"color": color, "source": "harvest"}
    now = utc_now()
    return TrapRecord(
        trap_id=make_trap_id(model, sample_id, probe_type, discriminator),
        model=model,
        sample_id=sample_id,
        image_hash=image_hash,
        prompt=prompt,
        expected_objects=objects,
        absent_objects=absent,
        probe_type=probe_type,
        last_outcome="fail",
        fail_count=1,
        consecutive_passes=0,
        created_at=now,
        updated_at=now,
        retired=False,
        extra=extra,
    )


def _pope_traps(row: Mapping[str, Any]) -> list[TrapRecord]:
    pope = row.get("pope")
    if not isinstance(pope, dict):
        return []
    probes = pope.get("probes")
    traps: list[TrapRecord] = []
    if isinstance(probes, list) and probes:
        for probe in probes:
            if not isinstance(probe, dict):
                continue
            if probe.get("correct") in (True, 1):
                continue
            object_name = str(probe.get("object") or probe.get("object_name") or "object")
            prompt = str(probe.get("prompt") or row.get("prompt") or "")
            trap = _base(row, "pope", object_name, prompt)
            trap.extra.update(
                {
                    "object_name": object_name,
                    "expected_present": bool(probe.get("expected_present")),
                    "answer": probe.get("answer"),
                }
            )
            traps.append(trap)
        return traps
    accuracy = pope.get("accuracy")
    f1 = pope.get("f1")
    missed = (isinstance(accuracy, (int, float)) and accuracy < 1.0) or (
        isinstance(f1, (int, float)) and f1 < 1.0
    )
    if missed and (pope.get("total") or 0):
        trap = _base(row, "pope", "aggregate", str(row.get("prompt") or ""))
        trap.extra.update({"accuracy": accuracy, "f1": f1, "expected_present": False})
        traps.append(trap)
    return traps


def _judge_trap(row: Mapping[str, Any], *, factual_threshold: float) -> TrapRecord | None:
    judge = row.get("judge")
    if not isinstance(judge, dict):
        metrics = row.get("metrics") or {}
        details = (metrics.get("llm_judge") or {}).get("details") if isinstance(metrics, dict) else None
        judge = details if isinstance(details, dict) else None
    if not isinstance(judge, dict):
        return None
    flags = judge.get("flags") or []
    if isinstance(flags, str):
        flags = [flags]
    factual = judge.get("factual_consistency")
    low_factual = isinstance(factual, (int, float)) and factual < factual_threshold
    if not flags and not low_factual:
        return None
    trap = _base(row, "judge", "factual", str(row.get("prompt") or ""))
    trap.extra.update(
        {
            "flags": list(flags),
            "factual_consistency": factual,
            "factual_threshold": factual_threshold,
        }
    )
    return trap


def _caption_trap(row: Mapping[str, Any]) -> TrapRecord | None:
    objects = [str(item) for item in (row.get("objects") or ())]
    response = str(row.get("response") or "")
    if not objects:
        flags = []
        judge = row.get("judge")
        if isinstance(judge, dict):
            flags = list(judge.get("flags") or [])
        missing_from_flags = _missing_from_flags(flags)
        if not missing_from_flags:
            return None
        objects = missing_from_flags
        missing = missing_from_flags
    else:
        lowered = response.lower()
        missing = [obj for obj in objects if obj.lower() not in lowered]
        if not missing:
            return None
    trap = _base(row, "caption", "objects", str(row.get("prompt") or ""))
    trap.extra.update({"missing": missing, "response": response})
    return trap


def _missing_from_flags(flags: list[Any]) -> list[str]:
    missing: list[str] = []
    for flag in flags:
        text = str(flag)
        if text.startswith("missing_objects:"):
            missing.extend(part.strip() for part in text.split(":", 1)[1].split(",") if part.strip())
    return missing
