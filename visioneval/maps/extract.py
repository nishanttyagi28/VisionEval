"""Extract failure events from multimodal reports and living-traps DBs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from visioneval.maps.types import FailureEvent

DEFAULT_FACTUAL_THRESHOLD = 0.5


def events_from_report(
    report: Mapping[str, Any],
    *,
    factual_threshold: float = DEFAULT_FACTUAL_THRESHOLD,
) -> list[FailureEvent]:
    """Extract failure events from a multimodal report payload. Corrupted rows skipped."""
    events: list[FailureEvent] = []
    for row in report.get("samples") or []:
        if not isinstance(row, dict):
            continue
        if not _is_clean(row):
            continue
        events.extend(_events_from_row(row, factual_threshold=factual_threshold))
    return events


def events_from_traps_db(path: Path) -> list[FailureEvent]:
    """Open (non-retired) traps become failure events keyed like harvest metrics."""
    from visioneval.traps.store import TrapStore

    events: list[FailureEvent] = []
    for trap in TrapStore(path).list_open():
        metric = {
            "pope": "pope_miss",
            "judge": "judge_flag",
            "caption": "caption_mismatch",
        }.get(trap.probe_type, trap.probe_type)
        object_name = None
        if trap.extra.get("object_name"):
            object_name = str(trap.extra["object_name"])
        elif trap.probe_type == "pope" and trap.absent_objects:
            object_name = trap.absent_objects[0]
        elif trap.probe_type == "caption" and trap.expected_objects:
            object_name = trap.expected_objects[0]
        detail = trap.prompt
        if trap.extra.get("flags"):
            detail = ",".join(str(item) for item in trap.extra["flags"])
        events.append(
            FailureEvent(
                sample_id=trap.sample_id,
                model=trap.model,
                probe_type=trap.probe_type,
                metric=metric,
                object_name=object_name,
                detail=detail,
                source="traps_db",
            )
        )
    return events


def _is_clean(row: Mapping[str, Any]) -> bool:
    corruption = row.get("corruption")
    return corruption in (None, "", "clean", "none")


def _events_from_row(row: Mapping[str, Any], *, factual_threshold: float) -> list[FailureEvent]:
    sample_id = str(row.get("sample_id") or row.get("id") or "sample")
    model = str(row.get("model") or "unknown")
    events: list[FailureEvent] = []
    pope = row.get("pope")
    if isinstance(pope, dict):
        for probe in pope.get("probes") or []:
            if not isinstance(probe, dict) or probe.get("correct", True):
                continue
            object_name = str(probe.get("object") or probe.get("object_name") or "")
            events.append(
                FailureEvent(
                    sample_id=sample_id,
                    model=model,
                    probe_type="pope",
                    metric="pope_miss",
                    object_name=object_name or None,
                    detail=str(probe.get("prompt") or ""),
                    source="report",
                )
            )
    judge = row.get("judge")
    if isinstance(judge, dict):
        flags = [str(item) for item in (judge.get("flags") or [])]
        factual = float(judge.get("factual_consistency", 1.0))
        if flags or factual < factual_threshold:
            detail = ",".join(flags) if flags else f"factual_consistency={factual}"
            events.append(
                FailureEvent(
                    sample_id=sample_id,
                    model=model,
                    probe_type="judge",
                    metric="judge_flag",
                    object_name=_first_missing_object(flags),
                    detail=detail,
                    source="report",
                )
            )
    objects = [str(item) for item in (row.get("objects") or ())]
    response = str(row.get("response") or row.get("caption") or "").lower()
    if objects and response:
        missing = [obj for obj in objects if obj.lower() not in response]
        leaked = [
            str(item)
            for item in (row.get("absent_objects") or ())
            if str(item).lower() in response
        ]
        if missing or leaked:
            detail_parts = []
            if missing:
                detail_parts.append("missing:" + ",".join(missing))
            if leaked:
                detail_parts.append("leaked:" + ",".join(leaked))
            events.append(
                FailureEvent(
                    sample_id=sample_id,
                    model=model,
                    probe_type="caption",
                    metric="caption_mismatch",
                    object_name=(missing or leaked)[0],
                    detail=";".join(detail_parts),
                    source="report",
                )
            )
    return events


def _first_missing_object(flags: Iterable[str]) -> str | None:
    for flag in flags:
        if flag.startswith("missing_objects:"):
            parts = flag.split(":", 1)[1].split(",")
            if parts and parts[0]:
                return parts[0]
    return None
