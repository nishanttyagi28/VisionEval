"""CPU-only black-box hallucination maps from multimodal reports and/or traps DB.

No model inference, no network. Aggregates WHERE a model consistently fails:
by object, probe type, sample id, metric (POPE miss / judge flag / caption mismatch).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from visioneval.maps.extract import (
    DEFAULT_FACTUAL_THRESHOLD,
    events_from_report,
    events_from_traps_db,
)
from visioneval.maps.types import FailureEvent, HallucinationMap


def build_map(
    report: Mapping[str, Any] | Path | str | None = None,
    *,
    traps_db: Path | None = None,
    factual_threshold: float = DEFAULT_FACTUAL_THRESHOLD,
) -> HallucinationMap:
    """Build a map from a multimodal JSON report and/or a living-traps SQLite DB."""
    events: list[FailureEvent] = []
    sources: list[str] = []
    if report is not None:
        payload = _load_report(report)
        events.extend(events_from_report(payload, factual_threshold=factual_threshold))
        sources.append("report")
    if traps_db is not None and Path(traps_db).is_file():
        events.extend(events_from_traps_db(Path(traps_db)))
        sources.append("traps_db")
    return aggregate_events(events, sources=tuple(sources))


def aggregate_events(
    events: Sequence[FailureEvent],
    *,
    sources: tuple[str, ...] = (),
) -> HallucinationMap:
    """Deduplicate and count. Sort keys for deterministic JSON/human output."""
    unique = _dedupe(events)
    by_object: Counter[str] = Counter()
    by_probe: Counter[str] = Counter()
    by_sample: Counter[str] = Counter()
    by_metric: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    for event in unique:
        if event.object_name:
            by_object[event.object_name] += 1
        by_probe[event.probe_type] += 1
        by_sample[event.sample_id] += 1
        by_metric[event.metric] += 1
        by_model[event.model] += 1
    ordered = tuple(
        sorted(
            unique,
            key=lambda item: (
                item.sample_id,
                item.probe_type,
                item.metric,
                item.object_name or "",
                item.model,
                item.source,
                item.detail,
            ),
        )
    )
    return HallucinationMap(
        events=ordered,
        by_object=dict(sorted(by_object.items())),
        by_probe_type=dict(sorted(by_probe.items())),
        by_sample_id=dict(sorted(by_sample.items())),
        by_metric=dict(sorted(by_metric.items())),
        by_model=dict(sorted(by_model.items())),
        sources=sources or _infer_sources(ordered),
    )


def format_human(hallucination_map: HallucinationMap) -> str:
    lines = [
        "VisionEval hallucination map",
        "",
        f"Total failures:  {hallucination_map.total}",
        f"Sources:         {', '.join(hallucination_map.sources) or '(none)'}",
        "",
        "By metric:",
    ]
    lines.extend(_bucket_lines(hallucination_map.by_metric))
    lines.append("")
    lines.append("By probe type:")
    lines.extend(_bucket_lines(hallucination_map.by_probe_type))
    lines.append("")
    lines.append("By object:")
    lines.extend(_bucket_lines(hallucination_map.by_object))
    lines.append("")
    lines.append("By sample id:")
    lines.extend(_bucket_lines(hallucination_map.by_sample_id))
    lines.append("")
    lines.append("By model:")
    lines.extend(_bucket_lines(hallucination_map.by_model))
    if hallucination_map.events:
        lines.extend(["", "Events:"])
        for event in hallucination_map.events:
            obj = event.object_name or "-"
            lines.append(
                f"  {event.sample_id}  {event.probe_type}/{event.metric}  "
                f"obj={obj}  model={event.model}  [{event.source}]"
            )
    return "\n".join(lines) + "\n"


def to_json(hallucination_map: HallucinationMap) -> str:
    return json.dumps(hallucination_map.as_dict(), indent=2, sort_keys=True) + "\n"


def _bucket_lines(counts: Mapping[str, int]) -> list[str]:
    if not counts:
        return ["  (none)"]
    width = max(len(key) for key in counts)
    return [f"  {key:<{width}}  {count}" for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _load_report(report: Mapping[str, Any] | Path | str) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    path = Path(report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"samples": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object or array")
    return payload


def _dedupe(events: Sequence[FailureEvent]) -> list[FailureEvent]:
    seen: set[tuple[str, ...]] = set()
    unique: list[FailureEvent] = []
    for event in events:
        key = (
            event.sample_id,
            event.model,
            event.probe_type,
            event.metric,
            event.object_name or "",
            event.source,
            event.detail,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def _infer_sources(events: Sequence[FailureEvent]) -> tuple[str, ...]:
    return tuple(sorted({event.source for event in events}))
