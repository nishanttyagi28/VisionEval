"""Datatypes for black-box hallucination maps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FailureEvent:
    """One machine-actionable failure locus."""

    sample_id: str
    model: str
    probe_type: str
    metric: str
    object_name: str | None = None
    detail: str = ""
    source: str = "report"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HallucinationMap:
    """Deterministic aggregation of failure events."""

    events: tuple[FailureEvent, ...]
    by_object: dict[str, int]
    by_probe_type: dict[str, int]
    by_sample_id: dict[str, int]
    by_metric: dict[str, int]
    by_model: dict[str, int]
    sources: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.events)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "sources": list(self.sources),
            "by_object": dict(sorted(self.by_object.items())),
            "by_probe_type": dict(sorted(self.by_probe_type.items())),
            "by_sample_id": dict(sorted(self.by_sample_id.items())),
            "by_metric": dict(sorted(self.by_metric.items())),
            "by_model": dict(sorted(self.by_model.items())),
            "events": [event.as_dict() for event in self.events],
        }
