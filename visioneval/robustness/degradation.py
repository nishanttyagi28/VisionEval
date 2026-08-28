"""Resilience and drop-off scores versus a clean baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


def drop_off(clean: float, corrupted: float) -> float:
    """Relative degradation of ``corrupted`` versus ``clean``.

    * ``clean > 0``: ``max(0, (clean - corrupted) / abs(clean))``. An
      *improvement* under corruption therefore scores 0, not a negative drop.
    * ``clean == 0`` and ``corrupted == 0``: 0 (no information, no drop).
    * ``clean == 0`` and ``corrupted != 0``: 1 (undefined ratio; treat as full drop).
    """
    clean_f = float(clean)
    corrupted_f = float(corrupted)
    if clean_f == 0.0:
        return 0.0 if corrupted_f == 0.0 else 1.0
    return max(0.0, (clean_f - corrupted_f) / abs(clean_f))


def resilience_score(clean: float, corrupted_values: Sequence[float]) -> float:
    """Mean of ``1 - drop_off`` across corrupted scores (higher is more robust)."""
    values = list(corrupted_values)
    if not values:
        return 1.0
    return sum(1.0 - drop_off(clean, value) for value in values) / len(values)


@dataclass(frozen=True)
class SeverityPoint:
    severity: float
    score: float
    drop_off: float


@dataclass(frozen=True)
class DegradationReport:
    """Per-metric, per-corruption degradation versus the clean baseline."""

    metric: str
    corruption: str
    clean_score: float
    resilience: float
    points: tuple[SeverityPoint, ...] = ()
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "corruption": self.corruption,
            "clean_score": self.clean_score,
            "resilience": self.resilience,
            "points": [
                {"severity": p.severity, "score": p.score, "drop_off": p.drop_off}
                for p in self.points
            ],
            "extra": dict(self.extra),
        }


def summarize_degradation(
    *,
    metric: str,
    corruption: str,
    clean_score: float,
    scores_by_severity: Mapping[float, float],
) -> DegradationReport:
    """Build a report from a clean score and ``{severity: score}`` mapping."""
    points = []
    corrupted_values = []
    for severity in sorted(scores_by_severity):
        score = float(scores_by_severity[severity])
        points.append(
            SeverityPoint(
                severity=float(severity),
                score=score,
                drop_off=drop_off(clean_score, score),
            )
        )
        if float(severity) > 0.0:
            corrupted_values.append(score)
    return DegradationReport(
        metric=metric,
        corruption=corruption,
        clean_score=float(clean_score),
        resilience=resilience_score(clean_score, corrupted_values),
        points=tuple(points),
    )
