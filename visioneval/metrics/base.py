"""Metric contracts so CLIP, BLIP, and custom scorers stay interchangeable."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class MetricResult:
    """A named scalar plus optional structured extras."""

    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)


class PairMetric(ABC):
    """Score a single image-text pair. Implementations must be swappable."""

    name: str

    @abstractmethod
    def score(self, image: Image.Image, text: str) -> MetricResult:
        """Return a metric result for ``image`` aligned with ``text``."""


@runtime_checkable
class AlignmentBackend(Protocol):
    """Produces an image-text similarity in roughly ``[-1, 1]`` or ``[0, 1]``.

    Mock backends must be deterministic and must never download weights.
    """

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        """Return cosine similarity (or equivalent) between image and text."""
