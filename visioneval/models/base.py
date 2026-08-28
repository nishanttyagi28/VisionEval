"""Shared VLM contract. Heavy backends (HF, API) stay in optional extras."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class GenerationResult:
    """Text plus the profiling fields the rest of the stack expects."""

    text: str
    ttft_ms: float
    total_ms: float
    vram_mb: float | None = None
    throughput_tps: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        extra_count = self.extra.get("token_count")
        if isinstance(extra_count, int):
            return extra_count
        return len(self.text.split())


@runtime_checkable
class VisionLanguageModel(Protocol):
    """Minimal protocol so tests can inject fakes without subclassing."""

    name: str

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        sample_id: str | None = None,
    ) -> GenerationResult:
        """Caption or answer ``prompt`` for ``image``."""


class BaseVLM(ABC):
    """ABC variant of :class:`VisionLanguageModel` for concrete adapters."""

    name: str

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        sample_id: str | None = None,
    ) -> GenerationResult:
        """Caption or answer ``prompt`` for ``image``."""
