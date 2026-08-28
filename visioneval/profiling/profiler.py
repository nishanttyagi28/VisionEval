"""Lightweight inference timers and optional CUDA VRAM snapshots."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ProfileStats:
    """Timing and resource snapshot for a single generation call."""

    ttft_ms: float
    total_ms: float
    vram_mb: float | None
    throughput_tps: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def peak_vram_mb() -> float | None:
    """Current CUDA peak allocation in MiB, or ``None`` without a GPU/torch."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2)


def reset_peak_vram() -> None:
    """Reset the CUDA peak-memory counter when torch+CUDA are present."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def estimate_throughput(token_count: int, total_ms: float) -> float | None:
    """Tokens per second from a completed generation. ``None`` if time is 0."""
    if total_ms <= 0:
        return None
    return float(token_count) / (total_ms / 1000.0)


def profile_generation(fn: Callable[[], T]) -> tuple[T, ProfileStats]:
    """Time a callable that returns an object with optional ``ttft_ms``/text.

    If the callable already returns a ``GenerationResult``-like object with
    ``ttft_ms`` and ``total_ms``, those values are preferred over the outer
    wall clock so streaming adapters can report true time-to-first-token.
    """
    reset_peak_vram()
    started = time.perf_counter()
    result = fn()
    wall_ms = (time.perf_counter() - started) * 1000.0
    ttft = float(getattr(result, "ttft_ms", wall_ms))
    total = float(getattr(result, "total_ms", wall_ms))
    vram = getattr(result, "vram_mb", None)
    if vram is None:
        vram = peak_vram_mb()
    throughput = getattr(result, "throughput_tps", None)
    if throughput is None:
        tokens = getattr(result, "token_count", None)
        if tokens is None:
            text = getattr(result, "text", "")
            tokens = len(str(text).split())
        throughput = estimate_throughput(int(tokens), total)
    stats = ProfileStats(
        ttft_ms=ttft,
        total_ms=total,
        vram_mb=None if vram is None else float(vram),
        throughput_tps=None if throughput is None else float(throughput),
    )
    return result, stats
