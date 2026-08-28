"""TTFT, total time, throughput, and VRAM helpers."""

from visioneval.models.base import GenerationResult
from visioneval.profiling.profiler import estimate_throughput, peak_vram_mb, profile_generation


def test_estimate_throughput_tokens_per_second() -> None:
    assert estimate_throughput(20, 2000.0) == 10.0
    assert estimate_throughput(10, 0.0) is None


def test_peak_vram_without_cuda_is_none() -> None:
    # Laptops without GPU (and CI) must still import and call this.
    value = peak_vram_mb()
    assert value is None or value >= 0.0


def test_profile_generation_prefers_result_timings() -> None:
    def _run() -> GenerationResult:
        return GenerationResult(text="hello world", ttft_ms=3.0, total_ms=12.0, extra={"token_count": 2})

    result, stats = profile_generation(_run)
    assert result.text == "hello world"
    assert stats.ttft_ms == 3.0
    assert stats.total_ms == 12.0
    assert stats.throughput_tps == estimate_throughput(2, 12.0)
