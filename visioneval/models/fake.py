"""Offline VLM used by pytest and the Streamlit demo. No downloads, no keys."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence

from PIL import Image

from visioneval.models.base import BaseVLM, GenerationResult
from visioneval.profiling.profiler import estimate_throughput, peak_vram_mb

_POPE_OBJECT = re.compile(r"is there (?:an? )?(.+?) in the image", re.IGNORECASE)


class FakeVLM(BaseVLM):
    """Lookup-table captioner that also answers POPE yes/no probes.

    * Caption prompts use ``responses[sample_id]`` or ``default_response``.
    * Prompts matching ``Is there a {object} in the image?`` return Yes/No
      from ``object_map[sample_id]``.
    """

    def __init__(
        self,
        name: str = "fake",
        responses: Mapping[str, str] | None = None,
        default_response: str = "A simple geometric scene.",
        *,
        latency_ms: float = 0.0,
        object_map: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.name = name
        self.responses = dict(responses or {})
        self.default_response = default_response
        self.latency_ms = latency_ms
        self.object_map = {
            key: {item.lower() for item in values} for key, values in (object_map or {}).items()
        }

    def _pope_answer(self, prompt: str, sample_id: str | None) -> str | None:
        match = _POPE_OBJECT.search(prompt)
        if match is None and "is there" not in prompt.lower():
            return None
        queried = (match.group(1) if match else prompt).strip().lower().rstrip("?!. ")
        present: set[str] = set()
        if sample_id and sample_id in self.object_map:
            present = self.object_map[sample_id]
        else:
            for values in self.object_map.values():
                present |= values
        if any(obj == queried or obj in queried or queried in obj for obj in present):
            return "Yes."
        return "No."

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        sample_id: str | None = None,
    ) -> GenerationResult:
        del image
        started = time.perf_counter()
        text = None
        pope = self._pope_answer(prompt, sample_id)
        if pope is not None:
            text = pope
        elif prompt in self.responses:
            text = self.responses[prompt]
        elif sample_id and sample_id in self.responses:
            text = self.responses[sample_id]
        else:
            text = self.default_response
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        total_ms = (time.perf_counter() - started) * 1000.0
        tokens = len(text.split())
        return GenerationResult(
            text=text,
            ttft_ms=total_ms,
            total_ms=total_ms,
            vram_mb=peak_vram_mb(),
            throughput_tps=estimate_throughput(tokens, total_ms),
            extra={"token_count": tokens, "backend": "fake"},
        )
