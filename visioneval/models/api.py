"""OpenAI-compatible vision chat adapter. Optional ``api`` extra."""

from __future__ import annotations

import base64
import io
import os
import time
from typing import Any

from PIL import Image

from visioneval.models.base import BaseVLM, GenerationResult
from visioneval.profiling.profiler import estimate_throughput, peak_vram_mb


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode an in-memory image as a data URL for vision chat APIs."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{encoded}"


class OpenAICompatibleVLM(BaseVLM):
    """Vision chat via the OpenAI Python SDK (any compatible ``base_url``).

    The API key is read from an environment variable (default ``OPENAI_API_KEY``).
    Nothing is read from disk and no secrets are stored on the instance.
    """

    def __init__(
        self,
        model: str,
        *,
        name: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        max_tokens: int = 256,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.name = name or model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}

    def _client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI-compatible VLM adapters require the 'api' extra: pip install -e '.[api]'"
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": os.environ.get(self.api_key_env, "") or "EMPTY",
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers
        return OpenAI(**kwargs)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        sample_id: str | None = None,
    ) -> GenerationResult:
        del sample_id
        client = self._client()
        data_url = image_to_data_url(image)
        started = time.perf_counter()
        ttft_ms = 0.0
        chunks: list[str] = []
        stream = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        )
        for event in stream:
            choice = event.choices[0] if event.choices else None
            delta = getattr(choice, "delta", None) if choice is not None else None
            piece = getattr(delta, "content", None) if delta is not None else None
            if piece:
                if not chunks:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                chunks.append(piece)
        total_ms = (time.perf_counter() - started) * 1000.0
        text = "".join(chunks).strip()
        if not chunks:
            ttft_ms = total_ms
        tokens = len(text.split())
        return GenerationResult(
            text=text,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            vram_mb=peak_vram_mb(),
            throughput_tps=estimate_throughput(tokens, total_ms),
            extra={"token_count": tokens, "backend": "api", "model": self.model},
        )
