"""HuggingFace transformers VLM adapters (Qwen2-VL, LLaVA, Auto). Optional extra."""

from __future__ import annotations

import time
from typing import Any, Literal

from PIL import Image

from visioneval.models.base import BaseVLM, GenerationResult
from visioneval.profiling.profiler import estimate_throughput, peak_vram_mb, reset_peak_vram

HfKind = Literal["auto", "qwen2_vl", "llava"]


class HuggingFaceVLM(BaseVLM):
    """Lazy-loading HuggingFace vision-language model.

    Importing this module does **not** import ``transformers`` or ``torch``.
    Those packages are loaded on first :meth:`generate` so a CPU-only laptop
    can still ``import visioneval.models``.
    """

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        kind: HfKind = "auto",
        device: str | None = None,
        max_new_tokens: int = 128,
        extra_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = name or model_id
        self.kind = kind
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.extra_kwargs = extra_kwargs or {}
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    def _require_transformers(self) -> Any:
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ImportError(
                "HuggingFace VLM adapters require the 'hf' extra: pip install -e '.[hf]'"
            ) from exc
        return torch, transformers

    def _load(self) -> None:
        if self._model is not None:
            return
        torch, transformers = self._require_transformers()
        self._torch = torch
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        kind = self.kind
        if kind == "auto":
            lowered = self.model_id.lower()
            if "qwen2-vl" in lowered or "qwen2_vl" in lowered:
                kind = "qwen2_vl"
            elif "llava" in lowered:
                kind = "llava"
        if kind == "qwen2_vl":
            model_cls = getattr(transformers, "Qwen2VLForConditionalGeneration")
            processor_cls = getattr(transformers, "AutoProcessor")
        elif kind == "llava":
            model_cls = getattr(transformers, "LlavaForConditionalGeneration")
            processor_cls = getattr(transformers, "AutoProcessor")
        else:
            model_cls = getattr(transformers, "AutoModelForVision2Seq")
            processor_cls = getattr(transformers, "AutoProcessor")
        self._processor = processor_cls.from_pretrained(self.model_id)
        self._model = model_cls.from_pretrained(self.model_id, **self.extra_kwargs)
        self._model.to(device).eval()

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        *,
        sample_id: str | None = None,
    ) -> GenerationResult:
        del sample_id
        self._load()
        torch = self._torch
        reset_peak_vram()
        started = time.perf_counter()
        rgb = image.convert("RGB")
        inputs = self._processor(text=prompt, images=rgb, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        first_token_at: list[float] = []

        def _mark_first_token(module: Any, args: Any, output: Any) -> None:  # pragma: no cover - hook
            if not first_token_at:
                first_token_at.append(time.perf_counter())

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        total_ms = (time.perf_counter() - started) * 1000.0
        ttft_ms = ((first_token_at[0] - started) * 1000.0) if first_token_at else total_ms
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        tokens = int(generated.shape[-1]) if hasattr(generated, "shape") else len(text.split())
        return GenerationResult(
            text=text.strip(),
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            vram_mb=peak_vram_mb(),
            throughput_tps=estimate_throughput(tokens, total_ms),
            extra={"token_count": tokens, "backend": "hf", "model_id": self.model_id, "kind": self.kind},
        )
