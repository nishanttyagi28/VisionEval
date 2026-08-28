"""Build a VLM from a config mapping without importing optional extras eagerly."""

from __future__ import annotations

from typing import Any

from visioneval.models.base import BaseVLM
from visioneval.models.fake import FakeVLM


def describe_available_backends() -> dict[str, bool]:
    """Report which optional extras imported cleanly on this machine."""
    available = {"fake": True, "hf": False, "api": False}
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        available["hf"] = True
    except ImportError:
        pass
    try:
        import openai  # noqa: F401

        available["api"] = True
    except ImportError:
        pass
    return available


def build_model(spec: dict[str, Any]) -> BaseVLM:
    """Instantiate a VLM from a JSON/YAML mapping.

    Recognised ``kind`` values: ``fake``, ``hf``, ``api`` (alias ``openai``).
    """
    kind = str(spec.get("kind", "fake")).lower()
    name = spec.get("name")
    if kind == "fake":
        return FakeVLM(
            name=name or "fake",
            responses=spec.get("responses") or {},
            default_response=spec.get("default_response") or "A simple geometric scene.",
            latency_ms=float(spec.get("latency_ms", 0.0)),
            object_map=spec.get("object_map") or {},
        )
    if kind == "hf":
        from visioneval.models.hf import HuggingFaceVLM

        model_id = spec.get("model_id") or spec.get("model")
        if not model_id:
            raise ValueError("hf models require 'model_id'")
        return HuggingFaceVLM(
            model_id=model_id,
            name=name,
            kind=spec.get("hf_kind", "auto"),
            device=spec.get("device"),
            max_new_tokens=int(spec.get("max_new_tokens", 128)),
            extra_kwargs=spec.get("extra_kwargs") or {},
        )
    if kind in {"api", "openai"}:
        from visioneval.models.api import OpenAICompatibleVLM

        model = spec.get("model") or spec.get("model_id")
        if not model:
            raise ValueError("api models require 'model'")
        return OpenAICompatibleVLM(
            model=model,
            name=name,
            api_key_env=spec.get("api_key_env", "OPENAI_API_KEY"),
            base_url=spec.get("base_url"),
            max_tokens=int(spec.get("max_tokens", 256)),
        )
    raise ValueError(f"unknown model kind {kind!r}; expected fake, hf, or api")
