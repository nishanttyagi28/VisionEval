"""Unified vision-language model wrappers (fake, HuggingFace, OpenAI-compatible)."""

from visioneval.models.base import BaseVLM, GenerationResult, VisionLanguageModel
from visioneval.models.factory import build_model, describe_available_backends
from visioneval.models.fake import FakeVLM

__all__ = [
    "BaseVLM",
    "FakeVLM",
    "GenerationResult",
    "VisionLanguageModel",
    "build_model",
    "describe_available_backends",
]
