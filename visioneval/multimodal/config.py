"""YAML / pydantic configuration for a multimodal evaluation run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_CAPTION_PROMPT = "Describe the image in detail. Mention objects and their spatial layout."
DEFAULT_POPE_PROMPT = "Answer yes or no. {question}"


class SampleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    image: str | None = None
    caption: str = ""
    objects: list[str] = Field(default_factory=list)
    absent_objects: list[str] = Field(default_factory=list)
    spatial_notes: str = ""
    prompt: str | None = None
    color: str | None = None  # used by the in-memory demo fixture generator


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    kind: Literal["fake", "hf", "api", "openai"] = "fake"
    model_id: str | None = None
    model: str | None = None
    responses: dict[str, str] = Field(default_factory=dict)
    default_response: str = "A simple geometric scene."
    latency_ms: float = 1.0
    hf_kind: Literal["auto", "qwen2_vl", "llava"] = "auto"
    device: str | None = None
    max_new_tokens: int = 128
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    max_tokens: int = 256

    def to_factory_dict(self) -> dict[str, Any]:
        return self.model_dump()


class MetricsToggle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip: bool = True
    blip: bool = True
    pope: bool = True
    llm_judge: bool = True
    clip_backend: Literal["mock", "hf"] = "mock"
    blip_backend: Literal["mock", "hf"] = "mock"
    clip_model_id: str = "openai/clip-vit-base-patch32"
    blip_model_id: str = "Salesforce/blip-itm-base-coco"


class CorruptionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    types: list[Literal["gaussian_noise", "motion_blur", "contrast_jitter", "occlusion"]] = Field(
        default_factory=lambda: ["gaussian_noise", "motion_blur", "contrast_jitter", "occlusion"]
    )
    severities: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    seed: int = 0

    @model_validator(mode="after")
    def severities_in_unit_interval(self) -> "CorruptionConfig":
        for value in self.severities:
            if value < 0.0 or value > 1.0:
                raise ValueError("corruption severities must be in [0, 1]")
        return self


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["mock", "api"] = "mock"
    prompt_template: str | None = None
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None


class ReportPaths(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    json_path: str | None = None
    markdown_path: str | None = None


class MultimodalEvalConfig(BaseModel):
    """Top-level multimodal eval config. Independent of Phase 1 suite YAML."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    caption_prompt: str = DEFAULT_CAPTION_PROMPT
    models: list[ModelSpec] = Field(min_length=1)
    samples: list[SampleConfig] = Field(default_factory=list)
    samples_path: str | None = None
    metrics: MetricsToggle = Field(default_factory=MetricsToggle)
    corruptions: CorruptionConfig = Field(default_factory=CorruptionConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    report: ReportPaths | None = None

    @model_validator(mode="after")
    def samples_or_path(self) -> "MultimodalEvalConfig":
        if not self.samples and not self.samples_path:
            raise ValueError("provide samples or samples_path")
        return self


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        contents = yaml.safe_load(handle)
    if not isinstance(contents, dict):
        raise ValueError(f"{path} must contain a mapping at its root")
    return contents


def load_multimodal_config(path: Path) -> MultimodalEvalConfig:
    """Load a multimodal eval YAML and resolve nested sample manifests."""
    path = path.resolve()
    data = _load_yaml(path)
    samples_path = data.get("samples_path")
    if samples_path and not data.get("samples"):
        nested = path.parent / samples_path
        nested_data = _load_yaml(nested)
        raw_samples = nested_data.get("samples", nested_data)
        if not isinstance(raw_samples, list):
            raise ValueError("samples YAML must be a list or a mapping with a 'samples' key")
        data["samples"] = raw_samples
    return MultimodalEvalConfig.model_validate(data)
