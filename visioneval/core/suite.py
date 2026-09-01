"""YAML suite loading and strict Phase 1 configuration validation."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    adapter: str = Field(min_length=1)


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    manifest: str = Field(min_length=1)


class AttentionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budget: int = Field(default=100, gt=0)
    seed: int = 0
    previous_failures: bool = True
    high_risk_tags: list[str] = Field(default_factory=list)
    low_confidence: bool = True
    low_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    previous_failures_fraction: float = Field(default=0.40, ge=0.0, le=1.0)
    high_risk_fraction: float = Field(default=0.30, ge=0.0, le=1.0)
    low_confidence_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    random_coverage_fraction: float = Field(default=0.15, ge=0.0, le=1.0)
    use_budget: bool = False

    @model_validator(mode="after")
    def fractions_match_budget(self) -> "AttentionConfig":
        if abs(self.previous_failures_fraction + self.high_risk_fraction + self.low_confidence_fraction + self.random_coverage_fraction - 1.0) > 1e-9:
            raise ValueError("attention allocation fractions must sum to 1.0")
        return self


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    path: str = Field(min_length=1)
    allowed_accuracy_drop: float = Field(default=0.0, ge=0.0, le=1.0)


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    path: str = Field(min_length=1)


class ReportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    path: str = Field(min_length=1)
    markdown_path: str | None = None


class ExecutionConfig(BaseModel):
    """Sequential execution controls for CI quality gates."""
    model_config = ConfigDict(extra="forbid")
    fail_fast: bool = False


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    task: Literal["image_classification"]
    model: ModelConfig
    dataset: DatasetConfig
    attention: AttentionConfig = Field(default_factory=AttentionConfig)
    baseline: BaselineConfig
    cache: CacheConfig
    report: ReportConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


def load_suite(path: Path) -> SuiteConfig:
    with path.open("r", encoding="utf-8") as suite_file:
        contents = yaml.safe_load(suite_file)
    if not isinstance(contents, dict):
        raise ValueError("suite YAML must contain a mapping at its root")
    return SuiteConfig.model_validate(contents)
