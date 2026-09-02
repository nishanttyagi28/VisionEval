"""Pydantic models for claim verification (vendored from TruthGraph)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Claim(BaseModel):
    """A statement to verify (typically a VLM caption or asserted fact)."""

    text: str = Field(min_length=3, max_length=500)
    source_url: str | None = None
    status: str = "pending"


class Evidence(BaseModel):
    """One piece of ground-truth or supporting text."""

    text: str = Field(min_length=3, max_length=1000)
    source: str = Field(min_length=2, max_length=100)
    reliability: float = Field(default=0.7, ge=0.0, le=1.0)


class VerificationResult(BaseModel):
    """Structured verdict dossier for one claim."""

    claim: str
    verdict: Literal["supported", "contradicted", "insufficient"]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[Evidence]
    contradicting_evidence: list[Evidence]
    matched_keywords: list[str]
