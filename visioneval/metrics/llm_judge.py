"""Structured LLM-as-a-Judge for caption/VQA quality.

Scores three axes in ``[0, 1]`` and always returns JSON-serialisable output:

* ``detail_richness`` — how much grounded visual detail the response contains
* ``factual_consistency`` — agreement with the caption and object list
* ``spatial_accuracy`` — use of spatial relations that match the scene notes
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from visioneval.metrics.base import MetricResult

DEFAULT_JUDGE_PROMPT = """You are an expert multimodal evaluator.
Score the MODEL RESPONSE against the IMAGE CONTEXT.

Return a single JSON object with exactly these keys:
- detail_richness: number in [0, 1]
- factual_consistency: number in [0, 1]
- spatial_accuracy: number in [0, 1]
- rationale: short string
- flags: array of strings (hallucinated objects, missing objects, etc.)

IMAGE CAPTION: {caption}
GROUND-TRUTH OBJECTS: {objects}
SPATIAL NOTES: {spatial_notes}
MODEL RESPONSE: {response}
"""

_SPATIAL_WORDS = (
    "left",
    "right",
    "above",
    "below",
    "top",
    "bottom",
    "on",
    "under",
    "behind",
    "front",
    "next to",
    "beside",
    "between",
    "inside",
    "center",
    "middle",
)


@dataclass(frozen=True)
class JudgeVerdict:
    """Structured judge output. All scores are in ``[0, 1]``."""

    detail_richness: float
    factual_consistency: float
    spatial_accuracy: float
    rationale: str = ""
    flags: tuple[str, ...] = ()

    @property
    def overall(self) -> float:
        return (self.detail_richness + self.factual_consistency + self.spatial_accuracy) / 3.0

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        payload["overall"] = self.overall
        return payload


class JudgeBackend(ABC):
    """Swappable judge implementation (mock heuristic or paid API)."""

    @abstractmethod
    def judge(
        self,
        *,
        image: Image.Image,
        response: str,
        caption: str,
        objects: Sequence[str],
        spatial_notes: str,
        prompt_template: str,
    ) -> JudgeVerdict:
        """Return a structured verdict for one response."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class MockJudgeBackend(JudgeBackend):
    """Heuristic judge used in tests and offline demos. Never calls an API."""

    def judge(
        self,
        *,
        image: Image.Image,
        response: str,
        caption: str,
        objects: Sequence[str],
        spatial_notes: str,
        prompt_template: str,
    ) -> JudgeVerdict:
        del image, prompt_template  # unused; kept for the backend contract
        words = _tokens(response)
        unique = set(words)
        content = [w for w in unique if len(w) > 2]
        detail = _clamp01(len(content) / 18.0)

        object_list = [obj.lower() for obj in objects]
        mentioned = [obj for obj in object_list if obj in response.lower()]
        coverage = len(mentioned) / max(len(object_list), 1)
        caption_tokens = set(_tokens(caption))
        overlap = len(unique & caption_tokens) / max(len(caption_tokens), 1)
        factual = _clamp01(0.6 * coverage + 0.4 * overlap)

        spatial_hits = sum(1 for word in _SPATIAL_WORDS if word in response.lower())
        note_hits = 0
        if spatial_notes:
            note_hits = sum(1 for word in _tokens(spatial_notes) if word in unique)
        spatial = _clamp01(0.35 + 0.15 * min(spatial_hits, 3) + 0.2 * min(note_hits, 2))

        flags: list[str] = []
        missing = [obj for obj in object_list if obj not in response.lower()]
        if missing:
            flags.append("missing_objects:" + ",".join(missing))
        if len(words) < 4:
            flags.append("too_short")

        rationale = (
            f"heuristic detail={detail:.2f} factual={factual:.2f} spatial={spatial:.2f}; "
            f"mentioned {len(mentioned)}/{max(len(object_list), 1)} objects"
        )
        return JudgeVerdict(
            detail_richness=round(detail, 4),
            factual_consistency=round(factual, 4),
            spatial_accuracy=round(spatial, 4),
            rationale=rationale,
            flags=tuple(flags),
        )


class OpenAIJudgeBackend(JudgeBackend):
    """JSON judge via an OpenAI-compatible chat API. Requires the ``api`` extra."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url

    def judge(
        self,
        *,
        image: Image.Image,
        response: str,
        caption: str,
        objects: Sequence[str],
        spatial_notes: str,
        prompt_template: str,
    ) -> JudgeVerdict:
        del image
        try:
            import os

            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "LLM judge API backend requires the 'api' extra: pip install -e '.[api]'"
            ) from exc
        api_key = os.environ.get(self.api_key_env, "")
        client_kwargs: dict[str, Any] = {"api_key": api_key or "EMPTY"}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        prompt = prompt_template.format(
            caption=caption,
            objects=", ".join(objects),
            spatial_notes=spatial_notes or "(none)",
            response=response,
        )
        completion = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        content = completion.choices[0].message.content or "{}"
        return verdict_from_json(content)


def verdict_from_json(payload: str) -> JudgeVerdict:
    """Parse a judge JSON blob, including fenced markdown responses."""
    text = payload.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    data = json.loads(text)
    flags = data.get("flags") or []
    if isinstance(flags, str):
        flags = [flags]
    return JudgeVerdict(
        detail_richness=_clamp01(data.get("detail_richness", 0.0)),
        factual_consistency=_clamp01(data.get("factual_consistency", 0.0)),
        spatial_accuracy=_clamp01(data.get("spatial_accuracy", 0.0)),
        rationale=str(data.get("rationale", "")),
        flags=tuple(str(flag) for flag in flags),
    )


class LLMJudge:
    """Pair-style wrapper that exposes ``MetricResult`` plus a structured verdict."""

    name = "llm_judge"

    def __init__(
        self,
        backend: JudgeBackend | None = None,
        *,
        prompt_template: str = DEFAULT_JUDGE_PROMPT,
    ) -> None:
        self.backend = backend or MockJudgeBackend()
        self.prompt_template = prompt_template

    def score(
        self,
        image: Image.Image,
        response: str,
        *,
        caption: str = "",
        objects: Sequence[str] = (),
        spatial_notes: str = "",
    ) -> tuple[MetricResult, JudgeVerdict]:
        verdict = self.backend.judge(
            image=image,
            response=response,
            caption=caption,
            objects=objects,
            spatial_notes=spatial_notes,
            prompt_template=self.prompt_template,
        )
        result = MetricResult(
            name=self.name,
            value=verdict.overall,
            details=verdict.as_dict(),
        )
        return result, verdict
