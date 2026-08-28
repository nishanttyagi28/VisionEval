"""CLIPScore: scaled cosine similarity between image and caption embeddings.

Hessel et al., *CLIPScore: A Reference-free Evaluation Metric for Image
Captioning* (EMNLP 2021): ``CLIPScore(I, C) = w * max(cos(E_I, E_C), 0)``
with the paper's default ``w = 2.5``.
"""

from __future__ import annotations

from PIL import Image

from visioneval.metrics.backends import MockAlignmentBackend
from visioneval.metrics.base import AlignmentBackend, MetricResult, PairMetric

CLIP_SCORE_WEIGHT = 2.5


def clip_score_from_similarity(similarity: float, weight: float = CLIP_SCORE_WEIGHT) -> float:
    """Apply the CLIPScore transform to a cosine similarity in ``[-1, 1]``.
"""
    return weight * max(float(similarity), 0.0)


class CLIPScore(PairMetric):
    """Image-text alignment via CLIPScore. Defaults to a mocked backend."""

    name = "clip_score"

    def __init__(
        self,
        backend: AlignmentBackend | None = None,
        *,
        weight: float = CLIP_SCORE_WEIGHT,
    ) -> None:
        self.backend = backend or MockAlignmentBackend()
        self.weight = weight

    def score(self, image: Image.Image, text: str) -> MetricResult:
        similarity = float(self.backend.image_text_similarity(image, text))
        value = clip_score_from_similarity(similarity, self.weight)
        return MetricResult(
            name=self.name,
            value=value,
            details={"cosine_similarity": similarity, "weight": self.weight},
        )
