"""BLIP-Score: image-text matching probability from a BLIP (or mock) backend."""

from __future__ import annotations

from PIL import Image

from visioneval.metrics.backends import MockAlignmentBackend, sigmoid
from visioneval.metrics.base import AlignmentBackend, MetricResult, PairMetric


def blip_score_from_similarity(similarity: float, *, as_logit: bool = False) -> float:
    """Map a backend score onto ``[0, 1]``.

    When ``as_logit`` is true the value is passed through a sigmoid (BLIP ITM
    heads sometimes expose raw logits). Otherwise the score is clipped.
    """
    if as_logit:
        return sigmoid(float(similarity))
    return min(1.0, max(0.0, float(similarity)))


class BLIPScore(PairMetric):
    """Image-text matching score. Defaults to a mocked backend."""

    name = "blip_score"

    def __init__(
        self,
        backend: AlignmentBackend | None = None,
        *,
        as_logit: bool = False,
    ) -> None:
        self.backend = backend or MockAlignmentBackend()
        self.as_logit = as_logit

    def score(self, image: Image.Image, text: str) -> MetricResult:
        raw = float(self.backend.image_text_similarity(image, text))
        value = blip_score_from_similarity(raw, as_logit=self.as_logit)
        return MetricResult(
            name=self.name,
            value=value,
            details={"raw_similarity": raw, "as_logit": self.as_logit},
        )
