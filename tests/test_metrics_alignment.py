"""CLIPScore / BLIP-Score math against constant and mock backends."""

from visioneval.metrics.backends import ConstantAlignmentBackend, MockAlignmentBackend, sigmoid
from visioneval.metrics.base import PairMetric
from visioneval.metrics.blip_score import BLIPScore, blip_score_from_similarity
from visioneval.metrics.clip_score import CLIP_SCORE_WEIGHT, CLIPScore, clip_score_from_similarity


def test_clip_score_formula_clips_negative_similarity() -> None:
    assert clip_score_from_similarity(0.4) == CLIP_SCORE_WEIGHT * 0.4
    assert clip_score_from_similarity(-0.3) == 0.0
    assert clip_score_from_similarity(0.0) == 0.0


def test_clip_score_uses_backend_similarity(red_square) -> None:
    metric = CLIPScore(backend=ConstantAlignmentBackend(0.4))
    result = metric.score(red_square, "a red square")
    assert isinstance(metric, PairMetric)
    assert result.name == "clip_score"
    assert result.value == CLIP_SCORE_WEIGHT * 0.4
    assert result.details["cosine_similarity"] == 0.4


def test_blip_score_clips_and_optional_sigmoid(red_square) -> None:
    assert blip_score_from_similarity(1.7) == 1.0
    assert blip_score_from_similarity(-0.2) == 0.0
    assert abs(blip_score_from_similarity(0.0, as_logit=True) - sigmoid(0.0)) < 1e-9
    metric = BLIPScore(backend=ConstantAlignmentBackend(0.8))
    result = metric.score(red_square, "a red square")
    assert result.value == 0.8
    assert result.name == "blip_score"


def test_mock_backend_is_deterministic(red_square) -> None:
    backend = MockAlignmentBackend()
    first = backend.image_text_similarity(red_square, "hello")
    second = backend.image_text_similarity(red_square, "hello")
    other = backend.image_text_similarity(red_square, "goodbye")
    assert first == second
    assert first != other


def test_mock_backend_respects_explicit_text_map(red_square) -> None:
    backend = MockAlignmentBackend(scores={"caption": 0.91})
    assert backend.image_text_similarity(red_square, "caption") == 0.91
