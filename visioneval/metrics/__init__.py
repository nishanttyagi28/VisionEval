"""Swappable multimodal metrics: alignment, POPE, and LLM-as-a-judge."""

from visioneval.metrics.base import AlignmentBackend, MetricResult, PairMetric
from visioneval.metrics.blip_score import BLIPScore
from visioneval.metrics.clip_score import CLIPScore, clip_score_from_similarity
from visioneval.metrics.llm_judge import JudgeVerdict, LLMJudge, MockJudgeBackend
from visioneval.metrics.pope import PopeQuestion, PopeScores, aggregate_pope, parse_yes_no

__all__ = [
    "AlignmentBackend",
    "BLIPScore",
    "CLIPScore",
    "JudgeVerdict",
    "LLMJudge",
    "MetricResult",
    "MockJudgeBackend",
    "PairMetric",
    "PopeQuestion",
    "PopeScores",
    "aggregate_pope",
    "clip_score_from_similarity",
    "parse_yes_no",
]
