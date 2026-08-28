"""Structured LLM-as-a-judge: JSON parsing and heuristic mock backend."""

from visioneval.metrics.llm_judge import LLMJudge, MockJudgeBackend, verdict_from_json


def test_verdict_from_json_reads_fenced_payload() -> None:
    blob = """```json
    {"detail_richness": 0.8, "factual_consistency": 1.2, "spatial_accuracy": 0.4,
     "rationale": "ok", "flags": ["missing_objects:cat"]}
    ```"""
    verdict = verdict_from_json(blob)
    assert verdict.detail_richness == 0.8
    assert verdict.factual_consistency == 1.0  # clamped
    assert verdict.spatial_accuracy == 0.4
    assert verdict.flags == ("missing_objects:cat",)
    assert abs(verdict.overall - (0.8 + 1.0 + 0.4) / 3) < 1e-9


def test_mock_judge_rewards_detailed_grounded_responses(red_square) -> None:
    judge = LLMJudge(backend=MockJudgeBackend())
    rich = (
        "A red square sits in the center of a white background, "
        "with the square clearly visible."
    )
    poor = "thing"
    rich_result, rich_verdict = judge.score(
        red_square,
        rich,
        caption="A red square on a white background.",
        objects=["square", "red square"],
        spatial_notes="square in the center",
    )
    poor_result, poor_verdict = judge.score(
        red_square,
        poor,
        caption="A red square on a white background.",
        objects=["square", "red square"],
        spatial_notes="square in the center",
    )
    assert rich_result.value > poor_result.value
    assert rich_verdict.factual_consistency > poor_verdict.factual_consistency
    assert "too_short" in poor_verdict.flags
    assert set(rich_verdict.as_dict()) >= {
        "detail_richness",
        "factual_consistency",
        "spatial_accuracy",
        "rationale",
        "flags",
        "overall",
    }
