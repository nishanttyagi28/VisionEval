"""POPE yes/no parsing and accuracy / precision / recall / F1 aggregation."""

from visioneval.metrics.pope import (
    PopeQuestion,
    aggregate_pope,
    build_pope_questions,
    parse_yes_no,
)


def test_parse_yes_no_accepts_common_forms() -> None:
    assert parse_yes_no("Yes.") is True
    assert parse_yes_no("yes, there is a cat") is True
    assert parse_yes_no("No") is False
    assert parse_yes_no("No, there is not") is False
    assert parse_yes_no("Y") is True
    assert parse_yes_no("") is None
    assert parse_yes_no("maybe") is None


def test_aggregate_pope_perfect_scores() -> None:
    pairs = [
        (PopeQuestion("cat", True), "Yes"),
        (PopeQuestion("dog", False), "No"),
        (PopeQuestion("car", True), "Yes, a car is visible."),
        (PopeQuestion("tree", False), "No."),
    ]
    scores = aggregate_pope(pairs)
    assert scores.accuracy == 1.0
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0
    assert scores.true_positives == 2
    assert scores.true_negatives == 2
    assert scores.yes_ratio == 0.5


def test_aggregate_pope_known_confusion_matrix() -> None:
    pairs = [
        (PopeQuestion("cat", True), "Yes"),   # TP
        (PopeQuestion("dog", True), "No"),    # FN
        (PopeQuestion("car", False), "Yes"),  # FP
        (PopeQuestion("tree", False), "No"),  # TN
        (PopeQuestion("bus", True), "hmm"),   # unparseable expected-yes -> FN
    ]
    scores = aggregate_pope(pairs)
    assert scores.true_positives == 1
    assert scores.false_negatives == 2
    assert scores.false_positives == 1
    assert scores.true_negatives == 1
    assert scores.precision == 0.5          # 1 / (1+1)
    assert abs(scores.recall - 1 / 3) < 1e-9  # 1 / (1+2)
    expected_f1 = 2 * 0.5 * (1 / 3) / (0.5 + 1 / 3)
    assert abs(scores.f1 - expected_f1) < 1e-9
    assert scores.accuracy == 0.4           # (1+1)/5


def test_aggregate_pope_empty_is_zero() -> None:
    scores = aggregate_pope([])
    assert scores.total == 0
    assert scores.accuracy == 0.0
    assert scores.f1 == 0.0


def test_build_pope_questions_marks_presence() -> None:
    questions = build_pope_questions(["square"], ["cat"], split="adversarial")
    assert questions[0].expected_present is True
    assert questions[1].expected_present is False
    assert "square" in questions[0].rendered_prompt()
    assert questions[0].split == "adversarial"
