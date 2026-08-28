"""POPE: Polling-based Object Probing Evaluation for visual hallucinations.

Li et al., *Evaluating Object Hallucination in Large Vision-Language Models*
(EMNLP 2023). Each probe is a yes/no question of the form
"Is there a {object} in the image?". Ground truth comes from an object list.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


_YES_RE = re.compile(r"^\s*(yes|y|true|1)\b", re.IGNORECASE)
_NO_RE = re.compile(r"^\s*(no|n|false|0)\b", re.IGNORECASE)
_YES_ANYWHERE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_ANYWHERE = re.compile(r"\bno\b", re.IGNORECASE)


@dataclass(frozen=True)
class PopeQuestion:
    """A single yes/no object probe."""

    object_name: str
    expected_present: bool
    split: str = "random"
    prompt: str = ""

    def rendered_prompt(self) -> str:
        if self.prompt:
            return self.prompt
        article = "an" if self.object_name[:1].lower() in "aeiou" else "a"
        return f"Is there {article} {self.object_name} in the image?"


@dataclass(frozen=True)
class PopeScores:
    """POPE aggregation: accuracy, precision, recall, F1, plus confusion counts."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total: int
    yes_ratio: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "total": self.total,
            "yes_ratio": self.yes_ratio,
        }


def parse_yes_no(answer: str) -> bool | None:
    """Parse a free-form VLM answer into yes (True), no (False), or unknown."""
    stripped = answer.strip()
    if not stripped:
        return None
    if _YES_RE.match(stripped):
        return True
    if _NO_RE.match(stripped):
        return False
    yes_hit = _YES_ANYWHERE.search(stripped)
    no_hit = _NO_ANYWHERE.search(stripped)
    if yes_hit and not no_hit:
        return True
    if no_hit and not yes_hit:
        return False
    return None


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def aggregate_pope(pairs: Iterable[tuple[PopeQuestion, str]]) -> PopeScores:
    """Aggregate (question, model_answer) pairs into POPE metrics.

    Unparseable answers are treated as incorrect (they increment FP or FN
    depending on the expected label) so a silent model cannot inflate recall.
    """
    tp = fp = tn = fn = 0
    yes_answers = 0
    total = 0
    for question, answer in pairs:
        total += 1
        predicted = parse_yes_no(answer)
        expected = question.expected_present
        if predicted is True:
            yes_answers += 1
            if expected:
                tp += 1
            else:
                fp += 1
        elif predicted is False:
            if expected:
                fn += 1
            else:
                tn += 1
        else:
            # Unknown / garbled: count as the wrong class.
            if expected:
                fn += 1
            else:
                fp += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, total)
    yes_ratio = _safe_div(yes_answers, total)
    return PopeScores(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        total=total,
        yes_ratio=yes_ratio,
    )


def build_pope_questions(
    present: Sequence[str],
    absent: Sequence[str],
    *,
    split: str = "random",
) -> list[PopeQuestion]:
    """Build a balanced-style probe list from present and absent object names."""
    questions = [PopeQuestion(name, True, split) for name in present]
    questions.extend(PopeQuestion(name, False, split) for name in absent)
    return questions
