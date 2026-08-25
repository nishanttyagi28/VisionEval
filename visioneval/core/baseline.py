"""Git-trackable baseline persistence and per-sample regression comparison."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from visioneval.core.types import EvaluationRecord


@dataclass(frozen=True)
class Baseline:
    suite_name: str
    accuracy: float
    sample_count: int
    outcomes: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class RegressionResult:
    baseline_accuracy: float
    candidate_accuracy: float
    accuracy_drop: float
    is_regression: bool
    new_failures: tuple[str, ...] = ()
    fixed_failures: tuple[str, ...] = ()


def load_baseline(path: Path) -> Baseline:
    with path.open("r", encoding="utf-8") as baseline_file:
        return Baseline(**json.load(baseline_file))


def save_baseline(path: Path, baseline: Baseline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(baseline), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def compare_baseline(baseline: Baseline, candidate_accuracy: float, allowed_accuracy_drop: float, records: tuple[EvaluationRecord, ...] = ()) -> RegressionResult:
    accuracy_drop = round(baseline.accuracy - candidate_accuracy, 12)
    current = {record.sample_id: record.correct for record in records}
    new_failures = tuple(sorted(sample_id for sample_id, correct in current.items() if not correct and baseline.outcomes.get(sample_id, True)))
    fixed_failures = tuple(sorted(sample_id for sample_id, correct in current.items() if correct and baseline.outcomes.get(sample_id) is False))
    return RegressionResult(baseline.accuracy, candidate_accuracy, accuracy_drop, accuracy_drop > allowed_accuracy_drop or bool(new_failures), new_failures, fixed_failures)