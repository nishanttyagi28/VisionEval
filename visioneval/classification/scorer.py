"""Classification scoring for selected evaluation samples."""

from collections.abc import Iterable
from dataclasses import dataclass

from visioneval.classification.adapter import ClassificationAdapter
from visioneval.core.cache import SQLiteCache, prediction_cache_key
from visioneval.core.types import EvaluationRecord, SelectedSample


@dataclass(frozen=True)
class EvaluationSummary:
    records: tuple[EvaluationRecord, ...]
    accuracy: float
    cache_hits: int = 0
    cache_misses: int = 0


def evaluate(selected_samples: Iterable[SelectedSample], adapter: ClassificationAdapter, cache: SQLiteCache | None = None, model_hash: str = "", preprocess_hash: str = "") -> EvaluationSummary:
    """Evaluate selected samples with optional content-addressable prediction reuse."""
    records: list[EvaluationRecord] = []
    hits = misses = 0
    for selected in selected_samples:
        cache_key = prediction_cache_key(model_hash, preprocess_hash, selected.sample.image_path) if cache and selected.sample.image_path else None
        prediction = cache.get_prediction(cache_key) if cache_key else None
        cache_hit = prediction is not None
        if cache_hit:
            hits += 1
        else:
            prediction = adapter(selected.sample)
            misses += 1
            if cache_key:
                cache.put_prediction(cache_key, prediction)
        records.append(EvaluationRecord(selected.sample.sample_id, selected.sample.label, prediction.label, prediction.confidence, prediction.label == selected.sample.label, selected.reason, selected.attention_score, selected.risk_bucket, cache_hit))
    return EvaluationSummary(tuple(records), sum(record.correct for record in records) / len(records) if records else 0.0, hits, misses)