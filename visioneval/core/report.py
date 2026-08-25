"""JSON and Markdown reports for local and CI regression feedback."""

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from visioneval.classification.scorer import EvaluationSummary
from visioneval.core.baseline import RegressionResult
from visioneval.core.types import EvaluationRecord, SelectionReason


def write_reports(json_path: Path, markdown_path: Path | None, suite_name: str, summary: EvaluationSummary, regression: RegressionResult | None, partial: dict[str, object] | None = None) -> None:
    payload = {"suite_name": suite_name, "accuracy": summary.accuracy, "sample_count": len(summary.records), "cache": {"hits": summary.cache_hits, "misses": summary.cache_misses}, "partial": partial, "regression": asdict(regression) if regression else None, "records": [{**asdict(record), "selection_reason": record.selection_reason.value} for record in summary.records]}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        status = "BASELINE UPDATED" if regression is None else ("REGRESSION" if regression.is_regression else "PASS")
        buckets = Counter(record.selection_reason.value for record in summary.records)
        bucket_summary = ", ".join(f"`{reason.value}` `{buckets[reason.value]}`" for reason in SelectionReason if buckets[reason.value]) or "none"
        lines = [f"# VisionEval: {suite_name}", "", f"- Status: **{status}**", f"- Accuracy: `{summary.accuracy:.4f}`", f"- Evaluated samples: `{len(summary.records)}`", f"- Prediction cache: `{summary.cache_hits}` hits / `{summary.cache_misses}` misses", f"- Attention buckets: {bucket_summary}"]
        if partial:
            failing = partial["failing_sample"]
            lines.extend(["- Execution: **FAIL-FAST**", f"- Remaining samples: `{partial['remaining_count']}`", f"- Failing sample: `{failing['sample_id']}` ({failing['selection_reason']}, score `{failing['attention_score']}`)"])
        if regression:
            lines.extend([f"- Accuracy drop: `{regression.accuracy_drop:.4f}`", f"- New failures: `{len(regression.new_failures)}` ({_id_list(regression.new_failures)})", f"- Fixed failures: `{len(regression.fixed_failures)}` ({_id_list(regression.fixed_failures)})"])
        markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _id_list(sample_ids: tuple[str, ...]) -> str:
    return ", ".join(f"`{sample_id}`" for sample_id in sample_ids) if sample_ids else "none"


def partial_execution(record: EvaluationRecord, evaluated_count: int, remaining_count: int) -> dict[str, object]:
    """Return report-safe deterministic evidence for a fail-fast termination."""
    return {"evaluated_count": evaluated_count, "remaining_count": remaining_count, "failing_sample": {"sample_id": record.sample_id, "selection_reason": record.selection_reason.value, "attention_score": record.attention_score, "risk_bucket": record.risk_bucket, "cache_hit": record.cache_hit}}