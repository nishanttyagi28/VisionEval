"""JSON and Markdown serializers for multimodal evaluation runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _metric_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def report_to_json(payload: Mapping[str, Any]) -> str:
    """Pretty-printed, key-sorted JSON with a trailing newline."""
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def report_to_markdown(payload: Mapping[str, Any]) -> str:
    """Human-readable summary: models, metrics, POPE, judge, degradation."""
    name = payload.get("name", "multimodal-eval")
    lines = [
        f"# VisionEval multimodal: {name}",
        "",
        "This report is produced by the **multimodal evaluation layer**. "
        "It sits beside the Phase 1 classification CI harness "
        "(`visioneval run`) and does not replace it.",
        "",
        "## Models",
        "",
    ]
    models = payload.get("models") or []
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict):
                lines.append(f"- `{model.get('name', '?')}` ({model.get('kind', '?')})")
            else:
                lines.append(f"- `{model}`")
    else:
        lines.append("- (none)")

    lines.extend(["", "## Per-sample scores", ""])
    samples = payload.get("samples") or []
    if not samples:
        lines.append("_No samples evaluated._")
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        lines.append(f"### {sample.get('sample_id', 'sample')} · `{sample.get('model', '')}`")
        lines.append("")
        lines.append(
            f"- Corruption: `{sample.get('corruption') or 'clean'}` "
            f"(severity `{sample.get('severity', 0.0)}`)"
        )
        profile = sample.get("profile") or {}
        if isinstance(profile, dict):
            ttft = profile.get("ttft_ms")
            total = profile.get("total_ms")
            vram = profile.get("vram_mb")
            tps = profile.get("throughput_tps")
            lines.append(
                f"- Profile: TTFT `{_metric_cell(ttft) if ttft is not None else 'n/a'} ms`, "
                f"total `{_metric_cell(total) if total is not None else 'n/a'} ms`, "
                f"VRAM `{_metric_cell(vram) if vram is not None else 'n/a'} MiB`, "
                f"throughput `{_metric_cell(tps) if tps is not None else 'n/a'} tok/s"
            )
        response = sample.get("response", "")
        lines.append(f"- Response: {response}")
        metrics = sample.get("metrics") or {}
        if isinstance(metrics, dict):
            for key, raw in sorted(metrics.items()):
                if isinstance(raw, dict) and "value" in raw:
                    lines.append(f"- `{key}`: `{_metric_cell(raw['value'])}`")
                else:
                    lines.append(f"- `{key}`: `{_metric_cell(raw)}`")
        pope = sample.get("pope")
        if isinstance(pope, dict):
            lines.append(
                "- POPE: "
                f"acc `{_metric_cell(pope.get('accuracy', 0))}`, "
                f"P `{_metric_cell(pope.get('precision', 0))}`, "
                f"R `{_metric_cell(pope.get('recall', 0))}`, "
                f"F1 `{_metric_cell(pope.get('f1', 0))}`"
            )
        judge = sample.get("judge")
        if isinstance(judge, dict):
            lines.append(
                "- Judge: "
                f"detail `{_metric_cell(judge.get('detail_richness', 0))}`, "
                f"factual `{_metric_cell(judge.get('factual_consistency', 0))}`, "
                f"spatial `{_metric_cell(judge.get('spatial_accuracy', 0))}`"
            )
        lines.append("")

    degradation = payload.get("degradation") or []
    lines.extend(["## Robustness / degradation", ""])
    if not degradation:
        lines.append("_No corruption sweep was requested._")
        lines.append("")
    else:
        lines.append("| Metric | Corruption | Clean | Resilience |")
        lines.append("| --- | --- | --- | --- |")
        for row in degradation:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| `{row.get('metric', '')}` | `{row.get('corruption', '')}` | "
                f"`{_metric_cell(row.get('clean_score', 0))}` | "
                f"`{_metric_cell(row.get('resilience', 0))}` |"
            )
        lines.append("")
    return "\n".join(lines)


def write_multimodal_reports(
    payload: Mapping[str, Any],
    json_path: Path | None,
    markdown_path: Path | None,
) -> None:
    """Write JSON and/or Markdown reports, creating parent directories."""
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report_to_json(payload), encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report_to_markdown(payload), encoding="utf-8")
