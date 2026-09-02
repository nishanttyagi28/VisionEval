"""Map VisionEval multimodal rows / YAML cases → TruthGraph claim+evidence."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from visioneval.verify.models import Claim, Evidence, VerificationResult
from visioneval.verify.verifier import verify_claim

_CLEAN = {None, "", "clean", "none"}


def _as_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def _ensure_sentence(text: str, *, floor: int = 3) -> str:
    text = text.strip()
    if len(text) >= floor:
        return text
    padded = (text + ".").strip()
    if len(padded) >= floor:
        return padded
    return (text + " present.").strip()


def evidence_from_row(row: Mapping[str, Any]) -> list[Evidence]:
    """Build ground-truth evidence from expected objects, caption, absent, POPE."""
    items: list[Evidence] = []

    caption = _as_text(row.get("caption") or row.get("ground_truth") or row.get("gt_caption"))
    if caption:
        items.append(
            Evidence(
                text=_ensure_sentence(f"Ground truth caption: {caption}"),
                source="ground_truth_caption",
                reliability=1.0,
            )
        )

    objects = [str(item).strip() for item in (row.get("objects") or ()) if str(item).strip()]
    if objects:
        joined = ", ".join(objects)
        items.append(
            Evidence(
                text=_ensure_sentence(f"The image contains the following objects: {joined}."),
                source="expected_objects",
                reliability=1.0,
            )
        )

    absent = [str(item).strip() for item in (row.get("absent_objects") or ()) if str(item).strip()]
    if absent:
        joined = ", ".join(absent)
        items.append(
            Evidence(
                text=_ensure_sentence(
                    f"The image does not contain the following objects: {joined}."
                ),
                source="absent_objects",
                reliability=1.0,
            )
        )

    pope = row.get("pope")
    if isinstance(pope, dict):
        for probe in pope.get("probes") or []:
            if not isinstance(probe, dict):
                continue
            obj = _as_text(probe.get("object") or probe.get("object_name"))
            if not obj:
                continue
            if probe.get("expected_present"):
                text = f"The image contains a {obj} object in the scene."
            else:
                text = f"The image does not contain a {obj} object in the scene."
            items.append(
                Evidence(
                    text=_ensure_sentence(text),
                    source="pope_fact",
                    reliability=0.95,
                )
            )

    # Explicit evidence list (YAML cases / TruthGraph-shaped input).
    for raw in row.get("evidence") or []:
        if not isinstance(raw, Mapping):
            continue
        text = _as_text(raw.get("text"))
        source = _as_text(raw.get("source"), "evidence")
        if not text or not source:
            continue
        reliability = float(raw.get("reliability", 0.7))
        items.append(
            Evidence(
                text=_ensure_sentence(text),
                source=source[:100],
                reliability=max(0.0, min(1.0, reliability)),
            )
        )

    return items


def claim_from_row(row: Mapping[str, Any]) -> Claim | None:
    """Extract the model claim (response / caption prediction / explicit claim)."""
    text = _as_text(
        row.get("response")
        or row.get("claim")
        or row.get("caption_pred")
        or row.get("prediction")
    )
    if not text:
        return None
    return Claim(text=_ensure_sentence(text))


def verify_row(row: Mapping[str, Any]) -> VerificationResult | None:
    """Verify one multimodal sample row. Returns None when claim or evidence missing."""
    claim = claim_from_row(row)
    if claim is None:
        return None
    evidence = evidence_from_row(row)
    if not evidence:
        return None
    return verify_claim(claim, evidence)


def _is_clean(row: Mapping[str, Any]) -> bool:
    return row.get("corruption") in _CLEAN


def _case_entry(
    *,
    case_id: str,
    model: str,
    result: VerificationResult,
    row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": case_id,
        "model": model,
        "verdict": result.verdict,
        "confidence": result.confidence,
        "claim": result.claim,
        "matched_keywords": list(result.matched_keywords),
        "supporting_evidence": [item.model_dump() for item in result.supporting_evidence],
        "contradicting_evidence": [item.model_dump() for item in result.contradicting_evidence],
    }
    if row is not None:
        sample_id = _as_text(row.get("sample_id") or row.get("id"), case_id)
        entry["sample_id"] = sample_id
        if row.get("objects") is not None:
            entry["objects"] = list(row.get("objects") or [])
        if row.get("absent_objects") is not None:
            entry["absent_objects"] = list(row.get("absent_objects") or [])
    return entry


def load_verify_input(path: Path | str) -> dict[str, Any]:
    """Load a multimodal JSON report or a simple YAML/JSON case suite."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"verify input not found: {path}")
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if isinstance(payload, list):
        return {"cases": payload, "samples": payload}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON/YAML object or array")
    return payload


def iter_verify_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize report samples and YAML cases into row dicts."""
    rows: list[dict[str, Any]] = []
    cases = payload.get("cases")
    if isinstance(cases, list) and cases:
        for index, case in enumerate(cases):
            if not isinstance(case, Mapping):
                continue
            row = dict(case)
            row.setdefault("id", row.get("sample_id") or f"case_{index}")
            rows.append(row)
        return rows

    samples = payload.get("samples")
    if isinstance(samples, list):
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                continue
            row = dict(sample)
            row.setdefault("sample_id", row.get("id") or f"sample_{index}")
            rows.append(row)
    return rows


def build_dossier(
    source: Mapping[str, Any] | Path | str,
    *,
    skip_corrupted: bool = True,
) -> dict[str, Any]:
    """Run verification over a report or case suite; return a structured dossier."""
    if isinstance(source, (Path, str)):
        payload = load_verify_input(source)
        source_name = str(source)
    else:
        payload = dict(source)
        source_name = str(payload.get("name") or "inline")

    rows = iter_verify_rows(payload)
    cases: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        if skip_corrupted and not _is_clean(row):
            continue
        result = verify_row(row)
        if result is None:
            continue
        case_id = _as_text(row.get("id") or row.get("sample_id"), f"case_{index}")
        model = _as_text(row.get("model"), "unknown")
        entry = _case_entry(case_id=case_id, model=model, result=result, row=row)
        cases.append(entry)
        counts[result.verdict] += 1

    return {
        "name": payload.get("name") or Path(source_name).stem,
        "source": source_name,
        "total": len(cases),
        "by_verdict": {
            "supported": counts.get("supported", 0),
            "contradicted": counts.get("contradicted", 0),
            "insufficient": counts.get("insufficient", 0),
        },
        "cases": cases,
    }


def format_human(dossier: Mapping[str, Any]) -> str:
    """Human-readable TruthGraph-style verification summary."""
    by = dossier.get("by_verdict") or {}
    lines = [
        "VisionEval TruthGraph verify",
        "",
        f"Source:         {dossier.get('source', '')}",
        f"Total claims:   {dossier.get('total', 0)}",
        f"Supported:      {by.get('supported', 0)}",
        f"Contradicted:   {by.get('contradicted', 0)}",
        f"Insufficient:   {by.get('insufficient', 0)}",
    ]
    cases = dossier.get("cases") or []
    if cases:
        lines.extend(["", "Cases:"])
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            lines.append(
                f"  {case.get('id', '?')}  {case.get('verdict', '?')}  "
                f"conf={case.get('confidence', 0)}  model={case.get('model', 'unknown')}"
            )
            claim = _as_text(case.get("claim"))
            if claim:
                lines.append(f"    claim: {claim}")
            keywords = case.get("matched_keywords") or []
            if keywords:
                lines.append(f"    keywords: {', '.join(str(k) for k in keywords)}")
    return "\n".join(lines) + "\n"


def to_json(dossier: Mapping[str, Any]) -> str:
    return json.dumps(dossier, indent=2, sort_keys=True, default=str) + "\n"


def contradicted_object_hint(row: Mapping[str, Any], result: VerificationResult) -> str | None:
    """Best-effort object name for maps when a claim is contradicted."""
    absent = [str(item).lower() for item in (row.get("absent_objects") or ())]
    claim_l = result.claim.lower()
    for obj in absent:
        if obj and obj in claim_l:
            return obj
    objects = [str(item) for item in (row.get("objects") or ())]
    for obj in objects:
        if obj and obj.lower() not in claim_l:
            return obj
    if result.matched_keywords:
        return result.matched_keywords[0]
    return None
