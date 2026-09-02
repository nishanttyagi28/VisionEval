"""TruthGraph-style verification: core engine, adapter, CLI, maps hook."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from visioneval.cli import app
from visioneval.maps.extract import events_from_report
from visioneval.verify.adapter import build_dossier, evidence_from_row, verify_row
from visioneval.verify.models import Claim, Evidence
from visioneval.verify.text_analyzer import (
    calculate_relevance,
    contains_negation,
    extract_numbers,
    tokenize,
)
from visioneval.verify.verifier import verify_claim

FIXTURES = Path(__file__).resolve().parent.parent / "examples" / "verify"


def test_tokenize_removes_common_words() -> None:
    words = tokenize("The Earth has one natural satellite")
    assert "the" not in words
    assert "has" not in words
    assert "earth" in words
    assert "satellite" in words


def test_calculate_relevance_and_negation_numbers() -> None:
    score, keywords = calculate_relevance(
        "Earth has one natural satellite",
        "NASA confirms Earth has one natural satellite called the Moon",
    )
    assert score == 1.0
    assert "earth" in keywords
    assert contains_negation("Earth does not have two moons") is True
    assert contains_negation("Earth has one moon") is False
    assert extract_numbers("The scores were 42 and 98.5") == {"42", "98.5"}


def test_supported_contradicted_insufficient() -> None:
    claim = Claim(text="Earth has one natural satellite.")
    supported = verify_claim(
        claim,
        [
            Evidence(
                text="Earth has one natural satellite called the Moon.",
                source="NASA",
                reliability=0.95,
            )
        ],
    )
    assert supported.verdict == "supported"
    assert supported.confidence == 0.95

    contradicted = verify_claim(
        claim,
        [
            Evidence(
                text="Earth does not have one natural satellite.",
                source="Incorrect Source",
                reliability=0.80,
            )
        ],
    )
    assert contradicted.verdict == "contradicted"
    assert contradicted.confidence == 0.80

    insufficient = verify_claim(
        claim,
        [
            Evidence(
                text="Python is a programming language used for software development.",
                source="Programming Book",
                reliability=0.90,
            )
        ],
    )
    assert insufficient.verdict == "insufficient"
    assert insufficient.confidence == 0.0


def test_adapter_supported_caption_against_objects() -> None:
    row = {
        "sample_id": "red_square",
        "model": "fake-good",
        "response": "A red square on a white background.",
        "caption": "A red square on a white background.",
        "objects": ["square", "red square"],
        "absent_objects": ["cat", "car"],
    }
    result = verify_row(row)
    assert result is not None
    assert result.verdict == "supported"
    assert evidence_from_row(row)
    assert any(item.source == "expected_objects" for item in evidence_from_row(row))


def test_adapter_contradicted_when_absent_object_leaked() -> None:
    row = {
        "sample_id": "red_square",
        "model": "fake-bad",
        "response": "There is a cat sitting next to a car on the floor.",
        "caption": "A red square on a white background.",
        "objects": ["square", "red square"],
        "absent_objects": ["cat", "car"],
    }
    result = verify_row(row)
    assert result is not None
    assert result.verdict == "contradicted"


def test_build_dossier_from_yaml_cases() -> None:
    dossier = build_dossier(FIXTURES / "cases.yaml")
    assert dossier["total"] == 3
    by = dossier["by_verdict"]
    assert by["supported"] >= 1
    assert by["contradicted"] >= 1
    assert by["insufficient"] >= 1
    ids = {case["id"] for case in dossier["cases"]}
    assert ids == {"supported_square", "contradicted_cat", "insufficient_unrelated"}


def test_build_dossier_from_sample_report_skips_corrupted() -> None:
    dossier = build_dossier(FIXTURES / "sample_report.json")
    assert dossier["total"] == 2  # corrupted row skipped
    verdicts = {case["id"]: case["verdict"] for case in dossier["cases"]}
    assert verdicts["red_square"] == "supported"
    assert verdicts["hallucinated_cat"] == "contradicted"


def test_maps_surfaces_claim_contradicted() -> None:
    report = json.loads((FIXTURES / "sample_report.json").read_text(encoding="utf-8"))
    events = events_from_report(report)
    contradicted = [e for e in events if e.metric == "claim_contradicted"]
    assert contradicted
    assert all(e.probe_type == "verify" for e in contradicted)
    assert any(e.sample_id == "hallucinated_cat" for e in contradicted)


def test_verify_cli_human_and_json(tmp_path: Path) -> None:
    runner = CliRunner()
    human = runner.invoke(app, ["verify", str(FIXTURES / "cases.yaml")])
    assert human.exit_code == 0, human.output
    assert "VisionEval TruthGraph verify" in human.output
    assert "Supported:" in human.output

    json_result = runner.invoke(app, ["truth", str(FIXTURES / "cases.yaml"), "--json"])
    assert json_result.exit_code == 0, json_result.output
    payload = json.loads(json_result.output)
    assert payload["total"] == 3
    assert "by_verdict" in payload
