"""End-to-end multimodal pipeline with fake models and in-memory fixtures."""

from pathlib import Path

from visioneval.multimodal.config import MultimodalEvalConfig, load_multimodal_config
from visioneval.multimodal.pipeline import run_multimodal_eval


def _config() -> dict:
    return {
        "name": "pipeline-demo",
        "models": [
            {
                "name": "fake-good",
                "kind": "fake",
                "responses": {
                    "red_square": "A red square sits in the center of a white background.",
                    "blue_circle": "A blue circle sits in the center of a white background.",
                },
                "latency_ms": 0.0,
            }
        ],
        "samples": [
            {
                "id": "red_square",
                "color": "red_square",
                "caption": "A red square on a white background.",
                "objects": ["square"],
                "absent_objects": ["cat"],
                "spatial_notes": "square in the center",
            }
        ],
        "metrics": {
            "clip": True,
            "blip": True,
            "pope": True,
            "llm_judge": True,
            "clip_backend": "mock",
            "blip_backend": "mock",
        },
        "corruptions": {
            "enabled": True,
            "types": ["gaussian_noise"],
            "severities": [0.5],
            "seed": 0,
        },
        "judge": {"backend": "mock"},
    }


def test_load_multimodal_config_from_yaml(tmp_path: Path) -> None:
    samples = tmp_path / "samples.yaml"
    samples.write_text(
        "samples:\n  - id: red_square\n    color: red_square\n    caption: A red square.\n    objects: [square]\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "name: from-file\nmodels:\n  - {name: fake, kind: fake}\nsamples_path: samples.yaml\n",
        encoding="utf-8",
    )
    loaded = load_multimodal_config(config_path)
    assert loaded.name == "from-file"
    assert loaded.samples[0].id == "red_square"


def test_pipeline_scores_clean_and_corrupted(tmp_path: Path) -> None:
    config = MultimodalEvalConfig.model_validate(_config())
    json_path = tmp_path / "mm.json"
    md_path = tmp_path / "mm.md"
    result = run_multimodal_eval(config, json_path=json_path, markdown_path=md_path)
    sample_ids = {(row["sample_id"], row["corruption"], row["severity"]) for row in result["samples"]}
    assert ("red_square", None, 0.0) in sample_ids
    assert ("red_square", "gaussian_noise", 0.5) in sample_ids
    clean = next(row for row in result["samples"] if row["corruption"] is None)
    assert "clip_score" in clean["metrics"]
    assert "blip_score" in clean["metrics"]
    assert clean["pope"]["f1"] == 1.0
    assert "detail_richness" in clean["judge"]
    assert result["degradation"]
    assert json_path.is_file()
    markdown = md_path.read_text(encoding="utf-8")
    assert "pipeline-demo" in markdown
    assert "POPE" in markdown


def test_cli_multimodal_command(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from visioneval.cli import app

    config_path = tmp_path / "eval.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: cli-demo",
                "models:",
                "  - name: fake",
                "    kind: fake",
                "    latency_ms: 0",
                "    responses:",
                "      red_square: A red square in the center.",
                "samples:",
                "  - id: red_square",
                "    color: red_square",
                "    caption: A red square.",
                "    objects: [square]",
                "    absent_objects: [cat]",
                "corruptions:",
                "  enabled: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["multimodal", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "cli-demo" in result.output
