"""End-to-end multimodal evaluation: generate, score, corrupt, report."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from visioneval.metrics.backends import (
    HuggingFaceBLIPBackend,
    HuggingFaceCLIPBackend,
    MockAlignmentBackend,
)
from visioneval.metrics.blip_score import BLIPScore
from visioneval.metrics.clip_score import CLIPScore
from visioneval.metrics.llm_judge import (
    DEFAULT_JUDGE_PROMPT,
    LLMJudge,
    MockJudgeBackend,
    OpenAIJudgeBackend,
)
from visioneval.metrics.pope import PopeQuestion, aggregate_pope, build_pope_questions
from visioneval.models.base import BaseVLM
from visioneval.models.factory import build_model
from visioneval.multimodal.config import (
    DEFAULT_CAPTION_PROMPT,
    MultimodalEvalConfig,
    SampleConfig,
    load_multimodal_config,
)
from visioneval.multimodal.fixtures import load_sample_image
from visioneval.profiling.profiler import profile_generation
from visioneval.report.serializers import write_multimodal_reports
from visioneval.robustness.corruptions import apply_corruption
from visioneval.robustness.degradation import summarize_degradation


def _alignment_backend(kind: str, model_id: str, family: str) -> Any:
    if kind == "hf":
        if family == "clip":
            return HuggingFaceCLIPBackend(model_id=model_id)
        return HuggingFaceBLIPBackend(model_id=model_id)
    return MockAlignmentBackend(boost_terms=("red", "blue", "square", "circle", "green"))


def _build_metrics(config: MultimodalEvalConfig) -> dict[str, Any]:
    toggles = config.metrics
    metrics: dict[str, Any] = {}
    if toggles.clip:
        metrics["clip"] = CLIPScore(
            backend=_alignment_backend(toggles.clip_backend, toggles.clip_model_id, "clip")
        )
    if toggles.blip:
        metrics["blip"] = BLIPScore(
            backend=_alignment_backend(toggles.blip_backend, toggles.blip_model_id, "blip")
        )
    if toggles.llm_judge:
        if config.judge.backend == "api":
            judge_backend: Any = OpenAIJudgeBackend(
                model=config.judge.model,
                api_key_env=config.judge.api_key_env,
                base_url=config.judge.base_url,
            )
        else:
            judge_backend = MockJudgeBackend()
        metrics["judge"] = LLMJudge(
            backend=judge_backend,
            prompt_template=config.judge.prompt_template or DEFAULT_JUDGE_PROMPT,
        )
    return metrics


def _pope_for_response(sample: SampleConfig, response: str, ask) -> dict[str, Any]:
    questions = build_pope_questions(sample.objects, sample.absent_objects)
    pairs: list[tuple[PopeQuestion, str]] = []
    for question in questions:
        answer = ask(question.rendered_prompt())
        pairs.append((question, answer))
    scores = aggregate_pope(pairs)
    return scores.as_dict()


def _score_sample(
    *,
    image: Image.Image,
    sample: SampleConfig,
    response: str,
    metrics: dict[str, Any],
    config: MultimodalEvalConfig,
    model: BaseVLM,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"metrics": {}}
    if "clip" in metrics:
        payload["metrics"]["clip_score"] = metrics["clip"].score(image, response).__dict__
    if "blip" in metrics:
        payload["metrics"]["blip_score"] = metrics["blip"].score(image, response).__dict__
    if "judge" in metrics:
        result, verdict = metrics["judge"].score(
            image,
            response,
            caption=sample.caption,
            objects=sample.objects,
            spatial_notes=sample.spatial_notes,
        )
        payload["metrics"]["llm_judge"] = result.__dict__
        payload["judge"] = verdict.as_dict()
    if config.metrics.pope:
        def ask(question: str) -> str:
            # POPE is a yes/no probe; reuse the same VLM without re-profiling.
            return model.generate(image, question, sample_id=sample.id).text

        payload["pope"] = _pope_for_response(sample, response, ask)
    return payload


def _evaluate_image(
    *,
    image: Image.Image,
    sample: SampleConfig,
    model: BaseVLM,
    metrics: dict[str, Any],
    config: MultimodalEvalConfig,
    corruption: str | None,
    severity: float,
) -> dict[str, Any]:
    prompt = sample.prompt or config.caption_prompt or DEFAULT_CAPTION_PROMPT
    generation, profile = profile_generation(
        lambda: model.generate(image, prompt, sample_id=sample.id)
    )
    scored = _score_sample(
        image=image,
        sample=sample,
        response=generation.text,
        metrics=metrics,
        config=config,
        model=model,
    )
    return {
        "sample_id": sample.id,
        "model": model.name,
        "corruption": corruption,
        "severity": severity,
        "prompt": prompt,
        "response": generation.text,
        "profile": profile.as_dict(),
        **scored,
    }


def run_multimodal_eval(
    config: MultimodalEvalConfig | Path | str,
    *,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the four-pillar eval. ``config`` may be a loaded object or a YAML path."""
    if not isinstance(config, MultimodalEvalConfig):
        path = Path(config)
        config_dir = path.parent
        config = load_multimodal_config(path)
    root = str(config_dir) if config_dir is not None else None

    object_map = {sample.id: list(sample.objects) for sample in config.samples}
    models = []
    for spec in config.models:
        payload = spec.to_factory_dict()
        if spec.kind == "fake":
            payload["object_map"] = object_map
        models.append(build_model(payload))
    metrics = _build_metrics(config)
    records: list[dict[str, Any]] = []
    clean_scores: dict[tuple[str, str, str], float] = {}
    corrupted: dict[tuple[str, str, str, str], dict[float, float]] = defaultdict(dict)

    def _metric_values(record: dict[str, Any]) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, payload in (record.get("metrics") or {}).items():
            if isinstance(payload, dict) and "value" in payload:
                values[name] = float(payload["value"])
        pope = record.get("pope")
        if isinstance(pope, dict) and "f1" in pope:
            values["pope_f1"] = float(pope["f1"])
        return values

    for model in models:
        for sample in config.samples:
            image = load_sample_image(sample.image, sample.color, root)
            clean = _evaluate_image(
                image=image,
                sample=sample,
                model=model,
                metrics=metrics,
                config=config,
                corruption=None,
                severity=0.0,
            )
            records.append(clean)
            for metric_name, value in _metric_values(clean).items():
                clean_scores[(model.name, sample.id, metric_name)] = value

            if not config.corruptions.enabled:
                continue
            for corruption in config.corruptions.types:
                for severity in config.corruptions.severities:
                    if severity <= 0.0:
                        continue
                    corrupted_image = apply_corruption(
                        image, corruption, severity, seed=config.corruptions.seed
                    )
                    record = _evaluate_image(
                        image=corrupted_image,
                        sample=sample,
                        model=model,
                        metrics=metrics,
                        config=config,
                        corruption=corruption,
                        severity=float(severity),
                    )
                    records.append(record)
                    for metric_name, value in _metric_values(record).items():
                        corrupted[(model.name, sample.id, metric_name, corruption)][
                            float(severity)
                        ] = value

    degradation = []
    for (model_name, sample_id, metric_name), clean_value in sorted(clean_scores.items()):
        for corruption in config.corruptions.types if config.corruptions.enabled else []:
            series = corrupted.get((model_name, sample_id, metric_name, corruption))
            if not series:
                continue
            report = summarize_degradation(
                metric=metric_name,
                corruption=corruption,
                clean_score=clean_value,
                scores_by_severity=series,
            )
            payload = report.as_dict()
            payload["model"] = model_name
            payload["sample_id"] = sample_id
            degradation.append(payload)

    result: dict[str, Any] = {
        "name": config.name,
        "models": [spec.model_dump() for spec in config.models],
        "samples": records,
        "degradation": degradation,
        "layer": "multimodal",
        "composes_with": "visioneval run (Phase 1 classification CI harness)",
    }

    json_out = json_path
    md_out = markdown_path
    if config.report is not None:
        if json_out is None and config.report.json_path:
            json_out = Path(config.report.json_path)
        if md_out is None and config.report.markdown_path:
            md_out = Path(config.report.markdown_path)
    write_multimodal_reports(result, json_out, md_out)
    return result
