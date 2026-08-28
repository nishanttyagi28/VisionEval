"""Streamlit dashboard for side-by-side multimodal model comparison.

Run from the repository root (no GPU or API keys required for the demo):

    pip install -e ".[ui]"
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

from visioneval.metrics.backends import MockAlignmentBackend
from visioneval.metrics.blip_score import BLIPScore
from visioneval.metrics.clip_score import CLIPScore
from visioneval.metrics.llm_judge import LLMJudge
from visioneval.metrics.pope import aggregate_pope, build_pope_questions
from visioneval.models.fake import FakeVLM
from visioneval.models.factory import build_model, describe_available_backends
from visioneval.multimodal.fixtures import DEMO_SAMPLES, solid_scene
from visioneval.profiling.profiler import profile_generation
from visioneval.report.serializers import report_to_json, report_to_markdown

st.set_page_config(page_title="VisionEval multimodal", layout="wide")


def _demo_image(sample_id: str) -> Image.Image:
    for sample in DEMO_SAMPLES:
        if sample["id"] == sample_id:
            return solid_scene(str(sample["color"]))
    return solid_scene("red_square")


def _radar(values: dict[str, float]):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    labels = list(values.keys())
    scores = [values[key] for key in labels]
    labels.append(labels[0])
    scores.append(scores[0])
    fig = go.Figure(go.Scatterpolar(r=scores, theta=labels, fill="toself", name="metrics"))
    fig.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}}, showlegend=False, height=320)
    return fig


def _normalise_clip(value: float) -> float:
    return max(0.0, min(1.0, value / 2.5))


def _evaluate(image: Image.Image, sample: dict, model, caption: str, objects: list[str], absent: list[str]) -> dict:
    generation, profile = profile_generation(
        lambda: model.generate(image, "Describe the image in detail.", sample_id=sample.get("id"))
    )
    clip = CLIPScore(backend=MockAlignmentBackend(boost_terms=("red", "blue", "square", "circle")))
    blip = BLIPScore(backend=MockAlignmentBackend(boost_terms=("red", "blue", "square", "circle")))
    judge = LLMJudge()
    clip_result = clip.score(image, generation.text)
    blip_result = blip.score(image, generation.text)
    judge_result, verdict = judge.score(
        image, generation.text, caption=caption, objects=objects, spatial_notes=sample.get("spatial_notes", "")
    )
    questions = build_pope_questions(objects, absent)

    def ask(prompt: str) -> str:
        return model.generate(image, prompt, sample_id=sample.get("id")).text

    pope = aggregate_pope((q, ask(q.rendered_prompt())) for q in questions)
    return {
        "sample_id": sample.get("id", "upload"),
        "model": model.name,
        "corruption": None,
        "severity": 0.0,
        "response": generation.text,
        "profile": profile.as_dict(),
        "metrics": {
            "clip_score": clip_result.__dict__,
            "blip_score": blip_result.__dict__,
            "llm_judge": judge_result.__dict__,
        },
        "pope": pope.as_dict(),
        "judge": verdict.as_dict(),
        "radar": {
            "clip": _normalise_clip(clip_result.value),
            "blip": blip_result.value,
            "pope_f1": pope.f1,
            "detail": verdict.detail_richness,
            "factual": verdict.factual_consistency,
            "spatial": verdict.spatial_accuracy,
        },
    }


st.title("VisionEval — multimodal comparison")
st.caption(
    "Side-by-side VLM responses, hallucination (POPE) scores, and a metric radar. "
    "This dashboard is the UI for the multimodal eval layer; the Phase 1 CI harness "
    "is still `visioneval run`."
)

backends = describe_available_backends()
st.sidebar.markdown("**Available backends**")
st.sidebar.json(backends)
st.sidebar.markdown(
    "HF and API adapters are optional extras. The demo Fake VLM always works."
)

source = st.sidebar.radio("Image source", ["Demo fixture", "Upload"])
if source == "Upload":
    uploaded = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg", "webp"])
    caption = st.sidebar.text_input("Caption / reference", "A user-uploaded image.")
    objects_raw = st.sidebar.text_input("Present objects (comma)", "object")
    absent_raw = st.sidebar.text_input("Absent objects (comma)", "cat, car")
    if uploaded is None:
        st.info("Upload an image to compare models.")
        st.stop()
    image = Image.open(uploaded).convert("RGB")
    sample = {
        "id": "upload",
        "caption": caption,
        "objects": [p.strip() for p in objects_raw.split(",") if p.strip()],
        "absent_objects": [p.strip() for p in absent_raw.split(",") if p.strip()],
        "spatial_notes": "",
    }
else:
    options = {item["id"]: item for item in DEMO_SAMPLES}
    choice = st.sidebar.selectbox("Demo sample", list(options))
    sample = options[choice]
    image = _demo_image(choice)
    caption = str(sample["caption"])

objects = list(sample.get("objects") or [])
absent = list(sample.get("absent_objects") or [])

left_kind = st.sidebar.selectbox("Left model", ["fake"], index=0)
right_kind = st.sidebar.selectbox("Right model", ["fake", "hf (if installed)", "api (if installed)"], index=0)

fake_left = FakeVLM(
    name="fake-left",
    responses={
        "red_square": "A red square sits in the center of a white background.",
        "blue_circle": "A blue circle sits in the center of a white background.",
        "upload": "A user-uploaded photograph with several visible objects.",
    },
    object_map={sample.get("id", "upload"): objects},
)
models = [fake_left]

if right_kind.startswith("fake"):
    models.append(
        FakeVLM(
            name="fake-right",
            default_response="A geometric figure is visible.",
            object_map={sample.get("id", "upload"): objects},
        )
    )
elif right_kind.startswith("hf"):
    if not backends["hf"]:
        st.sidebar.warning("Install the hf extra to enable HuggingFace models.")
        models.append(FakeVLM(name="hf-unavailable-fallback", default_response="(hf extra not installed)"))
    else:
        model_id = st.sidebar.text_input("HF model id", "Qwen/Qwen2-VL-2B-Instruct")
        models.append(build_model({"kind": "hf", "name": "hf-right", "model_id": model_id}))
else:
    if not backends["api"]:
        st.sidebar.warning("Install the api extra to enable OpenAI-compatible models.")
        models.append(FakeVLM(name="api-unavailable-fallback", default_response="(api extra not installed)"))
    else:
        model_name = st.sidebar.text_input("API model", "gpt-4o-mini")
        base_url = st.sidebar.text_input("Base URL (optional)", "")
        models.append(
            build_model(
                {
                    "kind": "api",
                    "name": "api-right",
                    "model": model_name,
                    "base_url": base_url or None,
                }
            )
        )

cols = st.columns(len(models) + 1)
with cols[0]:
    st.subheader("Input")
    st.image(image, caption=caption, use_container_width=True)
    st.markdown("**Present objects:** " + (", ".join(objects) or "—"))
    st.markdown("**Absent objects (POPE negatives):** " + (", ".join(absent) or "—"))

rows = []
for column, model in zip(cols[1:], models):
    with column:
        st.subheader(model.name)
        try:
            row = _evaluate(image, sample, model, caption, objects, absent)
        except Exception as exc:  # pragma: no cover - UI fallback
            st.error(f"{type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        st.write(row["response"])
        pope = row["pope"]
        st.metric("POPE F1 (hallucination)", f"{pope['f1']:.3f}")
        st.caption(
            f"acc {pope['accuracy']:.3f} · P {pope['precision']:.3f} · "
            f"R {pope['recall']:.3f} · yes-ratio {pope['yes_ratio']:.3f}"
        )
        profile = row["profile"]
        st.caption(
            f"TTFT {profile['ttft_ms']:.1f} ms · total {profile['total_ms']:.1f} ms · "
            f"VRAM {profile['vram_mb'] if profile['vram_mb'] is not None else 'n/a'} MiB"
        )
        fig = _radar(row["radar"])
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(row["radar"])

if rows:
    payload = {
        "name": "streamlit-demo",
        "models": [{"name": row["model"], "kind": "interactive"} for row in rows],
        "samples": rows,
        "degradation": [],
    }
    st.divider()
    st.subheader("Export")
    md = report_to_markdown(payload)
    js = report_to_json(payload)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Markdown", md, file_name="visioneval-multimodal.md", mime="text/markdown")
    with c2:
        st.download_button("Download JSON", js, file_name="visioneval-multimodal.json", mime="application/json")
    with st.expander("Markdown preview"):
        st.code(md, language="markdown")
