# Multimodal evaluation layer

This package originally shipped a **CI-first image-classification regression harness**. The multimodal framework is a second layer: same repo, same `pytest` job, different CLI command (`visioneval multimodal`) and a Streamlit app.

It does **not** replace attention sampling, SQLite failure memory, or baseline lockfiles. Those remain the release gate for classification.

## Package map

| Module | Role |
| --- | --- |
| `visioneval.metrics` | `PairMetric` ABC, CLIPScore, BLIP-Score, POPE, structured LLM judge |
| `visioneval.metrics.backends` | Mock similarity (default) and HuggingFace CLIP/BLIP extras |
| `visioneval.robustness` | Gaussian noise, motion blur, contrast jitter, occlusion; drop-off / resilience |
| `visioneval.models` | `VisionLanguageModel` protocol, `FakeVLM`, `HuggingFaceVLM`, `OpenAICompatibleVLM` |
| `visioneval.profiling` | TTFT, total ms, peak VRAM, tokens/s |
| `visioneval.report` | Markdown + JSON for multimodal runs (separate from `visioneval.core.report`) |
| `visioneval.multimodal` | Pydantic YAML config + pipeline |
| `app/streamlit_app.py` | Side-by-side comparison, radar charts, export |

## Swappable backends

Metrics and models take an explicit backend. Tests inject `ConstantAlignmentBackend`, `MockAlignmentBackend`, and `FakeVLM`. Production code can swap in `HuggingFaceCLIPBackend` / `HuggingFaceVLM` / `OpenAICompatibleVLM` after `pip install -e ".[hf,api]"`.

Importing `visioneval.models` or `visioneval.metrics` never imports `torch` or `openai`. Those modules load on first use and raise an `ImportError` that names the extra if they are missing.

## Fixtures

`visioneval.multimodal.fixtures.solid_scene` builds 64×64 RGB diagrams. Sample YAML may set `color: red_square` instead of an `image` path so CI and the Streamlit demo stay binary-free.
