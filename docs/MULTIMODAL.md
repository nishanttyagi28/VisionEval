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
| `visioneval.traps` | Living hallucination traps: store, harvest, replay, hard-negatives, lockfile |
| `app/streamlit_app.py` | Side-by-side comparison, radar charts, export, open-trap panel |

## Swappable backends

Metrics and models take an explicit backend. Tests inject `ConstantAlignmentBackend`, `MockAlignmentBackend`, and `FakeVLM`. Production code can swap in `HuggingFaceCLIPBackend` / `HuggingFaceVLM` / `OpenAICompatibleVLM` after `pip install -e ".[hf,api]"`.

Importing `visioneval.models` or `visioneval.metrics` never imports `torch` or `openai`. Those modules load on first use and raise an `ImportError` that names the extra if they are missing.

## Fixtures

`visioneval.multimodal.fixtures.solid_scene` builds 64×64 RGB diagrams. Sample YAML may set `color: red_square` instead of an `image` path so CI and the Streamlit demo stay binary-free.

## Living traps

Hallucinations from a multimodal run can be stored as **living traps** until the model beats them twice in a row. This is opt-in (`traps.enabled` in the eval YAML) and uses table `vlm_traps` so it cannot clobber Phase 1 `sample_outcomes` / `predictions`. Traps do not replace the classification CI harness.

```yaml
traps:
  enabled: true
  db: artifacts/traps.sqlite3
  retire_after_consecutive_passes: 2
  generate_hard_negatives: true
```

Demo with the FakeVLM fixtures in `examples/multimodal` (no GPU, no keys):

```bash
visioneval multimodal examples/multimodal/config.yaml
visioneval traps list --db artifacts/traps.sqlite3
visioneval traps harvest reports/multimodal.json --db artifacts/traps.sqlite3
visioneval traps run --db artifacts/traps.sqlite3 --config examples/multimodal/config.yaml --budget 8
visioneval traps update-baseline --db artifacts/traps.sqlite3 --lockfile artifacts/baselines/traps.json
```

`fake-sparse` answers `A shape.` and typically harvests caption + judge traps. Open traps consume replay budget before seeded hard-negative POPE variants. A retired trap that reappears, or an open trap whose `fail_count` grows, is a lockfile regression.
