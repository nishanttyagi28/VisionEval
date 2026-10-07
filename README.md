# VisionEval

A regression-testing CLI for vision models: lock a baseline, re-run on every change, and keep past failures as tests until the model has fixed them.

[![CI](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml/badge.svg)](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A single accuracy number can go up while a model keeps failing the same important images. A failure gets fixed, drops out of the test set, and comes back a few versions later. VisionEval keeps those failures around and checks them again. It exits non-zero when results regress, so it can run as a CI step.

It covers two kinds of model:

- **Image classifiers.** You write a YAML suite that points at your predict function and an image manifest. A run is compared against a saved baseline and fails if accuracy drops or new failures appear.
- **Vision-language models.** Captioning and image question answering are scored with CLIP/BLIP-style metrics, POPE-style yes/no probes, an optional LLM judge, and robustness checks (noise, blur, contrast, occlusion).

A [live demo](https://visioneval.streamlit.app) on Streamlit Community Cloud compares two fake models on simple shapes. It needs no API key, but may take a moment to wake up.

## Install

Python 3.10 or newer. Not on PyPI yet, so install from a clone:

```bash
git clone https://github.com/nishanttyagi28/VisionEval.git
cd VisionEval
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
export PYTHONPATH="$(pwd)"         # Windows: $env:PYTHONPATH = (Get-Location).Path
```

The `PYTHONPATH` line lets suites import adapters from the repo root, such as `demo_adapter.py`. Without it, `visioneval run demo_suite.yaml` fails with `No module named 'demo_adapter'`.

Optional extras: `hf` (Hugging Face models, pulls in torch), `api` (OpenAI-compatible endpoints, reading `OPENAI_API_KEY` from the environment), `metrics` (real CLIP/BLIP), `ui` (Streamlit), or `all`.

## Quick start

Classification, using the bundled demo adapter and a three-image manifest:

```bash
visioneval run demo_suite.yaml --update-baseline   # save the current result as the baseline
visioneval run demo_suite.yaml                     # compare against it; non-zero exit on regression
```

Vision-language models, using fake models and generated images (runs on CPU, no keys needed):

```bash
visioneval multimodal examples/multimodal/config.yaml \
  --json-out reports/mm.json --markdown-out reports/mm.md
visioneval verify examples/verify/cases.yaml --json
```

## Features

- **Baseline gate:** `visioneval run SUITE` compares accuracy and per-sample results against a saved baseline, with an allowed accuracy drop set in the suite.
- **Smaller test runs:** `visioneval budget-analyze SUITE` ranks samples by risk: past failures, samples tagged high-risk, low-confidence predictions, and a share of new ones. `visioneval run SUITE --use-budget` evaluates only that subset. The reduction depends on your data and is reported per run.
- **Failure memory (`traps`):** VLM failures from a report go into a SQLite database. They are re-run later and retired only after passing twice in a row. `traps gate` fails if a retired failure comes back or new ones appear compared to a lockfile:

  ```bash
  visioneval traps harvest reports/mm.json --db artifacts/traps.sqlite3
  visioneval traps run --db artifacts/traps.sqlite3 --config examples/multimodal/config.yaml --budget 8
  visioneval traps update-baseline --db artifacts/traps.sqlite3 --lockfile artifacts/baselines/traps.json
  visioneval traps gate --db artifacts/traps.sqlite3 --lockfile artifacts/baselines/traps.json --json
  ```

- **Failure summary (`maps`):** `visioneval maps reports/mm.json --json` groups failures by model, metric, object, probe type and sample, so you can see where errors cluster.
- **Caption checks (`verify`):** uses the claim checker from [TruthGraph](https://github.com/nishanttyagi28/truthgraph), copied into `visioneval/verify/`. It treats a model's answer as a claim and checks it against ground truth (expected objects, objects that should be absent, reference captions). The result is supported, contradicted or insufficient. It runs on CPU without any API.
- **Dashboard:** `streamlit run app/streamlit_app.py` (needs the `ui` extra) shows a side-by-side comparison with Markdown/JSON export.

## How it works

`visioneval/core` holds the suite loader, runner, baseline comparison, budget selection and a SQLite prediction cache. `visioneval/classification` provides lazy Torchvision and ONNX Runtime adapters. The vision-language layer lives in `metrics`, `models` (a fake model, Hugging Face, and OpenAI-compatible clients), `robustness`, `traps` and `maps`. Importing those modules doesn't import torch or openai until a real backend is used.

Further reading: [QUICKSTART.md](QUICKSTART.md), [DEMO_GUIDE.md](DEMO_GUIDE.md), [docs/MULTIMODAL.md](docs/MULTIMODAL.md), [docs/BUDGET.md](docs/BUDGET.md), [docs/MAPS.md](docs/MAPS.md), [docs/TRUTHGRAPH.md](docs/TRUTHGRAPH.md) (the claim check used by `verify`), and [ARCHITECTURE_GAP_REPORT.md](ARCHITECTURE_GAP_REPORT.md) for known gaps.

## Limitations

- Version 0.1.0. Suite and report formats may still change.
- The bundled demos use fake models and tiny generated images. They show the workflow, not real model quality.
- The default metric backends are mocks. Real CLIP/BLIP scores need the `metrics` extra and a download of model weights.
- `verify` is keyword and rule based. It checks against the ground truth you provide and doesn't look at the image itself.
- Real VLM evaluation needs the `hf` extra (and usually a GPU) or an API key for an OpenAI-compatible endpoint.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

CI runs the test suite on Python 3.10 for every push and pull request.

## License

Apache-2.0. See [LICENSE](LICENSE).
