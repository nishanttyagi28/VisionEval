# VisionEval

An open-source tool for finding and preventing repeated mistakes in vision and multimodal AI models.

[![CI](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml/badge.svg)](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-visioneval.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://visioneval.streamlit.app)

AI models can look fine when you only watch the overall score. A model can still keep making the same mistake on the same kind of image — saying something is there when it isn’t, or missing something that is.

Those mistakes are easy to forget. You fix one, the suite moves on, and a few weeks later the same failure shows up again. VisionEval is a small toolkit I built so those failures stick around as tests, get re-checked, and can stop a bad change before it lands.

**[Live demo](https://visioneval.streamlit.app)** — side-by-side captions and scores, no login, runs on CPU with fake models (no API key).

---

## In simple terms

If someone asks “what does this actually do?”, here is the short version.

**1. It remembers mistakes.**
Suppose a model says there is a cat in an image when there isn’t one. VisionEval can turn that into a lasting test and ask the model again later. The test stays open until the model gets it right twice in a row.

**2. It shows where the model is struggling.**
Instead of only “accuracy went down,” you can see clusters: which objects, which samples, which kinds of checks failed.

**3. It avoids testing everything when the risky examples matter more.**
You can rank samples by risk (past failures, tagged high-risk cases, low-confidence rows, new ones) and run that smaller set first. The gate still works the same way — you just spend less time on the easy stuff.

---

## What I built

### Stops a quiet regression from getting merged

`visioneval run` compares this run to a baseline you saved in git. If a new failure shows up, or accuracy on the locked set drops, the command exits with code `1` — the same idea as a failing unit test.

```bash
visioneval run demo_suite.yaml --update-baseline   # promote once, when it looks right
visioneval run demo_suite.yaml                    # later: exit 0 pass, exit 1 regression
```

**Why it helps:** You don’t have to notice a bad vision change by eye in a PR review.

### Remembers mistakes the model has already made

After a multimodal run, VisionEval can harvest failures (wrong yes/no answers, weak judge scores, captions that miss expected objects) into a local SQLite store. Those “traps” get replayed. They only retire after **two consecutive passes**. `visioneval traps gate` fails CI if an old trap comes back or gets worse.

```bash
visioneval traps harvest reports/mm.json --db artifacts/traps.sqlite3
visioneval traps update-baseline --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json
visioneval traps gate --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json --json
```

**Why it helps:** A bug you already fixed is less likely to sneak back unnoticed.

### Shows where the model keeps going wrong

`visioneval maps` reads a report (and optionally the traps database) and builds a simple map of failures — by object, sample, and failure type. No model call. CPU only.

```bash
visioneval maps reports/mm.json --json
visioneval maps reports/mm.json --db artifacts/traps.sqlite3 --json
```

**Why it helps:** Debugging gets more specific than “it hallucinates sometimes.”

### Tests the risky examples first

`visioneval budget-analyze` ranks the catalog without running the model. `visioneval run --use-budget` then evaluates that recommended subset. Previous failures always stay in. How much smaller the set is depends on *your* data that run — the tool prints an `estimated_reduction` figure; there is no fixed “we cut X%” claim in this repo.

```bash
visioneval budget-analyze demo_suite.yaml --json
visioneval run demo_suite.yaml --use-budget
```

**Why it helps:** Less time and compute on samples that rarely catch regressions.

### Compares multimodal models in one place

CLIP/BLIP-style alignment scores, yes/no hallucination probes, a simple judge, image corruptions (noise, blur, contrast, occlusion), and a Streamlit UI. Fake models work offline so you can try the loop without a GPU.

```bash
visioneval multimodal examples/multimodal/config.yaml \
  --json-out reports/mm.json --markdown-out reports/mm.md
```

**Why it helps:** Classification CI and VLM checks live in the same project instead of a pile of notebooks.

---

## A few numbers

Only things I can point to in this repository:

| | |
| --- | --- |
| Automated tests | **100** pytest cases in **23** modules |
| Main CLI commands | `run`, `budget-analyze`, `multimodal`, `maps`, `traps` |
| Traps subcommands | `list`, `harvest`, `run`, `update-baseline`, `gate` |
| Failure types harvested | 3 (`pope`, `judge`, `caption`) |
| Map metrics | 3 (`pope_miss`, `judge_flag`, `caption_mismatch`) |
| Trap retire rule | default **2** consecutive passes |
| Robustness corruptions | 4 (gaussian noise, motion blur, contrast, occlusion) |
| Demo fixtures | 3-sample classification demo; 2 multimodal scenes; 12-sample example catalog |
| Python | 3.10+ |
| License | Apache-2.0 |
| Version | 0.1.0 |
| CI | one GitHub Actions job: Ubuntu, Python 3.10, pytest |
| Live demo | CPU-only fake models, no API key |

I am not claiming a fixed cost saving or speedup. Budget reduction is calculated per run from your suite.

---

## Why this matters

The useful part isn’t another score on a dashboard. It’s knowing that a mistake you fixed last week didn’t quietly come back this week — and having a boring, repeatable check for that before you release.

It also makes investigation less vague (“where does it fail?”) and lets you spend evaluation effort on the examples that already burned you, instead of always running the whole pile.

---

## How it works

```text
Model
  ↓
Test examples
  ↓
Find mistakes
  ↓
Remember the important ones
  ↓
Test them again
  ↓
Block regressions (exit 1)
```

For engineers, the pieces look like this:

```text
visioneval run            → select samples → adapter → SQLite → baseline lock → exit 0/1
visioneval multimodal     → metrics + corruptions + profiling → Markdown/JSON
visioneval traps harvest  → failures → vlm_traps (separate tables from classification)
visioneval traps gate     → compare DB to lockfile → exit 1 on new_open / reappeared / worse
visioneval maps           → CPU-only map from report and/or open traps (read-only)
visioneval budget-analyze → risk rank, no inference → optional input to run --use-budget
```

Living traps and maps do not write into the classification tables. Layers sit beside each other.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
export PYTHONPATH="$(pwd)"         # Windows: $env:PYTHONPATH = (Get-Location).Path
python -m pytest
```

Optional stacks:

```bash
python -m pip install -e ".[hf,api,ui]"     # real VLMs, OpenAI-compatible APIs, Streamlit
python -m pip install -e ".[metrics]"      # real CLIP/BLIP (pulls torch)
python -m pip install -e ".[all]"
```

API keys stay in the environment (`OPENAI_API_KEY` by default). Don’t commit secrets.

## CLI

```text
visioneval run SUITE [--update-baseline] [--use-budget] [--traps-db PATH]
visioneval budget-analyze SUITE [--json] [--top N] [--traps-db PATH]
visioneval multimodal CONFIG [--json-out PATH] [--markdown-out PATH]
visioneval maps [REPORT] [--json] [--db PATH]
visioneval traps list|harvest|run|gate|update-baseline
```

## Quick start

```bash
# Classification gate
visioneval run demo_suite.yaml --update-baseline
visioneval run demo_suite.yaml
visioneval budget-analyze demo_suite.yaml --json
visioneval run demo_suite.yaml --use-budget

# Multimodal + traps + maps (CPU, no GPU, no API keys)
visioneval multimodal examples/multimodal/config.yaml \
  --json-out reports/mm.json --markdown-out reports/mm.md
visioneval traps harvest reports/mm.json --db artifacts/traps.sqlite3
visioneval traps run --db artifacts/traps.sqlite3 \
  --config examples/multimodal/config.yaml --budget 8
visioneval traps update-baseline --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json
visioneval traps gate --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json --json
visioneval maps reports/mm.json --json
```

More detail: [QUICKSTART.md](QUICKSTART.md), [docs/BUDGET.md](docs/BUDGET.md), [docs/MULTIMODAL.md](docs/MULTIMODAL.md), [docs/MAPS.md](docs/MAPS.md), [DEMO_GUIDE.md](DEMO_GUIDE.md).

### Multimodal dashboard

```bash
python -m pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

### Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

---

## Example workflow

1. Lock a baseline when the classification results look right.
2. On every change, run the gate.
3. Optionally rank samples and evaluate the riskier subset.
4. For VLMs, harvest failures into traps, promote a lockfile, fail CI if they return.
5. Open a map when you want to see where failures cluster.

```bash
visioneval run demo_suite.yaml --update-baseline
visioneval run demo_suite.yaml
visioneval budget-analyze demo_suite.yaml --json
visioneval run demo_suite.yaml --use-budget

visioneval multimodal examples/multimodal/config.yaml \
  --json-out reports/mm.json --markdown-out reports/mm.md
visioneval traps harvest reports/mm.json --db artifacts/traps.sqlite3
visioneval traps run --db artifacts/traps.sqlite3 \
  --config examples/multimodal/config.yaml --budget 8
visioneval traps update-baseline --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json
visioneval traps gate --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json --json
visioneval maps reports/mm.json --json
```

---

## Demo

**Live:** [https://visioneval.streamlit.app](https://visioneval.streamlit.app)

You can compare two fake models on simple shapes, look at scores, and export Markdown/JSON. It runs on CPU and does not ask for an API key.

<p align="center">
  <img src="docs/assets/streamlit-red-square.svg" alt="Side-by-side VLM comparison on red_square" width="410">
  &nbsp;
  <img src="docs/assets/streamlit-blue-circle.svg" alt="Metric radar and Markdown/JSON export on blue_circle" width="410">
</p>

---

## Why I built this

I kept seeing model evaluation reduced to a score, while the individual mistakes were easy to forget. I wanted a small tool that could remember those failures and make them part of the normal development loop — closer to unit tests than to a one-off notebook.

---

## Status

**WIP · v0.1.0 · actively developed.**

Useful today for demos, experiments, and local CI-style checks. Schemas and APIs may still move. Pin a commit if you depend on specific behavior. I’m not calling this a finished product.

## License

Apache-2.0. See [LICENSE](LICENSE).
