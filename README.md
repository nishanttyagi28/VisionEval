# VisionEval

**Catch vision-model regressions and hallucinations before they ship — in CI, not in a slide deck.**

[![CI](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml/badge.svg)](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-visioneval.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://visioneval.streamlit.app)

**[Try the live demo →](https://visioneval.streamlit.app)** · no login · CPU-only fake models

---

## The problem

Static vision / VLM benchmarks go stale. Models memorize the set. A flaky hallucination that bit you last week never shows up on the next leaderboard cut. Teams need a **release gate** that fails the PR — and a memory of failures that does not reset every run.

## What VisionEval does

| Feature | In plain English | Why it matters |
| --- | --- | --- |
| **Classification release gate** | `visioneval run` compares this run to a git-trackable baseline and exits `1` on a new failure or accuracy drop. | Same habit as unit tests: regressions block merge instead of showing up in chat. |
| **Living traps + CI gate** | Hallucinations become durable SQLite tests. They retire only after **two consecutive passes**. `visioneval traps gate` fails CI on `new_open` / `reappeared` / `worse`. | Intermittent “yes, there is a cat” bugs stay in the suite until the model actually beats them. |
| **Failure maps** | `visioneval maps` builds a CPU-only map of *where* the model fails — by object, probe type, sample, and metric. | Debugging “it hallucinates sometimes” becomes “it fails POPE on *absent* objects in these samples.” |
| **Budget-aware eval** | `visioneval budget-analyze` ranks samples with no model call; `visioneval run --use-budget` evaluates that recommended subset. | Spend GPU/API budget on previous failures and high-risk tags first. Reduction is **per-run** (`estimated_reduction`) — not a fixed marketing percentage. |
| **Multimodal layer** | Alignment (CLIP/BLIP), POPE + LLM-judge probes, image corruptions, VLM adapters, Streamlit dashboard. | One repo covers classification CI and VLM hallucination / robustness work without a second toolchain. |

## What this improves

Evidence from this repo only — no invented speed or cost claims.

| Fact | Value |
| --- | --- |
| Automated tests | **100** pytest cases across **23** test modules |
| Top-level CLI | `run`, `budget-analyze`, `multimodal`, `maps`, `traps` |
| Traps subcommands | `list`, `harvest`, `run`, `update-baseline`, `gate` |
| Hallucination probe types harvested | `pope`, `judge`, `caption` |
| Map metrics | `pope_miss`, `judge_flag`, `caption_mismatch` |
| Trap retirement rule | `--retire-after` default **2** consecutive passes |
| Python | **3.10+** |
| License | **Apache-2.0** |
| Version | **0.1.0** (actively developed) |
| Demo fixtures | 3-sample classification demo; 2 multimodal scenes (`red_square`, `blue_circle`); 12-sample classification example catalog |
| CI | GitHub Actions `test` job — Ubuntu, Python 3.10, `pytest` |
| Optional extras | `dev`, `hf`, `api`, `ui`, `metrics`, `all` |
| Robustness corruptions | gaussian noise, motion blur, contrast jitter, occlusion |

## Why this is useful

If you ship a vision or VLM change, you want two things: (1) a **hard gate** that fails CI when quality drops, and (2) a **memory** of the specific hallucinations that bit you. VisionEval is built for that workflow — baselines and trap lockfiles are git-trackable; reports are Markdown + JSON; the demo runs without GPUs or API keys.

## How it works

**Simple flow**

1. Define a suite / eval YAML and a model adapter (or use the FakeVLM demo).
2. Run classification (`visioneval run`) and/or multimodal (`visioneval multimodal`).
3. Promote a baseline / trap lockfile when outcomes look right.
4. On every PR: re-run and fail on regression (`run` / `traps gate`).
5. Optionally: budget-analyze → `--use-budget`, harvest traps, inspect `maps`.

**Optional architecture** (for engineers)

```text
visioneval run            → attention / budget select → adapter → SQLite → baseline lock → exit 0/1
visioneval multimodal     → metrics + corruptions + profiling → Markdown/JSON
visioneval traps harvest  → POPE / judge / caption failures → vlm_traps (separate tables)
visioneval traps gate     → compare DB to lockfile → exit 1 on new_open / reappeared / worse
visioneval maps           → CPU-only locus map from report and/or open traps (read-only)
visioneval budget-analyze → risk rank, no inference → feeds run --use-budget
```

Layers sit beside each other. Living traps and maps never write classification tables.

---

## Install

Python 3.10+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
export PYTHONPATH="$(pwd)"         # Windows: $env:PYTHONPATH = (Get-Location).Path
python -m pytest
```

Real VLMs / CLIP / paid judge / dashboard:

```bash
python -m pip install -e ".[hf,api,ui]"
```

API keys come from the environment (`OPENAI_API_KEY` by default). Do not commit secrets.

## CLI

```text
visioneval run SUITE [--update-baseline] [--use-budget] [--traps-db PATH]
visioneval budget-analyze SUITE [--json] [--top N] [--traps-db PATH]
visioneval multimodal CONFIG [--json-out PATH] [--markdown-out PATH]
visioneval maps [REPORT] [--json] [--db PATH]
visioneval traps list|harvest|run|gate|update-baseline
```

| Command | Role |
| --- | --- |
| `visioneval run` | Classification release gate. `--use-budget` evaluates the budget-analyzer subset. |
| `visioneval budget-analyze` | Deterministic risk ranking. No inference. |
| `visioneval multimodal` | Alignment, POPE, judge, robustness, profiling, reports. |
| `visioneval maps` | Black-box hallucination map (CPU-only). |
| `visioneval traps gate` | Visual red-team CI blocker (`new_open` / `reappeared` / `worse`). |

## Quick start

```bash
# Classification gate
visioneval run demo_suite.yaml --update-baseline
visioneval run demo_suite.yaml
visioneval budget-analyze demo_suite.yaml --json
visioneval run demo_suite.yaml --use-budget

# Multimodal + living traps + maps (CPU, no GPU, no API keys)
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

Full walkthrough: [QUICKSTART.md](QUICKSTART.md). Also: [docs/BUDGET.md](docs/BUDGET.md), [docs/MULTIMODAL.md](docs/MULTIMODAL.md), [docs/MAPS.md](docs/MAPS.md), [DEMO_GUIDE.md](DEMO_GUIDE.md).

### Living traps

Opt-in SQLite memory for VLM hallucinations. Default retire rule: **two consecutive passes**. Open traps consume replay budget first. Does not replace `visioneval run`.

### Failure maps

```bash
visioneval maps reports/mm.json
visioneval maps reports/mm.json --db artifacts/traps.sqlite3 --json
visioneval maps --db artifacts/traps.sqlite3
```

Read-only. Harvest still owns persistence.

### Budget-aware runs

```bash
visioneval budget-analyze demo_suite.yaml --json
visioneval run demo_suite.yaml --use-budget
# or attention.use_budget: true in suite YAML
```

`estimated_reduction` is computed **per run** from `recommended / total`. Do not treat any single percentage as a product claim.

### Multimodal + dashboard

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

**Plain English:** lock a good classification baseline, run the gate on every change, optionally spend less budget on low-risk samples, then for VLMs harvest hallucinations into traps and fail CI if they get worse.

```bash
# 1) Promote a known-good classification baseline
visioneval run demo_suite.yaml --update-baseline

# 2) CI-style check (exit 0 = PASS, 1 = REGRESSION)
visioneval run demo_suite.yaml

# 3) Optional: rank risk, then evaluate the recommended subset
visioneval budget-analyze demo_suite.yaml --json
visioneval run demo_suite.yaml --use-budget

# 4) Multimodal report → living traps → lockfile → gate
visioneval multimodal examples/multimodal/config.yaml \
  --json-out reports/mm.json --markdown-out reports/mm.md
visioneval traps harvest reports/mm.json --db artifacts/traps.sqlite3
visioneval traps run --db artifacts/traps.sqlite3 \
  --config examples/multimodal/config.yaml --budget 8
visioneval traps update-baseline --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json
visioneval traps gate --db artifacts/traps.sqlite3 \
  --lockfile artifacts/baselines/traps.json --json

# 5) See where failures cluster
visioneval maps reports/mm.json --json
```

---

## Demo

Live: **[visioneval.streamlit.app](https://visioneval.streamlit.app)**

<p align="center">
  <img src="docs/assets/streamlit-red-square.svg" alt="Side-by-side VLM comparison on red_square" width="410">
  &nbsp;
  <img src="docs/assets/streamlit-blue-circle.svg" alt="Metric radar and Markdown/JSON export on blue_circle" width="410">
</p>

<p align="center">
  <sub>
    Left: <code>red_square</code> — fake-left names the square, fake-right stays generic.
    Right: <code>blue_circle</code> — metric radar and export.
  </sub>
</p>

Local:

```bash
python -m pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

---

## Status

**WIP · v0.1.0 · actively developed.** Useful today for demos, CI experiments, and local VLM eval loops. Not claiming production-hardened multi-tenant SaaS. APIs and suite schemas may still move; pin a commit if you depend on behavior.

## Why I built this

I kept hitting the same gap: vision quality lived in notebooks and one-off scripts, while the rest of the stack had real CI. Intermittent VLM hallucinations were especially annoying — fixed once, gone from the suite, back next week. VisionEval is my attempt to make regression gates and failure memory as boring and reliable as unit tests, starting from a small builder’s toolkit rather than a platform pitch.

## License

Apache-2.0. See [LICENSE](LICENSE).
