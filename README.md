# VisionEval

**The eval harness that remembers when your VLM hallucinates.**

[![CI](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml/badge.svg)](https://github.com/nishanttyagi28/VisionEval/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-visioneval.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://visioneval.streamlit.app)

VisionEval is a **CI-native** evaluation stack for vision and VLMs:

1. A **classification release gate** that fails the job on a new failure or an accuracy drop against a git-trackable baseline.
2. An **adaptive evaluation budget** — `visioneval budget-analyze` ranks samples without running the model; `visioneval run --use-budget` evaluates that recommended subset.
3. A **four-pillar multimodal framework** (alignment, hallucination probes, robustness, models + dashboard).
4. **Living traps** with a **visual red-team CI gate** (`visioneval traps gate`): lockfile regressions (`new_open` / `reappeared` / `worse`) exit `1`.
5. **Black-box hallucination maps** (`visioneval maps`): CPU-only locus of consistent failures by object, probe type, sample, and metric.

The layers sit beside each other. `visioneval run` is still the classification blocker (optionally budget-aware). Living traps and maps never write classification tables.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
export PYTHONPATH="$(pwd)"
python -m pytest

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

Full walkthrough: [QUICKSTART.md](QUICKSTART.md). Architecture, Streamlit, and CLI reference: see also [docs/README.md](docs/README.md), [docs/BUDGET.md](docs/BUDGET.md), [docs/MULTIMODAL.md](docs/MULTIMODAL.md), [docs/MAPS.md](docs/MAPS.md).

## CLI

```text
visioneval run SUITE [--update-baseline] [--use-budget] [--traps-db PATH]
visioneval budget-analyze SUITE [--json] [--top N] [--traps-db PATH]
visioneval maps [REPORT] [--json] [--db PATH]
visioneval multimodal CONFIG [--json-out PATH] [--markdown-out PATH]
visioneval traps list|harvest|run|gate|update-baseline
```

| Command | Role |
| --- | --- |
| `visioneval run` | Classification release gate. `--use-budget` evaluates the budget-analyzer subset. |
| `visioneval budget-analyze` | Deterministic risk ranking. No inference. |
| `visioneval maps` | Black-box hallucination map (CPU-only). |
| `visioneval traps gate` | Visual red-team CI blocker (`new_open` / `reappeared` / `worse`). |
| `visioneval multimodal` / `traps *` | VLM eval and living-trap harvest/replay/lock. |

## Dashboard

```bash
python -m pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Live demo: [visioneval.streamlit.app](https://visioneval.streamlit.app).

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
