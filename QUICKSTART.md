# Quickstart

Phase 1 image-classification evaluation from a clean clone, plus the multimodal eval layer. Run all commands from the repository root so suite paths resolve.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

The installed `visioneval` script does not put the current directory on `PYTHONPATH`. Keep this in the same session:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
export PYTHONPATH="$(pwd)"
```

## Demo files (classification CI)

Create `demo_adapter.py`:

```python
from visioneval.core.types import ClassificationPrediction, ClassificationSample

def predict(sample: ClassificationSample) -> ClassificationPrediction:
    return ClassificationPrediction(sample.label, 0.9)
```

Create `demo_manifest.yaml`:

```yaml
samples:
  - {id: one, label: cat, confidence: 0.9, tags: [safety_critical]}
  - {id: two, label: cat, confidence: 0.4}
  - {id: three, label: cat, confidence: 0.9}
```

Create `demo_suite.yaml`:

```yaml
name: phase1-demo
task: image_classification
model:
  adapter: demo_adapter:predict
dataset:
  manifest: demo_manifest.yaml
attention:
  budget: 3
  seed: 42
  high_risk_tags: [safety_critical]
baseline:
  path: artifacts/baselines/demo.json
  allowed_accuracy_drop: 0.0
cache:
  path: artifacts/cache.sqlite3
report:
  path: reports/demo.json
  markdown_path: reports/demo.md
```

Images are optional. This adapter uses manifest labels only.

## Create baseline

```powershell
visioneval run demo_suite.yaml --update-baseline
```

Review `reports/demo.md`. Promote only after the outcomes look right.

## Run evaluation

```powershell
visioneval run demo_suite.yaml
```

Exit code `0` means no regression against the locked baseline.

## Trigger a regression

In `demo_adapter.py`, return a wrong label:

```python
return ClassificationPrediction("dog", 0.9)
```

```powershell
visioneval run demo_suite.yaml
```

Exit code `1`. Markdown lists **REGRESSION**, new-failure ids, and attention bucket counts.

## Interpret reports

| File | Use |
|------|-----|
| `reports/demo.md` | Status, accuracy, new/fixed failure ids, attention buckets |
| `reports/demo.json` | Per-sample labels, selection reason, cache hits |
| `artifacts/baselines/demo.json` | Locked identity and per-sample outcomes |

Restore `sample.label` in the adapter before the next baseline update.

---

## Multimodal eval layer

The classification harness above is unchanged. The multimodal layer adds CLIP/BLIP, POPE, an LLM judge, corruptions, VLM adapters, and a Streamlit dashboard. Tests and the demo use mocked backends — no weight downloads, no API keys.

```bash
python -m pip install -e ".[dev]"
visioneval multimodal examples/multimodal/config.yaml \
  --json-out reports/mm.json --markdown-out reports/mm.md
visioneval traps list --db artifacts/traps.sqlite3
visioneval traps harvest reports/mm.json --db artifacts/traps.sqlite3
visioneval traps run --db artifacts/traps.sqlite3 --config examples/multimodal/config.yaml --budget 8
python -m pytest
```

Living traps are opt-in SQLite memory for VLM hallucinations (POPE misses, judge flags, caption/object mismatches). They retire after two consecutive passes and do not replace `visioneval run`.

Dashboard (optional extra):

```bash
python -m pip install -e ".[ui]"
streamlit run app/streamlit_app.py
```

Real VLMs / CLIP / paid judge:

```bash
python -m pip install -e ".[hf,api,ui]"
```

API keys are read from the environment (`OPENAI_API_KEY` by default). Do not commit secrets.
