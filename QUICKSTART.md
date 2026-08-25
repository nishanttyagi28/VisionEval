# Quickstart

Phase 1 image-classification evaluation from a clean clone. Run all commands from the repository root so suite paths resolve.

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

## Demo files

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
