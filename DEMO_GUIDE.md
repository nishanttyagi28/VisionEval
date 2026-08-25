# Phase 1 demo

End-to-end classification regression check with a dummy adapter. No GPU, no extra packages. Working directory: repository root.

The in-repo `examples/classification_suite/suite.yaml` is a template (`your_project...` adapter). Use the files below for a runnable demo.

## 1. Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
$env:PYTHONPATH = (Get-Location).Path
```

Confirm the CLI:

```powershell
visioneval run --help
```

## 2. Adapter, manifest, suite

`demo_adapter.py` must be importable as `demo_adapter:predict` (same directory as the shell, with `PYTHONPATH` set):

```python
from visioneval.core.types import ClassificationPrediction, ClassificationSample

def predict(sample: ClassificationSample) -> ClassificationPrediction:
    return ClassificationPrediction(sample.label, 0.9)
```

`demo_manifest.yaml`:

```yaml
samples:
  - {id: one, label: cat, confidence: 0.9, tags: [safety_critical]}
  - {id: two, label: cat, confidence: 0.4}
  - {id: three, label: cat, confidence: 0.9}
```

`demo_suite.yaml` — paths are relative to the current working directory:

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
  low_confidence_threshold: 0.5
baseline:
  path: artifacts/baselines/demo.json
  allowed_accuracy_drop: 0.0
cache:
  path: artifacts/cache.sqlite3
report:
  path: reports/demo.json
  markdown_path: reports/demo.md
execution:
  fail_fast: true
```

## 3. Create the baseline

```powershell
visioneval run demo_suite.yaml --update-baseline
```

Expected: accuracy `1.0000`, `reports/demo.md` status **BASELINE UPDATED**. The JSON baseline stores suite hash, model id, seed, budget, selected ids, and per-sample pass/fail.

Do not use `--update-baseline` in CI.

## 4. Run evaluation (candidate)

```powershell
visioneval run demo_suite.yaml
```

Expected: exit `0`, status **PASS**. Same sample ids as the lockfile; overlap comparison is used.

## 5. Trigger a regression

Change `predict` to:

```python
return ClassificationPrediction("dog", 0.9)
```

```powershell
visioneval run demo_suite.yaml
```

Expected: exit `1`. With `fail_fast: true`, execution can stop on the first **new** failure and still write reports.

## 6. Interpret reports

**Markdown** (`reports/demo.md`)

- **Status** — `PASS`, `REGRESSION`, or `BASELINE UPDATED`
- **New failures** — ids that passed in the baseline and failed now
- **Fixed failures** — ids that failed in the baseline and passed now
- **Attention buckets** — counts for `previous_failure`, `high_risk`, `low_confidence`, `random_coverage`
- Fail-fast lines appear only when a run stopped early

**JSON** (`reports/demo.json`)

- `records[]` — expected vs predicted label, `selection_reason`, `attention_score`, `cache_hit`
- `regression` — accuracy drop, `new_failures`, `fixed_failures`

**CLI**

- Prints `Accuracy: ...`
- Exit `1` only when `regression.is_regression` is true

## 7. Optional: real images

Add `image_path` on each manifest sample and point `adapter` at a callable that reads the file (your module, or `visioneval.classification.backends` after installing torch/onnxruntime). Cache keys then include image bytes plus model/preprocess identity.

## 8. Reset

Delete `artifacts/` and `reports/` to clear cache and baselines. Restore a correct `predict` before promoting a new baseline.
