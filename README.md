# VisionEval

VisionEval is a CI-first evaluation harness for image-classification models. It selects a deterministic, risk-focused subset of samples, compares the result with a committed baseline, and emits evidence that can fail a deployment check.

## Problem Statement

Aggregate benchmark metrics can hide regressions on high-risk or previously failing samples. Running every image on every commit is often too slow for practical CI. VisionEval focuses evaluation budget where a regression is most likely while retaining seeded random coverage.

## Why VisionEval Exists

VisionEval is built for regression detection before deployment. It is not a training framework, serving system, data platform, or dashboard. Its job is to make a release decision reproducible, explainable, and suitable for local development or CI.

## Harness Engineering Philosophy

The harness prioritizes deterministic inputs, explicit baselines, local persistence, small reports, and non-zero CLI exits for regressions. Configuration is validated from YAML; output is JSON for automation and Markdown for review.

## Attention-Based Evaluation

The attention harness selects samples in this priority order:

1. Previous failures from local SQLite history
2. Samples with configured high-risk tags
3. Samples below the confidence threshold
4. Seeded random coverage

The default evaluation budget is allocated 40% / 30% / 15% / 15%. Unused quota from an earlier attention bucket is given to the next attention bucket before random coverage. Every selected sample records its selection reason, attention score, and risk bucket.

## Key Features

- Deterministic, seedable attention sampling
- Historical failure prioritization
- SQLite WAL failure memory and prediction cache
- Content-addressable prediction cache keys
- Baseline persistence and aggregate accuracy comparison
- New-failure and fixed-failure detection
- JSON and Markdown reports
- Sequential fail-fast on definite new failures
- Partial reports with cache and attention evidence

## Architecture Overview

```text
YAML suite + manifest
        │
        ▼
attention sampler ──► selected samples with provenance
        │
        ▼
classification adapter ──► SQLite prediction cache
        │
        ▼
scoring ──► baseline comparison ──► JSON / Markdown report / CI exit code
```

## Repository Structure

```text
visioneval/
  cli.py                 CLI entry point
  core/                  suite loading, sampling, runner, cache, baseline, reports
  classification/        adapter loading, scoring, optional Torchvision/ONNX adapters
examples/classification_suite/
  suite.yaml             Phase 1 suite configuration example
tests/                   unit and integration tests
.github/workflows/       GitHub Actions test workflow
```

## Installation

Python 3.10+ is required. Create the virtual environment inside the repository.

```powershell
Set-Location E:\VisionEval
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Torchvision and ONNX Runtime adapters are lazy imports; install their runtime dependencies only when using them.

## Quick Start

1. Provide a YAML dataset manifest with labelled samples and an adapter callable using `module:callable` notation.
2. Configure paths for a baseline, local SQLite cache, and reports.
3. Create the initial baseline after reviewing the model output.
4. Run the same suite in CI to detect regressions.

```powershell
visioneval run path\to\suite.yaml --update-baseline
visioneval run path\to\suite.yaml
```

The second command exits non-zero when the candidate is a regression.

## Example `suite.yaml`

```yaml
name: classification-regression-suite
task: image_classification
model:
  adapter: your_project.classification_adapter:predict
dataset:
  manifest: data/evaluation_manifest.yaml
attention:
  budget: 100
  seed: 42
  high_risk_tags: [safety_critical]
  low_confidence_threshold: 0.5
baseline:
  path: artifacts/baselines/production.json
  allowed_accuracy_drop: 0.01
cache:
  path: artifacts/cache.sqlite3
report:
  path: reports/classification.json
  markdown_path: reports/classification.md
execution:
  fail_fast: true
```

The manifest accepts `id`, `label`, `confidence`, optional `tags`, and optional `image_path` per sample. The adapter accepts a `ClassificationSample` and returns a `ClassificationPrediction`.

## Example CLI Usage

```powershell
visioneval run examples\classification_suite\suite.yaml --update-baseline
visioneval run examples\classification_suite\suite.yaml
python -m pytest
```

The bundled example references a user-provided adapter and manifest path; update those entries before running it.

## Attention Sampling Explanation

Selection starts from a stable sample-ID order and uses the suite seed for all draws. A sample is selected once, at its highest-priority applicable bucket. Unused quota cascades down the attention order and only then fills random coverage. This prevents duplicate computation and makes report provenance stable across identical runs.

## Regression Detection Workflow

A baseline stores aggregate accuracy and per-sample pass/fail outcomes. When outcomes are present, accuracy drop and new/fixed failures are computed only on sample ids that appear in both the baseline and the current run, so attention-driven selection changes do not compare different populations. A run that shares no ids with the locked baseline is a regression. A candidate is a regression when that overlap drop exceeds the configured accuracy-drop tolerance or introduces a new failure on a locked sample. Fixed failures are recorded separately for review. Baselines without per-sample outcomes still use aggregate accuracy.

With `execution.fail_fast: true`, VisionEval processes the deterministic selection sequence one sample at a time. It stops only on a definite new failure, writes a partial report, and returns a regression result. Aggregate accuracy is evaluated after a complete run because an interim value can recover.

## Cache Architecture

The local SQLite database uses WAL mode. It stores the latest sample outcome for attention prioritization and cached predictions for unchanged inputs. Prediction keys combine model identity, preprocessing identity, and a streaming SHA-256 hash of image content. Reports include cache hit and miss totals.

## Baseline Architecture

Baselines are small, sorted JSON files intended for review and source control. Promote a baseline deliberately with `--update-baseline`; do not update it automatically in CI.

## CI/CD Integration

Run the suite after a model or preprocessing change. Publish the JSON and Markdown reports as CI artifacts in the calling workflow, and treat a non-zero `visioneval run` exit as a quality-gate failure. The repository includes a GitHub Actions workflow that runs the test suite.

## Current Status

Phase 1 supports image classification evaluation with deterministic attention selection, local cache-backed execution, baseline regression comparison, and sequential fail-fast reporting. Detection, OCR, segmentation, distributed execution, dashboards, and cloud services are intentionally out of scope.

## Roadmap

The next highest-impact milestone is deterministic process-pool execution for complete, non-fail-fast runs. Fail-fast remains sequential to preserve termination order and unambiguous partial evidence.

## Contributing

Keep changes narrow, deterministic, and covered by tests. Prefer plain functions and dataclasses over new abstractions. Do not add future-modality infrastructure until a concrete Phase 1 need exists.

## License

License placeholder: Apache-2.0. Add the full license text before distributing release artifacts.