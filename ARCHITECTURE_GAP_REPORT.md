# VisionEval Architecture Gap Report

Audit of `E:\VisionEval` (Phase 1 tree) against the intended architecture. No code was changed.

Status uses three labels only: **Implemented**, **Partially implemented**, **Missing**.

---

## Component status

### Attention-based sampling — Partially implemented

**Present.** `visioneval/core/sampler.py` selects a seed-stable budget in priority order: previous failures, high-risk tags, low-confidence, then random. Quotas default to 40/30/15/15. Each `SelectedSample` carries `selection_reason`, a fixed `attention_score`, and `risk_bucket`. Duplicate `sample_id` values are rejected. Draw order is independent of input order.

**Gaps.** Scores are constants (`1.0 / 0.75 / 0.5 / 0.25`), not computed from model attention or learned risk. Unused quota in an earlier bucket is not given to the next attention bucket; leftover slots are filled as `random_coverage`. That weakens “spend budget where a regression is most likely” when failure or high-risk pools are smaller than their fractions.

### Failure memory — Partially implemented

**Present.** `SQLiteCache.sample_outcomes` stores the latest `correct` flag per `sample_id`. `runner._load_samples` sets `previous_failure` from `previous_failure_ids()`. Tests cover “failed sample is attention input on the next run.”

**Gaps.** Memory is latest-boolean only: no timestamps, no failure counts, no predicted/expected labels, no decay. A later pass overwrites the failure and drops the sample from attention. Outcomes live in local SQLite under `artifacts/` (gitignored), so CI machines do not share memory unless a workflow copies the file.

### High-risk tagging — Implemented

**Present.** Manifest `tags` intersect `attention.high_risk_tags`. Matching samples are selected under `SelectionReason.HIGH_RISK` before low-confidence and random. Provenance is recorded on the evaluation record.

**Limits (not a miss).** Tags are author-supplied; there is no auto-tagger. Default `high_risk_tags` is empty, so the bucket is inactive until configured.

### Low-confidence prioritization — Partially implemented

**Present.** Samples with manifest `confidence <= low_confidence_threshold` fill the low-confidence quota when `low_confidence` is true.

**Gaps.** Selection uses catalog confidence, not adapter/model confidence. Failure memory does not store prediction confidence, so a high-catalog / low-model-confidence sample is not prioritized on the next run unless it also failed. Sampling is pre-inference, so live confidence cannot affect the current run.

### Random exploration bucket — Implemented

**Present.** `SelectionReason.RANDOM_COVERAGE` uses the configured fraction plus any unused earlier quota, drawn with `random.Random(config.seed)`. Provenance score `0.25` / bucket `random_coverage`. Seed stability is tested.

### SQLite cache — Partially implemented

**Present.** WAL SQLite with `sample_outcomes` and `predictions`. Prediction keys hash model id, preprocess id, and streaming SHA-256 of image bytes. Reports include cache hits/misses. Content-addressable reuse is tested.

**Gaps.** `run_suite` always passes preprocess identity `"phase1"` and hashes the adapter import path string, not weights or a real preprocess graph. Samples without `image_path` never hit the prediction cache (always miss, never store). No schema version, TTL, or invalidation besides key change. Cache path is local and gitignored.

### Regression detection — Partially implemented

**Present.** `compare_baseline` flags a regression on accuracy drop above `allowed_accuracy_drop` **or** any new failure. Fixed failures are listed separately. CLI exits `1` on regression. Fail-fast defers aggregate accuracy until a complete run.

**Gaps.** Comparison is only over the current selected subset. Selection changes when failure memory changes, so candidate accuracy is not the same population as the locked baseline. Samples not in this run are not re-checked. Unknown sample ids default to “previously passing,” so first-seen failures count as new failures. No per-tag or per-bucket regression gates.

### Git-native baseline lockfile — Partially implemented

**Present.** `save_baseline` writes sorted, indented JSON (`suite_name`, `accuracy`, `sample_count`, `outcomes`). Promotion is explicit (`--update-baseline`). Intended for review and commit.

**Gaps.** This is a results snapshot, not a lockfile. It does not pin git SHA, suite YAML digest, manifest digest, image content hashes, adapter/weights identity, attention seed/budget, or the selected sample set. Example baseline path is `artifacts/baselines/production.json`; `.gitignore` ignores `artifacts/`, so the default layout cannot be git-native without path changes. CI workflow runs pytest only; it does not consume a committed baseline as a quality gate.

### Fail-fast execution — Implemented

**Present.** Sequential evaluation when `execution.fail_fast` is true and a baseline exists. Stops on a definite new failure, writes partial JSON evidence (`evaluated_count`, `remaining_count`, failing sample attention fields), still records outcomes and reports. Repeatable termination is tested. Does not stop on interim accuracy drop (by design).

**Limits (not a miss).** Default `fail_fast` is `false`. Example `examples/classification_suite/suite.yaml` omits `execution`. Parallel/process-pool execution is roadmap, not this component.

### Markdown and JSON reports — Partially implemented

**Present.** JSON includes suite name, accuracy, sample count, cache counts, optional `partial`, regression fields, and per-sample records (reason, scores, labels). Markdown states status (`PASS` / `REGRESSION` / `BASELINE UPDATED`), accuracy, counts, cache, fail-fast snippet, accuracy drop, and new/fixed failure **counts**.

**Gaps.** Markdown does not list sample ids, new/fixed failure ids, or an attention-bucket breakdown. Reviewers must open JSON for evidence. Markdown path is optional; if omitted, only JSON is written.

---

## Ranked gaps (missing or incomplete vs intended architecture)

Ranked by **impact on core value** (CI-first, risk-focused, reproducible release decision), then **implementation complexity** (L = low, M = medium, H = high).

| Rank | Gap | Parent component | Impact | Complexity | Why it matters |
|------|-----|------------------|--------|------------|----------------|
| 1 | Selection-set drift vs baseline population | Regression detection | High | M | Accuracy and “new failure” calls compare different samples across runs; the release gate can false-pass or false-fail. |
| 2 | Baseline is not an identity lockfile | Git-native baseline lockfile | High | M | Without hashes of suite, manifest, images, model/preprocess, and selected ids, a committed JSON does not prove the same evaluation was repeated. |
| 3 | Default baseline path is gitignored | Git-native baseline lockfile | High | L | `artifacts/` ignore blocks the documented “commit the baseline” workflow unless paths are changed. |
| 4 | Model/preprocess identity is a stub | SQLite cache | High | M | Adapter-path hash + hardcoded `"phase1"` can reuse predictions across real model or preprocess changes, hiding regressions. |
| 5 | Unused attention quota spills to random | Attention-based sampling | High | L | Empty failure/high-risk pools convert reserved risk budget into exploration, under-sampling remaining high-value buckets. |
| 6 | Failure memory is latest-boolean only | Failure memory | Medium | M | Recovered samples vanish from attention; intermittent failures are forgotten; CI cannot share history without extra plumbing. |
| 7 | Low-confidence ignores model confidence | Low-confidence prioritization | Medium | M | Catalog scores do not track actual model uncertainty; the bucket can miss the samples most likely to flip. |
| 8 | Prediction cache skipped without `image_path` | SQLite cache | Medium | L | Manifest-only suites never get content-addressable reuse. |
| 9 | Markdown lacks per-sample / id evidence | Markdown and JSON reports | Medium | L | CI reviewers see counts, not which samples or buckets failed, unless they parse JSON. |
| 10 | CI does not run `visioneval` against a locked baseline | Git-native baseline lockfile / Regression | Medium | M | Repo CI proves unit tests, not the product quality gate. |
| 11 | Static attention scores | Attention-based sampling | Low | H | Constants are enough for provenance; learned/spatial attention is not required for Phase 1 risk buckets. |
| 12 | No auto high-risk tagging | High-risk tagging | Low | H | Manual tags match the intended config contract. |

---

## Summary counts

| Status | Components |
|--------|------------|
| Implemented | High-risk tagging; Random exploration bucket; Fail-fast execution |
| Partially implemented | Attention-based sampling; Failure memory; Low-confidence prioritization; SQLite cache; Regression detection; Git-native baseline lockfile; Markdown and JSON reports |
| Missing (as a complete capability) | None of the ten named pillars is absent; the largest holes are **lockfile semantics**, **stable comparison population**, and **real model/preprocess identity** inside an otherwise present Phase 1 skeleton. |

Highest-leverage incomplete work: pin what a run means (inputs + sample set), then compare that same set, then stop leaking risk budget into random when higher buckets undershoot.
