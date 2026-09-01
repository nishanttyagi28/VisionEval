# Adaptive evaluation budget

`visioneval budget-analyze` ranks a Phase 1 classification catalog and recommends how many samples to run. It does **not** load an adapter, call a model, or hit the network.

```bash
visioneval budget-analyze demo_suite.yaml
visioneval budget-analyze suite.yaml --json
```

## Risk score

```text
risk_score = (previous_failure * 0.5)
           + (high_risk_tag * 0.2)
           + (low_confidence * 0.2)
           + (novelty * 0.1)
```

Each factor is `0` or `1` (values in `[0, 1]` are allowed). Weights live on `RiskWeights` and default to the formula above.

| Factor | `1` when | Source |
| --- | --- | --- |
| `previous_failure` | Sample is still in failure memory, or an **open** living trap points at its id | `sample_outcomes` (same two-consecutive-pass recovery as the attention sampler); optional `vlm_traps` |
| `high_risk_tag` | Manifest tags intersect suite `attention.high_risk_tags` | Suite YAML + catalog |
| `low_confidence` | Catalog `confidence` ≤ `low_confidence_threshold` | Suite YAML + catalog |
| `novelty` | No `sample_outcomes` row yet | Classification SQLite (missing DB ⇒ everything is novel) |

Tie-break: higher `risk_score` first, then `sample_id` ascending. Input order does not matter.

## Recommended budget

Deterministic coverage floor:

1. Include **every** previous-failure sample.
2. Walk the remaining samples in the rank order above, adding until:
   - every configured high-risk tag that appears in the catalog is represented by at least one selected sample, and
   - a novelty / random slice is included: `max(min_novelty_count, ceil(novelty_fraction * catalog_size))`, capped by how many novel samples exist. `novelty_fraction` defaults to the suite `random_coverage_fraction` (`0.15`).
3. `attention.budget` is a **cap after** previous failures. Failures always fit even if they exceed the cap.

The estimated reduction is `round(100 * (1 - recommended / total), 1)`.

A non-empty catalog that would otherwise select nothing still returns the single highest-risk sample.

## Output

Human and JSON both report:

- total sample count
- previous-failure count
- high-risk count
- low-confidence count
- top-risk samples
- recommended evaluation budget (sample count to run)
- estimated sample reduction percentage

JSON also includes `recommended_sample_ids`, per-sample factors, weights, and policy.

Optional flags: `--top N` (default 10), `--traps-db PATH` (default `artifacts/traps.sqlite3` when that file exists).

## Wire into `visioneval run`

CI can evaluate the recommended risk subset instead of the attention-fraction sampler:

```bash
visioneval run demo_suite.yaml --use-budget
visioneval run demo_suite.yaml --use-budget --traps-db artifacts/traps.sqlite3
```

Or set `attention.use_budget: true` in the suite YAML. Default behavior is unchanged when the flag/option is off.

Selection order matches `recommended_sample_ids` from `budget-analyze` (previous failures first, then risk ranking / coverage floor). Each selected sample gets a `selection_reason` from its highest risk factor.
