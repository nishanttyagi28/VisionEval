# Black-box hallucination maps

`visioneval maps` builds a deterministic map of **where** a multimodal model consistently fails. CPU-only: no adapter load, no model inference, no network.

```bash
visioneval maps reports/mm.json
visioneval maps reports/mm.json --json
visioneval maps reports/mm.json --db artifacts/traps.sqlite3 --json
visioneval maps --db artifacts/traps.sqlite3
```

## Inputs

| Source | What counts as a failure |
| --- | --- |
| Multimodal JSON report | Clean rows only. POPE probe `correct: false` → `pope_miss`; judge flags or low factual score → `judge_flag`; caption missing expected objects or leaking absent ones → `caption_mismatch`; TruthGraph contradicted claim → `claim_contradicted` (`probe_type=verify`) |
| Living-traps SQLite (`--db`) | Every **open** (non-retired) trap, mapped to the same metric names |

Corrupted / noise-sweep rows are skipped so a robustness pass does not flood the map.

## Aggregations

Counts (sorted keys) plus a deterministic event list:

- `by_metric` — `pope_miss` / `judge_flag` / `caption_mismatch` / `claim_contradicted`
- `by_probe_type` — `pope` / `judge` / `caption` / `verify`
- `by_object` — object name when known
- `by_sample_id`
- `by_model`

JSON also includes the full `events` array for machine consumers. Human output prints the same buckets plus a compact event listing.

## Relation to living traps

Maps do **not** write the traps DB. Harvest still owns persistence; maps are a read-only analysis layer you can run before or after `visioneval traps gate`.
