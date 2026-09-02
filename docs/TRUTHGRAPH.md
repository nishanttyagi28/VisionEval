# TruthGraph verify in VisionEval

VisionEval vendors the **core** claim-verification logic from
[TruthGraph](https://github.com/nishanttyagi28/truthgraph) so multimodal / VLM
outputs can be checked as **claims** against **visual ground-truth evidence**
(expected objects, captions, POPE-style facts).

No FastAPI process, no network, no paid APIs. CPU-only and deterministic.

```bash
visioneval verify examples/verify/cases.yaml
visioneval verify examples/verify/sample_report.json --json
visioneval truth reports/mm.json --json          # alias
```

## Design choice: vendored core

| Option | Choice |
| --- | --- |
| Git / path dependency on TruthGraph | Not used — CI stays self-contained |
| Run TruthGraph FastAPI from VisionEval | Not used — unnecessary for offline eval |
| **Vendor `text_analyzer` + `verifier` + models** | **Selected** under `visioneval/verify/` |

See `visioneval/verify/SOURCE.md` for the upstream mapping and small adaptations
(relaxed `min_length` for short visual phrases; package-local imports).

## Mapping

| VisionEval field | TruthGraph role |
| --- | --- |
| `response` / `claim` / `caption_pred` | Claim text |
| `caption` / `ground_truth` | Evidence (`ground_truth_caption`) |
| `objects` | Evidence (`expected_objects`) |
| `absent_objects` | Evidence (`absent_objects`, negated) |
| `pope.probes[]` | Evidence (`pope_fact`, present/absent) |
| explicit `evidence: [{text, source, reliability}]` | Passed through |

Verdicts: `supported` | `contradicted` | `insufficient`, plus confidence and matched keywords.

README gallery stills: `docs/assets/verify-supported.svg`, `verify-contradicted.svg`, `verify-cli.svg` (from the silent verify demo).

## Relation to maps / traps

- `visioneval maps` may surface `probe_type=verify` / `metric=claim_contradicted`
  when a clean report row’s claim is contradicted by ground truth.
- Living traps harvest is unchanged (still `pope` / `judge` / `caption` only).
  Contradicted claims are **not** auto-written as traps, so existing lockfiles
  and gates stay stable.

## Fixtures

- `examples/verify/cases.yaml` — three deterministic cases (supported / contradicted / insufficient)
- `examples/verify/sample_report.json` — multimodal-shaped report including a corrupted row (skipped by default)
