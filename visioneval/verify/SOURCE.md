# Vendored TruthGraph core

This package vendors the **pure verification logic** from
[TruthGraph](https://github.com/nishanttyagi28/truthgraph) so VisionEval can
check VLM captions / claims against visual ground-truth evidence without
running FastAPI or pulling a live dependency in CI.

| Upstream | Local |
| --- | --- |
| `app/services/text_analyzer.py` | `text_analyzer.py` |
| `app/services/verifier.py` | `verifier.py` |
| `app/models/{claim,evidence,results}.py` | `models.py` |

Adaptations for VisionEval:

- Imports use `visioneval.verify.*` instead of `app.*`.
- Claim / evidence `min_length` relaxed to `3` so short visual phrases validate.
- No FastAPI / HTTP layer.
- `adapter.py` maps multimodal report rows and YAML cases → claim + evidence.
