# Multimodal eval example

Offline demo of the four-pillar multimodal layer. No GPU, no API keys, no weight downloads.

```bash
python -m pip install -e ".[dev]"
visioneval multimodal examples/multimodal/config.yaml
```

Reports are written to `reports/multimodal.json` and `reports/multimodal.md` (gitignored).

To swap in a real HuggingFace or OpenAI-compatible model, copy a `kind: hf` or `kind: api` block from the root README and install `.[hf]` / `.[api]`.

## Living traps

Hallucinations harvested from this demo become durable SQLite traps (`vlm_traps`)
until the model beats them twice in a row. They do not replace Phase 1.

```bash
visioneval multimodal examples/multimodal/config.yaml
visioneval traps list --db artifacts/traps.sqlite3
visioneval traps harvest reports/multimodal.json --db artifacts/traps.sqlite3
visioneval traps run --db artifacts/traps.sqlite3 --config examples/multimodal/config.yaml --budget 8
visioneval traps update-baseline --db artifacts/traps.sqlite3 --lockfile artifacts/baselines/traps.json
```

`fake-sparse` (`default_response: A shape.`) is the model that typically mint caption/judge traps. Replay uses FakeVLM; no GPU or API keys.
