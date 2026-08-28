# Multimodal eval example

Offline demo of the four-pillar multimodal layer. No GPU, no API keys, no weight downloads.

```bash
python -m pip install -e ".[dev]"
visioneval multimodal examples/multimodal/config.yaml
```

Reports are written to `reports/multimodal.json` and `reports/multimodal.md` (gitignored).

To swap in a real HuggingFace or OpenAI-compatible model, copy a `kind: hf` or `kind: api` block from the root README and install `.[hf]` / `.[api]`.
