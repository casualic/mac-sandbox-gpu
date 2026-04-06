---
name: gpu-run
description: Run GPU-intensive Python code inside Claude Code sandbox on macOS using Apple Metal (MPS). Use when running ML training, PyTorch models, or any GPU-accelerated Python in sandbox mode.
user-invocable: true
argument-hint: [script.py]
---

# GPU Run — Metal/MPS in Claude Code Sandbox

Run GPU-intensive Python code inside Claude Code's sandbox on macOS Apple Silicon.

## Rules

1. **Run with the project's Python** — use `venv/bin/python3` if a `venv/` directory exists, otherwise fall back to `python3`.
2. **Use `torch.device("mps")`** — not CUDA. Check with `torch.backends.mps.is_available()`.

### Template:

If `venv/` exists in the project root:
```
venv/bin/python3 <script.py>
```

Otherwise:
```
python3 <script.py>
```

If the user provided an argument, replace `<script.py>` with `$ARGUMENTS`.

## MPS limitations

- `AdaptiveAvgPool1d` — use `AvgPool1d` instead
- `torch.svd` falls back to CPU automatically
