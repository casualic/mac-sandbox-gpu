---
name: gpu-run
description: Run GPU-intensive Python code inside Claude Code sandbox on macOS using Apple Metal (MPS). Use when running ML training, PyTorch models, or any GPU-accelerated Python in sandbox mode.
user-invocable: true
argument-hint: [script.py]
---

# GPU Run — Metal/MPS in Claude Code Sandbox

Run GPU-intensive Python code inside Claude Code's sandbox on macOS Apple Silicon.

## Rules

1. **Use `venv/bin/python3`** — never `source venv/bin/activate` (triggers permission prompts)
2. **Use `torch.device("mps")`** — not CUDA. Check with `torch.backends.mps.is_available()`.
3. **Filter warnings**: `2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"`

### Template:

```
venv/bin/python3 <script.py> 2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"
```

If the user provided an argument:
```
venv/bin/python3 $ARGUMENTS 2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"
```

## MPS limitations

- `AdaptiveAvgPool1d` — use `AvgPool1d` instead
- `torch.svd` falls back to CPU automatically
- LSTM/GRU show minimal GPU speedup
