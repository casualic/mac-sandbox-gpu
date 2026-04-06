Run GPU-intensive Python code inside Claude Code sandbox on macOS using Apple Metal (MPS).

## Rules

1. **Use `venv/bin/python3`** — never `source venv/bin/activate` (triggers permission prompts)
2. **Use `torch.device("mps")`** — not CUDA. Check with `torch.backends.mps.is_available()`.
3. **Filter warnings**: `2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"`

### Template:

```
venv/bin/python3 <script.py> 2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"
```

## MPS limitations

- `AdaptiveAvgPool1d` — use `AvgPool1d` instead
- `torch.svd` falls back to CPU automatically
- LSTM/GRU show minimal GPU speedup
