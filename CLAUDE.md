# GPU Sandbox Project

This project enables Apple Metal (MPS) GPU access inside Claude Code's sandbox on macOS.

## Running Python in this project

When running Python scripts (especially ML/GPU code) in sandbox mode:

1. **Always use** `venv/bin/python3` — never `source venv/bin/activate` (triggers permission prompts)
2. **Filter harmless warnings** with: `2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"`

Template:
```
venv/bin/python3 <script.py> 2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"
```

## GPU device

This is macOS Apple Silicon. Use `torch.device("mps")`, not CUDA. Check with `torch.backends.mps.is_available()`.

## Known MPS limitations

- `AdaptiveAvgPool1d` fails with non-divisible sizes — use `AvgPool1d`
- `torch.svd` / `torch.linalg.svd` falls back to CPU automatically
- LSTM/GRU show minimal GPU speedup
