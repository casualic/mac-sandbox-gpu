# GPU Sandbox Project

This project enables Apple Metal (MPS) GPU access inside Claude Code's sandbox on macOS.

## Running Python in this project

When running Python scripts (especially ML/GPU code) in sandbox mode:

```
python3 <script.py>
```

## GPU device

This is macOS Apple Silicon. Use `torch.device("mps")`, not CUDA. Check with `torch.backends.mps.is_available()`.

## Known MPS limitations

- `AdaptiveAvgPool1d` fails with non-divisible sizes — use `AvgPool1d`
- `torch.svd` / `torch.linalg.svd` falls back to CPU automatically
- LSTM/GRU show minimal GPU speedup
