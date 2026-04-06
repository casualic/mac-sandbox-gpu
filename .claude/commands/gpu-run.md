Run GPU-intensive Python code inside Claude Code sandbox on macOS using Apple Metal (MPS).

This skill ensures the GPU sandbox override is in place and provides the correct
invocation pattern for running ANY GPU/ML code in sandbox mode.

## Critical Rules for GPU in Sandbox

When running Python code that uses GPU (PyTorch MPS) inside sandbox, you MUST:

1. **Always prefix with** `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — without this,
   MPS will OOM on large models because the sandbox restricts memory signaling.

2. **Always use `venv/bin/python3`** instead of `source venv/bin/activate` — 
   `source` triggers a shell-eval permission prompt every time in sandbox mode.

3. **Always use `torch.device("mps")`** — this is Apple Silicon's GPU backend.
   CUDA does not exist on macOS. Check with `torch.backends.mps.is_available()`.

4. **Filter noisy but harmless warnings** with:
   `2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"`

### Template command for any GPU Python script:

```
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 venv/bin/python3 <script.py> 2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning"
```

## Pre-flight Checks

Before running GPU code, verify the setup:

1. Check that the sandbox-exec wrapper is installed and first in PATH:
   ```
   which sandbox-exec
   ```
   Expected: `~/.local/bin/sandbox-exec` (must come before `/usr/bin/sandbox-exec`)
   
   If it shows `/usr/bin/sandbox-exec`, the override is NOT installed.
   Tell the user to follow the install instructions in `INSTALL.md`, or run
   these commands in another terminal:
   ```
   mkdir -p ~/.local/bin
   cp gpu_sandbox_override.sh ~/.local/bin/sandbox-exec
   chmod +x ~/.local/bin/sandbox-exec
   ```
   Then restart Claude Code.

2. Verify MPS is available through the sandbox:
   ```
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 venv/bin/python3 -c "import torch; print('MPS:', torch.backends.mps.is_available())"
   ```
   If this prints `MPS: False`, the override is not working.

## Running the Test Suite

Once pre-flight passes, run:

1. Stress test (11 model architectures on GPU):
   ```
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 venv/bin/python3 gpu_stress_test.py 2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning" | tee stress_test_log.txt
   ```

2. CPU vs GPU benchmark:
   ```
   PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 venv/bin/python3 cpu_vs_gpu_benchmark.py 2>&1 | grep -v "Error creating directory\|don.*t have permission\|OMP: Warning" | tee benchmark_log.txt
   ```

3. Report results summary to the user.

## Writing New GPU Code

When writing new PyTorch scripts for this project, always include this device setup:

```python
import torch
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

And move tensors/models to device:
```python
model = MyModel().to(DEVICE)
X = torch.tensor(data, device=DEVICE)
```

Known MPS limitations:
- `AdaptiveAvgPool1d` fails with non-divisible sizes — use `AvgPool1d` instead
- `torch.svd` / `torch.linalg.svd` falls back to CPU automatically
- LSTM/GRU show minimal speedup over CPU due to sequential nature
- For memory-heavy models, `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` is mandatory
