# GPU Sandbox Override for Claude Code on macOS

Enables Apple Metal (MPS) GPU access inside Claude Code's sandbox on macOS.

## Problem

Claude Code's sandbox uses `sandbox-exec` (Apple Seatbelt) which blocks IOKit
GPU access. This means PyTorch, TensorFlow, and other ML frameworks fall back
to CPU even on Apple Silicon Macs with powerful GPUs.

## How It Works

A wrapper script intercepts `sandbox-exec` calls and injects additional Seatbelt
rules that allow:

- **IOKit access** — lets Metal talk to the GPU hardware
- **Sysctl reads** — CPU/cache info queries needed by PyTorch, NumPy, Arrow
- **Mach service lookups** — Metal shader compiler, GPU memory daemon
- **Shader cache writes** — Metal Performance Shaders graph cache
- **Framework reads** — Metal system frameworks (read-only)

None of these open network access or weaken file restrictions.

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- `~/.local/bin` must be in your PATH before `/usr/bin`
- Python venv with PyTorch installed (`pip install torch`)

## Installation

### Install the Claude Code plugin

```bash
/plugin marketplace add mateuszdelpercio/SandboxGPU
/plugin install gpu-sandbox@mateuszdelpercio
```

Then follow steps 1-4 below to set up the sandbox-exec wrapper.

### 1. Ensure `~/.local/bin` is first in PATH

Add to your `~/.zshrc` (or `~/.bashrc`):

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then reload: `source ~/.zshrc`

### 2. Install the wrapper

```bash
mkdir -p ~/.local/bin
cp gpu_sandbox_override.sh ~/.local/bin/sandbox-exec
chmod +x ~/.local/bin/sandbox-exec
```

### 3. Verify installation

```bash
which sandbox-exec
# Should output: /Users/<you>/.local/bin/sandbox-exec
```

### 4. Restart Claude Code

The wrapper is picked up on next sandbox-exec invocation. Restart Claude Code
or start a new session.

### 5. Test it

Enable sandbox in Claude Code, then run:

```bash
venv/bin/python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

Should print `MPS available: True`.

Or use the Claude skill:

```
/gpu-run
```

## Files

| File | Purpose |
|------|---------|
| `gpu_sandbox_override.sh` | The sandbox-exec wrapper (install to `~/.local/bin/sandbox-exec`) |
| `gpu_test.py` | Quick MPS sanity check — trains a simple model on test.parquet |
| `gpu_stress_test.py` | 11 model architectures (MLP, CNN, LSTM, GRU, Transformer, etc.) |
| `cpu_vs_gpu_benchmark.py` | Side-by-side CPU vs GPU timing comparison |
| `benchmark_log.txt` | Latest benchmark results |
| `stress_test_log.txt` | Latest stress test results |

## Benchmark Results (Apple Silicon)

```
Model                    CPU(s)     GPU(s)    Speedup
----------------------------------------------------------------------
MLP                        1.45       0.55       2.6x
1D-CNN                   118.69       2.33      51.0x
LSTM (50k)                31.74      31.39       1.0x
ResNet                     6.00       1.47       4.1x
Wide&Deep                  1.70       0.42       4.0x
Autoencoder                0.45       0.28       1.6x
MatrixMul                  5.83       0.46      12.6x
----------------------------------------------------------------------
Average speedup: 11.0x
```

## Tips

- Always use `venv/bin/python3` instead of `source venv/bin/activate` to avoid
  shell-eval permission prompts in sandbox mode
- Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` for memory-heavy models (LSTM, large batches)
- The `torchinductor_<username>/` directory is a PyTorch cache artifact — safe to delete

## Uninstall

```bash
rm ~/.local/bin/sandbox-exec
```

## Security Notes

The override only adds GPU-related permissions:
- IOKit device access (GPU hardware only)
- Read-only sysctl queries (CPU/cache info)
- Apple GPU mach services (shader compiler, memory daemon)
- Shader cache in `/private/var/folders` (macOS temp)
- Read-only access to Metal system frameworks

It does **not** open network access, bypass file write restrictions on your
project, or allow access to credentials/secrets. The sandbox remains enforced
for everything else.

## Related Issues

- [anthropics/claude-code#13108](https://github.com/anthropics/claude-code/issues/13108) — GPU device passthrough (Linux bwrap wrapper)
- [anthropics/claude-code#37481](https://github.com/anthropics/claude-code/issues/37481) — macOS Metal/IOKit sandbox blocking
