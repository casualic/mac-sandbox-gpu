# GPU Sandbox Override for Claude Code on macOS

Enables Apple Metal (MPS) GPU access inside Claude Code's sandbox on macOS Apple Silicon.

**Average speedup: ~11x** across ML workloads (up to 50x for convolutions).

## Problem

Claude Code's sandbox uses `sandbox-exec` (Apple Seatbelt) which blocks IOKit GPU access. PyTorch, TensorFlow, and other ML frameworks silently fall back to CPU — even on M1/M2/M3/M4 Macs.

## Solution

A `sandbox-exec` wrapper that injects scoped Metal GPU permissions into the Seatbelt profile. Only fires for Python commands. Includes a kill switch, audit mode, and has passed four rounds of security review.

## Quick Start

### 1. Install the Claude Code plugin

```
/plugin marketplace add casualic/mac-sandbox-gpu
/plugin install gpu-sandbox
```

### 2. Set up the wrapper

Ensure `~/.local/bin` is first in your PATH — add to `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then install:

```bash
mkdir -p ~/.local/bin
cp gpu_sandbox_override.sh ~/.local/bin/sandbox-exec
chmod +x ~/.local/bin/sandbox-exec
source ~/.zshrc
```

### 3. Verify

```bash
which sandbox-exec
# Should output: ~/.local/bin/sandbox-exec (NOT /usr/bin/sandbox-exec)
```

### 4. Restart Claude Code

Start a new session with sandbox enabled. The `/gpu-run` skill is now available.

## Usage

```
/gpu-run my_training_script.py
```

Or just ask Claude to run GPU code — the CLAUDE.md ensures it uses the right invocation pattern.

### Manual usage (without the skill)

```bash
python3 my_script.py
```

Key rules:
- Use `torch.device("mps")`, not CUDA

## Benchmark Results

```
Model                    CPU(s)     GPU(s)    Speedup
----------------------------------------------------------------------
1D-CNN                   118.69       2.33      50.5x
MatrixMul (4096x4096)      5.83       0.46      12.7x
ResNet                     6.00       1.47       4.1x
Wide&Deep                  1.70       0.42       4.0x
MLP                        1.45       0.55       2.6x
Autoencoder                0.45       0.28       1.6x
LSTM                      31.74      31.39       1.0x
----------------------------------------------------------------------
Average: ~11x speedup
```

## Environment Variables

| Variable | Effect |
|----------|--------|
| `GPU_SANDBOX_OVERRIDE_DISABLED=1` | Bypass override entirely, use original sandbox |
| `GPU_SANDBOX_OVERRIDE_AUDIT=1` | Log to stderr when GPU rules are injected/skipped |

## Security

The wrapper has been through **four rounds of security review**. It only adds:

- **IOKit access** — scoped to 5 specific GPU driver classes (AGX, IOGPU, IOSurface)
- **IOKit property reads** — scoped to 21 GPU-related properties only
- **Sysctl reads** — CPU/cache topology queries (read-only)
- **Mach lookups** — Metal shader compiler + GPU memory daemon
- **File access** — current user's shader cache dir only; Metal frameworks read-only

It does **not**:
- Open network access
- Bypass file write restrictions
- Allow access to non-GPU IOKit devices (USB, Bluetooth, HID, SMC)
- Inject rules into non-Python sandbox commands
- Allow reading other users' temp data

### Safety features
- Only injects for Python commands (other commands pass through untouched)
- Kill switch via environment variable
- Refuses injection if profile is malformed
- Refuses injection if user temp dir can't be safely resolved
- Audit mode for debugging

## Files

| File | Purpose |
|------|---------|
| `gpu_sandbox_override.sh` | The sandbox-exec wrapper |
| `skills/gpu-run/SKILL.md` | Claude Code plugin skill |
| `.claude-plugin/plugin.json` | Plugin manifest |
| `CLAUDE.md` | Project-level Claude instructions |
| `gpu_test.py` | Quick MPS sanity check |
| `gpu_stress_test.py` | 11 model architectures stress test |
| `cpu_vs_gpu_benchmark.py` | CPU vs GPU timing comparison |

## Uninstall

```bash
rm ~/.local/bin/sandbox-exec
```

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Claude Code with sandbox enabled
- Python 3.10+ with PyTorch (`pip install torch`)

## Known MPS Limitations

- `AdaptiveAvgPool1d` — use `AvgPool1d` instead
- `torch.svd` falls back to CPU automatically
- LSTM/GRU show minimal GPU speedup (sequential bottleneck)

## Disclaimer

**Use at your own risk.** This tool modifies the behavior of macOS sandbox-exec, which is a security boundary. While it has been through four rounds of security review and permissions are tightly scoped, it does weaken the sandbox for Python processes. The authors are not responsible for any security issues, data loss, or other damages arising from its use. Review the source code and understand the permissions before installing.

## Related

- [anthropics/claude-code#13108](https://github.com/anthropics/claude-code/issues/13108) — GPU passthrough feature request
- [anthropics/claude-code#37481](https://github.com/anthropics/claude-code/issues/37481) — macOS Metal sandbox blocking
