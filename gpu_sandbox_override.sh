#!/bin/bash
# gpu_sandbox_override.sh — wrapper for sandbox-exec that injects Metal GPU permissions
#
# Claude Code on macOS uses sandbox-exec with a Seatbelt profile that blocks
# IOKit GPU classes and sysctl calls needed for Metal/MPS (PyTorch, TensorFlow).
#
# This wrapper intercepts sandbox-exec calls, detects the profile string (-p),
# and injects additional rules to allow GPU access before passing to real sandbox-exec.
#
# INSTALL:
#   1. chmod +x gpu_sandbox_override.sh
#   2. mkdir -p ~/.local/bin
#   3. cp gpu_sandbox_override.sh ~/.local/bin/sandbox-exec
#   4. Ensure ~/.local/bin is FIRST in PATH (before /usr/bin)
#   5. Restart Claude Code
#
# UNINSTALL:
#   rm ~/.local/bin/sandbox-exec
#
# ENVIRONMENT VARIABLES:
#   GPU_SANDBOX_OVERRIDE_DISABLED=1  — bypass override, use original sandbox-exec
#   GPU_SANDBOX_OVERRIDE_AUDIT=1     — log when GPU rules are injected (to stderr)

set -euo pipefail

REAL_SANDBOX_EXEC=/usr/bin/sandbox-exec

# --- Kill switch ---
if [[ "${GPU_SANDBOX_OVERRIDE_DISABLED:-0}" == "1" ]]; then
    exec "$REAL_SANDBOX_EXEC" "$@"
fi

# --- Resolve current user's temp directory for scoped /var/folders access ---
USER_TMPDIR="$(getconf DARWIN_USER_DIR 2>/dev/null || echo "/private/var/folders")"
# Strip trailing slash for seatbelt subpath
USER_TMPDIR="${USER_TMPDIR%/}"

# --- Detect if command is ML/Python related ---
cmd_needs_gpu() {
    for arg in "$@"; do
        case "$arg" in
            *python* | *Python* | *torch* | *tensorflow* | *keras* | *mps* | \
            *metal* | *gpu_test* | *gpu_stress* | *benchmark* | *train* | \
            *import\ torch* | *import\ tensorflow*)
                return 0 ;;
        esac
    done
    return 1
}

if ! cmd_needs_gpu "$@"; then
    # Not ML-related — pass through without GPU rules
    if [[ "${GPU_SANDBOX_OVERRIDE_AUDIT:-0}" == "1" ]]; then
        echo "[gpu_sandbox_override] SKIP (non-ML command): $*" >&2
    fi
    exec "$REAL_SANDBOX_EXEC" "$@"
fi

# --- GPU rules to inject — Metal/MPS on Apple Silicon ---
GPU_RULES="
; --- GPU/Metal override (injected by gpu_sandbox_override.sh) ---

; IOKit: allow Metal GPU device access — scoped to GPU driver classes only
(allow iokit-open
  (iokit-user-client-class \"AGXDeviceUserClient\")
  (iokit-user-client-class \"AGXAcceleratorUserClient\")
  (iokit-user-client-class \"AGXCommandQueue\")
  (iokit-user-client-class \"IOGPUDeviceUserClient\")
  (iokit-user-client-class \"IOAccelerator\")
  (iokit-user-client-class \"IOAccelerationUserClient\")
  (iokit-user-client-class \"IOSurfaceRootUserClient\")
  (iokit-user-client-class \"IOSurfaceSendRight\")
)
(allow iokit-get-properties)

; Sysctl: allow hardware/CPU queries needed by PyTorch, NumPy, Arrow, scipy
(allow sysctl-read
  (sysctl-name \"hw.l1dcachesize\")
  (sysctl-name \"hw.l1icachesize\")
  (sysctl-name \"hw.l2cachesize\")
  (sysctl-name \"hw.l3cachesize\")
  (sysctl-name \"hw.cachelinesize\")
  (sysctl-name \"hw.optional.neon\")
  (sysctl-name \"hw.optional.AdvSIMD\")
  (sysctl-name \"hw.optional.floatingpoint\")
  (sysctl-name-prefix \"hw.optional.arm.\")
  (sysctl-name-prefix \"hw.perflevel\")
  (sysctl-name \"machdep.cpu.core_count\")
  (sysctl-name \"machdep.cpu.thread_count\")
  (sysctl-name \"hw.cpusubtype\")
  (sysctl-name \"hw.gpu_cores\")
  (sysctl-name \"hw.memsize_usable\")
)

; Mach services needed by Metal runtime and shader compilation
(allow mach-lookup
  (global-name \"com.apple.gpumemd.source\")
  (global-name \"com.apple.MTLCompilerService\")
  (global-name-prefix \"com.apple.AGX\")
  (global-name-prefix \"com.apple.iogpu\")
  (global-name-prefix \"com.apple.gpu\")
  (global-name-prefix \"com.apple.metal\")
)

; File access: Metal shader cache — scoped to current user's temp dir only
(allow file-read* file-write*
  (subpath \"${USER_TMPDIR}\")
  (subpath \"/Library/GPUBundles\")
)
(allow file-read*
  (subpath \"/System/Library/Frameworks/Metal.framework\")
  (subpath \"/System/Library/Frameworks/MetalPerformanceShaders.framework\")
  (subpath \"/System/Library/Frameworks/MetalPerformanceShadersGraph.framework\")
  (subpath \"/System/Library/PrivateFrameworks/GPUCompiler.framework\")
  (subpath \"/System/Library/Extensions\")
)

; Temp file access for OMP and Metal shader compilation
(allow file-read* file-write*
  (subpath \"/tmp\")
  (subpath \"/private/tmp\")
)
; --- end GPU/Metal override ---
"

# --- Parse args to find -p (profile string) mode ---
args=("$@")
profile_index=-1

for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "-p" ]] && (( i+1 < ${#args[@]} )); then
        profile_index=$((i+1))
        break
    fi
done

if [[ $profile_index -ge 0 ]]; then
    original_profile="${args[$profile_index]}"

    # Validate: count occurrences of (version 1) — reject if more than one
    version_count=$(echo "$original_profile" | grep -c '(version 1)' || true)
    if [[ "$version_count" -gt 1 ]]; then
        echo "[gpu_sandbox_override] WARNING: profile contains multiple (version 1) — refusing to inject, passing through unmodified" >&2
        exec "$REAL_SANDBOX_EXEC" "$@"
    fi

    # Inject GPU rules right after (version 1)
    modified_profile="${original_profile/\(version 1\)/(version 1)${GPU_RULES}}"

    # If no (version 1) found, pass through unmodified rather than blindly prepending
    if [[ "$modified_profile" == "$original_profile" ]]; then
        echo "[gpu_sandbox_override] WARNING: no (version 1) found in profile — passing through unmodified" >&2
        exec "$REAL_SANDBOX_EXEC" "$@"
    fi

    args[$profile_index]="$modified_profile"

    if [[ "${GPU_SANDBOX_OVERRIDE_AUDIT:-0}" == "1" ]]; then
        echo "[gpu_sandbox_override] INJECT GPU rules for: $*" >&2
    fi
fi

exec "$REAL_SANDBOX_EXEC" "${args[@]}"
