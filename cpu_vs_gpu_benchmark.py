"""
CPU vs GPU (MPS) Benchmark — side-by-side timing comparison.
Runs each model on both devices and reports speedup.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

HAS_MPS = torch.backends.mps.is_available()
print(f"MPS available: {HAS_MPS}")
if not HAS_MPS:
    print("ERROR: MPS not available. Cannot benchmark GPU.")
    exit(1)

# ── Load data ────────────────────────────────────────────────────────────
print("Loading test.parquet ...")
df = pd.read_parquet("test.parquet")
feature_cols = [c for c in df.columns if c.startswith("feature_") and df[c].dtype in ("float64", "float32", "int64", "int32")]
X_np = df[feature_cols].values.astype("float32")
X_np = np.nan_to_num(X_np, 0.0)
y_np = (X_np[:, 0] > np.median(X_np[:, 0])).astype("float32")

N = min(200_000, len(X_np))
X_np, y_np = X_np[:N], y_np[:N]
N_FEAT = X_np.shape[1]
print(f"Data: {N:,} rows, {N_FEAT} features\n")

EPOCHS = 3
BATCH = 2048


import gc

def bench(name, model_fn, reshape_fn=None):
    """Train on CPU then GPU, return times."""
    times = {}
    for dev_name in ["cpu", "mps"]:
        # Aggressive cleanup before each run
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        device = torch.device(dev_name)
        X_t = torch.tensor(X_np, device=device)
        y_t = torch.tensor(y_np, device=device)
        model = model_fn().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        # Warmup (1 batch)
        xb = X_t[:BATCH]
        if reshape_fn:
            xb = reshape_fn(xb)
        _ = model(xb)
        if dev_name == "mps":
            torch.mps.synchronize()

        t0 = time.time()
        for ep in range(EPOCHS):
            for i in range(0, N, BATCH):
                xb = X_t[i:i+BATCH]
                yb = y_t[i:i+BATCH]
                if reshape_fn:
                    xb = reshape_fn(xb)
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        if dev_name == "mps":
            torch.mps.synchronize()
        times[dev_name] = time.time() - t0

        # Free memory
        del X_t, y_t, model, opt
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    speedup = times["cpu"] / times["mps"] if times["mps"] > 0 else 0
    print(f"  {name:<20} CPU={times['cpu']:>7.2f}s  GPU={times['mps']:>7.2f}s  Speedup={speedup:>5.1f}x")
    return name, times["cpu"], times["mps"], speedup


# ── Model definitions ────────────────────────────────────────────────────

def mlp():
    return nn.Sequential(
        nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),
        )
        self.fc = nn.Sequential(nn.Linear(64 * (N_FEAT // 2), 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim), nn.BatchNorm1d(dim))
    def forward(self, x):
        return F.relu(self.block(x) + x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(N_FEAT, 256), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(256), ResBlock(256), ResBlock(256), ResBlock(256))
        self.head = nn.Linear(256, 1)
    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))

class WideDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.Linear(N_FEAT, 1)
        self.deep = nn.Sequential(nn.Linear(N_FEAT, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        return self.wide(x) + self.deep(x)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(N_FEAT, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU())
        self.cls = nn.Linear(32, 1)
    def forward(self, x):
        return self.cls(self.enc(x))


# ── Run benchmarks ───────────────────────────────────────────────────────
print("Running CPU vs GPU benchmarks (3 epochs each)...")
print("="*70)

all_results = []
all_results.append(bench("MLP", mlp))
all_results.append(bench("1D-CNN", CNN1D, reshape_fn=lambda x: x.unsqueeze(1)))
# LSTM uses more memory per sample (seq_len=86), use smaller subset
N_saved, X_np_saved = N, X_np.copy()
y_np_saved = y_np.copy()
N = min(50_000, N)
X_np = X_np[:N]
y_np = y_np[:N]
all_results.append(bench("LSTM (50k)", LSTMNet, reshape_fn=lambda x: x.unsqueeze(-1)))
N = N_saved
X_np = X_np_saved
y_np = y_np_saved
all_results.append(bench("ResNet", ResNet))
all_results.append(bench("Wide&Deep", WideDeep))
all_results.append(bench("Autoencoder", Autoencoder))

# Matrix ops benchmark
print("\n  Large matrix multiply (4096x4096, 20 iterations):")
for dev_name in ["cpu", "mps"]:
    device = torch.device(dev_name)
    # warmup
    _ = torch.mm(torch.randn(256, 256, device=device), torch.randn(256, 256, device=device))
    if dev_name == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(20):
        a = torch.randn(4096, 4096, device=device)
        b = torch.randn(4096, 4096, device=device)
        c = torch.mm(a, b)
    if dev_name == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    if dev_name == "cpu":
        cpu_t = dt
    else:
        gpu_t = dt
speedup = cpu_t / gpu_t if gpu_t > 0 else 0
print(f"  {'MatrixMul':<20} CPU={cpu_t:>7.2f}s  GPU={gpu_t:>7.2f}s  Speedup={speedup:>5.1f}x")
all_results.append(("MatrixMul", cpu_t, gpu_t, speedup))

# ── Final summary ────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"{'Model':<20} {'CPU(s)':>10} {'GPU(s)':>10} {'Speedup':>10}")
print("-"*70)
for name, ct, gt, sp in all_results:
    marker = " ***" if sp > 2 else ""
    print(f"{name:<20} {ct:>10.2f} {gt:>10.2f} {sp:>9.1f}x{marker}")
print("="*70)

avg_speedup = np.mean([r[3] for r in all_results])
print(f"\nAverage speedup: {avg_speedup:.1f}x")
print("GPU sandbox override: WORKING")
