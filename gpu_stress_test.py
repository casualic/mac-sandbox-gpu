"""
GPU Stress Test — Multiple PyTorch model architectures on MPS (Apple Metal).
Tests: CNN, LSTM, Transformer, Autoencoder, ResNet-style, GAN, Attention.
Each model trains on the test.parquet data to verify GPU works end-to-end.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

# ── Setup ────────────────────────────────────────────────────────────────
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cpu":
    print("WARNING: MPS not available — running on CPU (sandbox may be blocking GPU)")

# ── Load data ────────────────────────────────────────────────────────────
print("Loading test.parquet ...")
df = pd.read_parquet("test.parquet")
feature_cols = [c for c in df.columns if c.startswith("feature_") and df[c].dtype in ("float64", "float32", "int64", "int32")]
X_np = df[feature_cols].values.astype("float32")
X_np = np.nan_to_num(X_np, 0.0)
y_np = (X_np[:, 0] > np.median(X_np[:, 0])).astype("float32")

# Use a subset for faster iteration
N = min(200_000, len(X_np))
X_np, y_np = X_np[:N], y_np[:N]
N_FEAT = X_np.shape[1]

X_t = torch.tensor(X_np, device=DEVICE)
y_t = torch.tensor(y_np, device=DEVICE)

print(f"Data: {N} rows, {N_FEAT} features\n")

EPOCHS = 3
BATCH = 2048
results = []


def train_model(name, model, reshape_fn=None, epochs=EPOCHS):
    """Generic training loop. Returns elapsed time."""
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    t0 = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        nb = 0
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
            ep_loss += loss.item()
            nb += 1
    dt = time.time() - t0
    avg_loss = ep_loss / max(nb, 1)
    params = sum(p.numel() for p in model.parameters())
    print(f"  [{name}] {epochs} epochs, {params:,} params, loss={avg_loss:.4f}, time={dt:.2f}s")
    results.append((name, params, avg_loss, dt))
    return dt


# ── 1. MLP (baseline) ───────────────────────────────────────────────────
print("1. MLP (3-layer)")
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

train_model("MLP", MLP())


# ── 2. 1D-CNN ───────────────────────────────────────────────────────────
print("2. 1D-CNN")
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
        x = x.view(x.size(0), -1)
        return self.fc(x)

train_model("1D-CNN", CNN1D(), reshape_fn=lambda x: x.unsqueeze(1))


# ── 3. LSTM ─────────────────────────────────────────────────────────────
print("3. LSTM")
class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Reshape features as a sequence of length N_FEAT, each with 1 channel
train_model("LSTM", LSTMNet(), reshape_fn=lambda x: x.unsqueeze(-1))


# ── 4. GRU ──────────────────────────────────────────────────────────────
print("4. GRU")
class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

train_model("GRU", GRUNet(), reshape_fn=lambda x: x.unsqueeze(-1))


# ── 5. Transformer Encoder ──────────────────────────────────────────────
print("5. Transformer Encoder")
class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1, 32)
        encoder_layer = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)

train_model("Transformer", TransformerModel(), reshape_fn=lambda x: x.unsqueeze(-1))


# ── 6. Autoencoder (reconstruction + classify from bottleneck) ──────────
print("6. Autoencoder")
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(N_FEAT, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, N_FEAT),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)

train_model("Autoencoder", Autoencoder())


# ── 7. ResNet-style (skip connections) ──────────────────────────────────
print("7. ResNet-style")
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
    def forward(self, x):
        return F.relu(self.block(x) + x)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(N_FEAT, 128), nn.ReLU())
        self.blocks = nn.Sequential(ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128))
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

train_model("ResNet", ResNet())


# ── 8. GAN (Generator + Discriminator) ──────────────────────────────────
print("8. GAN (mini)")
class Generator(nn.Module):
    def __init__(self, z_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, N_FEAT), nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEAT, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

Z_DIM = 32
gen = Generator(Z_DIM).to(DEVICE)
disc = Discriminator().to(DEVICE)
opt_g = torch.optim.Adam(gen.parameters(), lr=2e-4)
opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4)
bce = nn.BCEWithLogitsLoss()

t0 = time.time()
for ep in range(EPOCHS):
    for i in range(0, N, BATCH):
        real = X_t[i:i+BATCH]
        bs = real.size(0)
        # Discriminator
        z = torch.randn(bs, Z_DIM, device=DEVICE)
        fake = gen(z).detach()
        d_loss = bce(disc(real), torch.ones(bs, 1, device=DEVICE)) + bce(disc(fake), torch.zeros(bs, 1, device=DEVICE))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        # Generator
        z = torch.randn(bs, Z_DIM, device=DEVICE)
        g_loss = bce(disc(gen(z)), torch.ones(bs, 1, device=DEVICE))
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
dt = time.time() - t0
params = sum(p.numel() for p in gen.parameters()) + sum(p.numel() for p in disc.parameters())
print(f"  [GAN] {EPOCHS} epochs, {params:,} params, d_loss={d_loss.item():.4f}, time={dt:.2f}s")
results.append(("GAN", params, d_loss.item(), dt))


# ── 9. Multi-Head Attention (standalone) ─────────────────────────────────
print("9. Multi-Head Attention")
class AttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.proj(x)
        x, _ = self.attn(x, x, x)
        x = x.mean(dim=1)
        return self.fc(x)

train_model("Attention", AttentionClassifier(), reshape_fn=lambda x: x.unsqueeze(-1))


# ── 10. Wide & Deep ─────────────────────────────────────────────────────
print("10. Wide & Deep")
class WideAndDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.wide = nn.Linear(N_FEAT, 1)
        self.deep = nn.Sequential(
            nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.wide(x) + self.deep(x)

train_model("Wide&Deep", WideAndDeep())


# ── 11. Large matrix ops (pure GPU compute stress) ──────────────────────
print("11. Large Matrix Ops (stress test)")
t0 = time.time()
for _ in range(10):
    a = torch.randn(4096, 4096, device=DEVICE)
    b = torch.randn(4096, 4096, device=DEVICE)
    c = torch.mm(a, b)
    _ = torch.svd(c[:512, :512])
torch.mps.synchronize()
dt = time.time() - t0
print(f"  [MatrixOps] 10x (4096x4096 matmul + 512x512 SVD), time={dt:.2f}s")
results.append(("MatrixOps", 0, 0, dt))


# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<16} {'Params':>10} {'Loss':>10} {'Time(s)':>10}")
print("-"*65)
for name, params, loss, dt in results:
    print(f"{name:<16} {params:>10,} {loss:>10.4f} {dt:>10.2f}")
print("="*65)
print(f"\nAll models trained on: {DEVICE}")
if DEVICE.type == "mps":
    print("GPU sandbox override: WORKING")
else:
    print("GPU sandbox override: NOT WORKING (fell back to CPU)")
