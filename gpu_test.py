"""GPU (MPS) test on macOS with PyTorch — trains a simple model on test.parquet."""

import time
import torch
import torch.nn as nn
import pandas as pd

# --- Device setup ---
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using GPU: MPS (Apple Metal)")
else:
    DEVICE = torch.device("cpu")
    print("MPS not available, falling back to CPU")

# --- Load data ---
print("Loading test.parquet ...")
df = pd.read_parquet("test.parquet")

feature_cols = [c for c in df.columns if c.startswith("feature_")]
numeric_features = [c for c in feature_cols if df[c].dtype in ("float64", "float32", "int64", "int32")]

X = df[numeric_features].values.astype("float32")
# Fake binary target from median split of first feature for demo
y = (X[:, 0] > float(pd.Series(X[:, 0]).median())).astype("float32")

print(f"Data: {X.shape[0]} rows, {X.shape[1]} features")

# --- Convert to tensors ---
X_t = torch.tensor(X, device=DEVICE)
y_t = torch.tensor(y, device=DEVICE).unsqueeze(1)

# --- Simple model ---
class SimpleNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNet(X_t.shape[1]).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# --- Train ---
EPOCHS = 5
BATCH = 4096

print(f"\nTraining {EPOCHS} epochs, batch size {BATCH}, device={DEVICE}")
for epoch in range(EPOCHS):
    t0 = time.time()
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, len(X_t), BATCH):
        xb = X_t[i : i + BATCH]
        yb = y_t[i : i + BATCH]
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    dt = time.time() - t0
    print(f"  epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/n_batches:.4f}  time={dt:.2f}s")

print("\nDone — GPU training completed successfully." if DEVICE.type == "mps" else "\nDone — CPU training completed.")
