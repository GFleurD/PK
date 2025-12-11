#!/usr/bin/env python3
"""
Property-driven training (PDT) for your PK/PD model.

Fixed autograd/backward issues:
 - single backward per optimizer.step()
 - adversarial examples detached after creation
 - empirical Lipschitz + domain constraint penalties preserved
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Flinkow-ish imports (used earlier in your environment)
import property_driven_ml.logics as logics
import property_driven_ml.constraints as constraints
import property_driven_ml.training as training

# -----------------------------
# Paths & data load
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
VIS_DIR  = os.path.join(THIS_DIR, "../data/visualisations")
os.makedirs(VIS_DIR, exist_ok=True)

X_path = os.path.join(DATA_DIR, "patient_states.csv")
y_path = os.path.join(DATA_DIR, "dose_targets.csv")
SCALER_PATH = os.path.join(os.path.dirname(THIS_DIR), "scaler.pkl")  # scaler saved in project root

df_X = pd.read_csv(X_path)
df_y = pd.read_csv(y_path)

X = df_X.values.astype(np.float32)   # [C, T, WBC, Age, Weight, Sex]
y = df_y.values.astype(np.float32).reshape(-1, 1)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

X_scaled = scaler.transform(X).astype(np.float32)

# Torch dataset
X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

dataset = TensorDataset(X_t, y_t)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# Model (same architecture)
# -----------------------------
input_size = X_t.shape[1]

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()   # keep non-negative output
        )

    def forward(self, x):
        return self.model(x)

model = Net(input_size)

# -----------------------------
# Device, optimizer, criterion
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -----------------------------
# Hyperrectangles (original units -> scaled)
# -----------------------------
raw_eps_dict = {0:0.1, 1:0.05, 2:0.05, 3:0.5, 4:0.0, 5:1.5}
eps_scaled = {i: raw_eps_dict[i] / scaler.scale_[i] for i in raw_eps_dict}

# helper to sample within hyperrectangle (batch-wise)
def sample_hyperrect_noise(batch_size, eps_scaled_dict, device):
    n = len(eps_scaled_dict)
    noise = torch.zeros((batch_size, n), dtype=torch.float32, device=device)
    for i, eps in eps_scaled_dict.items():
        if eps == 0:
            continue
        noise[:, i] = (2.0 * torch.rand(batch_size, device=device) - 1.0) * eps
    return noise

# -----------------------------
# PGD oracle (use Flinkow's training.PGD if present)
# -----------------------------
logic = logics.GoedelFuzzyLogic()
try:
    oracle = training.PGD(logic, device, steps=5, restarts=2, step_size=0.01)
    have_oracle = True
    print("Using PGD oracle from property_driven_ml.training")
except Exception:
    oracle = None
    have_oracle = False
    print("PGD oracle unavailable; continuing without it")

# -----------------------------
# GradNorm detection (kept read-only)
# -----------------------------
try:
    grad_norm = training.GradNorm(model, device, optimizer, lr=1e-3, alpha=1.0)
    have_gradnorm = True
    print("GradNorm available (will not call internal backward).")
except Exception:
    grad_norm = None
    have_gradnorm = False

# -----------------------------
# Domain constraint helpers (implemented as differentiable PyTorch penalties)
# -----------------------------
# indices mapping
IDX_CONC = 0
IDX_TEMP = 1
IDX_WBC = 2
IDX_AGE = 3
IDX_WEIGHT = 4
IDX_SEX = 5

TEMP_HEALTHY = 37.5
WBC_HEALTHY = 8.0
C_MAX = 30.0   # concentration safety threshold (raw units) -- tune as needed

def domain_losses(model, x_batch, y_batch, eps_scaled_local):
    """
    Compute domain-related penalty terms on a batch (all tensors on device).
    Returns: (healthy_loss, conc_loss, sex_loss)
    All are scalar tensors (can be zero).
    """
    preds = model(x_batch)  # shape (B,1)

    # Convert scaled features back to raw for masking:
    x_raw = (x_batch.cpu().numpy() * scaler.scale_) + scaler.mean_
    x_raw = torch.tensor(x_raw, dtype=torch.float32, device=x_batch.device)

    # Healthy mask: if TEMP <= 37.5 OR WBC <= 8 then healthy -> expect dose 0
    healthy_mask = ((x_raw[:, IDX_TEMP] <= TEMP_HEALTHY) | (x_raw[:, IDX_WBC] <= WBC_HEALTHY))
    if healthy_mask.any():
        healthy_preds = preds[healthy_mask]
        healthy_loss = healthy_preds.relu().mean()
    else:
        healthy_loss = torch.tensor(0.0, device=x_batch.device)

    # Concentration safety: approximate next concentration
    k_conc = 1.0 / 30.0
    x_raw_C = x_raw[:, IDX_CONC]
    next_conc_approx = x_raw_C + (preds.squeeze() * k_conc)
    conc_violation = (next_conc_approx - C_MAX).clamp(min=0.0)
    conc_loss = conc_violation.mean()

    # Sex invariance: flip sex bit and compute difference
    x_flipped = x_batch.clone()
    x_flipped[:, IDX_SEX] = 1.0 - x_flipped[:, IDX_SEX]
    preds_flipped = model(x_flipped)
    sex_loss = (preds_flipped - preds).abs().mean()

    return healthy_loss, conc_loss, sex_loss

# -----------------------------
# Empirical Lipschitz soft penalty (hyperrectangle-aware)
# -----------------------------
def lipschitz_penalty(model, x_batch, eps_scaled_local, k_allowed=1.0):
    """
    Soft Lipschitz penalty: for a random perturbation within eps hyperrectangle,
    compute violation = relu(|f(x') - f(x)| - k_allowed * ||x'-x||_inf)
    Return mean violation (scalar).
    """
    batch_size = x_batch.shape[0]
    noise = sample_hyperrect_noise(batch_size, eps_scaled_local, device)
    x_pert = x_batch + noise

    y = model(x_batch)
    yp = model(x_pert)
    dy = (yp - y).abs().squeeze()    # (B,)
    dx = (noise).abs().amax(dim=1)  # L-inf per sample

    dx_safe = dx + 1e-12
    violation = (dy - k_allowed * dx_safe).relu()
    return violation.mean()

# -----------------------------
# Training loop (PDT style)
# -----------------------------
EPOCHS = 20
alpha_weights = {"pred": 1.0, "lip": 1.0, "dom": 1.0}  # fallback weights

model.train()
for epoch in range(EPOCHS):
    total_pred = 0.0
    total_lip = 0.0
    total_dom = 0.0
    n_batches = 0

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        # 1) Prediction loss
        preds = model(xb)
        pred_loss = criterion(preds, yb)

        # 2) Lipschitz penalty
        lip_loss = lipschitz_penalty(model, xb, eps_scaled, k_allowed=1.0)

        # 3) Domain losses (on clean data)
        healthy_loss, conc_loss, sex_loss = domain_losses(model, xb, yb, eps_scaled)
        dom_loss = healthy_loss + conc_loss + sex_loss

        # 4) Optionally get adversarial sample (oracle) to evaluate domain loss more strictly.
        #    Ensure returned adversarial examples are detached so they don't hold old graphs.
        adv_dom_loss = torch.tensor(0.0, device=device)
        if have_oracle:
            try:
                xb_adv = oracle.attack(model, xb, yb, constraints.StandardRobustnessConstraint(device=device, epsilon=0.0, delta=0.0))
                if isinstance(xb_adv, torch.Tensor):
                    xb_adv = xb_adv.detach()
                    h_la, c_la, s_la = domain_losses(model, xb_adv, yb, eps_scaled)
                    adv_dom_loss = h_la + c_la + s_la
            except Exception:
                # if oracle fails for any reason, just skip adversarial domain loss
                adv_dom_loss = torch.tensor(0.0, device=device)

        # 5) Compose final loss (single scalar) and backward once
        total_dom_loss = dom_loss + adv_dom_loss
        total_loss = (alpha_weights["pred"] * pred_loss
                      + alpha_weights["lip"] * lip_loss
                      + alpha_weights["dom"] * total_dom_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_pred += pred_loss.item()
        total_lip += lip_loss.item()
        total_dom += total_dom_loss.item()
        n_batches += 1

    print(f"Epoch {epoch+1}/{EPOCHS} | pred {total_pred/n_batches:.4f} | lip {total_lip/n_batches:.6f} | dom {total_dom/n_batches:.6f}")

# -----------------------------
# Evaluation and save
# -----------------------------
model.eval()
with torch.no_grad():
    preds_all = model(X_t.to(device)).cpu().numpy()

plt.figure(figsize=(6,6))
plt.scatter(y, preds_all, alpha=0.5)
plt.plot([0, 1500], [0, 1500], 'r--')
plt.xlabel("True Dose")
plt.ylabel("Predicted Dose")
plt.title("PDT-trained model: True vs Predicted Dose")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "pdt_true_vs_predicted.png"))
plt.show()

# save weights and ONNX
OUT_PTH = os.path.join(os.path.dirname(THIS_DIR), "pk_pdt_trained.pth")
torch.save(model.state_dict(), OUT_PTH)

dummy_in = torch.randn(1, input_size, dtype=torch.float32)
OUT_ONNX = os.path.join(os.path.dirname(THIS_DIR), "pk_pdt.onnx")
torch.onnx.export(model.cpu(), dummy_in, OUT_ONNX, input_names=["input"], output_names=["output"], opset_version=18)
print("Saved PDT model:", OUT_PTH, OUT_ONNX)
