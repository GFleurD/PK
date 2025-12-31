#!/usr/bin/env python3
"""
Minimal PDT example with Lipschitz + stabilized domain constraints (no sex).
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

# -----------------------------
# Paths and data
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
SCALER_PATH = os.path.join(os.path.dirname(THIS_DIR), "scaler.pkl")
X_PATH = os.path.join(DATA_DIR, "patient_states_no_sex.csv")
Y_PATH = os.path.join(DATA_DIR, "dose_targets_no_sex.csv")

# Load data
df_X = pd.read_csv(X_PATH)
df_y = pd.read_csv(Y_PATH)
X = df_X.values.astype(np.float32)
y = df_y.values.astype(np.float32).reshape(-1,1)

# Load scaler and scale X
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)
X_scaled = scaler.transform(X).astype(np.float32)

# Torch tensors
X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X_t, y_t)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# Model
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
            nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)

model = Net(input_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -----------------------------
# Hyperrectangles for Lipschitz
# -----------------------------
raw_eps = {0:0.1,1:0.05,2:0.2,3:0.5,4:1.5}  # C, T, WBC, Age, Weight
eps_scaled = {i: raw_eps[i]/scaler.scale_[i] for i in raw_eps}

def sample_hyperrect(batch_size, eps_scaled):
    n = len(eps_scaled)
    noise = torch.zeros(batch_size,n, device=device)
    for i, eps in eps_scaled.items():
        noise[:,i] = (2*torch.rand(batch_size, device=device)-1)*eps
    return noise

def lipschitz_loss(model, x, eps_scaled, k_allowed=1.0):
    noise = sample_hyperrect(x.size(0), eps_scaled)
    x_pert = x + noise
    y = model(x)
    y_pert = model(x_pert)
    dy = (y_pert - y).abs().squeeze()
    dx = noise.abs().amax(dim=1)
    violation = (dy - k_allowed*dx).relu()
    return violation.mean()

# -----------------------------
# Domain constraints
# -----------------------------
IDX_CONC = 0
IDX_TEMP = 1
IDX_WBC = 2
TEMP_HEALTHY = 37.0
WBC_HEALTHY = 8.0
MAX_CONC = 40.0
DOMAIN_MULTIPLIER = 20.0  # strong but stable

def domain_loss(model, x):
    preds = model(x)
    
    # Convert back to original scale
    x_raw = x.cpu().numpy() * scaler.scale_ + scaler.mean_
    x_raw = torch.tensor(x_raw, dtype=torch.float32, device=x.device)
    
    # Healthy: dose should be ~0
    healthy_mask = ((x_raw[:,IDX_TEMP] <= TEMP_HEALTHY) | 
                    (x_raw[:,IDX_WBC] <= WBC_HEALTHY) | 
                    (x_raw[:,IDX_CONC] >= MAX_CONC))
    
    if healthy_mask.any():
        healthy_penalty = nn.SmoothL1Loss(reduction='mean')(
            preds[healthy_mask], torch.zeros_like(preds[healthy_mask])
        )
    else:
        healthy_penalty = torch.tensor(0.0, device=x.device)
    
    # Unhealthy: dose should be >0
    unhealthy_mask = ~healthy_mask
    if unhealthy_mask.any():
        target_unhealthy = torch.ones_like(preds[unhealthy_mask]) * 1e-3
        unhealthy_penalty = nn.SmoothL1Loss(reduction='mean')(
            preds[unhealthy_mask], target_unhealthy
        )
    else:
        unhealthy_penalty = torch.tensor(0.0, device=x.device)
    
    return DOMAIN_MULTIPLIER * (healthy_penalty + unhealthy_penalty)

# -----------------------------
# Training loop
# -----------------------------
EPOCHS = 20
for epoch in range(EPOCHS):
    total_loss = 0
    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()

        pred_loss = criterion(model(xb), yb)
        lip_loss = lipschitz_loss(model, xb, eps_scaled)
        dom_loss = domain_loss(model, xb)

        loss = pred_loss + 0.1*lip_loss + dom_loss
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    preds_all = model(X_t.to(device)).cpu().numpy()

mae = np.mean(np.abs(preds_all - y))
rmse = np.sqrt(np.mean((preds_all - y)**2))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Domain checks
x_raw = X_t.cpu().numpy() * scaler.scale_ + scaler.mean_
healthy_mask = ((x_raw[:,IDX_TEMP]<=TEMP_HEALTHY) | (x_raw[:,IDX_WBC]<=WBC_HEALTHY))
healthy_violation = preds_all[healthy_mask].clip(min=0).mean() if healthy_mask.any() else 0.0
print(f"Healthy dose violation (should be 0): {healthy_violation:.4f}")

# Robustness check
noise = sample_hyperrect(X_t.size(0), eps_scaled)
X_pert = X_t + noise
with torch.no_grad():
    preds_pert = model(X_pert.to(device)).cpu().numpy()
robust_change = np.max(np.abs(preds_pert - preds_all))
print(f"Max change under hyperrectangle perturbation: {robust_change:.4f}")

# -----------------------------
# Plotting
# -----------------------------
VIS_DIR  = os.path.join(THIS_DIR, "../data/visualisations")
os.makedirs(VIS_DIR, exist_ok=True)
plt.scatter(y, preds_all, alpha=0.5)
plt.plot([0,1500],[0,1500],'r--')
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Minimal PDT model")
plt.savefig(os.path.join(VIS_DIR, "pdt_true_vs_predicted.png"))
plt.show()
print("Saved minimal pdt true vs predicted.")

# -----------------------------
# Export to ONNX
# -----------------------------
model.eval()
input_tensor = torch.randn(1, input_size, dtype=torch.float32)
torch.onnx.export(
    model,
    input_tensor,
    "pk_pdt_no_sex.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True,
    export_params=True,
    external_data=False
)
print("ONNX model exported successfully!")

MODEL_PATH = "pk_pdt_no_sex.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"PyTorch model saved to {MODEL_PATH}")
