#!/usr/bin/env python3
"""
Evaluation script for PK/PD models (vanilla or PDT-trained).

Features:
- Standard regression metrics (MAE, RMSE, R²)
- Domain constraint adherence:
    * Healthy -> dose ~ 0
    * Sex invariance
    * Concentration safety (approx)
- Robustness check: small perturbations within hyperrectangles
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split

# -----------------------------
# Model definition (must match training)
# -----------------------------
import torch.nn as nn

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

# -----------------------------
# Metrics & domain checks
# -----------------------------
IDX_CONC = 0
IDX_TEMP = 1
IDX_WBC  = 2
IDX_AGE  = 3
IDX_WEIGHT = 4
IDX_SEX = 5

TEMP_HEALTHY = 37.5
WBC_HEALTHY  = 8.0
C_MAX = 30.0

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler, eps_scaled=None, plot=True):
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train).cpu().numpy()
        y_test_pred  = model(X_test).cpu().numpy()

    y_train_np = y_train.cpu().numpy()
    y_test_np  = y_test.cpu().numpy()

    # Standard regression metrics
    train_mae = np.mean(np.abs(y_train_pred - y_train_np))
    test_mae  = np.mean(np.abs(y_test_pred - y_test_np))
    train_rmse = np.sqrt(np.mean((y_train_pred - y_train_np)**2))
    test_rmse  = np.sqrt(np.mean((y_test_pred - y_test_np)**2))
    r2_train = 1 - np.sum((y_train_np - y_train_pred)**2)/np.sum((y_train_np - np.mean(y_train_np))**2)
    r2_test  = 1 - np.sum((y_test_np - y_test_pred)**2)/np.sum((y_test_np - np.mean(y_test_np))**2)

    # Domain adherence
    X_test_raw = X_test.cpu().numpy() * scaler.scale_ + scaler.mean_
    X_test_raw_torch = torch.tensor(X_test_raw, dtype=torch.float32)

    y_test_pred_torch = torch.tensor(y_test_pred, dtype=torch.float32)

    # Healthy dose ~0
    healthy_mask = (X_test_raw_torch[:, IDX_TEMP] <= TEMP_HEALTHY) | (X_test_raw_torch[:, IDX_WBC] <= WBC_HEALTHY)
    healthy_violation = y_test_pred_torch[healthy_mask].relu().mean().item() if healthy_mask.any() else 0.0

    # Sex invariance
    X_flip = X_test.clone()
    X_flip[:, IDX_SEX] = 1 - X_flip[:, IDX_SEX]
    y_flip_pred = model(X_flip).cpu()
    sex_diff = (y_flip_pred - y_test_pred_torch).abs().mean().item()

    # Concentration safety approx
    k_conc = 1.0/30.0
    next_conc = X_test_raw_torch[:, IDX_CONC] + y_test_pred_torch.squeeze() * k_conc
    conc_violation = (next_conc - C_MAX).clamp(min=0.0).mean().item()

    # Robustness if hyperrectangle eps given
    robustness_violation = None
    if eps_scaled is not None:
        batch_size = X_test.shape[0]
        noise = torch.zeros_like(X_test)
        for i, eps in eps_scaled.items():
            if eps>0:
                noise[:, i] = (2*torch.rand(batch_size) -1) * eps
        y_pert = model(X_test + noise)
        robustness_violation = (y_pert - y_test_pred_torch).abs().amax().item()

    print("\n=== Evaluation Metrics ===")
    print(f"Train MAE / RMSE / R²: {train_mae:.4f} / {train_rmse:.4f} / {r2_train:.4f}")
    print(f"Test  MAE / RMSE / R²: {test_mae:.4f} / {test_rmse:.4f} / {r2_test:.4f}")
    print(f"Healthy dose violation (should be 0): {healthy_violation:.4f}")
    print(f"Sex invariance violation: {sex_diff:.4f}")
    print(f"Concentration safety violation: {conc_violation:.4f}")
    if robustness_violation is not None:
        print(f"Max change under hyperrectangle perturbation: {robustness_violation:.4f}")

    if plot:
        plt.figure(figsize=(6,6))
        plt.scatter(y_test_np, y_test_pred, alpha=0.5)
        plt.plot([0, max(y_test_np.max(), y_test_pred.max())],
                 [0, max(y_test_np.max(), y_test_pred.max())], 'r--')
        plt.xlabel("True Dose")
        plt.ylabel("Predicted Dose")
        plt.title("True vs Predicted Dose")
        plt.tight_layout()
        plt.show()

# -----------------------------
# Main entry
# -----------------------------
def run_evaluation(model_path, scaler_path, states_csv, doses_csv, eps_scaled=None):
    df_X = pd.read_csv(states_csv)
    df_y = pd.read_csv(doses_csv)
    X = df_X.values.astype(np.float32)
    y = df_y.values.astype(np.float32).reshape(-1,1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_t = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    X_test_t  = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.float32)

    model = Net(X_train_t.shape[1])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    evaluate_model(model, X_train_t, y_train_t, X_test_t, y_test_t, scaler, eps_scaled=eps_scaled)

# -----------------------------
if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../data/csv")

    # Optional: hyperrectangle eps for robustness check
    eps_scaled = {0:0.1, 1:0.05, 2:0.05, 3:0.5, 4:0.0, 5:1.5}

    run_evaluation(
        model_path=os.path.join(THIS_DIR, "../pk_pdt_trained.pth"),
        scaler_path=os.path.join(THIS_DIR, "../scaler.pkl"),
        states_csv=os.path.join(DATA_DIR, "patient_states.csv"),
        doses_csv=os.path.join(DATA_DIR, "dose_targets.csv"),
        eps_scaled=eps_scaled
    )
