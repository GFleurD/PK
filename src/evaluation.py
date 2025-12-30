# evaluation.py

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
# Metrics function
# -----------------------------
def evaluate_regression_model(model, X_train, y_train, X_test, y_test, plot=True):
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train).cpu().numpy()
        y_test_pred  = model(X_test).cpu().numpy()

    y_train_np = y_train.cpu().numpy()
    y_test_np  = y_test.cpu().numpy()

    train_mae = np.mean(np.abs(y_train_pred - y_train_np))
    test_mae  = np.mean(np.abs(y_test_pred - y_test_np))

    train_rmse = np.sqrt(np.mean((y_train_pred - y_train_np)**2))
    test_rmse  = np.sqrt(np.mean((y_test_pred - y_test_np)**2))

    ss_res_train = np.sum((y_train_np - y_train_pred)**2)
    ss_tot_train = np.sum((y_train_np - np.mean(y_train_np))**2)
    r2_train = 1 - ss_res_train / ss_tot_train

    ss_res_test = np.sum((y_test_np - y_test_pred)**2)
    ss_tot_test = np.sum((y_test_np - np.mean(y_test_np))**2)
    r2_test = 1 - ss_res_test / ss_tot_test

    print("\n=== Evaluation ===")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test  MAE: {test_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Test  RMSE: {test_rmse:.4f}")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test  R²: {r2_test:.4f}")

    if plot:
        plt.figure(figsize=(6,6))
        plt.scatter(y_test_np, y_test_pred, alpha=0.5)
        plt.plot([y_test_np.min(), y_test_np.max()],
                 [y_test_np.min(), y_test_np.max()], 'r--')
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("True vs Predicted")
        plt.tight_layout()
        plt.show()

# -----------------------------
# Main evaluation entry
# -----------------------------
def run_evaluation(model_path, scaler_path, states_csv, doses_csv):
    # Load data
    df_X = pd.read_csv(states_csv)
    df_y = pd.read_csv(doses_csv)
    X = df_X.values.astype(np.float32)
    y = df_y.values.astype(np.float32)

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    X_scaled = scaler.transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)

    # Load model
    input_size = X_train.shape[1]
    model = Net(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Evaluate
    evaluate_regression_model(model, X_train, y_train, X_test, y_test)

# -----------------------------
if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../data/csv")

    run_evaluation(
        model_path=os.path.join(THIS_DIR, "../pk.pth"),
        scaler_path=os.path.join(THIS_DIR, "../scaler.pkl"),
        states_csv=os.path.join(DATA_DIR, "patient_states.csv"),
        doses_csv=os.path.join(DATA_DIR, "dose_targets.csv")
    )
