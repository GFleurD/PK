# explainability.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths
# -----------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
EXPLAIN_DIR = os.path.join(THIS_DIR, "../data/explainability")
os.makedirs(EXPLAIN_DIR, exist_ok=True)

STATES_CSV = os.path.join(DATA_DIR, "patient_states.csv")
DOSES_CSV = os.path.join(DATA_DIR, "dose_targets.csv")
SCALER_PATH = os.path.join(THIS_DIR, "../scaler.pkl")
MODEL_PATH = os.path.join(THIS_DIR, "../pk_trained.pth")

FEATURE_NAMES = ['C','T','WBC','Age','Weight','Sex']

# -----------------------------
# Model definition
# -----------------------------
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
# Metrics / plots
# -----------------------------
def evaluate_regression_model(model, X_train, y_train, X_test, y_test, feature_names=FEATURE_NAMES):
    model.eval()
    with torch.no_grad():
        y_train_pred = model(X_train).cpu().numpy()
        y_test_pred  = model(X_test).cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    y_test_np  = y_test.cpu().numpy()

    # --- Basic metrics ---
    train_mae = np.mean(np.abs(y_train_pred - y_train_np))
    test_mae  = np.mean(np.abs(y_test_pred - y_test_np))
    train_rmse = np.sqrt(np.mean((y_train_pred - y_train_np)**2))
    test_rmse  = np.sqrt(np.mean((y_test_pred - y_test_np)**2))
    r2_train = 1 - np.sum((y_train_np - y_train_pred)**2)/np.sum((y_train_np - np.mean(y_train_np))**2)
    r2_test  = 1 - np.sum((y_test_np - y_test_pred)**2)/np.sum((y_test_np - np.mean(y_test_np))**2)

    print("\n=== Evaluation ===")
    print(f"Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
    print(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
    print(f"Train R²: {r2_train:.4f} | Test R²: {r2_test:.4f}")

    # --- Residual plot ---
    residuals = y_test_np - y_test_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_test_np, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("True Dose")
    plt.ylabel("Residual")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, "residual_plot.png"))
    plt.close()

    # --- SHAP explanations ---
    print("Computing SHAP values (may take a few seconds)...")
    def predict_fn_numpy(X_np):
        with torch.no_grad():
            X_torch = torch.tensor(X_np, dtype=torch.float32)
            return model(X_torch).numpy()
    explainer = shap.Explainer(predict_fn_numpy, X_test.cpu().numpy())
    shap_values = explainer(X_test.cpu().numpy())

    # SHAP summary
    shap.summary_plot(shap_values, X_test.cpu().numpy(), feature_names=feature_names, show=False)
    plt.savefig(os.path.join(EXPLAIN_DIR, "shap_summary.png"))
    plt.close()

    # SHAP dependence plots for each feature
    for i, feat in enumerate(feature_names):
        shap.dependence_plot(i, shap_values.values, X_test.cpu().numpy(), feature_names=feature_names, show=False)
        plt.savefig(os.path.join(EXPLAIN_DIR, f"shap_dependence_{feat}.png"))
        plt.close()

    # --- Optional: LIME explanation for one instance ---
    print("Generating LIME explanation for one random test instance...")
    explainer_lime = lime_tabular.LimeTabularExplainer(
        training_data=X_train.cpu().numpy(),
        feature_names=feature_names,
        mode='regression'
    )
    idx = np.random.randint(0, X_test.shape[0])
    exp = explainer_lime.explain_instance(
        X_test[idx].cpu().numpy(),
        predict_fn_numpy,
        num_features=len(feature_names)
    )
    exp.save_to_file(os.path.join(EXPLAIN_DIR, f"lime_instance_{idx}.html"))
    print(f"LIME explanation saved for test index {idx}")

# -----------------------------
# Main entry point
# -----------------------------
def run_explainability():
    # Load data
    df_X = pd.read_csv(STATES_CSV)
    df_y = pd.read_csv(DOSES_CSV)
    X = df_X.values.astype(np.float32)
    y = df_y.values.astype(np.float32)

    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)

    # Split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Torch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    X_test  = torch.tensor(X_test_np, dtype=torch.float32)
    y_test  = torch.tensor(y_test_np, dtype=torch.float32)

    # Load model
    input_size = X_train.shape[1]
    model = Net(input_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Evaluate and explain
    evaluate_regression_model(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    run_explainability()
