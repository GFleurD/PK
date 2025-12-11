# explainability.py

import os
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle

from evaluation import Net  # reuse the model definition

# -----------------------------
# SHAP explainability for regression
# -----------------------------
def run_shap(model_path, scaler_path, states_csv, doses_csv, plot=True, n_samples=50):
    # Load data
    df_X = pd.read_csv(states_csv)
    df_y = pd.read_csv(doses_csv)
    X = df_X.values.astype(np.float32)
    y = df_y.values.astype(np.float32)

    # Load scaler
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)

    # Torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Load model
    input_size = X_tensor.shape[1]
    model = Net(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Select subset for SHAP (for speed)
    if n_samples < X_tensor.shape[0]:
        X_tensor = X_tensor[:n_samples]

    # Wrap model for SHAP
    def predict_fn(x_numpy):
        with torch.no_grad():
            x_tensor = torch.tensor(x_numpy, dtype=torch.float32)
            return model(x_tensor).cpu().numpy()

    # SHAP explainer
    explainer = shap.Explainer(predict_fn, X_tensor.numpy())
    shap_values = explainer(X_tensor.numpy())

    if plot:
        shap.summary_plot(shap_values, X_tensor.numpy(), feature_names=df_X.columns)

    return shap_values

# -----------------------------
if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
    run_shap(
        model_path=os.path.join(THIS_DIR, "../pk_trained.pth"),
        scaler_path=os.path.join(THIS_DIR, "../scaler.pkl"),
        states_csv=os.path.join(DATA_DIR, "patient_states.csv"),
        doses_csv=os.path.join(DATA_DIR, "dose_targets.csv"),
        n_samples=100
    )
