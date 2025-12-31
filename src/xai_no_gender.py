# explainability_real_units_with_lime_neutral.py

import os
import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
from lime.lime_tabular import LimeTabularExplainer
from evaluation import Net  # your model definition

# Optional: map feature names to units for nicer labeling
FEATURE_UNITS = {
    "C": "Conc",
    "T": "°C",
    "WBC": "10^9/L",
    "Age": "years",
    "Weight": "kg"
}

def run_explainability(model_path, scaler_path, states_csv, doses_csv, vis_dir, plot=True, n_samples=50):
    os.makedirs(vis_dir, exist_ok=True)

    # --- Load data ---
    df_X = pd.read_csv(states_csv)
    df_y = pd.read_csv(doses_csv)
    X_real = df_X.values.astype(np.float32)   # real units
    y = df_y.values.astype(np.float32)

    # --- Load scaler ---
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X_real)  # scaled for model input

    # --- Torch tensors ---
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # --- Load model ---
    input_size = X_tensor.shape[1]
    model = Net(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- Subset for speed ---
    if n_samples < X_tensor.shape[0]:
        X_tensor = X_tensor[:n_samples]
        X_scaled = X_scaled[:n_samples]
        X_real = X_real[:n_samples]

    # --- Wrapper for model prediction ---
    def predict_fn(x_real_input):
        # x_real_input in real units
        x_scaled_input = scaler.transform(x_real_input)
        with torch.no_grad():
            x_tensor_input = torch.tensor(x_scaled_input, dtype=torch.float32)
            return model(x_tensor_input).cpu().numpy()

    # --- SHAP Explainer ---
    explainer_shap = shap.Explainer(lambda x: predict_fn(x), X_real)
    shap_values = explainer_shap(X_real)

    if plot:
        feature_names = df_X.columns.tolist()

        # SHAP summary plot
        plt.figure(figsize=(10,6))
        shap.summary_plot(shap_values, X_real,
                          feature_names=[f"{n} ({FEATURE_UNITS.get(n,'')})" for n in feature_names],
                          show=False)
        summary_file = os.path.join(vis_dir, "shap_summary_real_units_no_sex.png")
        plt.tight_layout()
        plt.savefig(summary_file)
        plt.close()
        print(f"Saved SHAP summary plot to {summary_file}")

        # SHAP dependence plots
        for i, feature in enumerate(feature_names):
            plt.figure(figsize=(6,4))
            shap.dependence_plot(
                i, shap_values.values, X_real,
                feature_names=[f"{n} ({FEATURE_UNITS.get(n,'')})" for n in feature_names],
                show=False
            )
            dep_file = os.path.join(vis_dir, f"shap_dependence_{feature}_no_sex.png")
            plt.tight_layout()
            plt.savefig(dep_file)
            plt.close()
            print(f"Saved SHAP dependence plot for {feature} to {dep_file}")

        # --- LIME Explainer ---
        lime_explainer = LimeTabularExplainer(
            training_data=X_real,
            feature_names=[f"{n} ({FEATURE_UNITS.get(n,'')})" for n in feature_names],
            mode='regression',
            verbose=False,
            random_state=42
        )

        # Pick 5 random points to explain
        sample_indices = np.random.choice(X_real.shape[0], min(5, X_real.shape[0]), replace=False)
        for idx in sample_indices:
            exp = lime_explainer.explain_instance(
                X_real[idx],
                predict_fn,
                num_features=len(feature_names)
            )
            lime_file = os.path.join(vis_dir, f"lime_explanation_{idx}_no_sex.html")
            exp.save_to_file(lime_file)
            print(f"Saved LIME explanation for sample {idx} to {lime_file}")

    return shap_values

# -----------------------------
if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
    EXP_DIR = os.path.join(THIS_DIR, "../data/explainability")

    run_explainability(
        model_path=os.path.join(THIS_DIR, "../pk_trained_no_sex.pth"),
        scaler_path=os.path.join(THIS_DIR, "../scaler.pkl"),
        states_csv=os.path.join(DATA_DIR, "patient_states_no_sex.csv"),
        doses_csv=os.path.join(DATA_DIR, "dose_targets_no_sex.csv"),
        vis_dir=EXP_DIR,
        n_samples=100
    )
