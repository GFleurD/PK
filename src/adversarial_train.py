import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

# -----------------------------------------
# Model definition
# -----------------------------------------
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


# -----------------------------------------
# PGD ATTACK
# -----------------------------------------
def pgd_attack(model, X, y, eps_scaled, alpha=0.01, iters=5, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    X_adv = X.clone().detach()
    X_adv.requires_grad = True

    for _ in range(iters):
        model.zero_grad()
        preds = model(X_adv)
        loss = loss_fn(preds, y)
        loss.backward()

        with torch.no_grad():
            grad_sign = X_adv.grad.sign()
            for i, eps_i in eps_scaled.items():
                X_adv[:, i] += alpha * grad_sign[:, i]
                X_adv[:, i] = torch.clamp(
                    X_adv[:, i],
                    X[:, i] - eps_i,
                    X[:, i] + eps_i
                )

        X_adv.grad.zero_()
    return X_adv.detach()


# -----------------------------------------
# ADVERSARIAL TRAINING
# -----------------------------------------
def adversarial_train(model, X, y, eps_scaled, epochs=20, batch_size=64, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N = len(X)

    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y[idx]

            # generate adversarial batch
            xb_adv = pgd_attack(model, xb, yb, eps_scaled, alpha=0.01, iters=5, loss_fn=loss_fn)

            optimizer.zero_grad()
            preds = model(xb_adv)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs} | train loss: {epoch_loss:.4f}")

    return model


# -----------------------------------------
# MAIN
# -----------------------------------------
def main():
    print("\n=== Running PGD Evaluation + Adversarial Training ===\n")

    # --- Paths ---
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
    VIS_DIR  = os.path.join(THIS_DIR, "../data/visualisations")
    os.makedirs(VIS_DIR, exist_ok=True)

    states_path = os.path.join(DATA_DIR, "patient_states.csv")
    doses_path  = os.path.join(DATA_DIR, "dose_targets.csv")
    PROJECT_DIR = os.path.dirname(THIS_DIR)
    scaler_path = os.path.join(PROJECT_DIR, "scaler.pkl")
    model_path  = os.path.join(PROJECT_DIR, "pk_trained.pth")

    # --- Load data ---
    X = np.loadtxt(states_path, delimiter=",", skiprows=1, usecols=range(6)).astype(np.float32)
    y = np.loadtxt(doses_path, delimiter=",", skiprows=1).astype(np.float32).reshape(-1,1)

    # --- Load scaler ---
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X).astype(np.float32)

    # --- Convert to tensors ---
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # --- Load original model ---
    model = Net(input_size=X_t.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print("Loaded model, scaler, and data successfully.\n")

    # --- Scale epsilons ---
    raw_eps_dict = {0:0.1, 1:0.05, 2:0.05, 3:0.5, 4:0.0, 5:1.5}
    eps_scaled = {i: raw_eps_dict[i]/scaler.scale_[i] for i in raw_eps_dict}

    # --- PGD Evaluation ---
    X_adv = pgd_attack(model, X_t, y_t, eps_scaled, alpha=0.01, iters=8)
    with torch.no_grad():
        preds_orig = model(X_t).numpy()
        preds_adv  = model(X_adv).numpy()

    print("=== Sample of adversarial changes ===")
    for i in range(5):
        print(f"True y={y[i,0]:.2f}, orig pred={preds_orig[i,0]:.2f}, adv pred={preds_adv[i,0]:.2f}")

    # --- Visualize PGD shift ---
    plt.figure(figsize=(6,6))
    plt.scatter(preds_orig, preds_adv, alpha=0.4)
    plt.xlabel("Original Prediction")
    plt.ylabel("Adversarial Prediction")
    plt.title("PGD Shift in Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "pgd_shift.png"))
    print("Saved PGD scatter plot.\n")

    # --- Adversarial Training ---
    adv_model = Net(input_size=X_t.shape[1])
    adv_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    adv_model.train()
    adv_model = adversarial_train(adv_model, X_t, y_t, eps_scaled, epochs=20, batch_size=64, lr=1e-3)

    # --- Save adversarially trained model ---
    adv_model_path = os.path.join(PROJECT_DIR, "pk_adv.pth")
    torch.save(adv_model.state_dict(), adv_model_path)

    # --- Export to ONNX ---
    dummy_input = torch.rand(1, X_t.shape[1], dtype=torch.float32)
    onnx_path = os.path.join(PROJECT_DIR, "pk_adv.onnx")
    torch.onnx.export(
        adv_model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        external_data=False 
    )

    print(f"Adversarially trained model saved as:\n - {adv_model_path}\n - {onnx_path}\n")

    # --- Visualize True vs Original vs Adversarially Trained Predictions ---
    
    adv_model.eval()
    with torch.no_grad():
        preds_adv_train = adv_model(X_t).numpy()

    y_flat = y.flatten()  # Flatten to 1D
    plt.figure(figsize=(6,6))
    plt.scatter(y_flat, preds_orig, alpha=0.4, label="Original Model")
    plt.scatter(y_flat, preds_adv_train, alpha=0.4, label="Adversarially Trained Model")
    plt.plot([0, max(y_flat)], [0, max(y_flat)], 'r--', label="Ideal x=y")
    plt.xlabel("True Dose")
    plt.ylabel("Predicted Dose")
    plt.title("True vs Model Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "true_vs_adv_vs_orig.png"))
    print("Saved True vs Original vs Adversarial predictions plot.\nDone!\n")

    # -----------------------------
    # Lipschitz estimation
    # -----------------------------
    def estimate_lipschitz(model):
        """
        Rough upper bound on the Lipschitz constant of a feedforward ReLU network.
        Multiplies spectral norms of linear layers; ReLU slope <= 1.
        """
        L = 1.0
        for layer in model.model:
            if isinstance(layer, nn.Linear):
                weight = layer.weight.detach()
                # Spectral norm (largest singular value)
                weight_norm = torch.linalg.norm(weight, 2)
                L *= weight_norm.item()
        return L

    # After loading your original model:
    L_orig = estimate_lipschitz(model)
    print(f"Estimated Lipschitz constant of original model: {L_orig:.4f}")

    # After adversarial training:
    adv_model.eval()
    L_adv = estimate_lipschitz(adv_model)
    print(f"Estimated Lipschitz constant of adversarially trained model: {L_adv:.4f}")

    def empirical_lipschitz_hyperrect(model, X, eps_scaled):
        """
        Estimate empirical Lipschitz over variable eps per feature.
        """
        X_torch = torch.tensor(X, dtype=torch.float32)
        X_adv = X_torch.clone()
        
        # Apply per-feature perturbation: uniform random in [-eps_i, eps_i]
        for i, eps_i in eps_scaled.items():
            noise = (2*torch.rand(X_torch[:,i].shape)-1) * eps_i
            X_adv[:, i] += noise

        with torch.no_grad():
            y_orig = model(X_torch)
            y_pert = model(X_adv)

        delta_y = (y_pert - y_orig).abs()
        delta_x = torch.sqrt(((X_adv - X_torch)**2).sum(dim=1))
        lipschitz_est = delta_y.squeeze() / delta_x

        return lipschitz_est.mean().item(), lipschitz_est.max().item()


    mean_L_orig, max_L_orig = empirical_lipschitz_hyperrect(model, X_scaled, eps_scaled)
    mean_L_adv, max_L_adv   = empirical_lipschitz_hyperrect(adv_model, X_scaled, eps_scaled)

    print(f"Original model → mean L ≈ {mean_L_orig:.2f}, max L ≈ {max_L_orig:.2f}")
    print(f"Adversarial model → mean L ≈ {mean_L_adv:.2f}, max L ≈ {max_L_adv:.2f}")

    # -----------------------------
    # Per-feature Lipschitz contributions
    # -----------------------------
    def per_feature_lipschitz(model, X, eps_scaled):
        """
        Estimate the empirical Lipschitz per feature.
        Returns a dict {feature_idx: mean contribution}.
        """
        X_torch = torch.tensor(X, dtype=torch.float32)
        contributions = {}

        with torch.no_grad():
            y_orig = model(X_torch)

            for i, eps_i in eps_scaled.items():
                X_pert = X_torch.clone()
                noise = (2*torch.rand(X_torch[:,i].shape)-1) * eps_i
                X_pert[:, i] += noise

                y_pert = model(X_pert)
                delta_y = (y_pert - y_orig).abs().squeeze()
                delta_x = torch.abs(X_pert[:,i] - X_torch[:,i])  # L1 along this feature
                contributions[i] = (delta_y / delta_x).mean().item()

        return contributions

    # --- Compute per-feature Lipschitz ---
    feature_L = per_feature_lipschitz(adv_model, X_scaled, eps_scaled)

    # --- Plot ---
    plt.figure(figsize=(6,4))
    plt.bar([str(i) for i in feature_L.keys()], feature_L.values(), alpha=0.7)
    plt.xlabel("Feature index")
    plt.ylabel("Mean Lipschitz contribution")
    plt.title("Per-feature Sensitivity of Adversarially Trained Model")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "per_feature_lipschitz.png"))
    print("Saved per-feature Lipschitz plot.\n")







if __name__ == "__main__":
    main()
    


