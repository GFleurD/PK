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
            nn.ReLU()   # dosage must be non-negative
        )

    def forward(self, x):
        return self.model(x)


# -----------------------------------------
# PGD ATTACK (regression, MSE loss)
# -----------------------------------------
def pgd_attack(model, X, y, eps_scaled, alpha=0.01, iters=5):
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

            xb_adv = pgd_attack(model, xb, yb, eps_scaled, iters=5)

            optimizer.zero_grad()
            preds = model(xb_adv)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1:02d}/{epochs} | train loss: {epoch_loss:.4f}")

    return model


# -----------------------------------------
# Lipschitz estimates
# -----------------------------------------
def estimate_lipschitz_spectral(model):
    """
    Upper bound via product of spectral norms of linear layers.
    """
    L = 1.0
    for layer in model.model:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.detach()
            L *= torch.linalg.norm(weight, 2).item()
    return L


def empirical_lipschitz(model, X, eps_scaled):
    """
    Empirical Lipschitz over hyper-rectangular perturbations.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    X_pert = X_t.clone()

    for i, eps_i in eps_scaled.items():
        noise = (2 * torch.rand_like(X_t[:, i]) - 1) * eps_i
        X_pert[:, i] += noise

    with torch.no_grad():
        y_orig = model(X_t)
        y_pert = model(X_pert)

    delta_y = (y_pert - y_orig).abs().squeeze()
    delta_x = torch.norm(X_pert - X_t, dim=1)

    L = delta_y / delta_x
    return L.mean().item(), L.max().item()


def per_feature_lipschitz(model, X, eps_scaled):
    """
    Empirical per-feature sensitivity.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        y_orig = model(X_t)

    contributions = {}
    for i, eps_i in eps_scaled.items():
        X_pert = X_t.clone()
        noise = (2 * torch.rand_like(X_t[:, i]) - 1) * eps_i
        X_pert[:, i] += noise

        with torch.no_grad():
            y_pert = model(X_pert)

        delta_y = (y_pert - y_orig).abs().squeeze()
        delta_x = (X_pert[:, i] - X_t[:, i]).abs()
        contributions[i] = (delta_y / delta_x).mean().item()

    return contributions


# -----------------------------------------
# MAIN
# -----------------------------------------
def main():
    print("\n=== PGD Evaluation + Adversarial Training (No Sex) ===\n")

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
    VIS_DIR  = os.path.join(THIS_DIR, "../data/visualisations")
    PROJECT_DIR = os.path.dirname(THIS_DIR)
    os.makedirs(VIS_DIR, exist_ok=True)

    states_path = os.path.join(DATA_DIR, "patient_states_no_sex.csv")
    doses_path  = os.path.join(DATA_DIR, "dose_targets_no_sex.csv")
    scaler_path = os.path.join(PROJECT_DIR, "scaler.pkl")
    model_path  = os.path.join(PROJECT_DIR, "pk_trained_no_sex.pth")

    # --- Load data ---
    X = np.loadtxt(states_path, delimiter=",", skiprows=1).astype(np.float32)
    y = np.loadtxt(doses_path, delimiter=",", skiprows=1).astype(np.float32).reshape(-1,1)

    # --- Load scaler ---
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X).astype(np.float32)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # --- Load model ---
    model = Net(input_size=X_t.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # --- Feature-wise epsilons (no sex) ---
    raw_eps = {
        0: 0.1,   # C
        1: 0.05,  # T
        2: 0.2,  # WBC
        3: 0.5,   # Age
        4: 1.5    # Weight
    }
    eps_scaled = {i: raw_eps[i] / scaler.scale_[i] for i in raw_eps}

    # --- PGD evaluation ---
    X_adv = pgd_attack(model, X_t, y_t, eps_scaled, iters=8)
    with torch.no_grad():
        preds_orig = model(X_t).numpy()
        preds_adv  = model(X_adv).numpy()

    plt.figure(figsize=(6,6))
    plt.scatter(preds_orig, preds_adv, alpha=0.4)
    plt.xlabel("Original prediction")
    plt.ylabel("Adversarial prediction")
    plt.title("PGD shift in predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "pgd_shift_no_sex.png"))

    # --- Adversarial training ---
    adv_model = Net(input_size=X_t.shape[1])
    adv_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    adv_model = adversarial_train(adv_model, X_t, y_t, eps_scaled)

    adv_model_path = os.path.join(PROJECT_DIR, "pk_adv_no_sex.pth")
    torch.save(adv_model.state_dict(), adv_model_path)

    # --- Lipschitz metrics ---
    L_spec_orig = estimate_lipschitz_spectral(model)
    L_spec_adv  = estimate_lipschitz_spectral(adv_model)

    mean_L_orig, max_L_orig = empirical_lipschitz(model, X_scaled, eps_scaled)
    mean_L_adv,  max_L_adv  = empirical_lipschitz(adv_model, X_scaled, eps_scaled)

    print(f"Spectral L (orig): {L_spec_orig:.2f}")
    print(f"Spectral L (adv):  {L_spec_adv:.2f}")
    print(f"Empirical L orig → mean {mean_L_orig:.2f}, max {max_L_orig:.2f}")
    print(f"Empirical L adv  → mean {mean_L_adv:.2f}, max {max_L_adv:.2f}")

    # --- Per-feature Lipschitz ---
    feature_L = per_feature_lipschitz(adv_model, X_scaled, eps_scaled)

    plt.figure(figsize=(6,4))
    plt.bar([str(i) for i in feature_L], feature_L.values())
    plt.xlabel("Feature index")
    plt.ylabel("Mean Lipschitz contribution")
    plt.title("Per-feature sensitivity (adversarial model)")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "per_feature_lipschitz_no_sex.png"))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
