import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt

# -----------------------------------------
# Model definition (UNCHANGED)
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
# PGD ATTACK (UNCHANGED)
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
# DL2 / GÖDEL DOMAIN LOSS
# -----------------------------------------
def dl2_domain_loss(preds,
                    safe_max=2.0,
                    unhealthy_min=6.0,
                    weight=1.0):
    safe_violation = torch.relu(preds - safe_max)
    unhealthy_violation = torch.relu(unhealthy_min - preds)
    healthy_center = 0.5 * (safe_max + unhealthy_min)
    healthy_band = torch.abs(preds - healthy_center)
    loss = safe_violation.mean() + unhealthy_violation.mean() + 0.1 * healthy_band.mean()
    return weight * loss


# -----------------------------------------
# ADVERSARIAL TRAINING + DL2 (CONVEX, NORMALIZED)
# -----------------------------------------
def adversarial_train_dl2_convex(model,
                                 X,
                                 y,
                                 eps_scaled,
                                 epochs=20,
                                 batch_size=64,
                                 lr=1e-3,
                                 alpha=0.5,      # convex weight for MSE vs DL2
                                 dl2_weight=1.0):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    N = len(X)

    for epoch in range(epochs):
        perm = torch.randperm(N)
        epoch_mse = 0.0
        epoch_dl2 = 0.0
        epoch_total = 0.0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y[idx]

            xb_adv = pgd_attack(model, xb, yb, eps_scaled, iters=5)
            preds = model(xb_adv)

            mse = mse_loss(preds, yb)
            dl2 = dl2_domain_loss(preds, weight=dl2_weight)

            # Normalize DL2 per batch
            dl2_norm = dl2 / (dl2.detach().mean() + 1e-6) * mse.detach().mean()

            # Convex combination
            loss = alpha * mse + (1 - alpha) * dl2_norm

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_mse += mse.item()
            epoch_dl2 += dl2.item()
            epoch_total += loss.item()

        print(f"[Convex] Epoch {epoch+1:02d}/{epochs} | "
              f"MSE: {epoch_mse:.4f} | DL2: {epoch_dl2:.4f} | Total: {epoch_total:.4f}")

    return model


# -----------------------------------------
# Lipschitz metrics
# -----------------------------------------
def estimate_lipschitz_spectral(model):
    L = 1.0
    for layer in model.model:
        if isinstance(layer, nn.Linear):
            L *= torch.linalg.norm(layer.weight.detach(), 2).item()
    return L


def empirical_lipschitz(model, X, eps_scaled):
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
    print("\n=== PGD + DL2 Convex Training ===\n")

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
    VIS_DIR  = os.path.join(THIS_DIR, "../data/visualisations")
    PROJECT_DIR = os.path.dirname(THIS_DIR)
    os.makedirs(VIS_DIR, exist_ok=True)

    states_path = os.path.join(DATA_DIR, "patient_states_no_sex.csv")
    doses_path  = os.path.join(DATA_DIR, "dose_targets_no_sex.csv")
    scaler_path = os.path.join(PROJECT_DIR, "scaler.pkl")
    model_path  = os.path.join(PROJECT_DIR, "pk_trained_no_sex.pth")

    X = np.loadtxt(states_path, delimiter=",", skiprows=1).astype(np.float32)
    y = np.loadtxt(doses_path, delimiter=",", skiprows=1).astype(np.float32).reshape(-1,1)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X).astype(np.float32)

    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    raw_eps = {0: 0.1, 1: 0.05, 2: 0.2, 3: 0.5, 4: 1.5}
    eps_scaled = {i: raw_eps[i] / scaler.scale_[i] for i in raw_eps}

    # Load pretrained model
    model = Net(input_size=X_t.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Train with convex DL2
    dl2_model = adversarial_train_dl2_convex(
        model, X_t, y_t, eps_scaled,
        epochs=20, lr=1e-3, alpha=0.5, dl2_weight=1.0
    )

    torch.save(dl2_model.state_dict(),
               os.path.join(PROJECT_DIR, "pk_adv_dl2_convex_no_sex.pth"))
    
        # -------------------------------
    # Save as ONNX
    # -------------------------------
    onnx_path = os.path.join(PROJECT_DIR, "pk_adv_dl2_convex_no_sex.onnx")
    dl2_model.eval()
    dummy_input = torch.tensor(X_scaled[:1], dtype=torch.float32)  # shape [1,5]
    torch.onnx.export(
        dl2_model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        export_params=True,
        external_data=False
    )
    print(f"Model exported to ONNX → {onnx_path}")


    # Lipschitz
    mean_L, max_L = empirical_lipschitz(dl2_model, X_scaled, eps_scaled)
    print(f"Empirical Lipschitz → mean {mean_L:.2f}, max {max_L:.2f}")
    feature_L = per_feature_lipschitz(dl2_model, X_scaled, eps_scaled)
    plt.figure(figsize=(6,4))
    plt.bar([str(i) for i in feature_L], feature_L.values())
    plt.xlabel("Feature index")
    plt.ylabel("Mean Lipschitz contribution")
    plt.title("Per-feature sensitivity (Convex DL2)")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "per_feature_lipschitz_convex_dl2.png"))

    # Evaluation + violation metrics
    dl2_model.eval()
    with torch.no_grad():
        preds = dl2_model(X_t).numpy()
        y_true = y_t.numpy()

    # True vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, preds, alpha=0.4)
    plt.plot([0, y_true.max()], [0, y_true.max()], "r--")
    plt.xlabel("True dose")
    plt.ylabel("Predicted dose")
    plt.title("True vs Predicted (Convex DL2)")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "true_vs_pred_convex_dl2.png"))
    plt.show()

    # Property-based violations
    SAFE_MAX = 2.0
    UNHEALTHY_MIN = 6.0
    HEALTHY_CENTER = 0.5 * (SAFE_MAX + UNHEALTHY_MIN)

    safe_mask      = preds <= SAFE_MAX
    healthy_mask   = (preds > SAFE_MAX) & (preds < UNHEALTHY_MIN)
    unhealthy_mask = preds >= UNHEALTHY_MIN

    safe_violation      = np.maximum(0.0, preds[safe_mask] - SAFE_MAX)
    healthy_violation   = np.abs(preds[healthy_mask] - HEALTHY_CENTER)
    unhealthy_violation = np.maximum(0.0, UNHEALTHY_MIN - preds[unhealthy_mask])

    safe_count      = len(safe_violation)
    healthy_count   = len(healthy_violation)
    unhealthy_count = len(unhealthy_violation)
    total_count     = safe_count + healthy_count + unhealthy_count

    print("\n--- Property-based DL2 Violations (Convex) ---")
    print(f"Safe region violations:      {safe_count}/{len(preds)}")
    print(f"Healthy region deviations:   {healthy_count}/{len(preds)}")
    print(f"Unhealthy region violations: {unhealthy_count}/{len(preds)}")
    print(f"Total considered samples:    {total_count}/{len(preds)}")
    print(f"Mean safe violation:      {safe_violation.mean() if safe_violation.size>0 else 0:.4f}")
    print(f"Mean healthy deviation:   {healthy_violation.mean() if healthy_violation.size>0 else 0:.4f}")
    print(f"Mean unhealthy violation: {unhealthy_violation.mean() if unhealthy_violation.size>0 else 0:.4f}")
    print(f"Max violation magnitude: {max(safe_violation.max() if safe_violation.size>0 else 0, healthy_violation.max() if healthy_violation.size>0 else 0, unhealthy_violation.max() if unhealthy_violation.size>0 else 0):.4f}")


    # Histogram
    plt.figure(figsize=(6,4))
    if safe_violation.size>0:
        plt.hist(safe_violation, bins=30, alpha=0.5, label="Safe violation")
    if healthy_violation.size>0:
        plt.hist(healthy_violation, bins=30, alpha=0.5, label="Healthy deviation")
    if unhealthy_violation.size>0:
        plt.hist(unhealthy_violation, bins=30, alpha=0.5, label="Unhealthy violation")
    plt.xlabel("Violation / deviation magnitude")
    plt.ylabel("Count")
    plt.title("Property-based DL2 Violations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "violation_hist_convex_dl2.png"))
    plt.show()


if __name__ == "__main__":
    main()

