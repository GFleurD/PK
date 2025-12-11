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

    # --- Visualize ---
    plt.figure(figsize=(6,6))
    plt.scatter(preds_orig, preds_adv, alpha=0.4)
    plt.xlabel("Original Prediction")
    plt.ylabel("Adversarial Prediction")
    plt.title("PGD Shift in Predictions")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "pgd_shift.png"))
    print("\nSaved PGD scatter plot.\n")

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

    print(f"Adversarially trained model saved as:\n - {adv_model_path}\n - {onnx_path}")
    print("\nDone!\n")


if __name__ == "__main__":
    main()
