import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import idx2numpy
import matplotlib.pyplot as plt

# Folder where this file lives
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the data folder 
DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
file_path_states = os.path.join(DATA_DIR, "patient_states.csv")
file_path_doses  = os.path.join(DATA_DIR, "dose_targets.csv")

df_X = pd.read_csv(file_path_states)
df_y = pd.read_csv(file_path_doses)
print(df_X.shape)
print(df_y.shape)

# Convert to numpy arrays
X = df_X.values.astype(np.float32)
y = df_y.values.astype(np.float32) # already shape (N,1)

idx2numpy.convert_to_file("X_vehicle.idx", X)
idx2numpy.convert_to_file("y_vehicle.idx", y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mean = scaler.mean_
std = scaler.scale_
print("mean:", mean)
print("std:", std)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

# -----------------------------
# PyTorch model
# -----------------------------
input_size = X_train.shape[1]

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU()  # keep if dosage must be non-negative
        )

    def forward(self, x):
        return self.model(x)

model = Net(input_size)

# -----------------------------
# Optimizer & loss
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Training loop
# -----------------------------
epochs = 30
batch_size = 60

# Simple batching function
def batch_iter(X, y, batch_size):
    idx = torch.randperm(len(X))
    for i in range(0, len(X), batch_size):
        batch_idx = idx[i:i+batch_size]
        yield X[batch_idx], y[batch_idx]

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for xb, yb in batch_iter(X_train, y_train, batch_size):
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_test)
        val_loss = criterion(val_preds, y_test).item()

    print(f"Epoch {epoch+1:02d} | train loss: {epoch_loss:.4f} | val loss: {val_loss:.4f}")


# --- Evaluate & Save Plot ---
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
y_pred_np = y_pred.numpy()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_np, alpha=0.5)
plt.xlabel("True Dose")
plt.ylabel("Predicted Dose")
plt.title("NN Controller: True vs Predicted Dose")
plt.plot([0, 1500], [0, 1500], 'r--', label='Ideal x=y')
plt.legend()
plt.tight_layout()

os.makedirs("nn_plots", exist_ok=True)
plt.savefig("nn_plots/true_vs_predicted_dose.png")
plt.show()

# # Save onnx

# dummy_input = torch.randn(1, 6, dtype=torch.float32)

# torch.onnx.export(
#     model,
#     dummy_input,
#     "pk.onnx",
#     input_names=["input"],
#     output_names=["output"],
#     opset_version=18,       # safest with Vehicle/Marabou
#     do_constant_folding=True,
#     export_params=True,     # <-- ensures all weights are stored in the file
# )


# print("ONNX model exported successfully!")



# # wrap the model so that it can be used in CORA

# mu = scaler.mean_.astype(np.float32)
# sigma = scaler.scale_.astype(np.float32)

# inp = tf.keras.Input(shape=(6,), name="raw_input")

# # Normalization layer: (x - mu) / sigma
# norm = tf.keras.layers.Lambda(lambda x: (x - mu) / sigma, name="normalization")(inp)

# # Pass through trained model
# out = model(norm)

# wrapped = tf.keras.Model(inputs=inp, outputs=out, name="NN_with_normalization")
# wrapped.summary()

# save_model_in_onnx(wrapped) 

# # -----------------------------
# # Inference on a new sample
# # -----------------------------
# X_test_vcl_raw = np.array([
#     28.99999886889077, 
#     37.999999826576, 
#     13.99999890156776, 
#     12.95275488369321, 
#     52.9822033387511, 
#     0.98053950031535
# ], dtype=np.float32)

# raw_sample = X_test_vcl_raw.reshape(1, -1)
# sample_scaled = scaler.transform(raw_sample)

# sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

# model.eval()
# with torch.no_grad():
#     prediction = model(sample_tensor).item()

# print("predicted dosage:", prediction)
