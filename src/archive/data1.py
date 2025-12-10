import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 


# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

# Preprocessing and train/test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Optional: evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score

import idx2numpy
import tf2onnx

# Random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Simulation parameters ---
timesteps = 48
dt = 0.5  # hours per timestep
num_patients = 50

# --- PK/PD parameters ---
ke = 0.1       # baseline elimination rate per hour
Vd = 30        # baseline volume of distribution
b_temp = 0.005  # drug effect on temp
c_wbc = 0.02    # drug effect on WBC
a_temp_inf = 0.05  # infection effect on temp if no drug
a_wbc_inf = 0.1    # infection effect on WBC if no drug
k_inf_T = 0.08 
k_home_T = 0.12 
k_inf_W = 0.12 
k_home_W = 0.04


# --- Normal ranges / targets ---
T_norm = (36.0, 38.0)
WBC_norm = (4.0, 12.0)
C_max = 30  # drug toxic max

# --- Dose bounds ---
D_min, D_max = 0, 1500  # mg

# --- Initialize storage ---
all_X = []
all_y = []

for p in range(num_patients):
    # Patient covariates
    age = np.random.randint(18, 90)
    weight = np.random.uniform(50, 100)
    sex = np.random.choice([0,1])  # 0=male, 1=female
    
    # Adjust PK/PD parameters based on covariates
    ke_eff = ke * (1 - 0.004 * (age - 50))          # slower clearance with age
    Vd_eff = Vd * (weight / 70)                     # scale with body weight
    a_temp_inf_eff = a_temp_inf * (1.1 if sex == 1 else 1.0)
    a_wbc_inf_eff = a_wbc_inf * (1.1 if sex == 1 else 1.0)
    # Patient-specific homeostasis setpoints
    T_set = np.random.normal(37.0, 0.2)   # e.g. mean=37, SD=0.2 → 36.8 to 37.2
    WBC_set = np.random.normal(8.0, 1.0)  # mean=8, SD=1 → 7–9 typically

        # PK/PD variability multipliers
    age_factor   = 1 - 0.003 * (age - 50)            # older → slightly weaker
    weight_factor = 70 / weight                      # heavier → slower response
    sex_factor    = 1.05 if sex == 1 else 1.0        # slight effect, optional

    # Patient-specific parameters
    k_inf_T_eff  = k_inf_T  * np.random.uniform(0.9, 1.1) * weight_factor
    k_home_T_eff = k_home_T * np.random.uniform(0.9, 1.1) * age_factor

    k_inf_W_eff  = k_inf_W  * np.random.uniform(0.9, 1.1) * weight_factor
    k_home_W_eff = k_home_W * np.random.uniform(0.9, 1.1) * age_factor

    # Initial sick state
    C = 0.0
    T = np.random.uniform(38.5, 40.0)
    WBC = np.random.uniform(12, 20)
    D_prev = 500  # initial dose

    for t in range(timesteps-1):
        # --- Dose calculation ---
        dose_temp = 50 * max(0, T - T_norm[1])
        dose_wbc  = 50 * max(0, WBC - WBC_norm[1])
        dose_from_vitals = dose_temp + dose_wbc
        
        # Reduce dose if concentration near max
        safety_factor = max(0, 1 - C / C_max)
        D_t_raw = dose_from_vitals * safety_factor

        # Smooth relative to previous dose
        D_t = 0.7 * D_prev + 0.3 * D_t_raw
        D_t = np.clip(D_t, D_min, D_max)
        D_prev = D_t

        # --- Store current state ---
        X_t = [C, T, WBC, age, weight, sex]
        all_X.append(X_t)
        all_y.append(D_t)

        # --- Update PK/PD ---
        C_next = C + dt * (-ke_eff * C + D_t / Vd_eff)
        C_next = max(0, C_next)

        T_next = T + dt * (
            k_inf_T_eff                      # infection raises temp
            - b_temp * C                     # drug reduces temp
            - k_home_T_eff * (T - T_set)     # homeostasis to patient-specific setpoint
        )
        T_next = np.clip(T_next, 35, 42)

        WBC_next = WBC + dt * (
            k_inf_W_eff                      # infection raises WBC
            - c_wbc * C                      # drug reduces WBC
            - k_home_W_eff * (WBC - WBC_set) # homeostasis to patient-specific setpoint
        )
        WBC_next = np.clip(WBC_next, 1, 30)



        C, T, WBC = C_next, T_next, WBC_next

# --- Convert to arrays ---
X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)

df_X = pd.DataFrame(X, columns=['C','T','WBC','Age','Weight','Sex'])
df_y = pd.DataFrame(y, columns=['Dose'])
df_X.to_csv("patient_states.csv", index=False)
df_y.to_csv("dose_targets.csv", index=False)

# --- Plot a few patients ---
num_plot = 5
samples_per_patient = timesteps-1
X_reshaped = X.reshape(num_patients, samples_per_patient, -1)
y_reshaped = y.reshape(num_patients, samples_per_patient)

for p in range(num_plot):
    plt.figure(figsize=(16,4))
    
    plt.subplot(1,4,1)
    plt.plot(X_reshaped[p,:,1], label='Temp')
    plt.axhline(T_norm[0], color='g', linestyle='--')
    plt.axhline(T_norm[1], color='r', linestyle='--')
    plt.title(f'Patient {p+1} Temperature')
    plt.legend()
    
    plt.subplot(1,4,2)
    plt.plot(X_reshaped[p,:,2], label='WBC')
    plt.axhline(WBC_norm[0], color='g', linestyle='--')
    plt.axhline(WBC_norm[1], color='r', linestyle='--')
    plt.title(f'Patient {p+1} WBC')
    plt.legend()
    
    plt.subplot(1,4,3)
    plt.plot(X_reshaped[p,:,0], label='Drug Conc (C)')
    plt.axhline(C_max, color='r', linestyle='--', label='C_max')
    plt.title(f'Patient {p+1} Drug Concentration')
    plt.legend()
    
    plt.subplot(1,4,4)
    plt.plot(y_reshaped[p,:], label='Dose')
    plt.title(f'Patient {p+1} Dose')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- Neural network training ---
X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.float32).reshape(-1,1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mean = scaler.mean_
print(mean)
std = scaler.scale_
print(std)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

input_size = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_size,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32
)

X_test_vcl_raw = np.array([ 28.99999886889077, 37.999999826576, 13.99999890156776, 12.95275488369321, 52.9822033387511, 0.98053950031535 ])
raw_sample = X_test_vcl_raw.reshape(1, -1)
sample_scaled = scaler.transform(raw_sample) 

prediction = model.predict(sample_scaled)
print("predicted dosage:", prediction)


# # --- Evaluate & Save Plot ---
# y_pred = model.predict(X_test)

# plt.figure(figsize=(6,6))
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel("True Dose")
# plt.ylabel("Predicted Dose")
# plt.title("NN Controller: True vs Predicted Dose")
# plt.plot([0, D_max], [0, D_max], 'r--', label='Ideal x=y')
# plt.legend()
# plt.tight_layout()

# os.makedirs("nn_plots", exist_ok=True)
# plt.savefig("nn_plots/true_vs_predicted_dose.png")
# plt.show()



# # Use the raw X and y (problem space)
# X_problem = X.astype(np.float32)  # keep raw units
# y_problem = y.astype(np.float32)  # only needed for plotting/evaluation

# # Save
# idx2numpy.convert_to_file("X_vehicle.idx", X_problem)
# idx2numpy.convert_to_file("y_vehicle.idx", y_problem)


# def save_model_in_onnx(model):
#     # Convert and save
#     out_path = "pk.onnx"
#     onnx_model, _ = tf2onnx.convert.from_keras(model, output_path=out_path)
#     print(f"Saved ONNX model to: {out_path}")


# save_model_in_onnx(model) 


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