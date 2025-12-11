# imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Folder where this file lives
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the data folder 
DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
VIS_DIR = os.path.join(THIS_DIR, "../data/visualisations")

# Random seed for reproducibility
np.random.seed(42)

# Parameters
timesteps = 24.0*4      # total hours (8 days)
dt_small = 0.05          # small dt for Euler updates
d_dt = 12.0             # hours between doses
num_patients = 200
V = 30                  # baseline volume of distribution

# Pharmacokinetics
t_half = 8.0 #hours
k_dose = np.log(2)/t_half

def next_conc(C, dt, V, D):
    return (C + (D / V)) * np.exp(-k_dose * dt)

# Pharmacodynamics: Scaled Sigmoids for Temperature and WBC
alpha_wbc = 1.5      # steepness for WBC
alpha_temp = 3.0     # steepness for Temp
k_t_inf_max = 0.02
k_wbc_inf_max = 0.03
r_max_wbc = k_wbc_inf_max
r_max_temp = k_t_inf_max
T_norm = 37.0
WBC_norm = 8.0
k_t_hom = 0.03 
k_t_c = 0.0025
k_wbc_hom = 0.03
k_wbc_c = 0.006

# Scaled sigmoid function
def scaled_sigmoid(x, x0, x1, r_max, alpha=1.0):
    x_center = (x0 + x1) / 2
    L = lambda x_: 1 / (1 + np.exp(-alpha * (x_ - x_center)))
    return r_max * (L(x) - L(x0)) / (L(x1) - L(x0))

# WBC and Temp scaling
def k_wbc_inf_scaled(WBC):
    return scaled_sigmoid(WBC, x0=12, x1=20, r_max=r_max_wbc, alpha=alpha_wbc)

def k_t_inf_scaled(T):
    return scaled_sigmoid(T, x0=37.5, x1=41, r_max=r_max_temp, alpha=alpha_temp)

# Update functions
def next_temp(T, C, dt_small):
    k_tinf = k_t_inf_scaled(T)
    return T + dt_small * (k_tinf - k_t_c*C - k_t_hom*(T-T_norm))

def next_WBC(WBC, C, dt_small):
    k_wbc_inf = k_wbc_inf_scaled(WBC)
    return WBC + dt_small * (k_wbc_inf - k_wbc_c*C - k_wbc_hom*(WBC-WBC_norm))

# Piecewise linear approximations
def sigmoid_inflection_bounds(x0, x1, alpha):
    x_center = (x0 + x1)/2
    dx = 2.0 / alpha
    return x_center - dx, x_center + dx

def k_wbc_inf_linear(WBC):
    x0, x1 = sigmoid_inflection_bounds(12, 20, alpha_wbc)
    if WBC <= x0:
        return 0.0
    elif WBC >= x1:
        return r_max_wbc
    else:
        return r_max_wbc / (x1 - x0) * (WBC - x0)

def k_t_inf_linear(T):
    x0, x1 = sigmoid_inflection_bounds(37.5, 41, alpha_temp)
    if T <= x0:
        return 0.0
    elif T >= x1:
        return r_max_temp
    else:
        return r_max_temp / (x1 - x0) * (T - x0)

# Dose function
D_min, D_max = 0, 750

def compute_dose_linear(T, WBC, D_max=750, T_norm=37.0, WBC_norm=8.0, T_max=41.0, WBC_max=20.0, w_T=0.5, w_WBC=0.5):
    # Only give dose if above normal
    if T <= T_norm or WBC <= WBC_norm:
        return 0.0
    temp_contrib = max(0.0, (T - T_norm)/(T_max - T_norm))
    wbc_contrib  = max(0.0, (WBC - WBC_norm)/(WBC_max - WBC_norm))
    
    # Weighted sum, capped at 1
    dose_fraction = min(1.0, w_T*temp_contrib + w_WBC*wbc_contrib)
    
    return D_max * dose_fraction


# --- Sigmoid verification plots (moved before simulation) ---
WBC_range = np.linspace(4, 22, 200)
T_range = np.linspace(36, 42, 200)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(WBC_range, [k_wbc_inf_scaled(w) for w in WBC_range], label='Scaled Sigmoid')
plt.plot(WBC_range, [k_wbc_inf_linear(w) for w in WBC_range], 'r--', label='Piecewise Linear')
plt.xlabel('WBC')
plt.ylabel('k_wbc_inf')
plt.title('WBC Scaling')
plt.legend()

plt.subplot(1,2,2)
plt.plot(T_range, [k_t_inf_scaled(t) for t in T_range], label='Scaled Sigmoid')
plt.plot(T_range, [k_t_inf_linear(t) for t in T_range], 'r--', label='Piecewise Linear')
plt.xlabel('Temperature (°C)')
plt.ylabel('k_t_inf')
plt.title('Temperature Scaling')
plt.legend()

plt.tight_layout()
plt.show()

# --- Simulation ---
all_X_csv = []       # for CSV (dose times only)
all_y_csv = []

all_X_full = []      # for plotting (every dt_small)
all_time_full = []

for p in range(num_patients):
    age = np.random.randint(18, 90)
    weight = np.random.uniform(50, 100)
    sex = np.random.choice([0,1])
    Vd_eff = V * (weight / 70)

    C = 0.0
    T = np.random.uniform(38.5, 40.0)
    WBC = np.random.uniform(12, 20)
    D = 0.0
    t_current = 0.0
    next_dose_time = 0.0

    iter_limit = int(timesteps / dt_small) + 1000
    iter_count = 0

    while t_current < timesteps and iter_count < iter_limit:
        iter_count += 1

        # Store full state for plotting
        all_X_full.append([C, T, WBC])
        all_time_full.append(t_current)

        # Check dose event
        if abs(t_current - next_dose_time) < dt_small/2:
            D = compute_dose_linear(T, WBC, D_max=750, T_norm=37.0, WBC_norm=8.0, T_max=41.0, WBC_max=20.0, w_T=0.5, w_WBC=0.5)
            all_X_csv.append([C, T, WBC, age, weight, sex])
            all_y_csv.append(D)
            next_dose_time += d_dt
        else:
            D = 0.0

        # Update dynamics
        C = next_conc(C, dt_small, Vd_eff, D)
        T = next_temp(T, C, dt_small)
        WBC = next_WBC(WBC, C, dt_small)

        t_current += dt_small

    if iter_count >= iter_limit:
        print(f"Warning: patient {p} loop reached iter limit.")

# --- Save CSV for training (dose times only) ---
X_csv = np.array(all_X_csv, dtype=np.float32)
y_csv = np.array(all_y_csv, dtype=np.float32)
df_X_csv = pd.DataFrame(X_csv, columns=['C','T','WBC','Age','Weight','Sex'])
df_y_csv = pd.DataFrame(y_csv, columns=['Dose'])
os.makedirs(DATA_DIR, exist_ok=True)
df_X_csv.to_csv(os.path.join(DATA_DIR, "patient_states.csv"), index=False)
df_y_csv.to_csv(os.path.join(DATA_DIR, "dose_targets.csv"), index=False)

# --- Plot examples using all dt_small points ---
all_X_full = np.array(all_X_full, dtype=np.float32)
all_time_full = np.array(all_time_full, dtype=np.float32)

n_queries = 5
patient_indices = np.random.choice(num_patients, n_queries, replace=False)
rows_per_patient = len(all_time_full) // num_patients

for p in patient_indices:
    start_idx = p * rows_per_patient
    end_idx = (p+1) * rows_per_patient
    t_vec = all_time_full[start_idx:end_idx]
    states = all_X_full[start_idx:end_idx]

    # Dose indices for scatter (still only every 12h)
    dose_indices = np.arange(0, len(t_vec), int(d_dt/dt_small))
    dose_times = t_vec[dose_indices]

    plt.figure(figsize=(16,4))

    # Temperature
    plt.subplot(1,4,1)
    plt.plot(t_vec, states[:,1], label='Temp')
    plt.axhline(T_norm, color='g', linestyle='--')
    plt.title(f'Patient {p+1} Temp')
    plt.xlabel('Time (hours)')
    plt.ylabel('°C')
    plt.legend()

    # WBC
    plt.subplot(1,4,2)
    plt.plot(t_vec, states[:,2], label='WBC')
    plt.axhline(WBC_norm, color='g', linestyle='--')
    plt.title(f'Patient {p+1} WBC')
    plt.xlabel('Time (hours)')
    plt.ylabel('10^9/L')
    plt.legend()

    # Drug concentration
    plt.subplot(1,4,3)
    plt.plot(t_vec, states[:,0], label='C', color='purple')
    plt.title(f'Patient {p+1} Drug Conc')
    plt.xlabel('Time (hours)')
    plt.ylabel('Conc')
    plt.legend()

    # Dose scatter at 12 h intervals
    plt.subplot(1,4,4)
    plt.scatter(dose_times, y_csv[p*dose_indices.size:(p+1)*dose_indices.size], color='r', label='Dose')
    plt.title(f'Patient {p+1} Dose')
    plt.xlabel('Time (hours)')
    plt.ylabel('mg')
    plt.legend()

    plt.tight_layout()
    plt.show()
