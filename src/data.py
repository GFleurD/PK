#imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Folder where this file lives
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the data folder 
DATA_DIR = os.path.join(THIS_DIR, "../data/csv")
VIS_DIR = os.path.join(THIS_DIR, "../data/visualisations")

# Random seed for reproducibility
np.random.seed(42)

# Parameters
timesteps = 24.0*8      # total hours (8 days)
dt = 1.0                # hours per timestep for concentration
dt_small = 0.05          # small dt for Euler updates
d_dt = 12.0             # hours between doses
num_patients = 200
V = 30                  # baseline volume of distribution

# Pharmacokinetics
t_half = 8.0 #hours
k_dose = np.log(2)/t_half

def next_conc(C, dt, V, D):
    return (C+(D/V))*(np.exp(-k_dose*dt))

# Pharmacodynamics (Temperature)
k_t_inf_max = 0.02
r_t = 3.0
T_norm = 37.0
k_t_hom = 0.04
k_t_c = 0.005

def k_t_inf_sigmoid(T):
    return k_t_inf_max * (1/(1 + np.exp(-r_t*(T-T_norm))))

def next_temp(T, C, dt_small):
    k_tinf = k_t_inf_sigmoid(T)
    return T + dt_small * (k_tinf - k_t_c*C - k_t_hom*(T-T_norm))

# Pharmacodynamics (WBC)
k_wbc_inf_max = 0.03
r_wbc = 0.7
WBC_norm = 8.0
k_wbc_hom = 0.12
k_wbc_c = 0.06

def k_wbc_inf_sigmoid(WBC):
    return k_wbc_inf_max * (1/(1 + np.exp(-r_wbc*(WBC-WBC_norm))))

def next_WBC(WBC, C, dt_small):
    k_wbc_inf = k_wbc_inf_sigmoid(WBC)
    return WBC + dt_small * (k_wbc_inf - k_wbc_c*C - k_wbc_hom*(WBC-WBC_norm))

#  Dose bounds
D_min, D_max = 0, 750

def compute_dose(C, T, WBC, D_max=750):
    """
    Compute dose based on temperature and WBC thresholds.
    Only dose if either value exceeds healthy range.
    Dose is proportional to how far values are above thresholds.
    """
    # Thresholds
    T_thresh = 37.5
    WBC_thresh = 12
    T_max = 40.0       # Max clinically relevant temp
    WBC_max = 20.0     # Max clinically relevant WBC

    # Only consider values above thresholds
    temp_factor = max(0.0, (T - T_thresh) / (T_max - T_thresh))
    wbc_factor = max(0.0, (WBC - WBC_thresh) / (WBC_max - WBC_thresh))

    # Combine factors (simple sum or weighted sum)
    dose = D_max * (temp_factor + wbc_factor)
    dose = np.clip(dose, 0.0, D_max)
    return dose

# Example: smooth sigmoid scaling
def compute_dose_smooth(T, WBC, D_max=1500, T_thresh=37.5, WBC_thresh=12, alpha=1.0):
    temp_factor = 1 / (1 + np.exp(-alpha * (T - T_thresh)))
    wbc_factor = 1 / (1 + np.exp(-alpha * (WBC - WBC_thresh)))
    dose = D_max * (temp_factor + wbc_factor) / 2  # normalize sum to stay <= D_max
    return dose


# begin storage
all_X = []
all_y = []

for p in range(num_patients):
    # Patient covariates
    age = np.random.randint(18, 90)
    weight = np.random.uniform(50, 100)
    sex = np.random.choice([0,1])  # 0=male, 1=female
    Vd_eff = V * (weight / 70)  # scale with body weight

    # Initial sick state
    C = 0.0
    T = np.random.uniform(38.5, 40.0)
    WBC = np.random.uniform(12, 20)
    D = 0.0  # initial dose
    t_current = 0.0
    next_dose_time = 0.0

    while t_current < timesteps:
        # Compute dose only at scheduled times
        if t_current >= next_dose_time:
            D = compute_dose(C, T, WBC, D_max)
            all_X.append([C, T, WBC, age, weight, sex])
            all_y.append(D)
            next_dose_time += d_dt
        else:
            D = 0.0

        # --- Store current state ---
        

        # --- Update states at every small dt ---
        C = next_conc(C, dt_small, Vd_eff, D)  # use dt_small now
        T = next_temp(T, C, dt_small)
        WBC = next_WBC(WBC, C, dt_small)

        t_current += dt_small

# Convert to arrays
X = np.array(all_X, dtype=np.float32)
y = np.array(all_y, dtype=np.float32)

# --- Save CSVs ---
df_X = pd.DataFrame(X, columns=['C','T','WBC','Age','Weight','Sex'])
df_y = pd.DataFrame(y, columns=['Dose'])

file_path_states= os.path.join(DATA_DIR, "patient_states.csv")
file_path_doses = os.path.join(DATA_DIR, "dose_targets.csv")
df_X.to_csv(file_path_states, index=False)
df_y.to_csv(file_path_doses, index=False)

# After generating X and y
rows_per_patient = len(X) // num_patients
X_reshaped = X.reshape(num_patients, rows_per_patient, -1)
y_reshaped = y.reshape(num_patients, rows_per_patient)


# Plot

n_queries = 5
random_indices = np.random.choice(len(X_reshaped), size=n_queries, replace=False)
for p in random_indices:
    plt.figure(figsize=(16,4))
    
    plt.subplot(1,4,1)
    plt.plot(X_reshaped[p,:,1], label='Temp')
    plt.axhline(T_norm, color='g', linestyle='--')
    plt.title(f'Patient {p+1} Temperature')
    plt.legend()
    
    plt.subplot(1,4,2)
    plt.plot(X_reshaped[p,:,2], label='WBC')
    plt.axhline(WBC_norm, color='g', linestyle='--')
    plt.title(f'Patient {p+1} WBC')
    plt.legend()
    
    plt.subplot(1,4,3)
    plt.plot(X_reshaped[p,:,0], label='Drug Conc (C)')
    plt.title(f'Patient {p+1} Drug Concentration')
    plt.legend()
    
    plt.subplot(1,4,4)
    plt.plot(y_reshaped[p,:], label='Dose')
    plt.title(f'Patient {p+1} Dose')
    plt.legend()
    
    plt.tight_layout()
    title = f'Patient {p+1}'
    file_path_toydata = os.path.join(VIS_DIR, f"{title}.png")
    plt.savefig(file_path_toydata, dpi=300, bbox_inches='tight')
    plt.show()

