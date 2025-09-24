import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ===================== Configuration =====================
breakin_out_dir = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_breakin"
# Column name conventions: X = EFC; y_pure_cycle = Capacity loss due to cycling (calendar effect removed); optional LABLE
# If your column names are different, change them here:
COL_EFC = "X"
COL_Y   = "y_pure_cycle"
COL_LAB = "LABLE"

# Global parameter initial values (from literature)
INIT_q5 = 3.04e-3
INIT_q6 = 1.43

# Parameter bounds
BOUNDS = {
    "q5": (1e-8, 1e-1),
    "q6": (0.01, 10.0),
    "q4": (0.0, 1.0),
}

# ===================== Read break-in subsets =====================
def read_breakin_groups(breakin_dir):
    groups = []  # list of dict: {"label": int, "file": str, "E": np.array, "y": np.array}
    for fn in os.listdir(breakin_dir):
        if not fn.lower().endswith(".csv"):
            continue
        fpath = os.path.join(breakin_dir, fn)
        df = pd.read_csv(fpath)
        if COL_EFC not in df.columns or COL_Y not in df.columns:
            print(f"[WARN] {fn} is missing columns {COL_EFC}/{COL_Y}, skipping")
            continue
        # Group number is taken from LABLE first, otherwise extracted from the filename
        if COL_LAB in df.columns and pd.notna(df[COL_LAB].iloc[0]):
            label = int(df[COL_LAB].iloc[0])
        else:
            m = re.match(r"(\d+)", os.path.splitext(fn)[0])
            label = int(m.group(1)) if m else None

        # Basic cleaning and sorting
        sub = df[[COL_EFC, COL_Y]].dropna().copy()
        sub = sub.sort_values(COL_EFC, kind="mergesort").reset_index(drop=True)
        E = sub[COL_EFC].to_numpy(dtype=float)
        y_loss = 1 - sub[COL_Y].to_numpy(dtype=float)

        if len(E) < 3:
            print(f"[WARN] {fn} has too few points ({len(E)}), fitting may be unreliable")
        groups.append({"label": label, "file": fn, "E": E, "y_loss": y_loss})
    # Sort by label for easier reading
    groups = sorted(groups, key=lambda g: (g["label"] if g["label"] is not None else 1e9))
    return groups

groups = read_breakin_groups(breakin_out_dir)
assert len(groups) > 0, "No valid CSV files found in the break-in directory"

# ===================== Model Definition =====================
import numpy as np

# def breakin_model(E, q4, q5, q6):
#     """
#     Eq.(11):
#       q_loss_breakin(E) = 2*q4 * [ 1/2 - 1/(1 + exp((q5*E)**q6)) ]
#     Returns the saturation drop from 0 to q4.
#     """
#     E   = np.asarray(E, dtype=float)
#     q5  = float(q5); q6 = float(q6); q4 = float(q4)

#     # s = (q5*E)**q6  (ensure non-negativity, avoid singularities)
#     s = np.power(np.maximum(q5, 1e-12) * np.maximum(E, 0.0), np.maximum(q6, 1e-12))

#     # Numerically stable computation of 1/(1+exp(s)): exp(-s)/(1+exp(-s))
#     # clip(s, 0, 50) to prevent overflow; for s>~50, this term is approximately 0
#     s_clip = np.clip(s, 0.0, 50.0)
#     exp_neg_s = np.exp(-s_clip)
#     inv_logistic = exp_neg_s / (1.0 + exp_neg_s)  # == 1/(1+exp(s))

#     return 2.0 * q4 * (0.5 - inv_logistic)

import numpy as np

def breakin_model(E, q4, q5, q6):
    """
    Original form of Eq.(11):
      q_loss_breakin(E) = 2*q4 * [ 0.5 - 1/(1 + exp((q5*E)**q6)) ]
    No clipping or numerical stabilization applied.
    """
    E   = np.asarray(E, dtype=float)
    return 2.0 * q4 * (0.5 - 1.0 / (1.0 + np.exp((q5 * E) ** q6)))


# ===================== Pack and Unpack Parameter Vector =====================
# Vector structure: theta = [q5, q6, q4_1, q4_2, ..., q4_G]
G = len(groups)

def pack_theta(q5, q6, q4_list):
    return np.r_[q5, q6, np.array(q4_list, dtype=float)]

def unpack_theta(theta):
    q5 = theta[0]; q6 = theta[1]
    q4_list = theta[2:]
    return q5, q6, q4_list

# Initial values: q4_i is set to the maximum or percentile value of y for the group
q4_init = []
for g in groups:
    if len(g["y_loss"]) == 0:
        q4_init.append(0.02)
    else:
        q4_init.append(float(np.clip(np.nanpercentile(g["y_loss"], 90), 1e-6, 0.5)))
theta0 = pack_theta(INIT_q5, INIT_q6, q4_init)

# Bounds
lb = [BOUNDS["q5"][0], BOUNDS["q6"][0]] + [BOUNDS["q4"][0]]*G
ub = [BOUNDS["q5"][1], BOUNDS["q6"][1]] + [BOUNDS["q4"][1]]*G
lb = np.array(lb, float); ub = np.array(ub, float)

# ===================== Residuals Function (Combined) =====================
def residuals(theta, groups):
    q5, q6, q4s = unpack_theta(theta)
    res_all = []
    for i, g in enumerate(groups):
        E = g["E"]; y_loss = g["y_loss"]
        if len(E) == 0:
            continue
        y_loss_hat = breakin_model(E, q4s[i], q5, q6)
        res = y_loss - y_loss_hat
        # Weights can be added here (e.g., by E range or number of points), currently unweighted
        res_all.append(res)
    if len(res_all) == 0:
        return np.array([0.0])
    return np.concatenate(res_all)

# ===================== Fitting =====================
out = least_squares(residuals, theta0, bounds=(lb, ub), args=(groups,), verbose=2)
q5_opt, q6_opt, q4s_opt = unpack_theta(out.x)

print("\n=== Break-in fitted globals ===")
print(f"q5 = {q5_opt:.6g}")
print(f"q6 = {q6_opt:.6g}")
print("=== Per-group q4 (amplitude) ===")
for i, g in enumerate(groups):
    print(f"Group {g['label']}: q4 = {q4s_opt[i]:.6g} | file={g['file']}")

# Save parameters to CSV
param_rows = [{"label": g["label"], "file": g["file"], "q4": q4s_opt[i],
               "q5_global": q5_opt, "q6_global": q6_opt} for i, g in enumerate(groups)]
param_df = pd.DataFrame(param_rows)
param_csv = os.path.join(breakin_out_dir, "breakin_fit_params.csv")
param_df.to_csv(param_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved break-in params to: {param_csv}")

# ===================== Visualization (one plot per group) =====================
plot_dir = os.path.join(breakin_out_dir, "fig_breakin_fit")
os.makedirs(plot_dir, exist_ok=True)

for i, g in enumerate(groups):
    E = g["E"]; y_loss = g["y_loss"]
    if len(E) == 0:
        continue
    q4i = q4s_opt[i]
    y_loss_hat = breakin_model(E, q4i, q5_opt, q6_opt)

    y_hat = 1-y_loss_hat
    y = 1-y_loss

    plt.figure(figsize=(7,4))
    plt.plot(E, y, 'o', ms=4, label='data')
    plt.plot(E, y_hat, '-', lw=2, label='break-in fit')
    plt.xlabel("EFC")
    plt.ylabel("Break-in loss (subset y)")
    plt.title(f"Group {g['label']} | q4={q4i:.3g}, q5={q5_opt:.3g}, q6={q6_opt:.3g}")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    savepath = os.path.join(plot_dir, f"group_{g['label']}_breakin_fit.png")
    plt.savefig(savepath, dpi=160)
    plt.close()


print(f"Saved per-group fit figures to: {plot_dir}")

# ===================== Diagnostics: Overall Error and Fit Quality =====================
res_vec = residuals(out.x, groups)
rmse = np.sqrt(np.mean(res_vec**2))
mae  = np.mean(np.abs(res_vec))
print(f"\nOverall RMSE={rmse:.4g}, MAE={mae:.4g}, cost={out.cost:.4g}, nfev={out.nfev}")

# ===================== Overall R² =====================
y_all = np.concatenate([g["y_loss"] for g in groups if len(g["y_loss"]) > 0])
yhat_all = np.concatenate([
    breakin_model(g["E"], q4s_opt[i], q5_opt, q6_opt)
    for i, g in enumerate(groups) if len(g["E"]) > 0
])
R2_all = 1 - np.sum((y_all - yhat_all)**2) / np.sum((y_all - np.mean(y_all))**2)

print(f"\nOverall RMSE={rmse:.4g}, MAE={mae:.4g}, R²={R2_all:.4f}, cost={out.cost:.4g}, nfev={out.nfev}")

# ===================== Per-group R² =====================
print("\n=== Per-group R² ===")
for i, g in enumerate(groups):
    if len(g["E"]) == 0:
        continue
    yhat = breakin_model(g["E"], q4s_opt[i], q5_opt, q6_opt)
    R2 = 1 - np.sum((g["y_loss"] - yhat)**2) / np.sum((g["y_loss"] - np.mean(g["y_loss"]))**2)
    print(f"Group {g['label']}: R²={R2:.4f}")

