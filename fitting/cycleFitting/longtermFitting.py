import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import least_squares

# ===================== Paths (modify as needed) =====================
breakin_out_dir  = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_breakin"
longterm_out_dir = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_longterm"
breakin_param_csv = os.path.join(breakin_out_dir, "breakin_fit_params.csv")

# ===================== Column Name Conventions =====================
COL_E = "X"                  # EFC
COL_Y = "y_pure_cycle"       # Note: This script will convert to y_loss = 1 - y_pure_cycle
COL_L = "LABLE"

# ===================== Utility Functions =====================
def metrics(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2) + eps
    r2  = 1 - ss_res/ss_tot
    rmse= np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    return r2, rmse, mae

def breakin_model(E, q4, q5, q6, E0=None):
    """
    Eq.(11) —— Cumulative break-in loss: Saturation drop from 0 to q4
      q_loss_breakin(E) = 2*q4 * [ 1/2 - 1/(1 + exp((q5*(E/E0))**q6)) ]
    If E0=None, no normalization is applied (EFC is used directly).
    """
    E = np.asarray(E, float)
    En = E/float(E0) if E0 else E
    s  = np.power(np.maximum(q5,1e-12)*np.maximum(En,0.0), np.maximum(q6,1e-12))
    s  = np.clip(s, 0.0, 50.0)                      # Numerical stability
    inv_logistic = np.exp(-s)/(1.0 + np.exp(-s))    # = 1/(1+exp(s))
    return 2.0 * q4 * (0.5 - inv_logistic)

def long_model(E, a, b, E0=None):
    """Long-term cyclic component (power law): a * (E/E0)^b"""
    E = np.asarray(E, float)
    En = E/float(E0) if E0 else E
    En = np.maximum(En, 1e-12)
    return a * np.power(En, b)

def read_groups(dir_path):
    """Read long-term subsets in the directory, return list[{'label','file','E','y_loss'}]"""
    groups = []
    for fn in os.listdir(dir_path):
        if not fn.lower().endswith(".csv"): 
            continue
        df = pd.read_csv(os.path.join(dir_path, fn))
        # Required columns
        if not {COL_E, COL_Y}.issubset(df.columns):
            print(f"[WARN] {fn} is missing columns {COL_E}/{COL_Y}, skipping")
            continue
        # Label
        if COL_L in df.columns and pd.notna(df[COL_L].iloc[0]):
            label = int(df[COL_L].iloc[0])
        else:
            m = re.match(r"(\d+)", os.path.splitext(fn)[0])
            label = int(m.group(1)) if m else None
        # Extract & Sort
        sub = df[[COL_E, COL_Y]].dropna().copy().sort_values(COL_E).reset_index(drop=True)
        E = sub[COL_E].to_numpy(float)
        # Convert to "loss quantity"
        y_loss = 1.0 - sub[COL_Y].to_numpy(float)
        groups.append({"label":label, "file":fn, "E":E, "y":y_loss})
    groups = sorted(groups, key=lambda g: g["label"] if g["label"] is not None else 10**9)
    return groups

# ===================== Read Break-in Global Parameters =====================
assert os.path.isfile(breakin_param_csv), f"File not found: {breakin_param_csv}"
bp = pd.read_csv(breakin_param_csv)

# label -> q4 (per group)
q4_map = {int(r["label"]): float(r["q4"]) for _, r in bp.iterrows()}
q5_global = float(bp["q5_global"].iloc[0])
q6_global = float(bp["q6_global"].iloc[0])

# ===================== Read Long-term Subsets (and convert y=1-y) =====================
lt_groups = read_groups(longterm_out_dir)
assert len(lt_groups)>0, f"No usable CSV files in {longterm_out_dir}"

# ===================== Long-term Joint Fitting =====================
E0 = 4000.0  # Common scale normalization for EFC, for stability

G = len(lt_groups)
# Initial guess: global b, group-specific a_i
b0 = 0.9
a0_list = []
for g in lt_groups:
    q4g = q4_map.get(int(g["label"]), 0.0)
    y_break = breakin_model(g["E"], q4g, q5_global, q6_global, E0=E0)
    y_target = np.clip(g["y"] - y_break, 0.0, None)
    a0 = np.nanpercentile(y_target, 90) if np.any(y_target>0) else 1e-4
    a0_list.append(max(a0, 1e-6))

def pack_theta(b, a_list): return np.r_[b, np.array(a_list, float)]
def unpack_theta(theta):   return float(theta[0]), theta[1:]

theta0 = pack_theta(b0, a0_list)
lb = np.r_[0.1, np.zeros(G)]          # b ∈ [0.1, 2], a_i ≥ 0
ub = np.r_[2.0,  np.ones(G)*1.0]

def residuals_long(theta, groups):
    b, a_list = unpack_theta(theta)
    res_all = []
    for i, g in enumerate(groups):
        E, y = g["E"], g["y"]             # Already in "loss quantity"
        q4g = q4_map.get(int(g["label"]), 0.0)
        y_break = breakin_model(E, q4g, q5_global, q6_global, E0=E0)
        y_target = np.clip(y - y_break, 0.0, None)
        yhat = long_model(E, a_list[i], b, E0=E0)
        res_all.append(y_target - yhat)
    return np.concatenate(res_all) if res_all else np.array([0.0])

fit = least_squares(residuals_long, theta0, bounds=(lb, ub), args=(lt_groups,), verbose=2)
b_opt, a_opts = unpack_theta(fit.x)

print("\n=== Long-term fitted globals ===")
print(f"b_global = {b_opt:.6g}")
for i,g in enumerate(lt_groups):
    print(f"Group {g['label']}: a = {a_opts[i]:.6g} | file={g['file']}")

# Save long-term parameter table
lt_param_df = pd.DataFrame([{
    "label": lt_groups[i]["label"], "file": lt_groups[i]["file"],
    "a_long": a_opts[i], "b_global": b_opt,
    "q4_breakin": q4_map.get(int(lt_groups[i]["label"]), 0.0),
    "q5_global": q5_global, "q6_global": q6_global
} for i in range(G)])
lt_param_csv = os.path.join(longterm_out_dir, "longterm_fit_params.csv")
lt_param_df.to_csv(lt_param_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved long-term params to: {lt_param_csv}")

# ===================== Diagnostics: How well is the long-term target fitted? =====================
plot_dir1 = os.path.join(longterm_out_dir, "fig_long_fit"); os.makedirs(plot_dir1, exist_ok=True)
print("\n=== Per-group metrics on long-term target (after removing break-in) ===")
for i,g in enumerate(lt_groups):
    E, y = g["E"], g["y"]
    q4g = q4_map.get(int(g["label"]), 0.0)
    y_break = breakin_model(E, q4g, q5_global, q6_global, E0=E0)
    y_target = np.clip(y - y_break, 0.0, None)
    yhat_long = long_model(E, a_opts[i], b_opt, E0=E0)
    r2, rmse, mae = metrics(y_target, yhat_long)
    print(f"Group {g['label']}: R²={r2:.3f}, RMSE={rmse:.4g}, MAE={mae:.4g}")

    # Plot
    plt.figure(figsize=(7,4))
    plt.plot(E, y, 'o', ms=3, label='cycle loss (y=1-y_pure_cycle)')
    plt.plot(E, y_break, '-', lw=2, label='break-in (fixed)')
    plt.plot(E, y_target, 'o', ms=3, label='target = cycle - break-in')
    plt.plot(E, yhat_long, '-', lw=2, label='long-term fit')
    plt.xlabel("EFC"); plt.ylabel("Loss")
    plt.title(f"Group {g['label']} | a={a_opts[i]:.3g}, b={b_opt:.3g}")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir1, f"group_{g['label']}_long_fit.png"), dpi=160)
    plt.close()

# ===================== Reconstruction: Can break-in + long-term restore the subset curve? =====================
plot_dir2 = os.path.join(longterm_out_dir, "fig_reconstruct"); os.makedirs(plot_dir2, exist_ok=True)
print("\n=== Reconstruction check on long-term subset (break-in + long-term) ===")
for i,g in enumerate(lt_groups):
    E, y = g["E"], g["y"]
    q4g = q4_map.get(int(g["label"]), 0.0)
    y_break = breakin_model(E, q4g, q5_global, q6_global, E0=E0)
    y_long  = long_model(E, a_opts[i], b_opt, E0=E0)
    y_hat   = y_break + y_long
    r2, rmse, mae = metrics(y, y_hat)
    print(f"Group {g['label']}: Recon R²={r2:.3f}, RMSE={rmse:.4g}, MAE={mae:.4g}")

    plt.figure(figsize=(7,4))
    plt.plot(E, y, 'o', ms=3, label='cycle loss (y=1-y_pure_cycle)')
    plt.plot(E, y_break, '-', lw=2, label='break-in')
    plt.plot(E, y_long,  '-', lw=2, label='long-term')
    plt.plot(E, y_hat,   '-', lw=2, label='sum (model)')
    plt.xlabel("EFC"); plt.ylabel("Loss")
    plt.title(f"Group {g['label']} | reconstruction")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir2, f"group_{g['label']}_recon.png"), dpi=160)
    plt.close()

# ===================== Diagnostics: Overall Error and Fitting Quality =====================

res_vec = residuals_long(fit.x, lt_groups)
rmse = np.sqrt(np.mean(res_vec**2))
mae  = np.mean(np.abs(res_vec))
print(f"\nOverall RMSE={rmse:.4g}, MAE={mae:.4g}, cost={fit.cost:.4g}, nfev={fit.nfev}")
print(f"Fitting success: {fit.success}, message: {fit.message}")
# ===================== Overall R² =====================
y_all = np.concatenate([g["y"] for g in lt_groups if len(g["y"]) > 0])
yhat_all = np.concatenate([
    breakin_model(g["E"], q4_map.get(int(g["label"]), 0.0), q5_global, q6_global, E0=E0) +
    long_model(g["E"], a_opts[i], b_opt, E0=E0)
    for i, g in enumerate(lt_groups) if len(g["E"]) > 0
])
R2_all = 1 - np.sum((y_all - yhat_all)**2) / np.sum((y_all - np.mean(y_all))**2)
print(f"\nOverall RMSE={rmse:.4g}, MAE={mae:.4g}, R²={R2_all:.4f}, cost={fit.cost:.4g}, nfev={fit.nfev}")
