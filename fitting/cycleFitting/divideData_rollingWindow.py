# ==========================================
# Rolling window data division script
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= Parameter Area =================
# Paper rules (empirical thresholds by EFC dimension)
EFC_UP_BREAKIN    = 4000.0          # Upper limit of EFC for break-in
EFC_LOW_LONGTERM  = 1000.0          # Lower limit of EFC for long-term
CURV_TH_BREAKIN   = 3e-6            # Second derivative threshold for EFC (break-in)
CURV_TH_LONGTERM  = 2.5e-7          # Second derivative threshold for EFC (long-term)

# Rolling mean window (odd number is recommended); will downgrade automatically for very few points
ROLL_WIN = 7

# ================= Utility Functions =================
def second_derivative_wrt_x(x, y):
    """
    Compute the second derivative with respect to x for unequally spaced x: d2y/dx2 = d/dx ( dy/dx )
    At least 3 points are required, otherwise NaN is returned.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return np.full_like(x, np.nan)
    dy_dx   = np.gradient(y, x)
    d2y_dx2 = np.gradient(dy_dx, x)
    return d2y_dx2

def robust_roll_mean(y, win=7):
    """
    Centered rolling mean; automatically shrinks window if larger than sample size; min_periods=1 to prevent NaN.
    """
    n = len(y)
    w = min(win, n if n % 2 == 1 else n - 1)  # Window should be odd
    if w < 3:  # Very few samples, return original series directly
        return np.asarray(y, dtype=float)
    return pd.Series(y, dtype=float).rolling(window=w, center=True, min_periods=1).mean().to_numpy()

def estimate_efc_per_day(EFC, t_day):
    """
    Estimate the EFC/day for the group (more robust with median slope).
    """
    if len(EFC) < 2 or np.ptp(t_day) <= 0:
        return np.nan
    inst_rate = np.gradient(EFC, t_day)  # Instantaneous EFC/day at each point
    return float(np.nanmedian(inst_rate))

# ================= Your Data Reading Function (Given) =================
def read_all_csv_data_seperate_drop_nan(root_dir):
    data_dict = {}
    for filename in os.listdir(root_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(root_dir, filename)
            data = pd.read_csv(file_path)
            if 'y_cleaned' in data.columns:
                data = data.dropna(subset=['y_cleaned'])
            data_dict[filename] = data
    return data_dict

# ====== Read Data (According to Your Path) and Add LABEL ======
root_dir = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_0"
all_data_dict_cleaned = read_all_csv_data_seperate_drop_nan(root_dir)

for idx, (k, v) in enumerate(all_data_dict_cleaned.items()):
    v['LABLE'] = idx + 1

# ====== Output Containers ======
breakin_dict  = {}
longterm_dict = {}
third_dict    = {}
summary_rows  = []

# ====== Main Loop: Each File as a Group, Rolling Mean + Second Derivative + Mask ======
for fname, df in all_data_dict_cleaned.items():
    # Check for necessary columns
    required_cols = {'X', 't', 'y_pure_cycle'}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] {fname} is missing columns: {missing}, skipping this file")
        continue

    # Extract necessary fields for the group and sort by EFC (to avoid impact of unequally spaced/unsorted data on derivatives)
    label = int(df['LABLE'].iloc[0]) if 'LABLE' in df.columns else None
    gdf = df[['X', 't', 'y_pure_cycle']].dropna().copy()
    gdf = gdf.sort_values('X', kind='mergesort').reset_index(drop=True)

    EFC    = gdf['X'].to_numpy(dtype=float)
    t_h    = gdf['t'].to_numpy(dtype=float)
    t_day  = t_h / 24.0
    q_loss = gdf['y_pure_cycle'].to_numpy(dtype=float)

    # 1) Rolling mean smoothing
    q_smooth = robust_roll_mean(q_loss, win=ROLL_WIN)

    # 2) Second derivative with respect to time: d^2 q / dt^2
    d2q_dt2 = second_derivative_wrt_x(t_day, q_smooth)

    # 3) Convert the curvature thresholds in the EFC domain to the time domain: d2q/dt^2 ≈ (dEFC/dt)^2 * d2q/dEFC^2
    efc_per_day = estimate_efc_per_day(EFC, t_day)
    if np.isfinite(efc_per_day):
        curv_th_breakin_time  = CURV_TH_BREAKIN  * (efc_per_day ** 2)
        curv_th_longterm_time = CURV_TH_LONGTERM * (efc_per_day ** 2)
    else:
        # Degradation protection: if unable to calculate, use the original values (approximation)
        curv_th_breakin_time  = CURV_TH_BREAKIN
        curv_th_longterm_time = CURV_TH_LONGTERM

    # 4) Mask criteria (paper rules: EFC window + curvature thresholds)
    mask_breakin = (EFC > 0) & (EFC < EFC_UP_BREAKIN)   & (d2q_dt2 >  curv_th_breakin_time)
    mask_long    = (EFC > EFC_LOW_LONGTERM)             & (d2q_dt2 <  curv_th_longterm_time)
    mask_third   = ~(mask_breakin | mask_long)

    # 5) Assemble output DataFrame (retain smoothing and second derivative for parameter tuning/plotting)
    out = gdf.copy()
    out['q_smooth'] = q_smooth
    out['d2q_dt2']  = d2q_dt2

    df_break = out[mask_breakin].reset_index(drop=True)
    df_long  = out[mask_long].reset_index(drop=True)
    df_third = out[mask_third].reset_index(drop=True)

    key_b = f"{label}_breakin"  if label is not None else f"{fname}_breakin"
    key_l = f"{label}_longterm" if label is not None else f"{fname}_longterm"
    key_t = f"{label}_third"    if label is not None else f"{fname}_third"

    if not df_break.empty:  breakin_dict[key_b]  = df_break
    if not df_long.empty:   longterm_dict[key_l] = df_long
    if not df_third.empty:  third_dict[key_t]    = df_third

    summary_rows.append({
        "label": label, "file": fname,
        "n_total": len(out), "n_breakin": len(df_break),
        "n_longterm": len(df_long), "n_third": len(df_third),
        "roll_win": ROLL_WIN, "efc_per_day": efc_per_day,
        "thr_break_time": curv_th_breakin_time, "thr_long_time": curv_th_longterm_time
    })

# ====== Summary Output ======
summary_df = pd.DataFrame(summary_rows).sort_values(by="label")
print("\n=== Mask split summary (by group) ===")
print(summary_df.to_string(index=False))
print(f"\nCollected {len(breakin_dict)} break-in subsets, {len(longterm_dict)} long-term subsets, {len(third_dict)} middle-zone subsets.")

# ====== Visualization: Plot break-in / long-term points separately ======
plt.figure(figsize=(10,6))
for key, sub in breakin_dict.items():
    lab = key.split("_")[0]
    plt.scatter(sub['X'], sub['y_pure_cycle'], s=15, label=f"Group {lab}")
plt.xlabel("EFC")
plt.ylabel("Capacity loss (y_pure_cycle)")
plt.title(f"Break-in points (rolling mean window={ROLL_WIN})")
plt.legend(markerscale=2, fontsize=8, ncol=2)
plt.grid(True); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,6))
for key, sub in longterm_dict.items():
    lab = key.split("_")[0]
    plt.scatter(sub['X'], sub['y_pure_cycle'], s=15, label=f"Group {lab}")
plt.xlabel("EFC")
plt.ylabel("Capacity loss (y_pure_cycle)")
plt.title(f"Long-term points (rolling mean window={ROLL_WIN})")
plt.legend(markerscale=2, fontsize=8, ncol=2)
plt.grid(True); plt.tight_layout(); plt.show()

# ====== (Optional) Overlay Original Curves & Mask Colors for Manual Inspection ======
# Choose a group name to check, for example, the first file:
if len(all_data_dict_cleaned) > 0:
    fname0, df0 = next(iter(all_data_dict_cleaned.items()))
    lab0 = int(df0['LABLE'].iloc[0])
    g0 = df0[['X','t','y_pure_cycle']].dropna().sort_values('X').reset_index(drop=True)
    E0 = g0['X'].to_numpy(float); t0 = (g0['t'].to_numpy(float))/24.0; y0 = g0['y_pure_cycle'].to_numpy(float)
    y0s = robust_roll_mean(y0, win=ROLL_WIN)
    d2 = second_derivative_wrt_x(t0, y0s)
    efc_day0 = estimate_efc_per_day(E0, t0)
    th_b = CURV_TH_BREAKIN*(efc_day0**2) if np.isfinite(efc_day0) else CURV_TH_BREAKIN
    th_l = CURV_TH_LONGTERM*(efc_day0**2) if np.isfinite(efc_day0) else CURV_TH_LONGTERM
    mB = (E0>0)&(E0<EFC_UP_BREAKIN)&(d2>th_b)
    mL = (E0>EFC_LOW_LONGTERM)&(d2<th_l)
    mM = ~(mB|mL)

    plt.figure(figsize=(11,6))
    plt.plot(E0, y0, '-o', ms=3, alpha=0.4, label='raw')
    plt.plot(E0, y0s, '-', lw=2, label='rolling mean')
    plt.scatter(E0[mB], y0[mB], c='tab:red', s=20, label='break-in mask')
    plt.scatter(E0[mL], y0[mL], c='tab:blue', s=20, label='long-term mask')
    plt.scatter(E0[mM], y0[mM], c='gray', s=15, alpha=0.6, label='middle')
    plt.xlabel("EFC"); plt.ylabel("Capacity loss"); plt.title(f"Group {lab0}: mask check")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
