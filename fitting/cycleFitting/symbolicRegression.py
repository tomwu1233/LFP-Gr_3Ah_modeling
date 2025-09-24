# -*- coding: utf-8 -*-
# ==========================================
# Symbolic regression script
# ==========================================
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor

# ================== Paths (update to your actual paths) ==================
BREAKIN_PARAMS_CSV  = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_breakin\breakin_fit_params.csv"
LONGTERM_PARAMS_CSV = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_longterm\longterm_fit_params.csv"
OUT_DIR = os.path.dirname(LONGTERM_PARAMS_CSV)

# ================== 1) Hard-coded Condition Table (provided by you) ==================
def parse_crate(pair: str):
    # e.g. "1C/1C", "0.2C/0.2C", "1C/2C"
    m = re.match(r"\s*([0-9]*\.?[0-9]+)\s*C\s*/\s*([0-9]*\.?[0-9]+)\s*C\s*", pair)
    if not m:
        raise ValueError(f"Cannot parse Crate: {pair}")
    cd, cc = float(m.group(1)), float(m.group(2))
    return cd, cc, 0.5*(cd+cc)

rows = [
    # Label, TempC, SOC%, DOD%, "dis/chg"
    ( 1, "25 °C",  "50%", "100%", "1C/1C"),
    ( 2, "25 °C",  "50%",  "80%", "1C/1C"),
    ( 3, "40 °C",  "25%",  "20%", "1C/1C"),
    ( 4, "40 °C",  "50%",  "10%", "1C/1C"),
    ( 5, "40 °C",  "50%", "100%", "1C/1C"),
    ( 6, "40 °C",  "50%", "100%", "1C/1C"),
    ( 7, "40 °C",  "50%",  "20%", "1C/1C"),
    ( 8, "40 °C",  "50%",  "40%", "1C/1C"),
    ( 9, "40 °C",  "50%",   "5%", "1C/1C"),
    (10, "40 °C",  "50%",  "80%", "0.2C/0.2C"),
    (11, "40 °C",  "50%",  "80%", "0.5C/0.5C"),
    (12, "40 °C",  "50%", "100%", "0.5C/0.5C"),
    (13, "40 °C",  "50%",  "80%", "1C/0.5C"),
    (14, "40 °C",  "50%",  "80%", "1C/1C"),
    (15, "40 °C",  "50%", "100%", "1C/2C"),
    (16, "40 °C",  "75%",  "20%", "1C/1C"),
]

labels_df = []
for lab, t_str, soc_str, dod_str, crate_str in rows:
    TempC = float(re.findall(r"-?\d+\.?\d*", t_str)[0])
    SOC   = float(re.findall(r"-?\d+\.?\d*", soc_str)[0]) / 100.0   # Convert to 0~1
    DOD   = float(re.findall(r"-?\d+\.?\d*", dod_str)[0]) / 100.0   # Convert to 0~1
    Cr_dis, Cr_chg, Cr_avg = parse_crate(crate_str)
    labels_df.append({
        "label": lab, "TempC": TempC, "SOC": SOC, "DOD": DOD,
        "Crate_dis": Cr_dis, "Crate_chg": Cr_chg, "Crate_avg": Cr_avg
    })
labels_df = pd.DataFrame(labels_df)

# ================== 2) Read Parameter Tables and Merge ==================
def read_params(csv_path, alias):
    df = pd.read_csv(csv_path)
    # Standardize label column
    for c in ["label","LABEL","LABLE","Group","group","Label"]:
        if c in df.columns:
            df = df.rename(columns={c:"label"})
            break
    if alias not in df.columns:
        # Error handling: breakin may have column name as q4; longterm as a_long
        # Add mapping here if different
        raise ValueError(f"{csv_path} is missing column {alias}")
    return df[["label", alias]]

bp = read_params(BREAKIN_PARAMS_CSV,  alias="q4")       # break-in magnitude
lp = read_params(LONGTERM_PARAMS_CSV, alias="a_long")   # long-term magnitude

df = labels_df.merge(bp, on="label", how="inner").merge(lp, on="label", how="inner")
df = df.sort_values("label").reset_index(drop=True)
print("\nMerged training table (head):")
print(df.head())

merged_csv = os.path.join(OUT_DIR, "per_group_params_and_stressors.csv")
df.to_csv(merged_csv, index=False, encoding="utf-8-sig")
print(f"Saved merged table -> {merged_csv}")

# ================== 3) PySR Symbolic Regression (fit q4 and a_long separately) ==================
X = df[["DOD","SOC","Crate_avg","TempC"]].to_numpy(dtype=float)
y_q4 = df["q4"].to_numpy(dtype=float)
y_a  = df["a_long"].to_numpy(dtype=float)

def run_pysr(X, y, name="target"):
    model = PySRRegressor(
        niterations=2000,
        binary_operators=["+", "-", "*", "/", "pow"],
        unary_operators=["exp","log"],
        model_selection="best",
        parsimony=1e-4,                 # Preference for simpler equations
        constraints={"pow": (3, 1)},    # Control exponent, improve robustness
        progress=True,
        batching=True,
    )
    model.fit(X, y)
    yhat = model.predict(X)
    # Values should be non-negative; use clip if strict non-negativity is required
    # yhat = np.maximum(yhat, 0.0)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    r2  = 1 - ss_res/ss_tot
    rmse= np.sqrt(np.mean((y - yhat)**2))
    mae = np.mean(np.abs(y - yhat))
    print(f"\n===== Best equation for {name} =====\n{model}")
    print(f"{name}: R²={r2:.4f}, RMSE={rmse:.4g}, MAE={mae:.4g}")
    return model, yhat, (r2, rmse, mae)

model_q4, yhat_q4, m_q4 = run_pysr(X, y_q4, name="q4_breakin")
model_a,  yhat_a,  m_a  = run_pysr(X, y_a,  name="a_long")

# ================== 4) Visualization: Predicted vs Actual ==================
def scatter_fit(y, yhat, title, savepath):
    plt.figure(figsize=(5,5))
    plt.scatter(y, yhat, s=28)
    mn, mx = min(np.min(y), np.min(yhat)), max(np.max(y), np.max(yhat))
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title)
    plt.grid(True); plt.tight_layout(); plt.savefig(savepath, dpi=160); plt.close()
    print(f"Saved figure -> {savepath}")

scatter_fit(y_q4, yhat_q4, f"PySR fit for q4 (R²={m_q4[0]:.3f})",
            os.path.join(OUT_DIR, "pysr_fit_q4.png"))
scatter_fit(y_a,  yhat_a,  f"PySR fit for a_long (R²={m_a[0]:.3f})",
            os.path.join(OUT_DIR, "pysr_fit_a_long.png"))

# ================== 5) Save Optimal Equation Texts for Future Reference ==================
eq_path = os.path.join(OUT_DIR, "pysr_equations.txt")
with open(eq_path, "w", encoding="utf-8") as f:
    f.write("Best equation for q4_breakin:\n")
    f.write(str(model_q4) + "\n\n")
    f.write("Best equation for a_long:\n")
    f.write(str(model_a) + "\n")
print(f"Saved PySR equations -> {eq_path}")
