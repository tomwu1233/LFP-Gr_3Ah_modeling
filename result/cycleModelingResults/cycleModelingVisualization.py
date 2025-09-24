# ==========================================
# Cycle aging heatmap style visualization
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sympy as sp

# Config
# change as needed
DATA_DIR = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_0"
SAVE_FIG = False
FIG_DIR = "./cycle_model_figs"
os.makedirs(FIG_DIR, exist_ok=True)

EQUATIONS_JSON = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_longterm\sr_outputs\pysr_equations.json"
BREAKIN_PARAMS_CSV  = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_breakin\breakin_fit_params.csv"
LONGTERM_PARAMS_CSV = r"D:\SELECT\StoreNow\code\model_openSourceData\cyclicAgingTestData\capacity\modeFitting\data_longterm\longterm_fit_params.csv"

E0 = 4000.0  # EFC normalization scale (set to None if not used)

# Load models & global parameters
def load_q4_a_funcs(eq_json_path):
    with open(eq_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    DOD, SOC, Crate_avg, TempC = sp.symbols("DOD SOC Crate_avg TempC")
    expr_q4 = sp.sympify(payload["q4_breakin"]["sympy"])
    expr_a  = sp.sympify(payload["a_long"]["sympy"])
    f_q4 = sp.lambdify((DOD, SOC, Crate_avg, TempC), expr_q4, "numpy")
    f_a  = sp.lambdify((DOD, SOC, Crate_avg, TempC), expr_a,  "numpy")
    print("[INFO] q4(DOD,SOC,Crate_avg,TempC) =", payload["q4_breakin"]["sympy"])
    print("[INFO] a_long(DOD,SOC,Crate_avg,TempC) =", payload["a_long"]["sympy"])
    return f_q4, f_a

def load_param_globals(breakin_csv, longterm_csv):
    bp = pd.read_csv(breakin_csv)
    lt = pd.read_csv(longterm_csv)
    q5 = float(bp["q5_global"].iloc[0]) if "q5_global" in bp.columns else float(bp["q5"].iloc[0])
    q6 = float(bp["q6_global"].iloc[0]) if "q6_global" in bp.columns else float(bp["q6"].iloc[0])
    b  = float(lt["b_global"].iloc[0])  if "b_global"  in lt.columns else float(lt["b"].iloc[0])
    print(f"[INFO] globals -> q5={q5:.6g}, q6={q6:.6g}, b={b:.6g}")
    return q5, q6, b

f_q4, f_a = load_q4_a_funcs(EQUATIONS_JSON)
q5_g, q6_g, b_g = load_param_globals(BREAKIN_PARAMS_CSV, LONGTERM_PARAMS_CSV)

def q_breakin(E, q4, q5, q6, E0=None):
    En = E/float(E0) if E0 else E
    s  = np.power(max(q5,1e-12)*np.maximum(En,0.0), max(q6,1e-12))
    s  = np.clip(s, 0.0, 50.0)
    inv_logistic = np.exp(-s)/(1.0 + np.exp(-s))
    return 2.0*q4*(0.5 - inv_logistic)

def q_longterm(E, a, b, E0=None):
    En = E/float(E0) if E0 else E
    En = np.maximum(En, 1e-12)
    return a * np.power(En, b)

# Read all cycle aging test data
all_data_dict = {}
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        all_data_dict[file] = df

def plot_cycle_model_heatmap(all_data_dict, title):
    plt.figure(figsize=(12, 6))
    n = len(all_data_dict)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / n) for i in range(n)]

    for idx, (fname, df) in enumerate(all_data_dict.items()):
        E   = df["X"].to_numpy(float)
        y   = df["y_pure_cycle"].to_numpy(float)
        y_true = y
        TempC = float(df["Temperature"].iloc[0])
        SOC   = float(df["SOC"].iloc[0])
        DoD   = float(df["DoD"].iloc[0])
        C_chg = float(df["C_chg"].iloc[0])
        C_dis = float(df["C_dis"].iloc[0])
        Crate_avg = 0.5*(C_chg + C_dis)
        q4 = float(np.maximum(f_q4(DoD, SOC, Crate_avg, TempC), 0.0))
        aL = float(np.maximum(f_a (DoD, SOC, Crate_avg, TempC),  0.0))
        y_hat = q_breakin(E, q4, q5_g, q6_g, E0=E0) + q_longterm(E, aL, b_g, E0=E0)
        y_pred = 1.0 - y_hat
        # Only add label to scatter, not to line
        plt.scatter(E, y_true, s=18, color=colors[idx], label=f"(DoD={DoD:.2f}, C̄={Crate_avg:.2f}, T={TempC:.1f}°C)")
        plt.plot(E, y_pred, linestyle="--", color=colors[idx])

    plt.xlabel("EFC")
    plt.ylabel("Remaining Capacity")
    plt.title(title)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    if SAVE_FIG:
        out_png = os.path.join(FIG_DIR, f"cycle_model_heatmap_{title}.png")
        plt.savefig(out_png, dpi=160)
    plt.show()

plot_cycle_model_heatmap(all_data_dict, "Cycle Aging Model result vs Test data")