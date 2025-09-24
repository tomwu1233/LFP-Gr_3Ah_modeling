# ==========================================
# bilevel_iterative.py
# Alternating bilevel optimization control script (Stage A→B→C, supporting multi-round iterations)
# - A: Outer layer q2 + inner layer local fitting (q1,q3) for each dataset
# - B: Symbolic regression q1=f(T,SOC), q3=g(T,SOC)
# - C: Fix f,g and re-optimize q2 (no more local fitting)
# Each round is recorded to log.txt, and outputs weighted R2 and global R2.
# ==========================================
import os, time, json
import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, minimize
from pysr import PySRRegressor

# ----------------------------
# Config (change as needed)
# ----------------------------
DATA_DIR   = r"D:\SELECT\StoreNow\code\model_openSourceData\calendarAgingTestData\cleanedData"
OUT_DIR    = "./sr_outputs"
FIG_DIR    = "./iter_figs"
LOG_FILE   = "./log.txt"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SAVE_FIG   = False

# <<< Modify total iteration count here (default 2 rounds) >>>
N_OUTER    = 20

USE_POW    = False   # Set to True when power operation is needed, add constraints
SEED_Q1    = 0
SEED_Q3    = 1

# ----------------------------
# Utils
# ----------------------------
def log_write(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)

def read_all_csv_data(root_dir):
    data_dict = {}
    for fn in os.listdir(root_dir):
        if not fn.lower().endswith(".csv"):
            continue
        fp = os.path.join(root_dir, fn)
        df = pd.read_csv(fp)
        must = {"Time", "capacityPercent", "TemperatureDeg", "SOC"}
        if not must.issubset(df.columns):
            continue
        df = df.dropna(subset=list(must)).sort_values("Time").reset_index(drop=True)
        data_dict[fn] = df
    return data_dict

def calendar_loss_curve(t, q1, q2, q3):
    z = -q2 * (t - q3)
    z = np.clip(z, -60.0, 60.0)  # Prevent overflow
    return q1 / (1.0 + np.exp(z))

def sse(y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    return float(np.sum((y_true - y_pred)**2))

def metrics_block(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred)**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2) + eps)
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)
    return r2, rmse, mae, mape

# ----------------------------
# Stage A: q2 global + local fits (q1,q3 per dataset)
# ----------------------------
def fit_q1_q3_given_q2(df, q2, q1_bounds=(0.0, 2.0), q3_bounds=None):
    t = df["Time"].values.astype(float)
    cap = df["capacityPercent"].values.astype(float)
    if cap.max() > 1.5:  # Auto normalize to 0~1
        cap = cap / 100.0
    y = 1.0 - cap

    q1_guess = float(np.clip(np.nanmax(y), 1e-6, 1.5))
    q3_guess = float(np.median(t))
    if q3_bounds is None:
        span = np.ptp(t) + 1.0
        q3_bounds = (float(np.min(t)-0.2*span), float(np.max(t)+0.2*span))

    def residuals(p):
        q1, q3 = p
        return calendar_loss_curve(t, q1, q2, q3) - y

    res = least_squares(
        residuals, x0=[q1_guess, q3_guess],
        bounds=([q1_bounds[0], q3_bounds[0]], [q1_bounds[1], q3_bounds[1]]),
        method="trf", loss="soft_l1", f_scale=0.02,  # Robust loss, stabilize q3
        max_nfev=4000
    )
    q1_opt, q3_opt = map(float, res.x)
    y_fit = calendar_loss_curve(t, q1_opt, q2, q3_opt)
    return q1_opt, q3_opt, y, y_fit

def fit_global_q2_bilevel(data_dict, q2_init=1e-4, bounds=(1e-7, 1e-2)):
    def objective(q2_scalar):
        q2 = float(q2_scalar[0]) if np.ndim(q2_scalar) else float(q2_scalar)
        if q2 <= 0: return 1e12
        total = 0.0
        for df in data_dict.values():
            q1,q3,y,yp = fit_q1_q3_given_q2(df, q2)
            total += sse(y, yp)
        return total

    res = minimize(
        fun=objective, x0=np.array([q2_init], float),
        bounds=[bounds], method="L-BFGS-B",
        options=dict(maxiter=100, ftol=1e-12)
    )
    return float(res.x[0])

# ----------------------------
# Stage B: symbolic regression for q1,q3
# ----------------------------
def build_sr_model(seed):
    binary = ["+","-","*","/"] + (["pow"] if USE_POW else [])
    kwargs = dict(
        niterations=1000,
        binary_operators=binary,
        unary_operators=["exp","log","sqrt"],
        elementwise_loss="loss(x, y) = (x - y)^2",
        model_selection="best",
        maxsize=20, populations=30,
        deterministic=True, parallelism="serial",
        random_state=seed
    )
    if USE_POW:
        kwargs["constraints"] = {'^': (-1, 1)}
    return PySRRegressor(**kwargs)

def sr_fit_q_functions(local_rows, out_dir=OUT_DIR):
    df = pd.DataFrame(local_rows)
    # Temperature→K, SOC→0~1
    T_K  = df["TemperatureDeg"].astype(float).values + 273.15
    SOC  = np.clip(df["SOC"].astype(float).values, 0.0, 1.0)
    X    = np.column_stack([T_K, SOC])

    # q1
    y1   = df["q1"].astype(float).values
    m1   = build_sr_model(SEED_Q1)
    m1.fit(X, y1)
    best1 = pd.DataFrame([m1.get_best()])
    eqs1  = pd.concat([best1, m1.equations_]).drop_duplicates().reset_index(drop=True)
    q1_csv= os.path.join(out_dir, "q1_equations.csv")
    eqs1.to_csv(q1_csv, index=False)

    # q3
    y3   = df["q3"].astype(float).values
    m3   = build_sr_model(SEED_Q3)
    m3.fit(X, y3)
    best3 = pd.DataFrame([m3.get_best()])
    eqs3  = pd.concat([best3, m3.equations_]).drop_duplicates().reset_index(drop=True)
    q3_csv= os.path.join(out_dir, "q3_equations.csv")
    eqs3.to_csv(q3_csv, index=False)

    # 返回可调用函数与“最佳公式字符串”
    def to_callable(eq_csv):
        tab = pd.read_csv(eq_csv)
        # 选最佳行：优先 loss 最小，其次 score 最大，否则第一行
        if "loss" in tab.columns:
            idx = tab["loss"].astype(float).idxmin()
        elif "score" in tab.columns:
            idx = tab["score"].astype(float).idxmax()
        else:
            idx = 0
        eq_str = str(tab.loc[idx, "equation"])
        x0,x1 = sp.symbols("x0 x1", real=True)
        expr  = sp.sympify(eq_str, locals={"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})
        func  = sp.lambdify((x0,x1), expr, modules={"numpy": np})
        return func, eq_str

    q1_func, q1_expr = to_callable(q1_csv)
    q3_func, q3_expr = to_callable(q3_csv)
    return q1_func, q3_func, q1_expr, q3_expr

# ----------------------------
# Stage C: re-opt q2 with SR functions
# ----------------------------
def refit_q2_with_sr(q1_func, q3_func, data_dict, q2_init=1e-4, bounds=(1e-7,1e-1)):
    def objective(q2_scalar):
        q2 = float(q2_scalar[0]) if np.ndim(q2_scalar) else float(q2_scalar)
        if q2 <= 0: return 1e12
        total = 0.0
        for df in data_dict.values():
            t   = df["Time"].values.astype(float)
            cap = df["capacityPercent"].values.astype(float)
            if cap.max() > 1.5:
                cap = cap/100.0
            y   = 1.0 - cap
            T_C = float(df["TemperatureDeg"].iloc[0])
            SOC = float(df["SOC"].iloc[0]);  SOC = SOC if 0<=SOC<=1 else SOC/100.0
            T_K = T_C + 273.15
            q1v = q1_func(T_K, SOC)
            q3v = q3_func(T_K, SOC)
            yp  = calendar_loss_curve(t, q1v, q2, q3v)
            total += sse(y, yp)
        return total

    res = minimize(
        fun=objective, x0=np.array([q2_init], float),
        bounds=[bounds], method="L-BFGS-B",
        options=dict(maxiter=200, ftol=1e-12)
    )
    return float(res.x[0])

# ----------------------------
# Evaluation (weighted R2 + global R2)
# ----------------------------
def evaluate_all(data_dict, q1_func, q3_func, q2, iter_tag="final", save_fig=SAVE_FIG):
    rows = []
    y_all, yhat_all = [], []
    for fname, df in data_dict.items():
        t   = df["Time"].values.astype(float)
        cap = df["capacityPercent"].values.astype(float)
        if cap.max() > 1.5:
            cap = cap/100.0
        y   = 1.0 - cap
        T_C = float(df["TemperatureDeg"].iloc[0])
        SOC = float(df["SOC"].iloc[0]); SOC = SOC if 0<=SOC<=1 else SOC/100.0
        T_K = T_C + 273.15
        q1v = q1_func(T_K, SOC)
        q3v = q3_func(T_K, SOC)
        yhat = calendar_loss_curve(t, q1v, q2, q3v)

        r2, rmse, mae, mape = metrics_block(y, yhat)
        rows.append(dict(file=fname, R2=r2, RMSE=rmse, MAE=mae, MAPE=mape,
                         TemperatureDeg=T_C, SOC=SOC, n_points=len(df)))

        y_all.append(y); yhat_all.append(yhat)

        # 绘图
        if save_fig:
            plt.figure()
            plt.scatter(t, y, s=18, label="Data (loss)")
            t_grid = np.linspace(t.min(), t.max(), 400)
            y_grid = calendar_loss_curve(t_grid, q1v, q2, q3v)
            plt.plot(t_grid, y_grid, label="Model (loss)")
            plt.xlabel("Time (days)")
            plt.ylabel("Capacity loss")
            plt.title(f"{fname}\nq2={q2:.2e} | R2={r2:.3f}, MAPE={mape:.2f}%")
            plt.legend()
            plt.tight_layout()
            out_png = os.path.join(FIG_DIR, f"{os.path.splitext(fname)[0]}_{iter_tag}.png")
            plt.savefig(out_png, dpi=160)
            plt.close()

    metrics_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    # Weighted R2
    if len(metrics_df):
        w = metrics_df["n_points"].values
        w = w / w.sum()
        weighted_r2 = float(np.sum(w * metrics_df["R2"].values))
    else:
        weighted_r2 = np.nan

    # Global R2
    if y_all:
        y_all = np.concatenate(y_all); yhat_all = np.concatenate(yhat_all)
        ss_res = np.sum((y_all - yhat_all)**2)
        ss_tot = np.sum((y_all - y_all.mean())**2) + 1e-8
        global_r2 = float(1.0 - ss_res / ss_tot)
    else:
        global_r2 = np.nan

    out_metrics_csv = os.path.join(OUT_DIR, f"metrics_calendar_model_{iter_tag}.csv")
    metrics_df.to_csv(out_metrics_csv, index=False)

    return weighted_r2, global_r2, out_metrics_csv

# ----------------------------
# Main alternate loop
# ----------------------------
def main():
    start_all = time.time()
    all_data = read_all_csv_data(DATA_DIR)
    log_write(f"Loaded datasets: {len(all_data)}")
    q2 = 1e-4  # 初始值（可按需改）

    for k in range(1, N_OUTER+1):
        iter_start = time.time()
        log_write(f"=== OUTER ITER {k} START ===")

        # A) 外层q2 + 内层局部q1,q3
        q2 = fit_global_q2_bilevel(all_data, q2_init=q2)
        log_write(f"[A] q2 (outer/global with local fits) = {q2:.6e}")

        local_rows = []
        for fname, df in all_data.items():
            q1, q3, y, yp = fit_q1_q3_given_q2(df, q2)
            T_C  = float(df["TemperatureDeg"].iloc[0])
            SOCp = float(df["SOC"].iloc[0]); SOCp = SOCp if 0<=SOCp<=1 else SOCp/100.0
            local_rows.append(dict(file=fname, q1=q1, q3=q3, TemperatureDeg=T_C, SOC=SOCp))
        round_res_csv = os.path.join(OUT_DIR, f"calendar_bilevel_results_iter{k}.csv")
        pd.DataFrame(local_rows).to_csv(round_res_csv, index=False)
        log_write(f"[A] saved locals -> {round_res_csv}")

        # B) 符号回归得到 q1=f, q3=g
        q1_func, q3_func, q1_expr, q3_expr = sr_fit_q_functions(local_rows, out_dir=OUT_DIR)
        log_write(f"[B] q1 best: {q1_expr}")
        log_write(f"[B] q3 best: {q3_expr}")

        # C) 固定f,g 重优化 q2
        q2_new = refit_q2_with_sr(q1_func, q3_func, all_data, q2_init=q2)
        log_write(f"[C] q2 re-optimized with SR: {q2_new:.6e}")
        q2 = q2_new

        # 评估并记录指标（加权R2与全局R2）
        wR2, gR2, csv_path = evaluate_all(all_data, q1_func, q3_func, q2, iter_tag=f"iter{k}", save_fig=SAVE_FIG)
        elapsed = time.time() - iter_start
        log_write(f"[EVAL] iter {k} weighted R2 = {wR2:.4f}, global R2 = {gR2:.4f}")
        log_write(f"[EVAL] iter {k} metrics -> {csv_path}")
        log_write(f"=== OUTER ITER {k} END | elapsed {elapsed:.2f}s ===\n")

    total = time.time() - start_all
    log_write(f"[FINAL] Done. Total duration = {total:.2f}s")

if __name__ == "__main__":
    main()
