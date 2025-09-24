import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp

Q1_EQ_CSV  = "./sr_outputs/q1_equations.csv"        # PySR 输出（候选式，取第一行）
Q3_EQ_CSV  = "./sr_outputs/q3_equations.csv"
# Q2_CSV     = "./sr_outputs/q2_equations.csv"         # 第一次globalOptilization的结果

Q2_CSV     = "./sr_outputs/q2_reOptimized.csv"         # reOptimization的结果


DATA_DIR   = r"D:\SELECT\StoreNow\code\model_openSourceData\calendarAgingTestData\cleanedData"
SAVE_FIG   = False
FIG_DIR    = "./calendar_model_figs_fromlog"
METRICS_CSV = "./metrics_calendar_model_fromlog.csv"
os.makedirs(FIG_DIR, exist_ok=True)

def plot_testData(oneSetData,title):
    """
    oneSetData: list 或 dict，每个元素为一个 DataFrame
    """
    plt.figure()
    for df in oneSetData:
        t = df["Time"].values.astype(float)
        cap = df["capacityPercent"].values.astype(float)
        # 自动从 0-100 转 0-1
        if cap.max() > 1.5:
            cap = cap / 100.0
        y_true_loss = 1.0 - cap

        SOCv = float(df["SOC"].iloc[0])
        if not (0.0 <= SOCv <= 1.0):
            SOCv = SOCv / 100.0
        SOCv = float(np.clip(SOCv, 0.0, 1.0))

        T_C = float(df["TemperatureDeg"].iloc[0])
        # T_K = T_C + 273.15  # 如果后续用到可以加上

        plt.scatter(t, 1-y_true_loss, s=18, label=f"{T_C:.1f}°C, SOC={SOCv:.2f}")

    plt.xlabel("Time (hours)")
    plt.ylabel("Capacity loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIG:
        out_png = os.path.join(FIG_DIR, f"testData_{title}.png")
        plt.savefig(out_png, dpi=160)
    plt.show()
# filepath: d:\SELECT\StoreNow\code\model_openSourceData\calendarAgingTestData\biLevelOptimization\visulization.py

#plot temperature effect
#把以下的文件在一个图中plotTP_0°C,50%SOC.csv, TP_25°C,50%SOC.csv, TP_40°C,50%SOC.csv, TP_60°C,50%SOC.csv
df_0C50SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_0°C,50%SOC.csv"))
df_25C50SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_25°C,50%SOC.csv"))
df_40C50SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,50%SOC.csv"))
df_60C50SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_60°C,50%SOC.csv"))

Datas_TemperatureComparison = [df_0C50SOC, df_25C50SOC, df_40C50SOC, df_60C50SOC]
# plot_testData(Datas_TemperatureComparison, 'Temperature Comparison at 50% SOC')

#把以下的文件在一个图中plot TP_40°C,0%SOC.csv,TP_40°C,12.5%SOC.csv，TP_40°C,25%SOC.csv，TP_40°C,37.5%SOC.csv，TP_40°C,50%SOC.csv，TP_40°C,62.5%SOC.csv，TP_40°C,75%SOC.csv，TP_40°C,87.5%SOC.csv，TP_40°C,100%SOC.csv
df_40C0SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,0%SOC.csv"))
df_40C12_5SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,12.5%SOC.csv"))
df_40C25SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,25%SOC.csv"))
df_40C37_5SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,37.5%SOC.csv"))
df_40C50SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,50%SOC.csv"))
df_40C62_5SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,62.5%SOC.csv"))
df_40C75SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,75%SOC.csv"))
df_40C87_5SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,87.5%SOC.csv"))
df_40C100SOC = pd.read_csv(os.path.join(DATA_DIR, "TP_40°C,100%SOC.csv"))

Datas_SOCComparison = [df_40C0SOC, df_40C12_5SOC, df_40C25SOC, df_40C37_5SOC, df_40C50SOC, df_40C62_5SOC, df_40C75SOC, df_40C87_5SOC, df_40C100SOC]
# plot_testData(Datas_SOCComparison, 'SOC Comparison at 40°C')

def load_q2_value(csv_path):
    with open(csv_path, "r", encoding="latin1") as f:   # encoding 可以换成 gbk, utf-8
        content = f.read().strip()
    # 去掉可能的逗号或空格，只保留数字
    try:
        return float(content.split(",")[0])
    except Exception:
        raise ValueError(f"无法从 {csv_path} 解析数值，内容是: {content}")

def load_best_equation_callable(eq_csv, var_order=("x0","x1")):
    """
    从 PySR equations.csv 读取第一行 'equation'，解析为可调用函数。
    约定 x0=T_K, x1=SOC (0~1)。
    支持常见函数 exp/log/sqrt、以及 pow(^)。
    """
    if not os.path.exists(eq_csv):
        raise FileNotFoundError(f"Equation CSV not found: {eq_csv}")
    tab = pd.read_csv(eq_csv)
    if "equation" not in tab.columns:
        raise ValueError(f"'equation' column not found in {eq_csv}")
    eq_str = str(tab.loc[0, "equation"])
    # 声明符号
    sym_vars = sp.symbols(" ".join(var_order), real=True)  # (x0, x1, ...)
    # 允许的函数环境
    sym_env = {"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}
    # 解析表达式
    expr = sp.sympify(eq_str, locals=sym_env)
    # 构造可调用函数（NumPy 后端）
    func = sp.lambdify(sym_vars, expr, modules={"numpy": np})
    return func, eq_str

q1_func, q1_expr = load_best_equation_callable(Q1_EQ_CSV, var_order=("x0","x1"))
q3_func, q3_expr = load_best_equation_callable(Q3_EQ_CSV, var_order=("x0","x1"))
q2_global = load_q2_value(Q2_CSV)

def qloss_calendar(t_days, T_C, SOC_01):
    """
    Inputs:
      t_days: ndarray, 时间(天)
      T_C   : float, 摄氏温度
      SOC_01: float, 0~1
    Returns:
      capacity loss ndarray
    """
    T_K = T_C + 273.15
    q1 = q1_func(T_K, SOC_01)
    q3 = q3_func(T_K, SOC_01)
    return q1 / (1.0 + np.exp(-q2_global * (t_days - q3)))

def plot_modelResult(oneSetData,title):
    """
    oneSetData: list 或 dict，每个元素为一个 DataFrame
    """
    plt.figure()
    for df in oneSetData:
        t = df["Time"].values.astype(float)
        cap = df["capacityPercent"].values.astype(float)
        # 自动从 0-100 转 0-1
        if cap.max() > 1.5:
            cap = cap / 100.0
        y_true_loss = 1.0 - cap
        SOCv = float(df["SOC"].iloc[0])
        T_C = float(df["TemperatureDeg"].iloc[0])

        # 计算预测
        y_pred_loss = qloss_calendar(t, T_C, SOCv)

        SOCv = float(df["SOC"].iloc[0])
        if not (0.0 <= SOCv <= 1.0):
            SOCv = SOCv / 100.0
        SOCv = float(np.clip(SOCv, 0.0, 1.0))

        T_C = float(df["TemperatureDeg"].iloc[0])
        # T_K = T_C + 273.15  # 如果后续用到可以加上

        # 绘图：散点(实验 loss) vs 连续曲线(模型 loss)
        plt.scatter(t, 1-y_true_loss, s=18, label=f"{T_C:.1f}°C, SOC={SOCv:.2f}")
        plt.plot(t, 1-y_pred_loss, linestyle="--", label=f"Model {T_C:.1f}°C, SOC={SOCv:.2f}")

    plt.figure(figsize=(12, 6))
    plt.xlabel("Time (hours)")
    plt.ylabel("Capacity loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if SAVE_FIG:
        out_png = os.path.join(FIG_DIR, f"testData_{title}.png")
        plt.savefig(out_png, dpi=160)
    plt.show()

def plot_modelResultColormap(oneSetData, title):
    """
    oneSetData: list 或 dict，每个元素为一个 DataFrame
    """
    plt.figure(figsize=(12, 6))
    n = len(oneSetData)
    cmap = plt.get_cmap('viridis')  # 可选：'plasma', 'rainbow', 'tab20', 等
    colors = [cmap(i / n) for i in range(n)]

    for idx, df in enumerate(oneSetData):
        t = df["Time"].values.astype(float)
        cap = df["capacityPercent"].values.astype(float)
        if cap.max() > 1.5:
            cap = cap / 100.0
        y_true_loss = 1.0 - cap
        SOCv = float(df["SOC"].iloc[0])
        T_C = float(df["TemperatureDeg"].iloc[0])
        y_pred_loss = qloss_calendar(t, T_C, SOCv)

        # 只用数字作为 label
        # label改成用SOC和Temperature
        plt.scatter(t/24, 1-y_true_loss, s=18, color=colors[idx], label=f"SOC={SOCv*100:.0f}%, T_C={T_C:.1f}°C")
        plt.plot(t/24, 1-y_pred_loss, linestyle="--", color=colors[idx])

    plt.xlabel("Time (days)")
    plt.ylabel("Remaining Capacity")
    plt.title(title)
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    if SAVE_FIG:
        out_png = os.path.join(FIG_DIR, f"testData_{title}.png")
        plt.savefig(out_png, dpi=160)
    plt.show()
# filepath: d:\SELECT\StoreNow\code\model_openSourceData\calendarAgingTestData\biLevelOptimization\visulization.py

def plot_testdataColormap(oneSetData, title):
    """
    oneSetData: list 或 dict，每个元素为一个 DataFrame
    """
    plt.figure(figsize=(12, 6))
    n = len(oneSetData)
    cmap = plt.get_cmap('viridis')  # 可选：'plasma', 'rainbow', 'tab20', 等
    colors = [cmap(i / n) for i in range(n)]

    for idx, df in enumerate(oneSetData):
        t = df["Time"].values.astype(float)
        cap = df["capacityPercent"].values.astype(float)
        if cap.max() > 1.5:
            cap = cap / 100.0
        y_true_loss = 1.0 - cap
        SOCv = float(df["SOC"].iloc[0])
        T_C = float(df["TemperatureDeg"].iloc[0])
        y_pred_loss = qloss_calendar(t, T_C, SOCv)

        # 只用数字作为 label
        plt.scatter(t, 1-y_true_loss, s=18, color=colors[idx], label=f"{idx+1}")
        # plt.plot(t, 1-y_pred_loss, linestyle="--", color=colors[idx])

    plt.xlabel("Time (hours)")
    plt.ylabel("Capacity loss")
    plt.title(title)
    plt.legend(title="Dataset No.", loc="best", ncol=2)
    plt.tight_layout()
    if SAVE_FIG:
        out_png = os.path.join(FIG_DIR, f"testData_{title}.png")
        plt.savefig(out_png, dpi=160)
    plt.show()

#把DATA_DIR中全部的csv文件读入到ALL_data中
All_data = {}
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        All_data[file] = df


plot_modelResultColormap(All_data.values(), 'Calendar Aging Model result vs Test data')


# plot_testdataColormap(All_data.values(), 'Time effect: 17 datasets')
