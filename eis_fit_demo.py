import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# 数据加载函数
def load_eis_data(file_path: Path):
    """
    加载EIS实验数据。要求csv文件包含列：'freq', 'Zreal', 'Zimag'。
    :param file_path: 数据文件路径
    :return: 频率数据与复阻抗数据
    """
    try:
        df = pd.read_csv(file_path)
        freq = df["freq"].values
        z_complex = df["Zreal"].values + 1j * df["Zimag"].values
        return freq, z_complex
    except Exception as e:
        print(f"加载数据时出错: {e}")
        sys.exit(1)

# 等效电路模型
def Z_cpe(Q, n, w): return 1 / (Q * (1j * w) ** n)  # 常见的CPE元件模型

def Z_w(sigma, w): return (sigma / np.sqrt(w)) * (1 - 1j)  # Warburg阻抗模型

def model_A(p, w):
    """模型 A: R_s + (R_ct ‖ C)"""
    R_s, R_ct, C_dl = p
    Z_c = 1 / (1j * w * C_dl)
    Z_par = 1 / (1 / R_ct + 1 / Z_c)
    return R_s + Z_par

def model_B(p, w):
    """模型 B: R_s + (R_ct ‖ CPE)"""
    R_s, R_ct, Q, n = p
    Z_q = Z_cpe(Q, n, w)
    Z_par = 1 / (1 / R_ct + 1 / Z_q)
    return R_s + Z_par

def model_C(p, w):
    """模型 C: R_s + (R1 ‖ CPE1) + (R2 ‖ CPE2)"""
    R_s, R1, Q1, n1, R2, Q2, n2 = p
    Z1 = 1 / (1 / R1 + 1 / Z_cpe(Q1, n1, w))
    Z2 = 1 / (1 / R2 + 1 / Z_cpe(Q2, n2, w))
    return R_s + Z1 + Z2

def model_D(p, w):
    """模型 D: R_s + (R_ct ‖ CPE) + Warburg"""
    R_s, R_ct, Q, n, sigma = p
    Z_q = Z_cpe(Q, n, w)
    Z_par = 1 / (1 / R_ct + 1 / Z_q)
    return R_s + Z_par + Z_w(sigma, w)

# 选择模型字典
MODEL_DICT = {
    "A": (model_A, [1, 100, 1e-5]),
    "B": (model_B, [1, 100, 1e-4, 0.8]),
    "C": (model_C, [1, 50, 1e-4, 0.9, 200, 1e-5, 0.7]),
    "D": (model_D, [1, 100, 1e-4, 0.8, 10]),
}

# 拟合函数
def residuals(p, w, z_exp, model_func):
    z_mod = model_func(p, w)
    return np.concatenate([(z_mod.real - z_exp.real), (z_mod.imag - z_exp.imag)])

# 绘图函数
def plot_eis(freq, z_exp, z_fit):
    """
    绘制Nyquist图与Bode图。
    :param freq: 频率数据
    :param z_exp: 实验数据
    :param z_fit: 拟合数据
    """
    # Nyquist图
    plt.figure(figsize=(5,4))
    plt.plot(z_exp.real, -z_exp.imag, 'o', ms=4, label='实验数据')
    plt.plot(z_fit.real, -z_fit.imag, '-', label='拟合数据')
    plt.xlabel('Z´ / Ω')
    plt.ylabel('-Z" / Ω')
    plt.title('Nyquist图')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nyquist_fit.png', dpi=300)
    plt.show()

    # Bode图 |Z|
    plt.figure(figsize=(5,4))
    plt.semilogx(freq, np.abs(z_exp), 'o', ms=4, label='实验数据')
    plt.semilogx(freq, np.abs(z_fit), '-', label='拟合数据')
    plt.xlabel('频率 / Hz')
    plt.ylabel('|Z| / Ω')
    plt.title('Bode |Z| 图')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bode_mag_fit.png', dpi=300)
    plt.show()

    # Bode图 相位
    plt.figure(figsize=(5,4))
    plt.semilogx(freq, np.angle(z_exp, deg=True), 'o', ms=4, label='实验数据')
    plt.semilogx(freq, np.angle(z_fit, deg=True), '-', label='拟合数据')
    plt.xlabel('频率 / Hz')
    plt.ylabel('相位 / °')
    plt.title('Bode Phase 图')
    plt.legend()
    plt.tight_layout()
    plt.savefig('bode_phase_fit.png', dpi=300)
    plt.show()

# 主函数
def main():
    if len(sys.argv) < 3:
        print("用法: python eis_fit.py data.csv A|B|C|D")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    model_key = sys.argv[2].upper()

    if model_key not in MODEL_DICT:
        print("模型必须是 A、B、C 或 D")
        sys.exit(1)

    # 加载数据
    freq, z_exp = load_eis_data(csv_path)
    w = 2 * np.pi * freq  # 角频率

    # 选择模型
    model_func, p0 = MODEL_DICT[model_key]
    
    # 拟合过程
    res = least_squares(residuals, p0, args=(w, z_exp, model_func), max_nfev=8000)
    z_fit = model_func(res.x, w)

    # 打印拟合参数
    print("拟合参数：")
    for i, param in enumerate(res.x, 1):
        print(f"p{i} = {param:.4e}")

    # 绘制图像
    plot_eis(freq, z_exp, z_fit)

if __name__ == "__main__":
    main()
