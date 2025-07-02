"""
EIS 拟合
读取阻抗谱 (freq, Zreal, Zimag)，根据选定的等效电路模型进行非线性最小二乘拟合，输出 Nyquist 图与 Bode 图。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def load_eis_csv(file_path: Path):
    """读取 csv，要求列：freq, Zreal, Zimag"""
    df = pd.read_csv(file_path)
    freq = df["freq"].values
    z_complex = df["Zreal"].values + 1j * df["Zimag"].values
    return freq, z_complex


Z_cpe = lambda Q, n, w: 1 / (Q * (1j * w) ** n)

Z_w = lambda sigma, w: (sigma / np.sqrt(w)) * (1 - 1j)


# 等效电路模型

def model_A(p, w):
    """R_s + (R_ct ‖ C)"""
    R_s, R_ct, C_dl = p
    Z_c = 1 / (1j * w * C_dl)
    Z_par = 1 / (1 / R_ct + 1 / Z_c)
    return R_s + Z_par

def model_B(p, w):
    """R_s + (R_ct ‖ CPE)"""
    R_s, R_ct, Q, n = p
    Z_q = Z_cpe(Q, n, w)
    Z_par = 1 / (1 / R_ct + 1 / Z_q)
    return R_s + Z_par

def model_C(p, w):
    """R_s + (R1 ‖ CPE1) + (R2 ‖ CPE2)"""
    R_s, R1, Q1, n1, R2, Q2, n2 = p
    Z1 = 1 / (1 / R1 + 1 / Z_cpe(Q1, n1, w))
    Z2 = 1 / (1 / R2 + 1 / Z_cpe(Q2, n2, w))
    return R_s + Z1 + Z2

def model_D(p, w):
    """R_s + (R_ct ‖ CPE) + Warburg"""
    R_s, R_ct, Q, n, sigma = p
    Z_q = Z_cpe(Q, n, w)
    Z_par = 1 / (1 / R_ct + 1 / Z_q)
    return R_s + Z_par + Z_w(sigma, w)

MODEL_DICT = {
    "A": (model_A, [1, 100, 1e-5]),
    "B": (model_B, [1, 100, 1e-4, 0.8]),
    "C": (model_C, [1, 50, 1e-4, 0.9, 200, 1e-5, 0.7]),
    "D": (model_D, [1, 100, 1e-4, 0.8, 10]),
}


# 拟合

def residuals(p, w, z_exp, model_func):
    z_mod = model_func(p, w)
    return np.concatenate([(z_mod.real - z_exp.real), (z_mod.imag - z_exp.imag)])

# 绘图


def plot_eis(freq, z_exp, z_fit):
    # Nyquist
    plt.figure(figsize=(5,4))
    plt.plot(z_exp.real, -z_exp.imag, 'o', ms=4, label='实验')
    plt.plot(z_fit.real, -z_fit.imag, '-', label='拟合')
    plt.xlabel('Z´ / Ω'); plt.ylabel('-Z" / Ω'); plt.title('Nyquist'); plt.legend()
    plt.tight_layout(); plt.savefig('nyquist_fit.png', dpi=300); plt.show()

    # Bode |Z|
    plt.figure(figsize=(5,4))
    plt.semilogx(freq, np.abs(z_exp), 'o', ms=4, label='实验')
    plt.semilogx(freq, np.abs(z_fit), '-', label='拟合')
    plt.xlabel('频率 / Hz'); plt.ylabel('|Z| / Ω'); plt.title('Bode |Z|'); plt.legend()
    plt.tight_layout(); plt.savefig('bode_mag_fit.png', dpi=300); plt.show()

    # Bode 相位
    plt.figure(figsize=(5,4))
    plt.semilogx(freq, np.angle(z_exp, deg=True), 'o', ms=4, label='实验')
    plt.semilogx(freq, np.angle(z_fit, deg=True), '-', label='拟合')
    plt.xlabel('频率 / Hz'); plt.ylabel('相位 / °'); plt.title('Bode Phase'); plt.legend()
    plt.tight_layout(); plt.savefig('bode_phase_fit.png', dpi=300); plt.show()



def main():
    if len(sys.argv) < 3:
        print("用法: python eis_fit_demo.py data.csv A|B|C|D")
        return

    csv_path = Path(sys.argv[1])
    key = sys.argv[2].upper()
    if key not in MODEL_DICT:
        print("模型代码必须为 A / B / C / D")
        return

    freq, z_exp = load_eis_csv(csv_path)
    w = 2 * np.pi * freq

    model_func, p0 = MODEL_DICT[key]
    res = least_squares(residuals, p0, args=(w, z_exp, model_func), max_nfev=8000)
    z_fit = model_func(res.x, w)

    print("拟合参数：")
    for i, v in enumerate(res.x, 1):
        print(f"p{i} = {v:.4e}")

    plot_eis(freq, z_exp, z_fit)

if __name__ == "__main__":
    main()
