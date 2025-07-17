import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_eis_csv(file_path: Path):
    """读取CSV格式的EIS数据。列名必须包含 freq, Zreal, Zimag"""
    df = pd.read_csv(file_path)
    freq = df['freq'].values
    z = df['Zreal'].values + 1j * df['Zimag'].values
    return freq, z


def kk_transform(freq, imag_part, z_inf=0.0):
    """根据K-K关系，由虚部预测实部。采用离散近似积分。"""
    w = 2 * np.pi * freq
    real_pred = np.zeros_like(imag_part)
    for i, wi in enumerate(w):
        integrand = (w * imag_part) / (w**2 - wi**2 + 1e-30)
        # 使用辛普森积分近似
        real_pred[i] = z_inf + (2 / np.pi) * np.trapz(integrand, w)
    return real_pred


def kk_check(freq, z):
    """执行K-K一致性检验，返回预测实部与测量实部之间的误差指标。"""
    z_real = z.real
    z_imag = z.imag

    z_inf = z_real[0]  # 高频截距近似
    z_real_pred = kk_transform(freq, z_imag, z_inf)

    residual = z_real - z_real_pred
    rmse = np.sqrt(np.mean(residual**2))
    nrmse = rmse / (z_real.max() - z_real.min() + 1e-12)
    return z_real_pred, residual, rmse, nrmse


def plot_kk(freq, z_real, z_real_pred, residual):
    plt.figure(figsize=(8, 6))
    plt.semilogx(freq, z_real, 'o', label='Measured Z´')
    plt.semilogx(freq, z_real_pred, '-', label='KK predicted Z´')
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Z´ / Ω')
    plt.title('Kramers-Kronig Validation')
    plt.legend()
    plt.tight_layout()
    plt.savefig('kk_validation.png', dpi=300)

    plt.figure(figsize=(8, 4))
    plt.semilogx(freq, residual, 'o-')
    plt.xlabel('Frequency / Hz')
    plt.ylabel('Residual / Ω')
    plt.title('KK Residual (Measured - Predicted)')
    plt.tight_layout()
    plt.savefig('kk_residual.png', dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Kramers-Kronig consistency check for EIS data')
    parser.add_argument('file', help='CSV file with columns freq,Zreal,Zimag')
    args = parser.parse_args()

    file_path = Path(args.file)
    freq, z = load_eis_csv(file_path)
    z_real_pred, residual, rmse, nrmse = kk_check(freq, z)

    print(f'KK RMSE: {rmse:.4e} Ω')
    print(f'KK NRMSE: {nrmse*100:.2f} % of dynamic range')

    plot_kk(freq, z.real, z_real_pred, residual)


if __name__ == '__main__':
    main() 