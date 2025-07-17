import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear


def load_eis_csv(file_path: Path):
    df = pd.read_csv(file_path)
    freq = df['freq'].values
    z = df['Zreal'].values + 1j * df['Zimag'].values
    return freq, z


def build_kernel(freq, tau):
    w = 2 * np.pi * freq[:, None]
    tau_mat = tau[None, :]
    K_real = 1 / (1 + (w * tau_mat)**2)
    K_imag = (w * tau_mat) / (1 + (w * tau_mat)**2)
    return K_real, K_imag


def solve_drt(freq, z, n_tau=100, lam=0.01):
    # log-spaced relaxation times (s)
    tau = np.logspace(-6, 4, n_tau)
    K_real, K_imag = build_kernel(freq, tau)

    # concatenate kernels and data (real & imag)
    A = np.vstack([K_real, -K_imag])
    b = np.hstack([z.real, z.imag])

    # Tikhonov regularization matrix
    L = lam * np.eye(n_tau)
    A_reg = np.vstack([A, L])
    b_reg = np.hstack([b, np.zeros(n_tau)])

    res = lsq_linear(A_reg, b_reg, bounds=(0, np.inf), lsmr_tol='auto')
    gamma = res.x
    return tau, gamma, res.cost


def plot_drt(tau, gamma):
    plt.figure(figsize=(6,4))
    plt.semilogx(tau, gamma, '-o')
    plt.xlabel('Relaxation time τ / s')
    plt.ylabel('DRT γ(τ)')
    plt.title('Distribution of Relaxation Times')
    plt.tight_layout()
    plt.savefig('drt_result.png', dpi=300)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='DRT analysis for EIS data via Tikhonov regularization')
    parser.add_argument('file', help='CSV file with columns freq,Zreal,Zimag')
    parser.add_argument('--ntau', type=int, default=100, help='Number of tau points')
    parser.add_argument('--lam', type=float, default=0.01, help='Regularization parameter lambda')
    args = parser.parse_args()

    freq, z = load_eis_csv(Path(args.file))
    tau, gamma, cost = solve_drt(freq, z, n_tau=args.ntau, lam=args.lam)

    print(f'DRT solved with cost: {cost:.4e}')
    plot_drt(tau, gamma)


if __name__ == '__main__':
    main() 