import numpy as np

def Z_cpe(Q, n, w): return 1 / (Q * (1j * w) ** n)  

def Z_w(sigma, w): return (sigma / np.sqrt(w)) * (1 - 1j)  

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

def model_E(p, w):
    """模型 E: R_s + (R_b ‖ CPE_b) + (R_ct ‖ CPE_dl) - 固态电解质模型"""
    R_s, R_b, Q_b, n_b, R_ct, Q_dl, n_dl = p
    Z_b = 1 / (1 / R_b + 1 / Z_cpe(Q_b, n_b, w))
    Z_ct = 1 / (1 / R_ct + 1 / Z_cpe(Q_dl, n_dl, w))
    return R_s + Z_b + Z_ct

MODEL_DICT = {
    "A": (model_A, [1, 100, 1e-5], ([0, 0, 0], [np.inf, np.inf, np.inf])),
    "B": (model_B, [1, 100, 1e-4, 0.8], ([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1])),
    "C": (model_C, [1, 50, 1e-4, 0.9, 200, 1e-5, 0.7], ([0, 0, 0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 1, np.inf, np.inf, 1])),
    "D": (model_D, [1, 100, 1e-4, 0.8, 10], ([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, 1, np.inf])),
    "E": (model_E, [1, 50, 1e-5, 0.9, 200, 1e-4, 0.8], ([0]*7, [np.inf, np.inf, np.inf, 1, np.inf, np.inf, 1])),
} 