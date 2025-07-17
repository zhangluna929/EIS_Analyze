import sys
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from models import MODEL_DICT
from parsers import load_eis
from scipy.optimize import differential_evolution

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

def residuals(p, w, z_exp, model_func):
    z_mod = model_func(p, w)
    return np.concatenate([(z_mod.real - z_exp.real), (z_mod.imag - z_exp.imag)])

def plot_eis(freq, z_exp, z_fit, fit_residuals, **kwargs):
    """
    绘制Nyquist图、Bode图以及残差图。
    :param freq: 频率数据
    :param z_exp: 实验数据
    :param z_fit: 拟合数据
    :param fit_residuals: 拟合残差
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('EIS 拟合与残差分析', fontsize=16)

    # Nyquist图
    ax1 = axs[0, 0]
    ax1.plot(z_exp.real, -z_exp.imag, 'o', ms=4, label='实验数据')
    ax1.plot(z_fit.real, -z_fit.imag, '-', lw=2, label='拟合数据')
    ax1.set_xlabel('Z´ / Ω')
    ax1.set_ylabel('-Z" / Ω')
    ax1.set_title('Nyquist 图')
    ax1.legend()
    ax1.axis('equal')

    # Bode |Z| 图
    ax2 = axs[0, 1]
    ax2.semilogx(freq, np.abs(z_exp), 'o', ms=4)
    ax2.semilogx(freq, np.abs(z_fit), '-', lw=2)
    ax2.set_xlabel('频率 / Hz')
    ax2.set_ylabel('|Z| / Ω')
    ax2.set_title('Bode |Z| 图')

    # Bode Phase 图
    ax3 = axs[1, 0]
    ax3.semilogx(freq, np.angle(z_exp, deg=True), 'o', ms=4)
    ax3.semilogx(freq, np.angle(z_fit, deg=True), '-', lw=2)
    ax3.set_xlabel('频率 / Hz')
    ax3.set_ylabel('相位 / °')
    ax3.set_title('Bode Phase 图')
    
    # 残差图
    ax4 = axs[1, 1]
    res_real = fit_residuals[:len(freq)]
    res_imag = fit_residuals[len(freq):]
    ax4.semilogx(freq, res_real, 'o', ms=4, label='实部残差 (Z´$_{exp}$-Z´$_{fit}$)')
    ax4.semilogx(freq, res_imag, 's', ms=4, label='虚部残差 (Z"$_{exp}$-Z"$_{fit}$)')
    ax4.set_xlabel('频率 / Hz')
    ax4.set_ylabel('残差 / Ω')
    ax4.set_title('拟合残差图')
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 从输入文件名生成输出文件名
    input_filename = kwargs.get('input_filename', 'fit')
    output_filepath = Path(input_filename).stem + '_analysis.png'
    
    plt.savefig(output_filepath, dpi=300)
    plt.close() # 关闭图像，防止在批量处理时显示出来

def guess_initial_params(freq, z_exp, model_key):
    """
    根据模型和实验数据估算初始参数。
    :param freq: 频率
    :param z_exp: 复数阻抗
    :param model_key: 模型标识
    :return: 估算的初始参数列表
    """
    # 估算 Rs: 取最高频阻抗的实部
    p0 = []
    Rs_guess = z_exp[np.argmax(freq)].real

    model_func, p_template, bounds = MODEL_DICT[model_key]

    if model_key == 'A':
        R_ct_guess = (np.max(z_exp.real) - Rs_guess)
        C_dl_guess = 1e-5 
        p0 = [Rs_guess, R_ct_guess, C_dl_guess]
    elif model_key == 'B':
        R_ct_guess = (np.max(z_exp.real) - Rs_guess)
        Q_guess = 1e-4
        n_guess = 0.8
        p0 = [Rs_guess, R_ct_guess, Q_guess, n_guess]
    # 对其他模型使用默认值
    else:
        p0 = p_template
        p0[0] = Rs_guess # 至少更新Rs

    # 保证初始值在边界内
    lb, ub = bounds
    p0 = np.maximum(p0, lb)
    p0 = np.minimum(p0, ub)
    
    return p0


def run_analysis(file_path: Path, model_key: str):
    """
    对单个文件执行完整的EIS分析和拟合。
    """
    print(f"\n--- 分析文件: {file_path.name} ---")
    try:
        freq, z_exp = load_eis(file_path)
    except Exception as e:
        print(f"无法处理文件 {file_path.name}: {e}")
        return None

    w = 2 * np.pi * freq

    if model_key not in MODEL_DICT:
        print(f"错误: 模型 '{model_key}' 不存在。")
        return None

    if args.auto:
        best_aic = np.inf
        best = None
        for mk, (mf, ptemp, bnds) in MODEL_DICT.items():
            p0 = guess_initial_params(freq, z_exp, mk)
            res = least_squares(residuals, p0, args=(w, z_exp, mf), bounds=bnds, max_nfev=4000)
            k = len(res.x)
            ssr = np.sum(res.fun**2)
            aic = len(w)*np.log(ssr/len(w)) + 2*k
            if aic < best_aic:
                best_aic, best = aic, (mk, mf, res, bnds)
        model_key, model_func, res, bounds = best
        z_fit = model_func(res.x, w)
    else:
        model_func, _, bounds = MODEL_DICT[model_key]
        p0 = guess_initial_params(freq, z_exp, model_key)
        if args.global_opt:
            def obj(p):
                return np.sum(residuals(p, w, z_exp, model_func)**2)
            res_de = differential_evolution(obj, bounds=list(zip(*bounds)))
            p0 = res_de.x
        res = least_squares(residuals, p0, args=(w, z_exp, model_func), bounds=bounds, max_nfev=8000)
    z_fit = model_func(res.x, w)

    ssr = np.sum(res.fun**2)

    print("拟合参数：")
    for i, param in enumerate(res.x, 1):
        print(f"p{i} = {param:.4e}")
    print(f"残差平方和 (SSR): {ssr:.4e}")

    final_residuals = residuals(res.x, w, z_exp, model_func)
    plot_eis(freq, z_exp, z_fit, final_residuals, input_filename=str(file_path))
    
    # 结果字典
    results = {'filename': file_path.name, 'model': model_key, 'ssr': ssr}

    # 从文件名提取循环次数
    match = re.search(r'cycle_(\d+)', file_path.name, re.IGNORECASE)
    if match:
        results['cycle_number'] = int(match.group(1))
    
    for i, param in enumerate(res.x, 1):
        results[f'p{i}'] = param
    return results

def setup_logging(log_queue=None):
    """
    如果提供了队列，则重定向print到一个自定义的处理器。
    """
    if log_queue:
        # 重定向stdout
        class QueueIO:
            def __init__(self, queue):
                self.queue = queue
            def write(self, text):
                self.queue.put(text)
            def flush(self):
                pass
        sys.stdout = QueueIO(log_queue)
        sys.stderr = QueueIO(log_queue)

def parse_cli():
    parser = argparse.ArgumentParser(description='EIS fitting utility')
    parser.add_argument('input', help='data file or folder')
    parser.add_argument('model', nargs='?', default='B', help='model key A-E, or auto when --auto')
    parser.add_argument('--global-opt', action='store_true', help='use differential evolution global optimizer')
    parser.add_argument('--auto', action='store_true', help='try all models and choose best AIC')
    return parser.parse_args()


def main():
    args = parse_cli()

    input_path = Path(args.input)
    model_key = args.model.upper()
    
    all_results = []

    if input_path.is_file():
        if input_path.suffix.lower() == '.csv':
            result = run_analysis(input_path, model_key)
            if result:
                all_results.append(result)
        else:
            print(f"错误: 输入文件 '{input_path.name}' 不是.csv文件。")
            sys.exit(1)
            
    elif input_path.is_dir():
        print(f"--- 开始在文件夹 '{input_path.name}' 中进行批量处理 ---")
        csv_files = sorted(input_path.glob('*.csv'))
        if not csv_files:
            print("未在文件夹中找到任何 .csv 文件。")
            sys.exit(0)
            
        for file_path in csv_files:
            result = run_analysis(file_path, model_key)
            if result:
                all_results.append(result)
    else:
        print(f"错误: 输入路径 '{input_path}' 不是一个有效的文件或文件夹。")
        sys.exit(1)
        
    if all_results:
        results_df = pd.DataFrame(all_results)
        # 如果包含cycle_number，则进行排序
        if 'cycle_number' in results_df.columns:
            results_df = results_df.sort_values(by='cycle_number').reset_index(drop=True)
            
        output_csv_path = 'batch_fit_results.csv'
        print(f"\n--- 所有分析完成 ---")
        print(f"拟合结果已保存到: {output_csv_path}")
        results_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()
