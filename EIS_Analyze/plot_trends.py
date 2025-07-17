import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_parameter_trends(csv_path: Path, param_to_plot: str):
    """
    读取拟合结果CSV文件，并绘制指定参数随循环次数变化的趋势图。

    :param csv_path: batch_fit_results.csv文件的路径
    :param param_to_plot: 要绘制的参数列名 (例如 'p2', 'p3', 'ssr')
    """
    if not csv_path.exists():
        print(f"错误: 结果文件 '{csv_path}' 不存在。")
        print("请先运行 eis_fit_demo.py 来生成拟合结果。")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # 检查必要的列是否存在
    if 'cycle_number' not in df.columns:
        print(f"错误: CSV文件中缺少 'cycle_number' 列。")
        print("请确保您的数据文件名包含 'cycle_xxx' 格式。")
        return
        
    if param_to_plot not in df.columns:
        print(f"错误: CSV文件中缺少参数 '{param_to_plot}' 列。")
        print(f"可用参数包括: {', '.join(df.columns)}")
        return

    # 移除缺少关键数据的行并排序
    df = df.dropna(subset=['cycle_number', param_to_plot]).sort_values(by='cycle_number')

    if df.empty:
        print("数据不足，无法绘制趋势图。")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(df['cycle_number'], df[param_to_plot], 'o-', label=param_to_plot)
    
    plt.title(f'参数 {param_to_plot} 随循环次数的演化趋势')
    plt.xlabel('循环圈数 (Cycle Number)')
    plt.ylabel(f'参数 {param_to_plot} 的值')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_filename = f'trend_{param_to_plot}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"趋势图已保存为: {output_filename}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="绘制EIS拟合参数随循环次数变化的趋势图。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--param',
        type=str,
        required=True,
        help="要绘制的参数名称。例如：\n"
             "p1 (通常是 Rs)\n"
             "p2 (通常是 Rct 或 Rb)\n"
             "ssr (残差平方和)"
    )
    parser.add_argument(
        '--file',
        type=str,
        default='batch_fit_results.csv',
        help="包含批量拟合结果的CSV文件名 (默认为 'batch_fit_results.csv')"
    )
    
    args = parser.parse_args()
    
    csv_file_path = Path(args.file)
    plot_parameter_trends(csv_file_path, args.param)

if __name__ == "__main__":
    main() 