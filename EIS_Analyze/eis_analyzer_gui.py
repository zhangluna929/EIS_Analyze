import PySimpleGUI as sg
import pandas as pd
from pathlib import Path
import threading
import queue

# 从我们自己的模块中导入功能
from models import MODEL_DICT
from eis_fit_demo import run_analysis, setup_logging

def analysis_thread(input_path_str, model_key, log_queue):
    """在单独的线程中运行EIS分析，以避免GUI冻结。"""
    setup_logging(log_queue)
    input_path = Path(input_path_str)
    all_results = []

    try:
        if input_path.is_file():
            if input_path.suffix.lower() == '.csv':
                result = run_analysis(input_path, model_key)
                if result:
                    all_results.append(result)
            else:
                print(f"错误: 输入文件 '{input_path.name}' 不是.csv文件。")

        elif input_path.is_dir():
            print(f"--- 开始在文件夹 '{input_path.name}' 中进行批量处理 ---")
            csv_files = sorted(input_path.glob('*.csv'))
            if not csv_files:
                print("未在文件夹中找到任何 .csv 文件。")
            
            for file_path in csv_files:
                result = run_analysis(file_path, model_key)
                if result:
                    all_results.append(result)
        else:
            print(f"错误: 输入路径 '{input_path}' 不是一个有效的文件或文件夹。")

        if all_results:
            results_df = pd.DataFrame(all_results)
            if 'cycle_number' in results_df.columns:
                results_df = results_df.sort_values(by='cycle_number').reset_index(drop=True)
            
            output_csv_path = 'batch_fit_results.csv'
            results_df.to_csv(output_csv_path, index=False)
            print(f"\n--- 所有分析完成 ---")
            print(f"拟合结果已保存到: {output_csv_path}")

    except Exception as e:
        print(f"分析过程中发生严重错误: {e}")
    finally:
        log_queue.put('---THREAD_DONE---')

def main():
    sg.theme('LightGrey1')

    layout = [
        [sg.Text('EIS分析工具', font=('Helvetica', 16))],
        [sg.HSeparator()],
        [sg.Text('选择数据文件或文件夹:'), sg.InputText(key='-INPUT_PATH-'), sg.FolderBrowse('浏览文件夹'), sg.FileBrowse('浏览文件')],
        [sg.Text('选择等效电路模型:'), sg.Combo(list(MODEL_DICT.keys()), default_value='B', key='-MODEL_KEY-', readonly=True)],
        [sg.Button('开始分析', key='-RUN-'), sg.Button('退出', key='-EXIT-')],
        [sg.HSeparator()],
        [sg.Text('日志输出:')],
        [sg.Multiline(size=(80, 20), key='-LOG-', autoscroll=True, disabled=True)],
    ]

    window = sg.Window('EIS分析器', layout)
    
    current_thread = None

    while True:
        event, values = window.read(timeout=100) # 使用超时来定期刷新日志

        if event == sg.WIN_CLOSED or event == '-EXIT-':
            break

        # 从队列中读取日志并更新GUI
        if current_thread:
            try:
                log_message = log_queue.get_nowait()
                if log_message == '---THREAD_DONE---':
                    current_thread.join()
                    current_thread = None
                    sg.popup('分析完成!', '所有任务已执行完毕。')
                    window['-RUN-'].update(disabled=False)
                else:
                    window['-LOG-'].update(log_message, append=True)
            except queue.Empty:
                pass

        if event == '-RUN-':
            input_path = values['-INPUT_PATH-']
            model_key = values['-MODEL_KEY-']

            if not input_path:
                sg.popup_error('错误', '请先选择一个数据文件或文件夹！')
                continue

            window['-LOG-'].update('') # 清空日志
            window['-RUN-'].update(disabled=True)

            log_queue = queue.Queue()
            current_thread = threading.Thread(
                target=analysis_thread,
                args=(input_path, model_key, log_queue),
                daemon=True
            )
            current_thread.start()

    window.close()

if __name__ == '__main__':
    main() 