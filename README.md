# EIS-Fit-Demo  
**电化学阻抗谱自动拟合工具 / Automated EIS Fitting Utility**

> 一个轻量级 Python 脚本，用于读取阻抗谱 (`freq, Z′, Z″`) 数据，选择常见等效电路模型，并自动拟合输出 Nyquist 与 Bode 图。  
> A lightweight Python script that loads impedance‐spectroscopy data (`freq, Z′, Z″`), fits it with selected equivalent‐circuit models, and exports Nyquist and Bode plots.

---

## 1. 项目简介 | Project Overview
- **中文**：实验室常用的 EIS 数据往往需要手动导入商业软件才能拟合。本脚本完全开源，依赖 SciPy 最小二乘算法即可完成 Rs、Rct、CPE 等参数提取。  
- **English**: Commercial EIS software can be expensive or restrictive. This open-source script uses SciPy’s least-squares engine to extract Rs, Rct, CPE and other parameters directly from CSV data.

---

## 2. 支持的等效电路 | Supported Equivalent-Circuit Models

| 代码 | Chinese 描述 | English Description |
|------|-------------|---------------------|
| **A** | Rs + (Rct‖C) | Classic Randles (Rs + parallel Rct & capacitor) |
| **B** | Rs + (Rct‖CPE) | Rs with Rct parallel CPE (non-ideal DL) |
| **C** | Rs + (R₁‖CPE₁) + (R₂‖CPE₂) | Two semicircles (typical for coatings or composite electrodes) |
| **D** | Rs + (Rct‖CPE) + W | Adds Warburg element for diffusion control |

> 需要更多模型,只需新增 `model_X()` 函数并在 `MODEL_DICT` 注册。  
> Need extra circuits? Add a `model_X()` function and register it in `MODEL_DICT`.

---

## 3. 安装依赖 | Installation

```bash
pip install numpy pandas matplotlib scipy
```
---

## 4.数据格式 | Data Format

freq,Zreal,Zimag
10000,2.31,-0.12
8000,2.40,-0.14
⋯

freq：频率 / Hz

Zreal：阻抗实部 / Ω

Zimag：阻抗虚部 / Ω（正负皆可，脚本可以自动处理哦~）

列名区分大小写，如需修改请编辑 load_eis_csv()。
Column names are case-sensitive; adjust them in load_eis_csv() if necessary.

## 5. 快速使用 | Quick Start
### 运行示例（B 模型：Rs + (Rct‖CPE)）
python eis_fit_demo.py eis_demo.csv B

拟合参数

p1 = 1.0210e+00

p2 = 5.1276e+01

p3 = 1.0340e-04

p4 = 7.9500e-01

自动生成 

Auto-generated files

nyquist_fit.png

bode_mag_fit.png

bode_phase_fit.png

## 7.未来计划 | Roadmap
NEWARE / Autolab 原始格式解析

图形界面（Tkinter / PySimpleGUI）

多线程批量拟合 + 汇总报告