# 电化学阻抗谱综合分析平台 / Comprehensive Electrochemical Impedance Spectroscopy Analysis Suite

Author: LunaZhang

---

## 1. 项目介绍 | Project Motivation

在固态锂电池及相关储能器件的研发过程中，电化学阻抗谱（Electrochemical Impedance Spectroscopy, EIS）是揭示界面、电荷传输与扩散多尺度过程的核心表征手段。然而，实验数据散落于不同仪器格式，后期分析缺乏统一、可信且可扩展的开源工具。本平台以严谨的数理框架和工业级的软件工程体系，提供从数据解析、质量检验、等效电路拟合、机理分离到健康状态（SOH）预测的“一站式”解决方案。

Electrochemical Impedance Spectroscopy is indispensable for deciphering interfacial, charge-transfer and diffusion processes in solid-state lithium batteries. Experimental records are dispersed across heterogeneous instrument formats, and post-processing often lacks a unified, trustworthy and extensible toolkit. This suite delivers an end-to-end workflow covering data parsing, quality validation, equivalent-circuit fitting, mechanistic deconvolution and state-of-health prognosis.

---

## 2. 关键特性 | Key Features

1. 多格式解析：原生支持 `.csv`、BioLogic `.mpt`、Autolab `.txt`，并保留插件接口。
2. 等效电路模型库：内置 A–E 五类模型，覆盖固态电解质晶界、电荷转移与 Warburg 扩散元件，支持自定义扩展。
3. 稳健拟合算法：局部 Levenberg-Marquardt 与全局 Differential Evolution 双引擎，AIC 自动模型优选。
4. DRT 与 K-K 校验：Tikhonov 正则化反演弛豫时间分布，Kramers-Kronig 一致性检验量化数据可靠性。
5. 批量与趋势分析：文件夹级自动处理，汇总 `batch_fit_results.csv` 并绘制参数演化曲线。
6. 机器学习 SOH 预测：梯度提升树映射阻抗参数-容量，输出置信度并保存模型。
7. 交互式 GUI：PySimpleGUI 前端零门槛操作，同时保留 CLI 适配自动化脚本。
8. 自动化报告：纯 Python HTML 渲染，生成自包含报告用于归档或打印。

---

## 3. 架构 | Repository Layout
```
models.py            # 等效电路模型
parsers.py           # 多格式数据解析
EIS_Analyze/
 ├─ eis_fit_demo.py  # 核心拟合与批量处理
 ├─ kk_validation.py # K-K 检验
 ├─ drt_analysis.py  # DRT 反演
 ├─ soh_ml.py        # SOH 机器学习
 ├─ plot_trends.py   # 参数趋势
 ├─ generate_report.py
 └─ eis_analyzer_gui.py
```

---

## 4. 快速入门 | Quick Start
```bash
pip install numpy pandas matplotlib scipy scikit-learn PySimpleGUI joblib
python eis_analyzer_gui.py                       # 图形界面
python eis_fit_demo.py data E --auto             # 批量拟合并自动选型
python plot_trends.py --param p2                 # 参数趋势图
python kk_validation.py sample.mpt               # K-K 校验
python drt_analysis.py sample.txt --ntau 120
python soh_ml.py --batch batch_fit_results.csv --capacity capacity.csv
python generate_report.py                        # 自动报告
```

---

## 5. 理论与实现 | Theory & Implementation Highlights

* 参数初值：Nyquist 高频截距估算 Rs，半圆跨度推得 Rct/Rb，CPE 指数初始化 0.8–0.9。
* 残差计算：实、虚部拼接保证权重均衡。
* 模型选择：Akaike 信息准则抑制过拟合。
* DRT 核：`1/(1+(ωτ)²)` 与 `ωτ/(1+(ωτ)²)`，非负约束 + Tikhonov 正则。
* K-K 数值积分：梯形积分输出归一化 RMSE。

---

## 6. 未来展望 | Outlook

计划加入 GPU 加速 DRT、温度-阻抗耦合、OPC-UA 产线集成与 LaTeX 自动成稿，进一步服务科研与工业场景。

---

> 本项目全部源码由 LunaZhang 编写与维护；若引用请注明出处。
