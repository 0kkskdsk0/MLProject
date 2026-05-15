# MLProject：时序异常检测 — v5 最终版

面向时序数据的异常检测任务。给定带标签的训练集 `train.csv` 和两个无标签测试集 `test_simple.csv`、`test_complex.csv`，训练模型并输出二值异常预测 `y_pred`。

**数据规模**：`train.csv` 137,192 行 / `test_simple.csv` 25,647 行 / `test_complex.csv` 34,542 行

## 仓库结构

```
MLProject/
├── code/
│   └── experiment_v5.py          # 训练脚本（v5 消融实验，17组配置）
├── data/                         # 原始数据
├── docs/
│   └── experiment_v5_report.md   # 详细实验报告
├── submission_v5/                # v5 产物与报告
├── experiment_v5_log.txt         # 训练日志
├── Project.pdf                   # 题目说明
└── README.md
```

## 核心方法

**XGBoost Focal 单模型** — 消融实验表明多模型集成冗余。

特征工程（310维）：
- 原始 33 维特征
- 滚动窗口 (5/10/20) 均值与标准差
- 一阶/五阶差分特征
- 滞后特征 (lag1, lag3)
- 前 3 特征两两交互
- 行统计量 (mean, std, max, min)

## 结果摘要

| 配置 | AUC-PR | F1 | FP | FN |
|------|:------:|:--:|:--:|:--:|
| **E15 B + smooth3（推荐）** | **0.9891** | **0.9356** | **4** | **11** |
| E17 B + nosmooth（备选） | 0.9974 | 0.9569 | 1 | 9 |
| E2 B 单模型 | 0.9825 | 0.9333 | 8 | 8 |

## 使用

```bash
cd MLProject
python code/experiment_v5.py
```

详细结果见 `docs/experiment_v5_report.md`。
