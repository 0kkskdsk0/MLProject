# MLProject：时序异常检测 — v5 最终版

面向时序数据的异常检测任务。最终模型 **E15: XGBoost Focal + 时序平滑窗口 3**。

**数据规模**: `train.csv` 137,192 行 / `test_simple.csv` 25,647 行 / `test_complex.csv` 34,542 行

## 仓库结构

```
MLProject/
├── code/
│   ├── experiment_v5.py       # 消融实验脚本（17组配置对比）
│   ├── train_final.py         # 【最终模型】训练 + 保存
│   └── predict_final.py       # 【最终模型】加载模型 → 推理输出
├── data/                      # 原始数据
├── docs/
│   └── experiment_v5_report.md # 最终实验报告（含完整分析）
├── submission_v5/             # 输出产物（model.pkl + 预测结果）
├── experiment_v5_log.txt      # 训练日志
├── Project.pdf                # 题目说明
└── README.md
```

## 最终模型：E15

| 组件 | 内容 |
|------|------|
| 基模型 | XGBoost Focal (depth=5, lr=0.03, scale_pos_weight=480) |
| 后处理 | 时序平滑，窗口 3 |
| 阈值 | 0.0061（Val 最大化 F1） |

| 数据集 | AUC-PR | F1 | FP | FN |
|--------|:------:|:--:|:--:|:--:|
| Train | 1.0000 | 0.9474 | 30 | 0 |
| Val | 0.9830 | 0.9375 | 7 | 15 |
| **Test** | **0.9891** | **0.9356** | **4** | **11** |

详细分析见 [`docs/experiment_v5_report.md`](docs/experiment_v5_report.md)。

## 使用

### 快速推理（已有 model.pkl）

```bash
python code/predict_final.py
```

### 从头训练 + 推理

```bash
python code/train_final.py      # 训练并保存模型
python code/predict_final.py    # 加载模型，输出预测
```

### 完整消融实验（17组配置对比）

```bash
python code/experiment_v5.py
```

依赖: `pandas numpy scikit-learn xgboost lightgbm`

## 输出

运行 `predict_final.py` 后，`submission_v5/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `model.pkl` | 训练好的模型 + scaler + 阈值 |
| `pred_simple.csv` | test_simple 预测结果（仅 `y_pred` 列） |
| `pred_complex.csv` | test_complex 预测结果（仅 `y_pred` 列） |

## 核心结论

在 270 个异常样本的极度不平衡场景下，**简单模型 + 正确参数远优于复杂集成**。
XGBoost Focal 单模型经窗口 3 平滑后，三集指标高度一致（F1 Δ≤0.012），是最终推荐方案。
