# MLProject：时序异常检测项目

面向时序数据的异常检测任务。给定带标签的训练集 `train.csv` 和两个无标签测试集 `test_simple.csv`、`test_complex.csv`，训练模型并输出二值异常预测 `y_pred`。

**数据规模**：`train.csv` 137,192 行 / `test_simple.csv` 25,647 行 / `test_complex.csv` 34,542 行

**最新版本 (v5)**：消融实验表明多模型集成冗余，最优方案为 **XGBoost Focal 单模型 + 平滑窗口 3**，Test AUC-PR = 0.9891，F1 = 0.9356。三集指标高度一致，无过拟合。

---

## 仓库结构

```text
MLProject/
├── code/                        # 训练脚本
│   ├── train_predict_v2.py          # 早期基线
│   ├── train_predict_v3_fast.py     # 稳定 baseline
│   ├── train_predict_v4_NoRegime.py # v4 主实验线（含 IF/Cascade/Selected）
│   ├── train_predict_v4_Regime(ErrorVersion).py  # 历史实验，保留参考
│   ├── ablation_study.py            # 消融实验（8组配置，含 IF）
│   ├── eval_all_configs.py          # IForest 诊断分析
│   └── experiment_v5.py             # v5 消融（去 IF，17组权重搜索）
├── data/                        # 原始数据
├── docs/
│   ├── handover.md                  # 技术交接文档
│   ├── metric.md                    # 指标说明
│   ├── v4result.md                  # v4 结果摘要
│   ├── experiment_v5_report.md      # v5 实验报告（17组配置详细结果）
│   └── iforest_vs_xgb_scores.png    # IF vs XGBoost 得分分布诊断图
├── notebooks/                   # v3/v4 分析 notebook
├── submission_v2/               # v2 输出
├── submission_v3/               # v3 输出
├── submission_v4_NoRegime/      # v4 主线输出
├── submission_v3notebook/       # notebook 相关输出
├── validation/                  # 辅助验证/复现脚本
├── Project.pdf                  # 题目说明
├── AGENTS.md                    # agent 协作说明
└── README.md
```

---

## 脚本说明

### 主线训练脚本

| 脚本 | 说明 |
|------|------|
| `train_predict_v2.py` | 早期基线，输出到 `submission_v2/` |
| `train_predict_v3_fast.py` | 稳定 baseline，结构简单，无 regime 风险 |
| `train_predict_v4_NoRegime.py` | v4 主线，7子模型集成（XGBstd + XGBfocal + LGB + IF + Cascade + XGBsel + LGBsel） |
| `train_predict_v4_Regime(ErrorVersion).py` | 历史实验，存在 regime_id 风险，不建议使用 |

### 实验/分析脚本

| 脚本 | 说明 |
|------|------|
| `ablation_study.py` | 首次消融实验（8组配置），对比含 IF 的各种组合 |
| `eval_all_configs.py` | IForest 诊断，6 种 IF 配置 + 得分分布对比，验证无数据泄露 |
| `experiment_v5.py` | **最终消融实验**，去掉 IF 后搜索 17 组模型组合与权重 |

### 关键发现（v5 实验）

1. **IForest 不适配** — 316维特征 + 30行连续异常块，IF AUC-PR ≤ 0.074
2. **XGBoost Focal > 标准版** — scale_pos_weight × 2 更重视异常样本
3. **LightGBM 过拟合** — leaf-wise 生长策略在 270 个异常样本上严重过拟合
4. **时间平滑有害** — 窗口 3/5/7 均降低 AUC-PR
5. **多模型集成无益** — 所有含 LGB 或 Selected 的组合在验证集上均不如单模型

**最佳配置**：`XGBoost Focal 单模型 + 平滑窗口 3`，阈值在验证集上选择

---

## 环境依赖

```powershell
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```

---

## 结果摘要

| 版本 | AUC-PR | F1 | 说明 |
|------|--------|----|------|
| v3 | 0.9285 | 0.9292 | 稳定 baseline |
| v4_NoRegime | 0.9723 (CV) | 0.9382 | 7模型集成，含 IF/Cascade |
| **v5 (E15)** | **0.9891** | **0.9356** | XGBoost Focal + smooth3，FP=4, FN=11 |

### v5 推荐配置排名

| 排名 | 配置 | Val AUC-PR | Test F1 | Train F1 |
|------|------|-----------|---------|---------|
| **1** | **E15 B + smooth3** | **0.9830** | **0.9356** | **0.9474** |
| 2 | E17 B 无平滑 | 0.9923 | 0.9569 | 0.7837 |
| 3 | E2 B (XGB focal) | 0.9756 | 0.9333 | 0.9294 |
| 4 | E8 B+C 0.5+0.5 | 0.9698 | 0.9053 | 0.8504 |

详细结果见 [`docs/experiment_v5_report.md`](docs/experiment_v5_report.md)。

---

## 推荐阅读顺序

1. `README.md`
2. `docs/handover.md`
3. `docs/experiment_v5_report.md` — 最新实验结论优先看
4. `notebooks/` 中的分析材料
5. `code/train_predict_v3_fast.py`
6. `code/train_predict_v4_NoRegime.py`

---

## 提交文件格式

每个 `pred_simple.csv` / `pred_complex.csv` 仅含一列：`y_pred`（0 或 1）。`model.pkl` 保存完整模型、阈值与元数据。
