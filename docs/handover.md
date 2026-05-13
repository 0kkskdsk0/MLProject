# Robust Anomaly Detection in Noisy Time-Series Data - 技术交接文档

**仓库路径**: `G:\MLProject\MLProject`  
**最后更新**: 2026-05-13  
**当前建议阅读顺序**: 先看本文档，再大致浏览 `notebooks/`理解v3v4版本模型构建细节，最后按需要进入 `code/` 和 `validation/`

---

## 1. 项目背景与目标

本项目面向**嘈杂时序数据中的鲁棒异常检测**。训练集来自一段按时间排序的金融市场观测序列，目标是在极度不平衡、噪声较强、且存在分布漂移的前提下，对两个测试集输出 `y_pred` 二值预测。

核心难点：
- **高噪声**：原始观测波动大，异常信号容易被局部噪声掩盖。
- **极度不平衡**：训练集仅 `570 / 137192` 为异常，约 `0.415%`。
- **强时序性**：异常更像持续片段或局部模式，而不是孤立点。
- **Task 2 分布漂移**：`test_complex.csv` 相比训练集更容易出现阈值偏移和泛化退化。

主要评估指标：
- **AUC-PR**：本任务最重要，适合极度不平衡数据。
- **F1**：用于阈值化后的整体精确率/召回率平衡。
- **AUC-ROC**：辅助看排序能力，不作为唯一判断依据。

---

## 2. 数据与文件

### 2.1 数据文件

| 文件 | 路径 | 行数 | 说明 |
|------|------|------|------|
| 训练集 | `data/train.csv` | 137,192 | 含标签 `y`，`0=正常, 1=异常` |
| 测试集1 | `data/test_simple.csv` | 25,647 | 同分布测试，无标签 |
| 测试集2 | `data/test_complex.csv` | 34,542 | 更复杂场景，无标签 |
| 项目说明 | `Project.pdf` | - | 题目要求 |

### 2.2 Schema

```text
f1 ~ f33 : 33 个数值特征
y        : 仅 train.csv 中存在，二分类标签
```

补充事实：
- 训练集共有 `570` 个异常，集中在时间序列尾部。
- 测试集只需要提交单列 `y_pred`。
- 缺失值很少，主要集中在训练集部分 `f22 ~ f29`。

---

## 3. 当前仓库结构

```text
MLProject/
├── code/
│   ├── train_predict_v2.py
│   ├── train_predict_v3_fast.py
│   ├── train_predict_v4_NoRegime.py
│   └── train_predict_v4_Regime.py
├── data/
├── docs/
│   ├── handover.md
│   ├── metric.md
│   └── v4result.md
├── notebooks/
│   ├── v3Notebook/v3model.ipynb #详细讲解v3版本模型的细节和结果可视化
│   └── v4Notebook/v4model.ipynb #详细讲解v3版本模型的细节和结果可视化
├── submission/                  # v2 产物
├── submission_v3/               # 当前 no-regime v3 产物
├── submission_v4_NoRegime/      # 当前 no-regime v4 产物
└── validation/                  # 验证与辅助脚本，并不重要，agent可以按需取用
```

---

## 4. 版本总览

这一节是给人和 agent 最重要的入口。每个版本都写清楚：脚本、特征工程、模型结构、集成方式、当前状态。

### 4.1 v2 基线

- **脚本**: `code/train_predict_v2.py`
- **输出目录**: `submission/`
- **定位**: 早期可运行基线，不建议继续迭代。

特征工程：
- rolling window 统计
- 差分特征
- 滞后特征
- 一部分交叉特征

模型组成：
- XGBoost
- LightGBM
- Isolation Forest

问题：
- 训练/验证切分非常差，验证集异常占比远高于训练集。
- 指标较弱，主要价值是提供最早端到端模板。

### 4.2 v3 当前稳定基线

- **脚本**: `code/train_predict_v3_fast.py`
- **输出目录**: `submission_v3/`
- **定位**: 当前推荐 baseline；结构简单、可解释、已去除 `regime_id` 风险。

特征工程：
- 原始 33 维特征
- rolling mean / std，窗口 `5/10/20`
- 一阶/五阶差分
- 前 3 个特征的 lag `1/3`
- 前 3 个特征两两交互
- LOF 异常分数
- PCA 5 维主成分
- 行统计：`row_mean / row_std / row_max / row_min`

当前特征维度：
- **316 维**
- 无`regime_id` 特征

模型组成：
- XGBoost
- LightGBM
- Isolation Forest

集成策略：
- `0.35 * XGB + 0.35 * LGBM + 0.30 * IF`

关键风险修复：
- 已删除 `regime_id` 特征，避免 train/val/test 编号语义不一致。
- 当前 LOF/PCA 在 v3 中仍按训练切分拟合，属于安全版本。

### 4.3 v4_Regime 实验版

- **脚本**: `code/train_predict_v4_Regime.py`
- **定位**: 保留原始 v4 full 思路，用于对照，不建议直接作为当前可信主线。

在 v3 基础上新增：
- 两折时序 CV
- 第二个 “focal 风格” XGBoost 分支
- Cascade：IF 粗筛 + XGBoost refine
- 时间平滑 `temporal_smooth`
- 孤立点过滤 `apply_temporal_consistency`
- Top-100 特征选择后再训练轻量子模型
- Task 2 的 `adaptive_threshold`

主要问题：
- 该版本的`regime_id` 特征构造不正确，会导致训练集和验证集的特征语义不对齐
- LOF/PCA 原逻辑在 CV 前全局拟合，存在信息泄露风险
- 因此它是“思路保留版”，不是当前可信版本，如果需要增加合理的regime特征可以优先在该版本上开刀

### 4.4 v4_NoRegime 当前最优版本

- **脚本**: `code/train_predict_v4_NoRegime.py`
- **输出目录**: `submission_v4_NoRegime/`
- **定位**: 当前 v4 主实验线；保留 v4 的复杂结构，同时修复主要风险。

特征工程：
- 与当前 v3 基本一致
- 不再构造 `regime_id`
- 当前特征维度同样为 **316 维**

模型组成：
- `xgb_std`
- `xgb_focal`
- `lgb`
- `iforest`
- `cascade`
- `xgb_std_sel`
- `lgb_sel`

集成策略：
- 基础分支：`0.25 * xgb_std + 0.25 * xgb_focal + 0.25 * lgb + 0.25 * iforest`
- 若 cascade 可用：`0.7 * base + 0.3 * cascade`
- 若 selected-feature 子模型可用：  
  `0.8 * base + 0.2 * (0.5 * xgb_std_sel + 0.5 * lgb_sel)`
- 最后统一经过 `temporal_smooth(window=5)`
- 二值化后再经过 `apply_temporal_consistency`

关键风险修复：
- 删除 `regime_id`
- 删除 Task 2 的 `adaptive_threshold`
- LOF/PCA 改为：
  - 每个 CV fold 只在该 fold 的训练切分上拟合
  - 最终模型只在最终训练段上拟合
- 这样避免验证集或测试集信息参与特征变换器训练

---

## 5. 当前指标与产物

详细数字以 `docs/metric.md` 为准，这里只给摘要。

### 5.1 v2
- 数据集切分方式
    - 验证集“最后 10K 行，含 480 个异常”
    - 训练集“前 127K 行，仅 90 个异常”
    - 训练集和验证集的异常分布严重不匹配。
- AUC-PR: `0.2634`
- F1: `0.5126`
- AUC-ROC: `0.7455`
- Task1 异常率: `3.54% (907/25647)`
- Task2 异常率: `2.70% (933/34542)`

### 5.2 v3
- 数据集切分方式
    - 训练集：前 134,035 行，含 450 个异常，异常率约 0.34%
    - 验证集：后 3,157 行，含 120 个异常，异常率约 3.80%
- AUC-PR: `0.9285`
- F1: `0.9292`
- AUC-ROC: `0.9781`
- Task1 异常率: `3.44% (883/25647)`
- Task2 异常率: `1.68% (580/34542)`

说明：
- `docs/metric.md` 中的 v3 指标沿用了历史表格；
- 当前 `submission_v3/` 对应的是已经去掉 `regime_id` 的 no-regime v3；
- 当前产物里的 `model.pkl` 已确认 `feature_count = 316` 且不含 `regime_id`

### 5.3 v4_NoRegime
- 数据集切分方式
  - CV切分：两折固定时序切分
      - [:131680] / [131680:]
      - [:134035] / [134035:]
  - 每折阈值确定：
    在该折验证集上，对 ensemble_smooth 跑 precision_recall_curve，选 F1 最大 的阈值
  - 每折评估 F1 用的阈值：
    该折自己验证集上找到的 best_thresh，并且二值化后还会经过
    apply_temporal_consistency
  - 最终测试集预测用的阈值：
    F1 最好的那一折的阈值
- Fold1: `AUC-PR=0.9579`, `F1=0.9268`, `threshold=0.1721`
- Fold2: `AUC-PR=0.9868`, `F1=0.9496`, `threshold=0.2040`
- CV Avg AUC-PR: `0.9723`
- CV Avg F1: `0.9382`
- AUC-ROC: 0.9993
- Task1 异常率: `3.34% (857/25647)`
- Task2 异常率: `1.86% (643/34542)`

补充：
- 结果摘要见 `docs/v4result.md`
- 当前 `model.pkl` 已确认 `feature_count = 316` 且不含 `regime_id`

---

## 6. 关键风险与已完成修复

### 6.1 `regime_id` 语义错位问题

问题：
- 原 notebook / 原 v4 的 local regime 做法是每个数据集自己从 0 开始编号；
- 这样 `train` 中的 `regime_id=0` 和 `val/test` 中的 `regime_id=0` 不是同一个语义；
- 树模型会把它当强特征，导致表现剧烈波动。

处理：
- 当前 v3 和 `train_predict_v4_NoRegime.py` 都已删除 `regime_id` 特征。

### 6.2 LOF / PCA 信息泄露问题

问题：
- 原 v4 曾在 CV 前对整份 `train_df` 拟合 LOF/PCA，再进入各 fold；
- 这会让验证集分布信息提前进入特征变换器。

处理：
- 当前 `train_predict_v4_NoRegime.py` 中，LOF/PCA 已改为 fold 内训练切分拟合。

### 6.3 Notebook 与主脚本不一致

问题：
- 历史 notebook 曾重新训练模型并给出与脚本不一致的结果；
- markdown 解读也有引用旧指标的问题。

处理：
- `validation/update_model_demo_notebook.py` 已把 v3 notebook 改为 no-regime 逻辑；
- 当前 notebook 代码不再保留旧输出。

---

## 7. 失败实验与经验

### 7.1 单独使用 Isolation Forest

- 对 Task 2 有一定鲁棒性
- 但单模型 AUC-PR 明显不如集成
- 结论：更适合作为特征或辅助分支，而不是单独交付模型

### 7.2 单独使用 LOF

- 高维下距离度量不稳定
- 作为特征有效，作为最终模型效果弱

### 7.3 使用不稳定的分段编号特征

- 这是这轮最重要的教训之一
- `regime_id` 如果不是跨 train/val/test 共享稳定语义，就会成为有害特征
- 它不是“普通无效特征”，而是“语义反转特征”

---

## 8. validation 目录说明

这些文件是给同伴和 agent 继续实验时最有用的索引。

| 文件 | 作用 |
|------|------|
| `validation/validate_submission_v3_pkl.py` | 验证 `submission_v3/model.pkl` 是否能复现历史指标 |
| `validation/run_regime_ablation.py` | 比较 global/local/no-regime 三种设置的影响 |
| `validation/fill_metric_prediction_rates.py` | 统计各 submission 的 Task1/Task2 异常率并回填 `docs/metric.md` |
| `validation/update_model_demo_notebook.py` | 把旧 notebook 同步到 no-regime 逻辑 |
| `validation/create_v4_notebook.py` | 构建 v4 notebook 的辅助脚本 |
| `validation/execute_notebook_locally.py` | 本地执行 notebook 的辅助脚本 |

---

## 9. 当前推荐工作流

如果你的目标是继续做实验或写报告，建议按下面优先级走。

### 9.1 如果你想继续优化模型

优先从：
- `code/train_predict_v4_NoRegime.py`
- `submission_v4_NoRegime`

原因：
- 当前最好版本
- 可以尝试添加合理的regime特征

### 9.2 如果你想要最稳的 baseline

优先使用：
- `code/train_predict_v3_fast.py`
- `submission_v3/`

原因：
- 结构简单
- 结果稳定
- 更容易解释和写报告

### 9.3 如果你要做对照实验

建议的最小对照组：
- v2 baseline
- 当前 v3_fast
- 当前 no-regime v4
- 如有必要，再加入 `v4_Regime` 说明风险版本(regime特征在训练集和验证集语义不对齐)

---

## 10. 后续可做的改进方向

### P1. 更严格的时序验证

- 继续扩展为 3-5 个切分点
- 不要只依赖最后一个短验证段
- 重点看指标方差，而不是只看单次最佳结果

### P2. 精简 v4 结构

- 现在 v4 组件很多，复杂度偏高
- 可以做逐项 ablation：
  - 去掉 cascade
  - 去掉 selected-feature 子模型
  - 去掉 focal 分支
- 观察哪些组件真正提供增益

### P3. 报告友好的可解释性

- 固定一套对比表：v2 / v3 / v4
- 固定一套图：PR / ROC / confusion matrix / 特征重要性
- 说明 no-regime 修复前后的风险与动机

---

## 11. 结论

当前仓库里，**真正推荐继续推进的版本有两个**：
- `train_predict_v3_fast.py`：当前稳健 baseline
- `train_predict_v4_NoRegime.py`：当前主要实验版

现在最重要的共识应该是：
- 先基于 **no-regime** 版本做实验
- 再讨论更复杂的后处理和集成结构是否真的带来收益
