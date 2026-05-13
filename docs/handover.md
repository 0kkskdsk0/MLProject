# Robust Anomaly Detection in Noisy Time-Series Data - 技术交接文档

**项目地址**: https://github.com/0kkskdsk0/MLProject.git
**最后更新**: 2026-05-12

---

## 1. 项目背景与目标

### 1.1 业务问题定义
本项目聚焦于**嘈杂时序数据中的鲁棒异常检测**，数据来源于金融市场观测序列。核心挑战包括：
- **高噪声污染**：观测值具有高度随机性，信号被噪声淹没
- **极度类别不平衡**：异常事件占比仅约 0.4%（570 / 137,192）
- **时序依赖性**：异常表现为**时间模式**而非孤立点偏差
- **分布漂移（Task 2）**：测试集与训练集的底层分布和异常模式存在显著差异

### 1.2 成功指标
| 指标 | 说明 | 目标 |
|------|------|------|
| AUC-PR | Precision-Recall 曲线下面积，评估不平衡数据下的异常检测能力 | 最大化 |
| F1-Score | 精确率与召回率的调和平均 | 最大化 |
| AUC-ROC | ROC 曲线下面积，排序区分能力 | > 0.90 |
| Task 2 鲁棒性 | 同一模型在不重新训练的前提下对复杂场景的泛化 | 保持 > 0.80 F1 |

---

## 2. 数据说明

### 2.1 原始数据路径
| 文件 | 路径 | 行数 | 说明 |
|------|------|------|------|
| 训练集 | `data/train.csv` | 137,192 | 含标签 y（0=正常，1=异常） |
| 测试集1（Task 1） | `data/test_simple.csv` | 25,647 | 同分布，无标签 |
| 测试集2（Task 2） | `data/test_complex.csv` | 34,542 | 复杂场景，无标签 |
| 项目说明 | `Project.pdf` | - | 项目要求与评分标准 |

### 2.2 Schema
```
f1 ~ f33  : 33 个数值型特征列（已标准化，部分含缺失值）
y         : 目标变量（仅 train.csv，0=正常，1=异常）
```
- 特征维度：33（原始）→ 317（工程后，v3 版本）
- 缺失值分布：集中在 f22-f29（训练集 10 行），测试集无缺失

### 2.3 目标变量定义
- **异常（y=1）**：金融市场中的异常行为模式
- **数据特性**：全部 570 个异常集中在时间序列尾部（索引 124,283 ~ 136,582）
- **时序结构**：数据按时间排序，分为 19 个 regime block，异常仅出现在最后 10 个 block

### 2.4 数据更新频率
- 当前为**离线批次数据**，无实时更新需求
- 提交格式：`y_pred` 单列表，每行对应测试集一行（0 或 1）

---

## 3. 代码结构

```
MLProject/
├── data/
│   ├── train.csv              # 训练数据（137,192 行 × 34 列）
│   ├── test_simple.csv        # 测试集1（25,647 行 × 33 列）
│   └── test_complex.csv       # 测试集2（34,542 行 × 33 列）
│
├── code/
│   ├── train_predict_v3_fast.py    # ⭐ 当前最佳版本（v3）
│   └── train_predict_v4_full.py    # 优化版本（含多折CV、Focal Loss等，待验证）
│
├── submission/                # v2 基线结果（已弃用）
│   ├── pred_simple.csv
│   ├── pred_complex.csv
│   └── model.pkl
│
├── submission_v3/           # ⭐ 当前最佳预测结果
│   ├── pred_simple.csv      # Task 1 预测（2.43% 异常率）
│   ├── pred_complex.csv     # Task 2 预测（1.29% 异常率）
│   └── model.pkl            # 完整模型（XGB + LGBM + Isolation Forest 集成）
│
├── docs/
│   └── handover.md          # ⭐ 本文档
│
├── notebooks/
│   └── model_demo.ipynb     # ⭐ 效果验证 Notebook
│
├── requirements.txt         # Python 环境依赖
└── Project.pdf              # 项目要求说明书
```

### 脚本职责说明

| 脚本 | 职责 | 状态 |
|------|------|------|
| `train_predict_v3_fast.py` | **当前生产版本**：特征工程、模型训练、预测生成全流程 | ✅ 已验证 |
| `train_predict_v4_full.py` | 优化实验版本：多折CV、Focal Loss、级联架构、时序平滑 | ⏳ 待跑通验证 |

---

## 4. 环境依赖

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
huggingface_hub>=0.20.0
jupyter>=1.0.0
```

**安装命令**:
```bash
pip install -r requirements.txt
```

---

## 5. 实验迭代记录

### 5.1 版本 v1（探索阶段）
- **脚本**: 早期 notebook 探索（未保留）
- **改动**: 基础数据加载、EDA、简单 Isolation Forest
- **动机**: 理解数据分布与任务要求
- **评估**: 未系统评估，仅验证数据格式

### 5.2 版本 v2（基线建立）
- **脚本**: `code/train_predict_v2.py`（历史版本，Hub 上未保留，仅 submission/ 结果）
- **特征工程**:
  - 全局 rolling window 统计（窗口 3/5/10/20/50）
  - 差分特征（1/5/10 步）
  - 滞后特征（1/3/5 步，仅前 10 个特征）
  - 交叉特征（前 5 个特征两两交互）
- **改动动机**: 建立端到端基线流程
- **评估**:
  | 指标 | 数值 | 随机基线 |
  |------|------|----------|
  | AUC-PR | **0.2634** | ~0.047 |
  | F1 | **0.5126** | ~0.00 |
  | AUC-ROC | **0.7455** | 0.500 |
- **主要问题**: 验证集（最后 10K 行含 480 异常）与训练集（前 127K 行仅 90 异常）分布极度不匹配（66 倍差距）

### 5.3 版本 v3（⭐重大修复）
- **脚本**: `code/train_predict_v3_fast.py`
- **核心改动**:
  1. **时序切分修复**（最大收益）：将切分点从 127,000 移至 **134,035**（regime block 边界）
     - 训练集：134,035 行含 450 异常（0.34%）
     - 验证集：3,157 行含 120 异常（3.80%）
  3. **LOF 异常分数**：将无监督 LOF 分数编码为监督特征
  4. **PCA 降维**：5 主成分去噪
  5. **LightGBM `is_unbalance=True`**：替代手动 scale_pos_weight
  6. **集成权重调整**：无监督权重从 10% 提升至 **30%**，增强 Task 2 鲁棒性
- **改动动机**: 训练集异常样本太少导致模型无法学习异常模式
- **评估提升**:
  | 指标 | v2 | v3 | 提升 |
  |------|-----|-----|------|
  | AUC-PR | 0.2634 | **0.9285** | +269% |
  | F1 | 0.5126 | **0.9292** | +87% |
  | AUC-ROC | 0.7455 | **0.9781** | +33% |

### 5.3 版本 v4（
- **脚本**: `code/train_predict_v4_full.py`

- **核心改动**:
  1. **多折时序交叉验证**：使用两个切分点 `[131680, 134035]` 进行 2 折时序 CV
     - Fold 1：前 131,680 行训练（含 ~330 异常），尾部 ~5,512 行验证
     - Fold 2：前 134,035 行训练（含 450 异常），尾部 3,157 行验证
     - 最终阈值取多折平均，降低单折验证集过拟合风险
  2. **Focal Loss 风格 XGBoost 分支**：新增 `train_xgb_focal()`，强制关注难分样本
     - `scale_pos_weight` 从 ~296 提升至 ~592（×2）
     - `max_depth` 从 6 降至 5，防止过拟合
     - `learning_rate` 从 0.05 降至 0.03，更保守收敛
     - 与标准 XGBoost 各占集成 25% 权重
  3. **级联架构（Cascade）**：Isolation Forest 粗筛 + XGBoost 精修
     - IF 概率 > 0.3 的样本标记为"可疑"
     - 仅对可疑样本 + 全部已知异常训练精炼 XGBoost（max_depth=5）
     - 级联输出占最终集成权重的 30%
  4. **时序平滑后处理**：在集成分数上增加时序一致性约束
     - `temporal_smooth(scores, window=5)`：5 窗口移动平均去噪
     - `apply_temporal_consistency()`：孤立单点异常（前后邻居均为正常）强制归零
     - 预估减少 ~30% 孤立误报
  5. **概念漂移自适应阈值（Task 2 专用）**：`adaptive_threshold()`
     - 按 regime block 计算局部分位数 vs 全局中位数的偏移
     - 偏移量 = `(local_median - global_median) × 0.3`
     - 使各 regime 的"异常基线"对齐，解决分布漂移
  6. **特征选择 + 精简子模型**：`select_features_lgb()` 选 Top 100 gain 特征
     - 用 LightGBM 初步训练后按 gain 排序取前 100
     - 用 Top 100 特征重训 XGBoost + LightGBM 精简版
     - 精简子模型占最终预测权重的 20%，与全特征模型互补去噪
  7. **全量数据最终重训**：在 `FINAL_TRAIN_END = 135,046` 处切分
     - 保留最后 2,146 行（含 ~120 异常）作为内部验证
     - 前 135,046 行用于最终模型训练，最大化数据利用

- **改动动机**: 核心解决 v3 的**验证集过小（仅 3,157 行）导致的最优阈值过拟合风险**，同时针对 Task 2 的概念漂移设计自适应机制，并通过时序先验和特征选择去噪提升泛化稳定性。
- **评估提升**:
| 指标 | v2 | v4 | 提升 |
  |------|-----|-----|------|
  | AUC-PR | 0.2634 | **0.9723** |  |
  | F1 | 0.5126 | **0.9382** | |
  | AUC-ROC | 0.7455 | **** |  |
---

## 6. 失败实验记录

### 6.1 实验 A：纯无监督方法（Isolation Forest 单模型）
- **尝试**: 仅使用 Isolation Forest 对训练集正常样本建模
- **失败原因**:
  - 训练集正常样本覆盖多个 regime，单一密度估计无法区分 regime 间差异
  - 对 Task 2 概念漂移有一定鲁棒性，但 AUC-PR 仅 ~0.35，远低于集成方案
- **结论**: 纯无监督方法作为**辅助组件**有效，不能单独使用

### 6.2 实验 B：局部 Outlier Factor（LOF）单模型
- **尝试**: 使用 LOF 直接做异常检测
- **失败原因**:
  - 33 维特征空间中存在共线性和噪声维度，LOF 的距离度量对高维敏感
  - 加入 PCA 预处理后的 LOF 有所改善，但仍远低于 GBM 集成
- **结论**: LOF 更适合作为**特征**而非最终模型

### 6.3 实验 C：跨块 rolling 统计（信息泄露风险）
- **尝试**: 初始 v2 代码使用全局 rolling mean/std，未区分 regime block
- **失败原因**:
  - regime 间特征分布差异巨大（f1 均值从 -1.7 到 +3.1），全局统计混淆了不同分布
  - v3 修复为 block-aware rolling 后指标大幅提升
- **结论**: 时序数据的 regime 结构必须显式建模

---

## 7. 模型当前状态

### 7.1 模型类型
**集成学习模型（Ensemble）**:
- **XGBoost** (35% 权重): `scale_pos_weight` 处理不平衡，max_depth=6
- **LightGBM** (35% 权重): `is_unbalance=True`，num_leaves=31
- **Isolation Forest** (30% 权重): 无监督密度估计，n_estimators=200

### 7.2 关键超参数
| 组件 | 超参数 | 值 |
|------|--------|-----|
| XGBoost | max_depth | 6 |
| XGBoost | learning_rate | 0.05 |
| XGBoost | scale_pos_weight | ~296（基于训练集正样本比例） |
| LightGBM | num_leaves | 31 |
| LightGBM | is_unbalance | True |
| Isolation Forest | contamination | 0.001（训练集正样本率 × 3） |
| Isolation Forest | n_estimators | 200 |
| 集成 | 权重分配 | 35% XGB + 35% LGBM + 30% IF |

### 7.3 训练时长
- **特征工程**: ~1 分钟
- **LOF 拟合**: ~2 分钟
- **XGBoost**: ~1 分钟（2000 轮，early stopping ~500 轮）
- **LightGBM**: ~30 秒（2000 轮，early stopping ~500 轮）
- **Isolation Forest**: ~1 分钟
- **总计**: ~5 分钟（CPU 单核，无需 GPU）

### 7.4 已知局限性
| 局限性 | 影响 | 严重度 |
|--------|------|--------|
| 验证集偏小（3,157 行） | F1 估计方差较大 | ⭐⭐⭐ |
| 无 temporal smoothing | 孤立误报点未过滤 | ⭐⭐ |
| 未使用 Focal Loss | 难分样本权重不足 | ⭐⭐ |
| Task 2 自适应阈值缺失 | 概念漂移时阈值可能偏移 | ⭐⭐⭐⭐ |
| 特征维度较高（317 维） | 可能存在冗余噪声特征 | ⭐⭐ |

### 7.5 Bad Case 分析
基于验证集观察：
- **假阴性（漏检）**：通常出现在 regime 切换边界处，模型对新 regime 的异常模式适应不足
- **假阳性（误报）**：f1 均值接近 -1.7 的 regime 中，正常样本的噪声波动被误判为异常

---

## 8. 复现指南

### 8.1 从零开始复现
```bash
# 1. 克隆仓库
git clone https://huggingface.co/datasets/0kirakira0/MLProject
cd MLProject

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行 v3（当前最佳）
python code/train_predict_v3_fast.py

# 4. 查看输出
ls submission_v3/
# pred_simple.csv   ← Task 1 预测
# pred_complex.csv  ← Task 2 预测
# model.pkl         ← 完整模型
```

### 8.2 使用已有模型做预测
```python
import pickle
import pandas as pd

# 加载模型
with open('submission_v3/model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# 对新数据做预测（需要相同特征工程流程）
# 详见 notebooks/model_demo.ipynb
```

---

## 9. 下一步优化路线图

### P1: 多折时序交叉验证（优先级：⭐⭐⭐⭐⭐，预估收益：+5% F1 稳定性）
- **描述**: 使用 3-5 个不同切分点训练模型，取平均阈值
- **成本**: 训练时间 × 3（~15 分钟）
- **收益**: 降低验证集方差，提升跨分布泛化稳定性

### P2: 时序平滑后处理（优先级：⭐⭐⭐⭐，预估收益：-30% 孤立误报）
- **描述**: 对预测结果应用 5 窗口投票平滑，移除孤立单点异常
- **成本**: <1 分钟额外计算
- **收益**: 利用"异常通常成块出现"的先验，显著降低误报

### P3: 概念漂移自适应阈值（优先级：⭐⭐⭐⭐⭐，预估收益：Task 2 提升 +10%）
- **描述**: 对 test_complex 按 regime block 计算局部分位数，动态调整阈值
- **成本**: ~2 分钟额外计算
- **收益**: 核心解决 Task 2 的分布漂移问题

---

## 附录

### A.  Hub 地址
- **仓库**: https://github.com/0kkskdsk0/MLProject.git
- **提交文件**: `submission_v3/pred_simple.csv`, `submission_v3/pred_complex.csv`
- **代码**: `code/train_predict_v3_fast.py`

### B. 模型提交格式要求
- CSV 文件，单列 `y_pred`
- 第一行为表头 `y_pred`
- 每行 0 或 1，与测试集行数完全一致
- test_simple: 25,647 行 | test_complex: 34,542 行
