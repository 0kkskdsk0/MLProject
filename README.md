# MLProject：时序异常检测项目

这是一个面向时序数据的异常检测项目。给定带标签的训练集 `train.csv` 和两个无标签测试集 `test_simple.csv`、`test_complex.csv`，目标是训练模型并输出二值异常预测结果 `y_pred`。

当前仓库已经保留了从早期基线到较强版本的多套实现。其中：

- `v3` 是当前稳定 baseline，结构更简单，适合复现和继续迭代
- `v4_NoRegime` 是当前主实验线，效果更强，包含多折验证和更复杂的集成设计
- `v4_Regime(ErrorVersion)` 仅作历史实验保留，不建议默认使用

## 1. 项目目标

任务核心是从高噪声、强时序、极度类别不平衡的数据中识别异常点。

- 训练集：`data/train.csv`
- 测试集1：`data/test_simple.csv`
- 测试集2：`data/test_complex.csv`
- 特征列：`f1 ~ f33`
- 标签列：`y`，只在训练集存在
- 提交格式：单列 `y_pred`

当前数据规模：

- `train.csv`：137,192 行，34 列
- `test_simple.csv`：25,647 行，33 列
- `test_complex.csv`：34,542 行，33 列

## 2. 仓库结构

```text
MLProject/
├── code/                        # 训练与预测主脚本
├── data/                        # 原始数据集
├── docs/                        # 指标、结果摘要、技术交接文档
├── notebooks/                   # v3/v4 结果展示与分析 notebook
├── submission/                  # v2 输出
├── submission_v3/               # v3 输出
├── submission_v4_NoRegime/      # 当前 v4 主线输出
├── submission_notebook/         # notebook 相关输出
├── validation/                  # 验证、复现、对比实验辅助脚本
├── Project.pdf                  # 题目说明
├── README.md                    # 项目入口说明
└── AGENTS.md                    # 给 agent 的仓库协作说明
```

## 3. 主要脚本说明

### `code/`

- `train_predict_v2.py`
  早期基线版本，输出到 `submission/`

- `train_predict_v3_fast.py`
  当前稳定 baseline，输出到 `submission_v3/`

- `train_predict_v4_NoRegime.py`
  当前主实验版本，输出到 `submission_v4_NoRegime/`

- `train_predict_v4_Regime(ErrorVersion).py`
  保留的历史实验版本，存在 `regime_id` 相关风险，不建议默认继续使用

### `validation/`
一些辅助进行实验的代码，不重要，agent可视需求使用
- `validate_submission_v3_pkl.py`
  回放 `submission_v3/model.pkl`，验证保存模型是否能复现 v3 指标逻辑

- `run_regime_ablation.py`
  对比 `global_regime`、`local_regime`、`no_regime` 三种设置

- `fill_metric_prediction_rates.py`
  统计提交文件中的异常比例，回填到指标文档

- `update_model_demo_notebook.py`
  同步更新 v3 notebook 展示逻辑

- `create_v4_notebook.py`
  生成或重建 v4 notebook 内容

- `execute_notebook_locally.py`
  本地直接执行 notebook，不依赖 `nbconvert`

## 4. 快速开始

建议在仓库根目录运行以下命令。

### 运行 v3 稳定 baseline

```powershell
python code\train_predict_v3_fast.py
```

预期输出：

- `submission_v3/pred_simple.csv`
- `submission_v3/pred_complex.csv`
- `submission_v3/model.pkl`

### 运行当前 v4 主线版本

```powershell
python code\train_predict_v4_NoRegime.py
```

预期输出：

- `submission_v4_NoRegime/pred_simple.csv`
- `submission_v4_NoRegime/pred_complex.csv`
- `submission_v4_NoRegime/model.pkl`

## 5. 环境依赖

当前代码中可见的主要依赖包括：

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `matplotlib`

可先安装这一组基础依赖：

```powershell
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```

如果你要运行 notebook 生成或展示流程，可能还需要补充 Jupyter 相关包。

## 6. 当前版本建议

如果你是第一次接手这个仓库，建议按下面顺序理解和使用：

1. 通过 `docs/handover.md`了解当前工作的主要成果
2. 浏览与运行`notebooks/`文件夹下的notebook，了解v3，v4版本的模型的构建细节


当前版本定位：

- `v2`
  早期可运行基线，主要价值是提供最初的端到端模板

- `v3`
  当前稳定 baseline，去掉了高风险的 `regime_id` 特征，适合继续做稳妥迭代

- `v4_NoRegime`
  当前主实验线，在 v3 基础上加入多折时序验证、更多分支模型和后处理逻辑

- `v4_Regime(ErrorVersion)`
  历史实验保留版本，不建议默认当作可交付主线

## 7. 当前结果摘要

详细指标请看 `docs/metric.md` 和 `docs/v4result.md`。这里仅保留最关键的版本结论。

### v3

- AUC-PR：`0.9285`
- F1：`0.9292`
- AUC-ROC：`0.9781`
- `submission_v3/pred_simple.csv`：883 / 25,647 预测为异常
- `submission_v3/pred_complex.csv`：580 / 34,542 预测为异常

### v4_NoRegime

- CV Avg AUC-PR：`0.9723`
- CV Avg F1：`0.9382`
- AUC-ROC：`0.9993`
- `submission_v4_NoRegime/pred_simple.csv`：857 / 25,647 预测为异常
- `submission_v4_NoRegime/pred_complex.csv`：643 / 34,542 预测为异常

## 8. 输出文件说明

训练脚本运行完成后，会生成以下文件：

- `pred_simple.csv`
- `pred_complex.csv`
- `model.pkl`

其中：

- `pred_simple.csv` 和 `pred_complex.csv` 只应包含一列：`y_pred`
- `model.pkl` 保存训练好的模型、阈值、特征列和部分复现所需元数据

## 9. 推荐阅读顺序

如果你想快速进入项目，建议这样看：

1. 本 `README.md`
2. `docs/handover.md`
3. `notebooks/` 中的展示和分析材料
4. `code/train_predict_v3_fast.py`
5. `code/train_predict_v4_NoRegime.py`

