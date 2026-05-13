"""Update notebooks/model_demo.ipynb to match the no-regime v3 pipeline."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "model_demo.ipynb"


def lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def set_source(nb: dict, index: int, text: str) -> None:
    nb["cells"][index]["source"] = lines(text)
    if nb["cells"][index].get("cell_type") == "code":
        nb["cells"][index]["outputs"] = []
        nb["cells"][index]["execution_count"] = None


def clear_code_outputs(nb: dict) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    clear_code_outputs(nb)

    set_source(
        nb,
        3,
        """# 2. 时间序列结构与异常分布可视化
# 注意：当前 v3 训练流程不再把 regime_id 作为模型特征。
# 这里仅用于观察 f1 的分段结构和异常在时间轴上的分布。

fig, axes = plt.subplots(2, 1, figsize=(16, 8))

# f1 时间序列
axes[0].plot(train_df['f1'].values, alpha=0.7, color='steelblue', linewidth=0.5)
axes[0].set_title('Feature f1 Time Series', fontsize=14)
axes[0].set_ylabel('f1 value')

# 异常标注
anomaly_idx = np.where(train_df['y'].values == 1)[0]
axes[1].scatter(anomaly_idx, [1]*len(anomaly_idx), c='red', s=2, alpha=0.6, label='Anomaly')
axes[1].set_title('Anomaly Distribution (red = anomaly)', fontsize=14)
axes[1].set_xlabel('Time Index')
axes[1].set_ylabel('Anomaly')
axes[1].set_ylim(0.8, 1.2)
axes[1].legend()

plt.tight_layout()
plt.savefig('time_series_anomalies.png', dpi=150, bbox_inches='tight')
plt.show()

print(f\"异常集中在索引 {anomaly_idx.min()} ~ {anomaly_idx.max()} (最后 {len(train_df)-anomaly_idx.min():,} 行)\")
""",
    )

    set_source(
        nb,
        4,
        """## 时间序列结构解读

上图用于观察 `f1` 的时间变化和异常标签分布。训练集异常高度集中在序列尾部，因此验证必须使用时间切分，不能随机划分。

当前 v3 版本已经移除 `regime_id` 特征，避免把全局分段编号作为模型输入。切分点仍保留为 134,035，使训练集包含足够异常样本，同时用最后 3,157 行做验证。
""",
    )

    set_source(
        nb,
        5,
        """# 3. 特征工程（v3 no-regime 版本）
def create_features(df, feature_cols, lof_model=None, scaler_lof=None, 
                    pca_model=None, pca_scaler=None):
    features = pd.DataFrame(index=df.index)
    
    # 原始特征
    for col in feature_cols:
        features[col] = df[col].values
    
    # 滚动统计
    for w in [5, 10, 20]:
        for col in feature_cols:
            features[f'{col}_rm{w}'] = df[col].rolling(window=w, min_periods=1).mean().values
            features[f'{col}_rs{w}'] = df[col].rolling(window=w, min_periods=1).std().fillna(0).values
    
    # 差分特征
    for col in feature_cols:
        features[f'{col}_d1'] = df[col].diff(1).fillna(0).values
        features[f'{col}_d5'] = df[col].diff(5).fillna(0).values
    
    # 滞后特征
    for lag in [1, 3]:
        for col in feature_cols[:3]:
            features[f'{col}_l{lag}'] = df[col].shift(lag).bfill().ffill().values
    
    # 交互特征
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(3, len(feature_cols))):
            features[f'i_{i}_{j}'] = (df[feature_cols[i]] * df[feature_cols[j]]).values
    
    # LOF 分数
    X_raw = df[feature_cols].values
    if lof_model is not None:
        X_scaled = scaler_lof.transform(X_raw)
        lof_scores = lof_model.decision_function(X_scaled)
        features['lof_score'] = -lof_scores
    else:
        features['lof_score'] = 0
    
    # 行统计
    row_data = df[feature_cols].values
    features['row_mean'] = row_data.mean(axis=1)
    features['row_std'] = row_data.std(axis=1)
    features['row_max'] = row_data.max(axis=1)
    features['row_min'] = row_data.min(axis=1)
    
    # PCA
    if pca_model is not None:
        X_scaled = pca_scaler.transform(X_raw)
        comps = pca_model.transform(X_scaled)
        for i in range(comps.shape[1]):
            features[f'pca_{i}'] = comps[:, i]
    
    return features

def preprocess(df, scaler=None, fit_scaler=False):
    df = df.copy().ffill().bfill()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    feature_cols_proc = [c for c in df.columns if c != 'y']
    if fit_scaler:
        scaler = StandardScaler()
        df[feature_cols_proc] = scaler.fit_transform(df[feature_cols_proc])
        return df, scaler
    df[feature_cols_proc] = scaler.transform(df[feature_cols_proc])
    return df

# 时序切分：前 134035 行训练，剩余验证
TRAIN_END = 134035
train_raw = train_df.iloc[:TRAIN_END].copy()
val_raw = train_df.iloc[TRAIN_END:].copy()
y_train = train_raw['y'].values
y_val = val_raw['y'].values

print(f\"训练: {len(train_raw)} 行, {y_train.sum()} 异常 ({y_train.mean()*100:.3f}%)\")
print(f\"验证: {len(val_raw)} 行, {y_val.sum()} 异常 ({y_val.mean()*100:.3f}%)\")

# LOF 和 PCA 仅在训练切分上拟合，避免验证集信息参与特征变换
X_train_sample = train_raw[feature_cols].values
scaler_lof = StandardScaler()
X_lof = scaler_lof.fit_transform(X_train_sample)
sample_size = min(15000, len(X_lof))
idx = np.random.choice(len(X_lof), sample_size, replace=False)
lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto', n_jobs=-1)
lof_model.fit(X_lof[idx])

pca_scaler = StandardScaler()
X_pca = pca_scaler.fit_transform(X_train_sample)
pca_model = PCA(n_components=5, random_state=42)
pca_model.fit(X_pca)

# 特征工程
train_fe = create_features(train_raw.drop(columns=['y']), feature_cols,
                           lof_model, scaler_lof, pca_model, pca_scaler)
val_fe = create_features(val_raw.drop(columns=['y']), feature_cols,
                         lof_model, scaler_lof, pca_model, pca_scaler)

common_cols = sorted(list(set(train_fe.columns) & set(val_fe.columns)))
train_fe = train_fe[common_cols]
val_fe = val_fe[common_cols]

train_scaled, scaler = preprocess(train_fe, fit_scaler=True)
val_scaled = preprocess(val_fe, scaler=scaler)

X_train = train_scaled.values
X_val = val_scaled.values

print(f\"特征维度: {len(common_cols)}\")
""",
    )

    set_source(
        nb,
        6,
        """## 特征工程说明

v3 no-regime 版本将原始 33 维特征扩展至 **316 维**，包含：
- **滚动统计**（窗口 5/10/20）：均值、标准差，捕捉局部趋势
- **差分特征**（1/5 步）：变化率，捕捉突变信号
- **滞后特征**（1/3 步，前 3 特征）：历史值，建模短期时序依赖
- **交互特征**：前 3 个原始特征的两两乘积
- **LOF 异常分数**：无监督密度视角，与监督模型互补
- **PCA 主成分**：5 维主成分，提供去噪后的主要变化方向
- **行统计**：均值、标准差、最大值、最小值，描述单样本整体形态
""",
    )

    source7 = "".join(nb["cells"][7]["source"])
    source7 = source7.replace("num_boost_round=2000", "num_boost_round=3000")
    source7 = source7.replace("'is_unbalance': True, 'seed': 42, 'verbose': -1\n", "'is_unbalance': True, 'seed': 42, 'verbose': -1,\n    'min_child_samples': 5\n")
    set_source(nb, 7, source7)

    set_source(
        nb,
        11,
        """## 混淆矩阵解读

混淆矩阵直观展示了模型在验证集上的分类表现：
- **左上角（TN）**：正常样本被正确识别为正常的数量
- **右下角（TP）**：异常样本被正确识别为异常的数量
- **右上角（FP）**：误报，正常样本被误判为异常
- **左下角（FN）**：漏报，异常样本被误判为正常

在不平衡场景下，精确率和召回率需要一起看。请以当前运行输出的 `F1-Score`、Precision 和 Recall 为准，不要复用旧运行结果。
""",
    )

    set_source(
        nb,
        13,
        """## ROC 曲线解读

ROC 曲线展示了模型在所有可能阈值下的排序能力：
- **X 轴（FPR）**：假阳性率，正常样本被误判的比例
- **Y 轴（TPR）**：真阳性率，即召回率
- **橙色曲线 vs 虚线**：曲线越靠近左上角，排序区分能力越强

请以当前运行输出和图例中的 AUC 为准。当前 notebook 代码已移除 `regime_id` 特征，因此不应再引用旧版 fixed AUC 数值。
""",
    )

    set_source(
        nb,
        16,
        """# 9. 测试集预测与概览
# 准备测试数据
test_simple_fe = create_features(test_simple_df, feature_cols,
                                 lof_model, scaler_lof, pca_model, pca_scaler)
test_complex_fe = create_features(test_complex_df, feature_cols,
                                  lof_model, scaler_lof, pca_model, pca_scaler)

# 对齐列
for col in common_cols:
    if col not in test_simple_fe.columns:
        test_simple_fe[col] = 0
        test_complex_fe[col] = 0
test_simple_fe = test_simple_fe[common_cols]
test_complex_fe = test_complex_fe[common_cols]

test_simple_scaled = preprocess(test_simple_fe, scaler=scaler)
test_complex_scaled = preprocess(test_complex_fe, scaler=scaler)

X_test_simple = test_simple_scaled.values
X_test_complex = test_complex_scaled.values

# 预测
pred_xgb_s = model_xgb.predict(xgb.DMatrix(X_test_simple))
pred_lgb_s = model_lgb.predict(X_test_simple, num_iteration=model_lgb.best_iteration)
if_scores_s = model_if.decision_function(X_test_simple)
pred_if_s = 1 - (if_scores_s - if_scores_s.min()) / (if_scores_s.max() - if_scores_s.min() + 1e-8)
scores_simple = 0.35 * pred_xgb_s + 0.35 * pred_lgb_s + 0.30 * pred_if_s

pred_xgb_c = model_xgb.predict(xgb.DMatrix(X_test_complex))
pred_lgb_c = model_lgb.predict(X_test_complex, num_iteration=model_lgb.best_iteration)
if_scores_c = model_if.decision_function(X_test_complex)
pred_if_c = 1 - (if_scores_c - if_scores_c.min()) / (if_scores_c.max() - if_scores_c.min() + 1e-8)
scores_complex = 0.35 * pred_xgb_c + 0.35 * pred_lgb_c + 0.30 * pred_if_c

pred_simple = (scores_simple >= best_thresh).astype(int)
pred_complex = (scores_complex >= best_thresh).astype(int)

print(\"测试集预测概览\")
print(\"=\" * 60)
print(f\"Task 1 (test_simple): {pred_simple.sum():,} 异常 / {len(pred_simple):,} 总行 ({pred_simple.mean()*100:.2f}%)\")
print(f\"Task 2 (test_complex): {pred_complex.sum():,} 异常 / {len(pred_complex):,} 总行 ({pred_complex.mean()*100:.2f}%)\")
print(f\"阈值: {best_thresh:.4f}\")
print(\"=\" * 60)

# 分数分布对比
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(scores_simple, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].axvline(best_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold={best_thresh:.3f}')
axes[0].set_title('Task 1: Prediction Score Distribution', fontsize=13)
axes[0].set_xlabel('Anomaly Score')
axes[0].set_ylabel('Frequency')
axes[0].legend()

axes[1].hist(scores_complex, bins=50, color='coral', alpha=0.7, edgecolor='black')
axes[1].axvline(best_thresh, color='red', linestyle='--', linewidth=2, label=f'Threshold={best_thresh:.3f}')
axes[1].set_title('Task 2: Prediction Score Distribution', fontsize=13)
axes[1].set_xlabel('Anomaly Score')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.savefig('test_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

print(f\"Task 1 分数范围: [{scores_simple.min():.3f}, {scores_simple.max():.3f}]\")
print(f\"Task 2 分数范围: [{scores_complex.min():.3f}, {scores_complex.max():.3f}]\")
""",
    )

    set_source(
        nb,
        17,
        """## 测试集预测概览解读

两个测试集的预测分布对比揭示了模型对概念漂移的响应：
- **Task 1（同分布）**：分数分布应与验证集相似，异常集中在高分段
- **Task 2（复杂场景）**：分数分布可能更分散，异常率偏低说明模型更保守

请以当前运行输出的异常数量和异常率为准。当前 notebook 不再使用 `regime_id` 特征，预测结果应与 no-regime v3 流程保持一致。
""",
    )

    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
