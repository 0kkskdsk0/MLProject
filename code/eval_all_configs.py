"""
Evaluate all 8 experiment configurations on the internal test set (Scheme C).
Trains on Train+Val, evaluates on Test. Reports full metrics.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score,
                             precision_recall_curve, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

# === CONFIG ===
TRAIN_PATH = 'data/train.csv'
SPLIT_TRAIN = 130816
SPLIT_VAL = 134545

# === DATA LOADING ===
train_df = pd.read_csv(TRAIN_PATH)
feature_cols = [c for c in train_df.columns if c.startswith('f')]
medians = train_df[feature_cols].median()
for col in feature_cols:
    train_df[col] = train_df[col].fillna(medians[col])

train_raw = train_df.iloc[:SPLIT_TRAIN].copy()
val_raw = train_df.iloc[SPLIT_TRAIN:SPLIT_VAL].copy()
test_raw = train_df.iloc[SPLIT_VAL:].copy()
all_train = pd.concat([train_raw, val_raw])
y_full = all_train['y'].values
y_test = test_raw['y'].values

print(f"Train+Val: {len(all_train)} rows, {y_full.sum()} anomalies")
print(f"Test:      {len(test_raw)} rows, {y_test.sum()} anomalies")

# === FEATURE ENGINEERING (full v4 version with LOF/PCA) ===
def create_features(df, feature_cols, lof_model=None, scaler_lof=None,
                    pca_model=None, pca_scaler=None):
    features = pd.DataFrame(index=df.index)
    for col in feature_cols:
        features[col] = df[col].values
    for w in [5, 10, 20]:
        for col in feature_cols:
            features[f"{col}_rm{w}"] = df[col].rolling(window=w, min_periods=1).mean().values
            features[f"{col}_rs{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0).values
    for col in feature_cols:
        features[f"{col}_d1"] = df[col].diff(1).fillna(0).values
        features[f"{col}_d5"] = df[col].diff(5).fillna(0).values
    for lag in [1, 3]:
        for col in feature_cols[:3]:
            features[f"{col}_l{lag}"] = df[col].shift(lag).bfill().fillna(0).values
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(3, len(feature_cols))):
            features[f"i_{i}_{j}"] = (df[feature_cols[i]] * df[feature_cols[j]]).values
    X_raw = df[feature_cols].values
    if lof_model is not None:
        X_scaled = scaler_lof.transform(X_raw)
        features["lof_score"] = -lof_model.decision_function(X_scaled)
    else:
        features["lof_score"] = 0
    row_data = df[feature_cols].values
    features["row_mean"] = row_data.mean(axis=1)
    features["row_std"] = row_data.std(axis=1)
    features["row_max"] = row_data.max(axis=1)
    features["row_min"] = row_data.min(axis=1)
    if pca_model is not None:
        X_scaled = pca_scaler.transform(X_raw)
        comps = pca_model.transform(X_scaled)
        for i in range(comps.shape[1]):
            features[f"pca_{i}"] = comps[:, i]
    return features

def fit_lof_pca(source_df, feature_cols):
    X_source = source_df[feature_cols].values
    scaler_lof = StandardScaler()
    X_lof = scaler_lof.fit_transform(X_source)
    sample_size = min(15000, len(X_lof))
    idx = np.random.choice(len(X_lof), sample_size, replace=False)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True,
                                   contamination='auto', n_jobs=-1)
    lof_model.fit(X_lof[idx])
    pca_scaler = StandardScaler()
    X_pca = pca_scaler.fit_transform(X_source)
    pca_model = PCA(n_components=5, random_state=42)
    pca_model.fit(X_pca)
    return lof_model, scaler_lof, pca_model, pca_scaler

print("\nFeature engineering...")
lof_model, scaler_lof, pca_model, pca_scaler = fit_lof_pca(train_raw, feature_cols)

full_fe = create_features(all_train.drop(columns=['y']), feature_cols,
                          lof_model, scaler_lof, pca_model, pca_scaler)
test_fe = create_features(test_raw.drop(columns=['y']), feature_cols,
                          lof_model, scaler_lof, pca_model, pca_scaler)

common_cols = sorted(set(full_fe.columns) & set(test_fe.columns))
full_fe = full_fe[list(common_cols)]
test_fe = test_fe[list(common_cols)]

scaler = StandardScaler()
X_full = scaler.fit_transform(full_fe.values)
X_test = scaler.transform(test_fe.values)
feature_names = list(full_fe.columns)
print(f"  Features: {len(feature_names)} dims")

# === MODEL TRAINING ===
print("\nTraining all models...")

# XGBoost standard
dt = xgb.DMatrix(X_full, label=y_full)
pos_ratio = y_full.mean()
scale_pos_weight = max(1.0, (1 - pos_ratio) / (pos_ratio + 1e-8))
params_xgb = {
    'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 6, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': scale_pos_weight,
    'seed': 42, 'tree_method': 'hist', 'min_child_weight': 3,
}
xgb_std = xgb.train(params_xgb, dt, num_boost_round=2000,
                    evals=[(dt, 'train')], verbose_eval=0)
print("  XGBoost std   OK")

# XGBoost focal
sw_focal = scale_pos_weight * 2
params_focal = {
    'objective': 'binary:logistic', 'eval_metric': 'aucpr',
    'max_depth': 5, 'learning_rate': 0.03,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': sw_focal,
    'seed': 42, 'tree_method': 'hist',
}
xgb_focal = xgb.train(params_focal, dt, num_boost_round=1500,
                      evals=[(dt, 'train')], verbose_eval=0)
print("  XGBoost focal  OK")

# LightGBM
lgb_data = lgb.Dataset(X_full, label=y_full)
params_lgb = {
    'objective': 'binary', 'metric': 'average_precision',
    'boosting_type': 'gbdt', 'num_leaves': 31,
    'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'is_unbalance': True, 'seed': 42, 'verbose': -1,
}
lgb_model = lgb.train(params_lgb, lgb_data, num_boost_round=2000,
                      callbacks=[lgb.log_evaluation(0)])
print("  LightGBM      OK")

# Isolation Forest
X_normal = X_full[y_full == 0]
contamination = max(0.001, min(0.1, y_full.mean() * 3))
if len(X_normal) > 50000:
    idx = np.random.choice(len(X_normal), 50000, replace=False)
    X_normal = X_normal[idx]
if_model = IsolationForest(n_estimators=200, contamination=contamination,
                           random_state=42, n_jobs=-1, max_samples=10000)
if_model.fit(X_normal)
print("  IForest       OK")

# Cascade
if_scores = if_model.decision_function(X_full)
if_probs = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
suspicious = if_probs > 0.3
train_mask = suspicious | (y_full == 1)
X_refine = X_full[train_mask]
y_refine = y_full[train_mask]
cascade_model = None
if y_refine.sum() >= 10:
    dt_r = xgb.DMatrix(X_refine, label=y_refine)
    pr_r = y_refine.mean()
    sw_r = max(1.0, (1 - pr_r) / (pr_r + 1e-8))
    params_cas = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'max_depth': 5, 'learning_rate': 0.05,
        'scale_pos_weight': sw_r, 'seed': 42, 'tree_method': 'hist',
    }
    refine_model = xgb.train(params_cas, dt_r, num_boost_round=1000,
                             evals=[(dt_r, 'train')], verbose_eval=0)
    cascade_model = {'if_model': if_model, 'refine_model': refine_model}
    print("  Cascade       OK")
else:
    print("  Cascade       NO (too few anomalies)")

# Feature selection sub-models
lgb_data_sel = lgb.Dataset(X_full, label=y_full)
lgb_sel = lgb.train(params_lgb, lgb_data_sel, num_boost_round=500,
                    callbacks=[lgb.log_evaluation(0)])
importance = lgb_sel.feature_importance(importance_type='gain')
ranked = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
selected_names = [name for name, _ in ranked[:100]]
selected_idx = [feature_names.index(n) for n in selected_names]
X_full_sel = X_full[:, selected_idx]
X_test_sel = X_test[:, selected_idx]

xgb_sel = xgb.train(params_xgb, xgb.DMatrix(X_full_sel, label=y_full),
                    num_boost_round=2000, evals=[(xgb.DMatrix(X_full_sel, label=y_full), 'train')],
                    verbose_eval=0)
lgb_sel_final = lgb.train(params_lgb, lgb.Dataset(X_full_sel, label=y_full),
                          num_boost_round=2000, callbacks=[lgb.log_evaluation(0)])
print("  Feature sel   OK")

# === PREDICTION HELPERS ===
def p_xgb(m, X): return m.predict(xgb.DMatrix(X))
def p_lgb(m, X): return m.predict(X)
def p_if(m, X):
    s = m.decision_function(X)
    return 1 - (s - s.min()) / (s.max() - s.min() + 1e-8)

def cascade_pred(cas, X):
    if cas is None:
        return None
    s = cas['if_model'].decision_function(X)
    p = 1 - (s - s.min()) / (s.max() - s.min() + 1e-8)
    r = cas['refine_model'].predict(xgb.DMatrix(X))
    return p * r

# === EVALUATE ALL CONFIGS ===
configs = [
    ("E1", ["xgb_std"], [1.0]),
    ("E2", ["lgb"], [1.0]),
    ("E3", ["iforest"], [1.0]),
    ("E4", ["xgb_std", "lgb"], [0.5, 0.5]),
    ("E5", ["xgb_std", "iforest"], [0.5, 0.5]),
    ("E6", ["xgb_std", "lgb", "iforest"], [1/3, 1/3, 1/3]),
    ("E7", ["xgb_std", "xgb_focal", "lgb", "iforest"], [0.25, 0.25, 0.25, 0.25]),
]

all_scores = {
    "xgb_std": p_xgb(xgb_std, X_test),
    "xgb_focal": p_xgb(xgb_focal, X_test),
    "lgb": p_lgb(lgb_model, X_test),
    "iforest": p_if(if_model, X_test),
}

# E8: full v4
base_4 = (0.25 * all_scores["xgb_std"] + 0.25 * all_scores["xgb_focal"]
          + 0.25 * all_scores["lgb"] + 0.25 * all_scores["iforest"])
pc = cascade_pred(cascade_model, X_test)
if pc is not None:
    base_4 = 0.7 * base_4 + 0.3 * pc
px = p_xgb(xgb_sel, X_test_sel)
pl = p_lgb(lgb_sel_final, X_test_sel)
e8_scores = 0.8 * base_4 + 0.2 * (0.5 * px + 0.5 * pl)
all_scores["e8"] = e8_scores

print()
header = f"{'名称':<5} {'AUC-PR':<7} {'Acc':<7} {'Prec':<7} {'Recall':<7} {'F1':<7} {'阈值':<7} {'预测异常':<12} {'FN':<5} {'FP':<5}"
print(header)
print("-" * 75)

def evaluate(name, scores):
    prec, rec, thrs = precision_recall_curve(y_test, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    best = np.argmax(f1s)
    thresh = thrs[min(best, len(thrs) - 1)] if best < len(thrs) else 0.5
    pred = (scores >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    ps = precision_score(y_test, pred, zero_division=0)
    rs = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    apr = average_precision_score(y_test, scores)
    fn = ((pred == 0) & (y_test == 1)).sum()
    fp = (pred == 1) & (y_test == 0)
    fp_count = fp.sum()
    print(f"{name:<5} {apr:<7.4f} {acc:<7.4f} {ps:<7.4f} {rs:<7.4f} {f1:<7.4f} {thresh:<7.4f} {pred.sum():>4}/{len(pred):<7} {fn:<5} {fp_count:<5}")

for name, models, weights in configs:
    scores = None
    for mname, w in zip(models, weights):
        p = all_scores[mname]
        scores = (w * p) if scores is None else scores + w * p
    evaluate(name, scores)

evaluate("E8", all_scores["e8"])

# Summary ranking by AUC-PR
print("\n=== Ranking by AUC-PR ===")
results = []
for name, models, weights in configs:
    scores = None
    for mname, w in zip(models, weights):
        p = all_scores[mname]
        scores = (w * p) if scores is None else scores + w * p
    apr = average_precision_score(y_test, scores)
    results.append((name, apr))
apr_e8 = average_precision_score(y_test, all_scores["e8"])
results.append(("E8", apr_e8))

for name, apr in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"  {name}: AUC-PR = {apr:.4f}")
