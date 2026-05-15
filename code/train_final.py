"""
E15: XGBoost Focal + 时序平滑窗口 3
最终模型训练脚本。

用法:
    python code/train_final.py

输出:
    submission_v5/model.pkl  — 模型 + scaler + 阈值 + 元数据
"""
import pandas as pd, numpy as np, warnings, pickle, os, time
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve
import xgboost as xgb

np.random.seed(42)

# === 配置 ===
TRAIN_PATH = 'data/train.csv'
OUTPUT_DIR = 'submission_v5'
os.makedirs(OUTPUT_DIR, exist_ok=True)

S1, S2 = 130816, 134545  # train/val 切分点

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# === 1. 数据加载 ===
log('Loading data...')
df = pd.read_csv(TRAIN_PATH)
fcols = [c for c in df.columns if c.startswith('f')]
med = df[fcols].median()
for c in fcols:
    df[c] = df[c].fillna(med[c])

tr = df.iloc[:S1].copy()
va = df.iloc[S1:S2].copy()
y_train, y_val = tr['y'].values, va['y'].values

log(f'Train: {len(tr)} rows, {y_train.sum()} anomalies')
log(f'Val:   {len(va)} rows, {y_val.sum()} anomalies')

# === 2. 特征工程 ===
log('Feature engineering...')
def mkfe(df, fcols):
    fe = pd.DataFrame(index=df.index)
    for c in fcols:
        fe[c] = df[c].values
    for w in [5, 10, 20]:
        for c in fcols:
            fe[f'{c}_rm{w}'] = df[c].rolling(w, min_periods=1).mean().values
            fe[f'{c}_rs{w}'] = df[c].rolling(w, min_periods=1).std().fillna(0).values
    for c in fcols:
        fe[f'{c}_d1'] = df[c].diff(1).fillna(0).values
        fe[f'{c}_d5'] = df[c].diff(5).fillna(0).values
    for lag in [1, 3]:
        for c in fcols[:3]:
            fe[f'{c}_l{lag}'] = df[c].shift(lag).bfill().fillna(0).values
    for i in range(min(3, len(fcols))):
        for j in range(i+1, min(3, len(fcols))):
            fe[f'i_{i}_{j}'] = (df[fcols[i]] * df[fcols[j]]).values
    fe['row_mean'] = df[fcols].values.mean(axis=1)
    fe['row_std'] = df[fcols].values.std(axis=1)
    fe['row_max'] = df[fcols].values.max(axis=1)
    fe['row_min'] = df[fcols].values.min(axis=1)
    return fe

fe_train = mkfe(tr.drop(columns=['y']), fcols)
fe_val = mkfe(va.drop(columns=['y']), fcols)

feature_cols = sorted(set(fe_train.columns) & set(fe_val.columns))
scaler = StandardScaler()
X_train = scaler.fit_transform(fe_train[feature_cols].values)
X_val = scaler.transform(fe_val[feature_cols].values)
log(f'Features: {len(feature_cols)} dims')

# === 3. 训练 XGBoost Focal (B) ===
log('Training XGBoost Focal (B)...')
dt = xgb.DMatrix(X_train, label=y_train)
pr = y_train.mean()
sw = max(1.0, (1 - pr) / (pr + 1e-8))
model = xgb.train({
    'objective': 'binary:logistic',
    'max_depth': 5,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': sw * 2,
    'seed': 42,
    'tree_method': 'hist',
}, dt, 1500, evals=[(dt, 'train')], verbose_eval=100)
# 注: 不设早停。Val 异常率(4.83%)远高于 Train(0.21%)，
# 用 Val logloss 早停会过早截断训练。实际 1500 轮 + Val F1 阈值选择表现一致。

# === 4. 阈值选择（Val 最大化 F1，加 smooth3）===
log('Selecting threshold on Val...')
def temporal_smooth(s, w=3):
    k = np.ones(w) / w
    return np.convolve(s, k, mode='same')

p_val = model.predict(xgb.DMatrix(X_val))
p_val_smooth = temporal_smooth(p_val, 3)

prec, rec, thrs = precision_recall_curve(y_val, p_val_smooth)
f1s = 2 * prec * rec / (prec + rec + 1e-10)
best = np.argmax(f1s)
threshold = thrs[min(best, len(thrs) - 1)]

pred_val = (p_val_smooth >= threshold).astype(int)
val_f1 = f1_score(y_val, pred_val)
val_fp = ((pred_val == 1) & (y_val == 0)).sum()
val_fn = ((pred_val == 0) & (y_val == 1)).sum()

log(f'Threshold = {threshold:.4f}')
log(f'Val F1 = {val_f1:.4f}, FP = {val_fp}, FN = {val_fn}')

# === 5. 保存 ===
artifact = {
    'model': model,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'fcols': fcols,
    'threshold': threshold,
    'smooth_window': 3,
    'val_metrics': {'f1': val_f1, 'fp': val_fp, 'fn': val_fn},
    'config': 'E15: XGBoost Focal + smooth3',
}
path = os.path.join(OUTPUT_DIR, 'model.pkl')
with open(path, 'wb') as f:
    pickle.dump(artifact, f)
log(f'Model saved to {path}')
log('DONE')
