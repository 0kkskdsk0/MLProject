"""
E15: XGBoost Focal + 时序平滑窗口 3
自测脚本：在 train/val/test 三集上评估模型准确率。

用法:
    python code/predict_final.py

输入:
    submission_v5/model.pkl  — 由 train_final.py 生成
    data/train.csv

输出:
    终端打印三集完整指标
"""
import pandas as pd, numpy as np, warnings, pickle, os, time
warnings.filterwarnings('ignore')
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score)
import xgboost as xgb

MODEL_PATH = 'submission_v5/model.pkl'
S1, S2 = 130816, 134545

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# === 1. 加载模型 ===
log('Loading model...')
with open(MODEL_PATH, 'rb') as f:
    art = pickle.load(f)

model = art['model']
scaler = art['scaler']
feature_cols = art['feature_cols']
fcols = art['fcols']
threshold = art['threshold']
smooth_window = art.get('smooth_window', 3)

# === 2. 特征工程 ===
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

def temporal_smooth(s, w):
    k = np.ones(w) / w
    return np.convolve(s, k, mode='same')

# === 3. 加载数据并切分 ===
log('Loading data...')
df = pd.read_csv('data/train.csv')
for c in fcols:
    df[c] = df[c].fillna(df[fcols].median())

splits = [
    ('Train', df.iloc[:S1]),
    ('Val',   df.iloc[S1:S2]),
    ('Test',  df.iloc[S2:]),
]

# === 4. 评估 ===
print()
print('=' * 90)
print(f'  E15: XGBoost Focal + smooth3  |  Threshold = {threshold:.4f}')
print('=' * 90)
print(f'{"Set":<8} {"AUC-PR":<10} {"Acc":<8} {"F1":<8} {"Prec":<8} {"Rec":<8} {"FP":<6} {"FN":<6} {"Pred":<8}')
print('-' * 90)

for name, split_df in splits:
    y_true = split_df['y'].values
    fe = mkfe(split_df.drop(columns=['y']), fcols)
    X = scaler.transform(fe[feature_cols].values)

    scores = model.predict(xgb.DMatrix(X))
    scores = temporal_smooth(scores, smooth_window)
    pred = (scores >= threshold).astype(int)

    aucpr = average_precision_score(y_true, scores)
    acc = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred)
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()

    print(f'{name:<8} {aucpr:<10.4f} {acc:<8.4f} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {fp:<6} {fn:<6} {pred.sum():<3}/{len(y_true):<4}')

print('=' * 90)
log('DONE')
