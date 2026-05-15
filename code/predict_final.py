"""
E15: XGBoost Focal + 时序平滑窗口 3
最终模型推理脚本。

用法:
    python code/predict_final.py

输入:
    submission_v5/model.pkl  — 由 train_final.py 生成
    data/test_simple.csv
    data/test_complex.csv

输出:
    submission_v5/pred_simple.csv   — 仅含 y_pred 列
    submission_v5/pred_complex.csv
"""
import pandas as pd, numpy as np, warnings, pickle, os, time
warnings.filterwarnings('ignore')
import xgboost as xgb

MODEL_PATH = 'submission_v5/model.pkl'
OUTPUT_DIR = 'submission_v5'

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

log(f'Config: {art.get("config", "E15")}')
log(f'Threshold = {threshold:.4f}, smooth = {smooth_window}')

# === 2. 特征工程（必须与训练完全一致）===
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

def predict_file(csv_path, output_path, label):
    log(f'Processing {label}...')
    df = pd.read_csv(csv_path)

    # 确保 test 集也有训练时见过的 fcols，缺失列补 0
    for c in fcols:
        if c not in df.columns:
            df[c] = 0.0

    fe = mkfe(df, fcols)
    # 补齐训练时有但测试没有的特征列
    for c in feature_cols:
        if c not in fe.columns:
            fe[c] = 0.0

    X = scaler.transform(fe[feature_cols].values)
    scores = model.predict(xgb.DMatrix(X))
    scores = temporal_smooth(scores, smooth_window)
    pred = (scores >= threshold).astype(int)

    out = pd.DataFrame({'y_pred': pred})
    out.to_csv(output_path, index=False)
    log(f'  {label}: {pred.sum()}/{len(pred)} anomalies predicted -> {output_path}')

# === 3. 推理 ===
predict_file('data/test_simple.csv',  os.path.join(OUTPUT_DIR, 'pred_simple.csv'),  'test_simple')
predict_file('data/test_complex.csv', os.path.join(OUTPUT_DIR, 'pred_complex.csv'), 'test_complex')

log('DONE')
