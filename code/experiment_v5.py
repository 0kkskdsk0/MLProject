"""
Experiment v5: IF-free ablation with multiple weight variants
Model pool: A=XGBstd, B=XGBfocal, C=LGB, D=XGBsel, E=LGBsel

FIXED DATA SPLIT: train on tr[:130816], threshold on va[130816:134545], test on te[134545:].
"""
import pandas as pd, numpy as np, warnings, time, sys
from datetime import datetime
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, average_precision_score, precision_recall_curve)
import xgboost as xgb, lightgbm as lgb

np.random.seed(42)
t_start = time.time()

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    print(f'[{t}] {msg}', flush=True)

# === DATA ===
log('Loading data...')
train_df = pd.read_csv('data/train.csv')
fcols = [c for c in train_df.columns if c.startswith('f')]
med = train_df[fcols].median()
for c in fcols: train_df[c] = train_df[c].fillna(med[c])

S1, S2 = 130816, 134545
tr = train_df.iloc[:S1].copy()
va = train_df.iloc[S1:S2].copy()
te = train_df.iloc[S2:].copy()

y_train = tr['y'].values
y_val = va['y'].values
y_test = te['y'].values

log(f'Train:     {len(tr)} rows, {y_train.sum()} anomalies')
log(f'Val:       {len(va)} rows, {y_val.sum()} anomalies')
log(f'Test:      {len(te)} rows, {y_test.sum()} anomalies')
log('Note: threshold selected on Val, test set NEVER touches training or threshold selection')

# === FEATURE ENGINEERING ===
log('Feature engineering...')
def mkfe(df, fcols):
    fe = pd.DataFrame(index=df.index)
    for c in fcols: fe[c] = df[c].values
    for w in [5,10,20]:
        for c in fcols:
            fe[f'{c}_rm{w}'] = df[c].rolling(w, min_periods=1).mean().values
            fe[f'{c}_rs{w}'] = df[c].rolling(w, min_periods=1).std().fillna(0).values
    for c in fcols:
        fe[f'{c}_d1'] = df[c].diff(1).fillna(0).values
        fe[f'{c}_d5'] = df[c].diff(5).fillna(0).values
    for lag in [1,3]:
        for c in fcols[:3]:
            fe[f'{c}_l{lag}'] = df[c].shift(lag).bfill().fillna(0).values
    for i in range(min(3,len(fcols))):
        for j in range(i+1,min(3,len(fcols))):
            fe[f'i_{i}_{j}'] = (df[fcols[i]] * df[fcols[j]]).values
    fe['row_mean'] = df[fcols].values.mean(axis=1)
    fe['row_std'] = df[fcols].values.std(axis=1)
    fe['row_max'] = df[fcols].values.max(axis=1)
    fe['row_min'] = df[fcols].values.min(axis=1)
    return fe

fe_train = mkfe(tr.drop(columns=['y']), fcols)
fe_val = mkfe(va.drop(columns=['y']), fcols)
fe_test = mkfe(te.drop(columns=['y']), fcols)

common = sorted(set(fe_train.columns) & set(fe_val.columns) & set(fe_test.columns))
scaler = StandardScaler()
X_train = scaler.fit_transform(fe_train[common].values)
X_val = scaler.transform(fe_val[common].values)
X_test = scaler.transform(fe_test[common].values)
log(f'Features: {len(common)} dims')

# === TRAIN MODELS ===
log('Training XGBoost std (A)...')
dt = xgb.DMatrix(X_train, label=y_train)
pr = y_train.mean()
sw = max(1.0, (1-pr)/(pr+1e-8))
A = xgb.train({'objective':'binary:logistic','max_depth':6,'learning_rate':0.05,
    'subsample':0.8,'colsample_bytree':0.8,'scale_pos_weight':sw,'seed':42,
    'tree_method':'hist','min_child_weight':3}, dt, 2000, evals=[(dt,'train')], verbose_eval=100)

log('Training XGBoost focal (B)...')
B = xgb.train({'objective':'binary:logistic','max_depth':5,'learning_rate':0.03,
    'subsample':0.8,'colsample_bytree':0.8,'scale_pos_weight':sw*2,'seed':42,
    'tree_method':'hist'}, dt, 1500, evals=[(dt,'train')], verbose_eval=100)

log('Training LightGBM (C)...')
C = lgb.train({'objective':'binary','boosting_type':'gbdt','num_leaves':31,
    'learning_rate':0.05,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'is_unbalance':True,'min_child_weight':5.0,'seed':42,'verbose':-1},
    lgb.Dataset(X_train,label=y_train), 2000, callbacks=[lgb.log_evaluation(50)])

log('Feature selection for D/E...')
sel = lgb.train({'objective':'binary','boosting_type':'gbdt','num_leaves':31,
    'learning_rate':0.05,'is_unbalance':True,'min_child_weight':5.0,'seed':42,'verbose':-1},
    lgb.Dataset(X_train,label=y_train), 500, callbacks=[lgb.log_evaluation(0)])
imp = sel.feature_importance(importance_type='gain')
sel_names = [n for _,n in sorted(zip(imp,common), reverse=True)][:100]
sel_idx = [common.index(n) for n in sel_names]
X_train_sel = X_train[:, sel_idx]
X_val_sel = X_val[:, sel_idx]
X_test_sel = X_test[:, sel_idx]

log('Training XGBoost sel (D)...')
D = xgb.train({'objective':'binary:logistic','max_depth':6,'learning_rate':0.05,
    'subsample':0.8,'colsample_bytree':0.8,'scale_pos_weight':sw,'seed':42,
    'tree_method':'hist','min_child_weight':3}, xgb.DMatrix(X_train_sel,label=y_train),
    2000, evals=[(xgb.DMatrix(X_train_sel,label=y_train),'train')], verbose_eval=0)

log('Training LightGBM sel (E)...')
E = lgb.train({'objective':'binary','boosting_type':'gbdt','num_leaves':31,
    'learning_rate':0.05,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'is_unbalance':True,'min_child_weight':5.0,'seed':42,'verbose':-1},
    lgb.Dataset(X_train_sel,label=y_train), 2000, callbacks=[lgb.log_evaluation(0)])

# === PRECOMPUTE PREDICTIONS ===
log('Computing predictions...')
px = lambda m, X: m.predict(xgb.DMatrix(X))
pl = lambda m, X: m.predict(X)

def get_preds(X, X_sel):
    return {
        'A': px(A, X), 'B': px(B, X), 'C': pl(C, X),
        'D': px(D, X_sel), 'E': pl(E, X_sel),
    }

p_train = get_preds(X_train, X_train_sel)
p_val   = get_preds(X_val, X_val_sel)
p_test  = get_preds(X_test, X_test_sel)

def temporal_smooth(s, w=5):
    k = np.ones(w)/w
    return np.convolve(s, k, mode='same')

# === EXPERIMENT CONFIGS ===
experiments = [
    ('E1  A (XGB std)',                lambda p: p['A'], True),
    ('E2  B (XGB focal)',              lambda p: p['B'], True),
    ('E3  C (LGB)',                    lambda p: p['C'], True),
    ('E4  A+B 0.5+0.5',               lambda p: 0.5*p['A']+0.5*p['B'], True),
    ('E5  A+B 0.7+0.3',               lambda p: 0.7*p['A']+0.3*p['B'], True),
    ('E6  A+B 0.3+0.7',               lambda p: 0.3*p['A']+0.7*p['B'], True),
    ('E7  A+C 0.5+0.5',               lambda p: 0.5*p['A']+0.5*p['C'], True),
    ('E8  B+C 0.5+0.5',               lambda p: 0.5*p['B']+0.5*p['C'], True),
    ('E9  A+C 0.7+0.3',               lambda p: 0.7*p['A']+0.3*p['C'], True),
    ('E10 A+B+C 0.34+0.33+0.33',      lambda p: 0.34*p['A']+0.33*p['B']+0.33*p['C'], True),
    ('E11 A+B+C 0.5+0.25+0.25',       lambda p: 0.5*p['A']+0.25*p['B']+0.25*p['C'], True),
    ('E12 A+B+C 0.2+0.4+0.4',         lambda p: 0.2*p['A']+0.4*p['B']+0.4*p['C'], True),
    ('E13 0.8*E10+0.2*sel',           lambda p: 0.8*(0.34*p['A']+0.33*p['B']+0.33*p['C'])+0.2*(0.5*p['D']+0.5*p['E']), True),
    ('E14 0.8*E4+0.2*sel',            lambda p: 0.8*(0.5*p['A']+0.5*p['B'])+0.2*(0.5*p['D']+0.5*p['E']), True),
]

def select_threshold(scores, y_true):
    prec, rec, thrs = precision_recall_curve(y_true, scores)
    f1s = 2*prec*rec/(prec+rec+1e-10)
    best = np.argmax(f1s)
    thresh = thrs[min(best, len(thrs)-1)] if best < len(thrs) else 0.5
    return thresh

def full_metrics(scores, y_true, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        'aucpr': average_precision_score(y_true, scores),
        'acc': accuracy_score(y_true, pred),
        'prec': precision_score(y_true, pred, zero_division=0),
        'rec': recall_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'fp': int(((pred==1)&(y_true==0)).sum()),
        'fn': int(((pred==0)&(y_true==1)).sum()),
        'pred_anom': int(pred.sum()),
    }

# === EVALUATE ALL CONFIGS ===
log('Evaluating configs...')
results = []

for name, fn, do_smooth in experiments:
    s_val_raw = fn(p_val)
    s_train_raw = fn(p_train)
    s_test_raw = fn(p_test)

    s_val = temporal_smooth(s_val_raw, 5) if do_smooth else s_val_raw
    thresh = select_threshold(s_val, y_val)

    s_train = temporal_smooth(s_train_raw, 5) if do_smooth else s_train_raw
    s_test  = temporal_smooth(s_test_raw, 5) if do_smooth else s_test_raw

    results.append({
        'name': name,
        'val': full_metrics(s_val, y_val, thresh),
        'test': full_metrics(s_test, y_test, thresh),
        'train': full_metrics(s_train, y_train, thresh),
        'threshold': thresh,
    })

# Add post-processing variants for best basic config
best_basic = max(results, key=lambda r: r['val']['aucpr'])
best_label = best_basic['name'].strip()
best_fn = None
for n, fn, _ in experiments:
    if n == best_basic['name']:
        best_fn = fn
        break

if best_fn:
    s_val_raw = best_fn(p_val)
    s_train_raw = best_fn(p_train)
    s_test_raw = best_fn(p_test)

    for pp_name, sw in [
        (f'E15 {best_label}+smooth3', 3),
        (f'E16 {best_label}+smooth7', 7),
        (f'E17 {best_label}+nosmooth', None),
    ]:
        s_val_pp = temporal_smooth(s_val_raw, sw) if sw else s_val_raw
        thresh_pp = select_threshold(s_val_pp, y_val)
        s_train_pp = temporal_smooth(s_train_raw, sw) if sw else s_train_raw
        s_test_pp  = temporal_smooth(s_test_raw, sw) if sw else s_test_raw
        results.append({
            'name': pp_name,
            'val': full_metrics(s_val_pp, y_val, thresh_pp),
            'test': full_metrics(s_test_pp, y_test, thresh_pp),
            'train': full_metrics(s_train_pp, y_train, thresh_pp),
            'threshold': thresh_pp,
        })

# Sort by Val AUC-PR
results.sort(key=lambda r: r['val']['aucpr'], reverse=True)

# === PRINT REPORT ===
print()
print('=' * 100)
print('  Experiment v5 Results')
print('  Data: Train={} rows, Val={} rows, Test={} rows'.format(len(tr), len(va), len(te)))
print(f'  Train anomalies: {y_train.sum()}/{len(y_train)} | Val: {y_val.sum()}/{len(y_val)} | Test: {y_test.sum()}/{len(y_test)}')
print('=' * 100)

# Train
print()
print('--- TRAIN METRICS (threshold from Val) ---')
print(f'{"Rank":<5} {"Exp":<35} {"AUC-PR":<10} {"Acc":<8} {"F1":<8} {"Prec":<8} {"Rec":<8} {"FP":<6} {"FN":<6} {"Pred":<7}')
print('-' * 96)
for i, r in enumerate(results, 1):
    t = r['train']
    print(f'{i:<5} {r["name"]:<35} {t["aucpr"]:<10.4f} {t["acc"]:<8.4f} {t["f1"]:<8.4f} {t["prec"]:<8.4f} {t["rec"]:<8.4f} {t["fp"]:<6} {t["fn"]:<6} {t["pred_anom"]:<3}/{len(y_train):<3}')

# Val
print()
print('--- VAL METRICS (ranking basis, threshold selected on Val) ---')
print(f'{"Rank":<5} {"Exp":<35} {"AUC-PR":<10} {"Acc":<8} {"F1":<8} {"Prec":<8} {"Rec":<8} {"FP":<6} {"FN":<6} {"Pred":<7}')
print('-' * 96)
for i, r in enumerate(results, 1):
    v = r['val']
    print(f'{i:<5} {r["name"]:<35} {v["aucpr"]:<10.4f} {v["acc"]:<8.4f} {v["f1"]:<8.4f} {v["prec"]:<8.4f} {v["rec"]:<8.4f} {v["fp"]:<6} {v["fn"]:<6} {v["pred_anom"]:<3}/{len(y_val):<3}')

# Test
print()
print('--- TEST METRICS (final hold-out) ---')
print(f'{"Rank":<5} {"Exp":<35} {"AUC-PR":<10} {"Acc":<8} {"F1":<8} {"Prec":<8} {"Rec":<8} {"FP":<6} {"FN":<6} {"Pred":<7}')
print('-' * 96)
for i, r in enumerate(results, 1):
    t = r['test']
    print(f'{i:<5} {r["name"]:<35} {t["aucpr"]:<10.4f} {t["acc"]:<8.4f} {t["f1"]:<8.4f} {t["prec"]:<8.4f} {t["rec"]:<8.4f} {t["fp"]:<6} {t["fn"]:<6} {t["pred_anom"]:<3}/{len(y_test):<3}')

# Summary
print()
print('=' * 100)
print('  SUMMARY')
print('=' * 100)
for i, r in enumerate(results[:5], 1):
    tr = r['train']; v = r['val']; t = r['test']
    print(f'  #{i}: {r["name"].strip()}')
    print(f'      Threshold = {r["threshold"]:.4f}')
    print(f'      Train: AUC-PR={tr["aucpr"]:.4f}  F1={tr["f1"]:.4f}  FP={tr["fp"]}  FN={tr["fn"]}')
    print(f'      Val:   AUC-PR={v["aucpr"]:.4f}  F1={v["f1"]:.4f}  FP={v["fp"]}  FN={v["fn"]}')
    print(f'      Test:  AUC-PR={t["aucpr"]:.4f}  F1={t["f1"]:.4f}  FP={t["fp"]}  FN={t["fn"]}')
    print()

# Overfitting analysis
print('--- OVERFITTING CHECK (Train F1 - Test F1 delta) ---')
for r in results:
    delta = r['train']['f1'] - r['test']['f1']
    flag = ' *** OVERFIT' if delta > 0.05 else ''
    print(f'  {r["name"]:<35} Train F1={r["train"]["f1"]:.4f}  Test F1={r["test"]["f1"]:.4f}  Delta={delta:+.4f}{flag}')

log(f'Total time: {time.time()-t_start:.0f}s')
log('DONE')
