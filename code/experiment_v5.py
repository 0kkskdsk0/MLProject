"""
Experiment: IF-free ablation with multiple weight variants
Model pool: A=XGBstd, B=XGBfocal, C=LGB, D=XGBsel, E=LGBsel
17 configs total.
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
all_tr = pd.concat([tr, va])
y_full, y_test = all_tr['y'].values, te['y'].values
log(f'Train+Val: {len(all_tr)} rows, {y_full.sum()} anomalies')
log(f'Test:      {len(te)} rows, {y_test.sum()} anomalies')

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

full_fe = mkfe(all_tr.drop(columns=['y']), fcols)
test_fe = mkfe(te.drop(columns=['y']), fcols)
common = sorted(set(full_fe.columns) & set(test_fe.columns))
scaler = StandardScaler()
X_full = scaler.fit_transform(full_fe[common].values)
X_test = scaler.transform(test_fe[common].values)
log(f'Features: {len(common)} dims')

# === TRAIN MODELS ===
log('Training XGBoost std (A)...')
dt = xgb.DMatrix(X_full, label=y_full)
pr = y_full.mean()
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
    'is_unbalance':True,'seed':42,'verbose':-1}, lgb.Dataset(X_full,label=y_full), 2000,
    callbacks=[lgb.log_evaluation(50)])

log('Feature selection for D/E...')
sel = lgb.train({'objective':'binary','boosting_type':'gbdt','num_leaves':31,
    'learning_rate':0.05,'is_unbalance':True,'seed':42,'verbose':-1},
    lgb.Dataset(X_full,label=y_full), 500, callbacks=[lgb.log_evaluation(0)])
imp = sel.feature_importance(importance_type='gain')
sel_names = [n for _,n in sorted(zip(imp,common), reverse=True)][:100]
sel_idx = [common.index(n) for n in sel_names]
X_full_sel = X_full[:, sel_idx]
X_test_sel = X_test[:, sel_idx]

log('Training XGBoost sel (D)...')
D = xgb.train({'objective':'binary:logistic','max_depth':6,'learning_rate':0.05,
    'subsample':0.8,'colsample_bytree':0.8,'scale_pos_weight':sw,'seed':42,
    'tree_method':'hist','min_child_weight':3}, xgb.DMatrix(X_full_sel,label=y_full),
    2000, evals=[(xgb.DMatrix(X_full_sel,label=y_full),'train')], verbose_eval=0)

log('Training LightGBM sel (E)...')
E = lgb.train({'objective':'binary','boosting_type':'gbdt','num_leaves':31,
    'learning_rate':0.05,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,
    'is_unbalance':True,'seed':42,'verbose':-1}, lgb.Dataset(X_full_sel,label=y_full),
    2000, callbacks=[lgb.log_evaluation(0)])

# === PRECOMPUTE PREDICTIONS ===
log('Computing predictions...')
px = lambda m, X: m.predict(xgb.DMatrix(X))
pl = lambda m, X: m.predict(X)
pa = px(A, X_test)
pb = px(B, X_test)
pc = pl(C, X_test)
pd_ = px(D, X_test_sel)
pe = pl(E, X_test_sel)

def temporal_smooth(s, w=5):
    k = np.ones(w)/w
    return np.convolve(s, k, mode='same')

# === EXPERIMENTS ===
experiments = [
    # (name, scores_func, post_process)
    # Single models
    ('E1  A (XGB std)', lambda: pa, True),
    ('E2  B (XGB focal)', lambda: pb, True),
    ('E3  C (LGB)', lambda: pc, True),
    # Dual models - different weights
    ('E4  A+B 0.5+0.5', lambda: 0.5*pa+0.5*pb, True),
    ('E5  A+B 0.7+0.3', lambda: 0.7*pa+0.3*pb, True),
    ('E6  A+B 0.3+0.7', lambda: 0.3*pa+0.7*pb, True),
    ('E7  A+C 0.5+0.5', lambda: 0.5*pa+0.5*pc, True),
    ('E8  B+C 0.5+0.5', lambda: 0.5*pb+0.5*pc, True),
    ('E9  A+C 0.7+0.3', lambda: 0.7*pa+0.3*pc, True),
    # Triple models - different weights
    ('E10 A+B+C 0.34+0.33+0.33', lambda: 0.34*pa+0.33*pb+0.33*pc, True),
    ('E11 A+B+C 0.5+0.25+0.25', lambda: 0.5*pa+0.25*pb+0.25*pc, True),
    ('E12 A+B+C 0.2+0.4+0.4', lambda: 0.2*pa+0.4*pb+0.4*pc, True),
    # With selected sub-models
    ('E13 0.8*E10+0.2*sel', lambda: 0.8*(0.34*pa+0.33*pb+0.33*pc)+0.2*(0.5*pd_+0.5*pe), True),
    ('E14 0.8*E4+0.2*sel', lambda: 0.8*(0.5*pa+0.5*pb)+0.2*(0.5*pd_+0.5*pe), True),
    # Post-processing variants (on best of above)
]

# Find best basic config first (no temporal smoothing to compare fairly)
log('')
log('Evaluating basic configs (no temporal smooth)...')
basic_results = []
for name, fn, _ in experiments[:14]:
    scores = fn()
    apr = average_precision_score(y_test, scores)
    basic_results.append((name, apr, scores))

# Rank by AUC-PR
basic_results.sort(key=lambda x: x[1], reverse=True)
best_name, best_apr, best_scores = basic_results[0]
log(f'Best basic: {best_name.strip()} (AUC-PR={best_apr:.4f})')

# Add post-processing experiments using best config
log('')
log('Adding post-processing variants...')
def run_pp(name, scores, smooth_w):
    s = temporal_smooth(scores, smooth_w) if smooth_w else scores
    prec, rec, thrs = precision_recall_curve(y_test, s)
    f1s = 2*prec*rec/(prec+rec+1e-10)
    best = np.argmax(f1s)
    thresh = thrs[min(best, len(thrs)-1)] if best < len(thrs) else 0.5
    pred = (s >= thresh).astype(int)
    return {
        'name': name, 'scores': s, 'threshold': thresh, 'pred': pred,
        'aucpr': average_precision_score(y_test, s),
        'acc': accuracy_score(y_test, pred),
        'prec': precision_score(y_test, pred, zero_division=0),
        'rec': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'fp': ((pred==1)&(y_test==0)).sum(),
        'fn': ((pred==0)&(y_test==1)).sum(),
        'pred_anom': pred.sum(),
    }

pp_exps = [
    (f'E15 {best_name.strip()}+smooth3', best_scores, 3),
    (f'E16 {best_name.strip()}+smooth7', best_scores, 7),
    (f'E17 {best_name.strip()}+nosmooth', best_scores, None),
]

# === FULL EVALUATION ===
log('')
log('='*70)
log('FULL RESULTS')
log('='*70)
all_results = []

# Run basic 14 with smoothing
for name, fn, do_smooth in experiments[:14]:
    scores = temporal_smooth(fn(), 5) if do_smooth else fn()
    prec, rec, thrs = precision_recall_curve(y_test, scores)
    f1s = 2*prec*rec/(prec+rec+1e-10)
    best = np.argmax(f1s)
    thresh = thrs[min(best, len(thrs)-1)] if best < len(thrs) else 0.5
    pred = (scores >= thresh).astype(int)
    all_results.append({
        'name': name,
        'aucpr': average_precision_score(y_test, scores),
        'acc': accuracy_score(y_test, pred),
        'prec': precision_score(y_test, pred, zero_division=0),
        'rec': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'fp': ((pred==1)&(y_test==0)).sum(),
        'fn': ((pred==0)&(y_test==1)).sum(),
        'thresh': thresh,
        'pred_anom': pred.sum(),
    })

# Run post-processing
for name, scores, w in pp_exps:
    r = run_pp(name, scores, w)
    all_results.append(r)

# Sort by AUC-PR
all_results.sort(key=lambda x: x['aucpr'], reverse=True)

print(f'{"Rank":<5} {"Exp":<35} {"AUC-PR":<8} {"Acc":<7} {"Prec":<7} {"Recall":<7} {"F1":<8} {"FP":<5} {"FN":<5} {"Pred":<7}')
print('-' * 95)
for i, r in enumerate(all_results, 1):
    print(f'{i:<5} {r["name"]:<35} {r["aucpr"]:<8.4f} {r["acc"]:<7.4f} {r["prec"]:<7.4f} {r["rec"]:<7.4f} {r["f1"]:<8.4f} {r["fp"]:<5} {r["fn"]:<5} {r["pred_anom"]:<3}/{len(y_test):<3}')

log('')
log(f'Total time: {time.time()-t_start:.0f}s')
log('DONE')
