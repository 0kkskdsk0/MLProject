"""
Experiment: IF-free ablation with multiple weight variants
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
    'is_unbalance':True,'seed':42,'verbose':-1}, lgb.Dataset(X_train,label=y_train), 2000,
    callbacks=[lgb.log_evaluation(50)])

log('Feature selection for D/E...')
sel = lgb.train({'objective':'binary','boosting_type':'gbdt','num_leaves':31,
    'learning_rate':0.05,'is_unbalance':True,'seed':42,'verbose':-1},
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
    'is_unbalance':True,'seed':42,'verbose':-1}, lgb.Dataset(X_train_sel,label=y_train),
    2000, callbacks=[lgb.log_evaluation(0)])

# === PRECOMPUTE PREDICTIONS ===
log('Computing predictions...')
px = lambda m, X: m.predict(xgb.DMatrix(X))
pl = lambda m, X: m.predict(X)

# Train set predictions (used only for training, not eval)
# Val set predictions (for threshold selection and comparison)
pa_val = px(A, X_val)
pb_val = px(B, X_val)
pc_val = pl(C, X_val)
pd_val = px(D, X_val_sel)
pe_val = pl(E, X_val_sel)

# Test set predictions (for final evaluation only)
pa_test = px(A, X_test)
pb_test = px(B, X_test)
pc_test = pl(C, X_test)
pd_test = px(D, X_test_sel)
pe_test = pl(E, X_test_sel)

def temporal_smooth(s, w=5):
    k = np.ones(w)/w
    return np.convolve(s, k, mode='same')

# === EXPERIMENT CONFIGS ===
# Using val predictions for threshold selection and ranking
experiments_val = [
    ('E1  A (XGB std)', lambda: pa_val, True),
    ('E2  B (XGB focal)', lambda: pb_val, True),
    ('E3  C (LGB)', lambda: pc_val, True),
    ('E4  A+B 0.5+0.5', lambda: 0.5*pa_val+0.5*pb_val, True),
    ('E5  A+B 0.7+0.3', lambda: 0.7*pa_val+0.3*pb_val, True),
    ('E6  A+B 0.3+0.7', lambda: 0.3*pa_val+0.7*pb_val, True),
    ('E7  A+C 0.5+0.5', lambda: 0.5*pa_val+0.5*pc_val, True),
    ('E8  B+C 0.5+0.5', lambda: 0.5*pb_val+0.5*pc_val, True),
    ('E9  A+C 0.7+0.3', lambda: 0.7*pa_val+0.3*pc_val, True),
    ('E10 A+B+C 0.34+0.33+0.33', lambda: 0.34*pa_val+0.33*pb_val+0.33*pc_val, True),
    ('E11 A+B+C 0.5+0.25+0.25', lambda: 0.5*pa_val+0.25*pb_val+0.25*pc_val, True),
    ('E12 A+B+C 0.2+0.4+0.4', lambda: 0.2*pa_val+0.4*pb_val+0.4*pc_val, True),
    ('E13 0.8*E10+0.2*sel', lambda: 0.8*(0.34*pa_val+0.33*pb_val+0.33*pc_val)+0.2*(0.5*pd_val+0.5*pe_val), True),
    ('E14 0.8*E4+0.2*sel', lambda: 0.8*(0.5*pa_val+0.5*pb_val)+0.2*(0.5*pd_val+0.5*pe_val), True),
]

# Corresponding test score functions (same weights, test data)
experiments_test = [
    ('E1  A (XGB std)', lambda: pa_test),
    ('E2  B (XGB focal)', lambda: pb_test),
    ('E3  C (LGB)', lambda: pc_test),
    ('E4  A+B 0.5+0.5', lambda: 0.5*pa_test+0.5*pb_test),
    ('E5  A+B 0.7+0.3', lambda: 0.7*pa_test+0.3*pb_test),
    ('E6  A+B 0.3+0.7', lambda: 0.3*pa_test+0.7*pb_test),
    ('E7  A+C 0.5+0.5', lambda: 0.5*pa_test+0.5*pc_test),
    ('E8  B+C 0.5+0.5', lambda: 0.5*pb_test+0.5*pc_test),
    ('E9  A+C 0.7+0.3', lambda: 0.7*pa_test+0.3*pc_test),
    ('E10 A+B+C 0.34+0.33+0.33', lambda: 0.34*pa_test+0.33*pb_test+0.33*pc_test),
    ('E11 A+B+C 0.5+0.25+0.25', lambda: 0.5*pa_test+0.25*pb_test+0.25*pc_test),
    ('E12 A+B+C 0.2+0.4+0.4', lambda: 0.2*pa_test+0.4*pb_test+0.4*pc_test),
    ('E13 0.8*E10+0.2*sel', lambda: 0.8*(0.34*pa_test+0.33*pb_test+0.33*pc_test)+0.2*(0.5*pd_test+0.5*pe_test)),
    ('E14 0.8*E4+0.2*sel', lambda: 0.8*(0.5*pa_test+0.5*pb_test)+0.2*(0.5*pd_test+0.5*pe_test)),
]

# === RANK ON VAL (NO SMOOTHING) ===
log('')
log('Ranking on Val (no temporal smooth)...')
val_basic = []
for (name, fn, _), (tname, tfn) in zip(experiments_val[:14], experiments_test[:14]):
    scores = fn()
    apr = average_precision_score(y_val, scores)
    val_basic.append((name, apr, scores, tfn))

val_basic.sort(key=lambda x: x[1], reverse=True)
best_val_name, best_val_apr, best_val_scores, best_test_fn = val_basic[0]
log(f'Best on Val: {best_val_name.strip()} (Val AUC-PR={best_val_apr:.4f})')

# === THRESHOLD SELECTION ON VAL ===
def select_threshold(scores, y_true):
    prec, rec, thrs = precision_recall_curve(y_true, scores)
    f1s = 2*prec*rec/(prec+rec+1e-10)
    best = np.argmax(f1s)
    thresh = thrs[min(best, len(thrs)-1)] if best < len(thrs) else 0.5
    # Select threshold that maximizes F1 on VAL
    pred = (scores >= thresh).astype(int)
    return thresh

def full_metrics(scores, y_true, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        'aucpr': average_precision_score(y_true, scores),
        'acc': accuracy_score(y_true, pred),
        'prec': precision_score(y_true, pred, zero_division=0),
        'rec': recall_score(y_true, pred),
        'f1': f1_score(y_true, pred),
        'fp': ((pred==1)&(y_true==0)).sum(),
        'fn': ((pred==0)&(y_true==1)).sum(),
        'pred_anom': pred.sum(),
        'threshold': threshold,
    }

# === FULL EVALUATION ===
log('')
log('='*70)
log('FULL RESULTS')
log('='*70)
log('Threshold selected on Val (max F1). Metrics reported on Val and Test separately.')
log('')

header_val  = f'{"Rank":<5} {"Exp":<35} {"Val AUC-PR":<10} {"Val Acc":<8} {"Val F1":<8} {"Thresh":<7} {"Val FP":<6} {"Val FN":<6}'
header_test = f'{"Rank":<5} {"Exp":<35} {"Test AUC-PR":<10} {"Test Acc":<8} {"Test F1":<8} {"Test Prec":<8} {"Test Rec":<8} {"Test FP":<6} {"Test FN":<6} {"Pred":<7}'
print('--- VAL METRICS (threshold selected here) ---')
print(header_val)
print('-' * 85)
print()
print('--- TEST METRICS (final evaluation, threshold from Val) ---')
print(header_test)
print('-' * 115)

all_results = []

for (name, fn_val, do_smooth), (tname, fn_test) in zip(experiments_val[:14], experiments_test[:14]):
    # Val scores with smoothing
    s_val = temporal_smooth(fn_val(), 5) if do_smooth else fn_val()
    # Select threshold on Val
    thresh = select_threshold(s_val, y_val)
    # Test scores (same config)
    s_test = fn_test()
    if do_smooth:
        s_test = temporal_smooth(s_test, 5)

    vm = full_metrics(s_val, y_val, thresh)
    tm = full_metrics(s_test, y_test, thresh)
    all_results.append({'name': name, 'val': vm, 'test': tm})

# Also evaluate best config with post-processing variants (E15-E17)
def add_pp_variants(val_scores, test_fn_base, config_label, all_results):
    for pp_name, smooth_w in [
        (f'E15 {config_label}+smooth3', 3),
        (f'E16 {config_label}+smooth7', 7),
        (f'E17 {config_label}+nosmooth', None),
    ]:
        s_val_pp = temporal_smooth(val_scores, smooth_w) if smooth_w else val_scores
        thresh = select_threshold(s_val_pp, y_val)
        s_test_pp = temporal_smooth(test_fn_base(), smooth_w) if smooth_w else test_fn_base()
        vm = full_metrics(s_val_pp, y_val, thresh)
        tm = full_metrics(s_test_pp, y_test, thresh)
        all_results.append({'name': pp_name, 'val': vm, 'test': tm})

best_config_label = best_val_name.strip()
# Find the matching test function for the best config
best_test_fn = None
for (name, fn_val, _), (tname, fn_test) in zip(experiments_val[:14], experiments_test[:14]):
    if name == best_val_name:
        best_test_fn = fn_test
        break

if best_test_fn:
    add_pp_variants(best_val_scores, best_test_fn, best_config_label, all_results)

# Sort by Val AUC-PR (the honest comparison metric)
all_results.sort(key=lambda x: x['val']['aucpr'], reverse=True)

print()
print('--- VAL METRICS (ranking basis) ---')
print(f'{"Rank":<5} {"Exp":<35} {"AUC-PR":<10} {"Acc":<8} {"F1":<8} {"FP":<6} {"FN":<6}')
print('-' * 78)
for i, r in enumerate(all_results, 1):
    v = r['val']
    print(f'{i:<5} {r["name"]:<35} {v["aucpr"]:<10.4f} {v["acc"]:<8.4f} {v["f1"]:<8.4f} {v["fp"]:<6} {v["fn"]:<6}')

print()
print('--- TEST METRICS (final) ---')
print(f'{"Rank":<5} {"Exp":<35} {"AUC-PR":<10} {"Acc":<8} {"F1":<8} {"Prec":<8} {"Rec":<8} {"FP":<6} {"FN":<6} {"Pred":<7}')
print('-' * 105)
for i, r in enumerate(all_results, 1):
    t = r['test']
    print(f'{i:<5} {r["name"]:<35} {t["aucpr"]:<10.4f} {t["acc"]:<8.4f} {t["f1"]:<8.4f} {t["prec"]:<8.4f} {t["rec"]:<8.4f} {t["fp"]:<6} {t["fn"]:<6} {t["pred_anom"]:<3}/{len(y_test):<3}')

winner = all_results[0]
print()
log('='*70)
log('FINAL RESULT (honest evaluation)')
log('='*70)
w = winner['test']
log(f'Best config: {winner["name"].strip()}')
log(f'  Threshold (from Val): {winner["val"]["threshold"]:.4f}')
log(f'  Test AUC-PR: {w["aucpr"]:.4f}')
log(f'  Test F1:     {w["f1"]:.4f}')
log(f'  Test FP:     {w["fp"]}  FN: {w["fn"]}')
log(f'')
log(f'Total time: {time.time()-t_start:.0f}s')
log('DONE')
