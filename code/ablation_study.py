"""
Ablation study for v4 anomaly detection pipeline.
Data split (scheme C): Train [:130816] / Val [130816:134545] / Test [134545:]

8 experiments, from single model to full ensemble.
Only evaluates on Val. Test set is touched once at the end.
"""
import pandas as pd
import numpy as np
import warnings, os, json, pickle, time
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb

np.random.seed(42)

# === CONFIG ===
TRAIN_PATH = 'data/train.csv'
TEST_SIMPLE_PATH = 'data/test_simple.csv'
TEST_COMPLEX_PATH = 'data/test_complex.csv'
OUTPUT_DIR = 'ablation_results'
SPLIT_TRAIN = 130816
SPLIT_VAL = 134545

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FEATURE ENGINEERING (identical to v4) ===
def create_features(df, feature_cols, lof_model=None, scaler_lof=None, pca_model=None, pca_scaler=None):
    features = pd.DataFrame(index=df.index)
    for col in feature_cols:
        features[col] = df[col].values
    for w in [5, 10, 20]:
        for col in feature_cols:
            features[f'{col}_rm{w}'] = df[col].rolling(window=w, min_periods=1).mean().values
            features[f'{col}_rs{w}'] = df[col].rolling(window=w, min_periods=1).std().fillna(0).values
    for col in feature_cols:
        features[f'{col}_d1'] = df[col].diff(1).fillna(0).values
        features[f'{col}_d5'] = df[col].diff(5).fillna(0).values
    for lag in [1, 3]:
        for col in feature_cols[:3]:
            features[f'{col}_l{lag}'] = df[col].shift(lag).bfill().ffill().values
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(3, len(feature_cols))):
            features[f'i_{i}_{j}'] = (df[feature_cols[i]] * df[feature_cols[j]]).values
    X_raw = df[feature_cols].values
    if lof_model is not None:
        X_scaled = scaler_lof.transform(X_raw)
        lof_scores = lof_model.decision_function(X_scaled)
        features['lof_score'] = -lof_scores
    else:
        features['lof_score'] = 0
    row_data = df[feature_cols].values
    features['row_mean'] = row_data.mean(axis=1)
    features['row_std'] = row_data.std(axis=1)
    features['row_max'] = row_data.max(axis=1)
    features['row_min'] = row_data.min(axis=1)
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
    feature_cols = [c for c in df.columns if c != 'y']
    if fit_scaler:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df, scaler
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df

def fit_lof_pca(source_df, feature_cols):
    X_source = source_df[feature_cols].values
    scaler_lof = StandardScaler()
    X_lof = scaler_lof.fit_transform(X_source)
    sample_size = min(15000, len(X_lof))
    idx = np.random.choice(len(X_lof), sample_size, replace=False)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto', n_jobs=-1)
    lof_model.fit(X_lof[idx])
    pca_scaler = StandardScaler()
    X_pca = pca_scaler.fit_transform(X_source)
    pca_model = PCA(n_components=5, random_state=42)
    pca_model.fit(X_pca)
    return lof_model, scaler_lof, pca_model, pca_scaler

def temporal_smooth(scores, window=5):
    kernel = np.ones(window) / window
    return np.convolve(scores, kernel, mode='same')

def apply_temporal_consistency(predictions):
    result = predictions.copy()
    n = len(predictions)
    for i in range(n):
        if predictions[i] == 1:
            left = result[max(0, i-1)] if i > 0 else 0
            right = result[min(n-1, i+1)] if i < n-1 else 0
            if left == 0 and right == 0:
                result[i] = 0
    return result

# === MODEL TRAINERS ===
def train_xgb_standard(X_train, y_train, X_val, y_val, verbosity=100):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    pos_ratio = y_train.mean()
    scale_pos_weight = max(1.0, (1 - pos_ratio) / (pos_ratio + 1e-8))
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42, 'tree_method': 'hist', 'min_child_weight': 3,
    }
    return xgb.train(params, dtrain, num_boost_round=2000,
                     evals=[(dtrain, 'train'), (dval, 'val')],
                     early_stopping_rounds=100, verbose_eval=verbosity)

def train_xgb_focal(X_train, y_train, X_val, y_val, verbosity=100):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    pos_ratio = y_train.mean()
    scale_pos_weight = max(1.0, (1 - pos_ratio) / (pos_ratio + 1e-8)) * 2
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'max_depth': 5, 'learning_rate': 0.03,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42, 'tree_method': 'hist',
    }
    return xgb.train(params, dtrain, num_boost_round=1500,
                     evals=[(dtrain, 'train'), (dval, 'val')],
                     early_stopping_rounds=100, verbose_eval=verbosity)

def train_lgb(X_train, y_train, X_val, y_val, verbosity=50):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        'objective': 'binary', 'metric': 'average_precision',
        'boosting_type': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.05, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'is_unbalance': True, 'seed': 42, 'verbose': -1,
    }
    return lgb.train(params, train_data, num_boost_round=2000,
                     valid_sets=[train_data, val_data],
                     valid_names=['train', 'val'],
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(verbosity)])

def train_iforest(X_train, y_train):
    X_normal = X_train[y_train == 0]
    contamination = max(0.001, min(0.1, y_train.mean() * 3))
    if len(X_normal) > 50000:
        idx = np.random.choice(len(X_normal), 50000, replace=False)
        X_normal = X_normal[idx]
    return IsolationForest(n_estimators=200, contamination=contamination,
                           random_state=42, n_jobs=-1, max_samples=10000).fit(X_normal)

def train_cascade(X_train, y_train, X_val, y_val):
    if_model = train_iforest(X_train, y_train)
    if_scores = if_model.decision_function(X_train)
    if_probs = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    suspicious = if_probs > 0.3
    train_mask = suspicious | (y_train == 1)
    X_refine = X_train[train_mask]
    y_refine = y_train[train_mask]
    if len(y_refine[y_refine == 1]) < 10:
        return None
    dtrain = xgb.DMatrix(X_refine, label=y_refine)
    dval = xgb.DMatrix(X_val, label=y_val)
    pos_ratio = y_refine.mean()
    scale_pos_weight = max(1.0, (1 - pos_ratio) / (pos_ratio + 1e-8))
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'aucpr',
        'max_depth': 5, 'learning_rate': 0.05,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42, 'tree_method': 'hist',
    }
    model = xgb.train(params, dtrain, num_boost_round=1000,
                      evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=50)
    return {'if_model': if_model, 'refine_model': model}

def cascade_predict(cascade, X):
    if cascade is None:
        return None
    if_scores = cascade['if_model'].decision_function(X)
    if_probs = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    refine_probs = cascade['refine_model'].predict(xgb.DMatrix(X))
    return if_probs * refine_probs

def select_features_lgb(X_train, y_train, feature_names, max_features=100):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary', 'metric': 'average_precision',
        'boosting_type': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.05, 'is_unbalance': True,
        'seed': 42, 'verbose': -1,
    }
    model = lgb.train(params, train_data, num_boost_round=500,
                      callbacks=[lgb.log_evaluation(50)])
    importance = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:max_features]]

# === PREDICTION HELPERS ===
def predict_xgb(model, X):
    return model.predict(xgb.DMatrix(X))

def predict_lgb(model, X):
    return model.predict(X, num_iteration=model.best_iteration)

def predict_if(model, X):
    scores = model.decision_function(X)
    return 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

def evaluate(y_true, scores, threshold=None):
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[min(best_idx, len(thresholds)-1)] if best_idx < len(thresholds) else 0.5
    pred = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, pred)
    auc_pr = average_precision_score(y_true, scores)
    return auc_pr, f1, threshold, pred

# === EXPERIMENT DEFINITIONS ===
EXPERIMENTS = [
    {
        'name': 'E1_xgb_std',
        'desc': 'XGBoost standard only',
        'models': ['xgb_std'],
        'weights': [1.0],
    },
    {
        'name': 'E2_lgb',
        'desc': 'LightGBM only',
        'models': ['lgb'],
        'weights': [1.0],
    },
    {
        'name': 'E3_iforest',
        'desc': 'Isolation Forest only',
        'models': ['iforest'],
        'weights': [1.0],
    },
    {
        'name': 'E4_xgb_lgb',
        'desc': 'XGBoost std + LightGBM (0.5 each)',
        'models': ['xgb_std', 'lgb'],
        'weights': [0.5, 0.5],
    },
    {
        'name': 'E5_xgb_if',
        'desc': 'XGBoost std + IForest (0.5 each)',
        'models': ['xgb_std', 'iforest'],
        'weights': [0.5, 0.5],
    },
    {
        'name': 'E6_xgb_lgb_if',
        'desc': 'XGBoost + LGB + IForest (0.33 each)',
        'models': ['xgb_std', 'lgb', 'iforest'],
        'weights': [1/3, 1/3, 1/3],
    },
    {
        'name': 'E7_4base',
        'desc': 'XGBoost std + focal + LGB + IForest (0.25 each)',
        'models': ['xgb_std', 'xgb_focal', 'lgb', 'iforest'],
        'weights': [0.25, 0.25, 0.25, 0.25],
    },
    {
        'name': 'E8_full_v4',
        'desc': 'Full v4: 4base + cascade + selected (0.8 base + 0.2 sel)',
        'models': ['xgb_std', 'xgb_focal', 'lgb', 'iforest', 'cascade', 'sel'],
        'weights': None,
    },
]

def run_experiment(exp, X_train, y_train, X_val, y_val, feature_names):
    """Train models for one experiment and evaluate on Val."""
    trained = {}
    result = {'name': exp['name'], 'desc': exp['desc']}
    t0 = time.time()

    # Train requested models
    if 'xgb_std' in exp['models']:
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Training XGBoost standard...")
        trained['xgb_std'] = train_xgb_standard(X_train, y_train, X_val, y_val)
    if 'xgb_focal' in exp['models']:
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Training XGBoost focal...")
        trained['xgb_focal'] = train_xgb_focal(X_train, y_train, X_val, y_val)
    if 'lgb' in exp['models']:
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Training LightGBM...")
        trained['lgb'] = train_lgb(X_train, y_train, X_val, y_val)
    if 'iforest' in exp['models']:
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Training Isolation Forest...")
        trained['iforest'] = train_iforest(X_train, y_train)
    if 'cascade' in exp['models']:
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Training Cascade...")
        trained['cascade'] = train_cascade(X_train, y_train, X_val, y_val)
    if 'sel' in exp['models']:
        trained['sel_xgb'] = None   # trained later if needed
        trained['sel_lgb'] = None

    # Build ensemble prediction on Val
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] Building ensemble prediction...")
    if exp['name'] == 'E8_full_v4':
        # Full v4 ensemble logic
        pred_xgb_std = predict_xgb(trained['xgb_std'], X_val)
        pred_xgb_focal = predict_xgb(trained['xgb_focal'], X_val)
        pred_lgb = predict_lgb(trained['lgb'], X_val)
        pred_if = predict_if(trained['iforest'], X_val)
        base = 0.25 * pred_xgb_std + 0.25 * pred_xgb_focal + 0.25 * pred_lgb + 0.25 * pred_if
        pred_cascade = cascade_predict(trained['cascade'], X_val)
        if pred_cascade is not None:
            base = 0.7 * base + 0.3 * pred_cascade
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Feature selection + retrain...")
        selected_names = select_features_lgb(X_train, y_train, feature_names, max_features=100)
        selected_idx = [feature_names.index(n) for n in selected_names]
        X_train_sel = X_train[:, selected_idx]
        X_val_sel = X_val[:, selected_idx]
        trained['sel_xgb'] = train_xgb_standard(X_train_sel, y_train, X_val_sel, y_val)
        trained['sel_lgb'] = train_lgb(X_train_sel, y_train, X_val_sel, y_val)
        pred_xgb_sel = predict_xgb(trained['sel_xgb'], X_val_sel)
        pred_lgb_sel = predict_lgb(trained['sel_lgb'], X_val_sel)
        sel_ensemble = 0.5 * pred_xgb_sel + 0.5 * pred_lgb_sel
        scores = 0.8 * base + 0.2 * sel_ensemble
    else:
        # Simple weighted ensemble
        scores = None
        for model_name, w in zip(exp['models'], exp['weights']):
            if model_name == 'xgb_std':
                p = predict_xgb(trained['xgb_std'], X_val)
            elif model_name == 'lgb':
                p = predict_lgb(trained['lgb'], X_val)
            elif model_name == 'iforest':
                p = predict_if(trained['iforest'], X_val)
            else:
                continue
            if scores is None:
                scores = w * p
            else:
                scores += w * p

    scores_smooth = temporal_smooth(scores, window=5)
    auc_pr, f1, thresh, pred = evaluate(y_val, scores_smooth)
    result['time'] = time.time() - t0
    result['auc_pr'] = round(auc_pr, 4)
    result['f1'] = round(f1, 4)
    result['threshold'] = round(thresh, 4)
    result['val_anomalies'] = int(pred.sum())

    print(f"  AUC-PR={result['auc_pr']:.4f}  F1={result['f1']:.4f}  "
          f"thresh={result['threshold']:.4f}  ({result['time']:.0f}s)")

    return result, trained

def main():
    print("=" * 70)
    print("ABLATION STUDY - v4 Component Analysis")
    print("Split: [:130816] / [130816:134545] / [134545:]")
    print("=" * 70)

    # Load
    print("\n[1] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_simple_df = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex_df = pd.read_csv(TEST_COMPLEX_PATH)
    feature_cols = [c for c in train_df.columns if c.startswith('f')]

    medians = train_df[feature_cols].median()
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(medians[col])
        test_simple_df[col] = test_simple_df[col].fillna(medians[col])
        test_complex_df[col] = test_complex_df[col].fillna(medians[col])

    # Split
    train_raw = train_df.iloc[:SPLIT_TRAIN].copy()
    val_raw = train_df.iloc[SPLIT_TRAIN:SPLIT_VAL].copy()
    test_raw = train_df.iloc[SPLIT_VAL:].copy()
    y_train = train_raw['y'].values
    y_val = val_raw['y'].values
    y_test = test_raw['y'].values

    print(f"  Train: {len(train_raw)} rows, {y_train.sum()} anomalies")
    print(f"  Val:   {len(val_raw)} rows, {y_val.sum()} anomalies ({y_val.mean()*100:.2f}%)")
    print(f"  Test:  {len(test_raw)} rows, {y_test.sum()} anomalies ({y_test.mean()*100:.2f}%)")

    # Feature engineering (shared across all experiments)
    print("\n[2] Feature engineering...")
    # Fit LOF/PCA on train only
    lof_model, scaler_lof, pca_model, pca_scaler = fit_lof_pca(train_raw, feature_cols)

    train_fe = create_features(train_raw.drop(columns=['y']), feature_cols,
                               lof_model, scaler_lof, pca_model, pca_scaler)
    val_fe = create_features(val_raw.drop(columns=['y']), feature_cols,
                             lof_model, scaler_lof, pca_model, pca_scaler)
    test_fe = create_features(test_raw.drop(columns=['y']), feature_cols,
                              lof_model, scaler_lof, pca_model, pca_scaler)

    common_cols = sorted(list(set(train_fe.columns) & set(val_fe.columns) & set(test_fe.columns)))
    train_fe = train_fe[common_cols]
    val_fe = val_fe[common_cols]
    test_fe = test_fe[common_cols]

    train_scaled, scaler = preprocess(train_fe, fit_scaler=True)
    val_scaled = preprocess(val_fe, scaler=scaler)
    test_scaled = preprocess(test_fe, scaler=scaler)

    X_train = train_scaled.values
    X_val = val_scaled.values
    X_test = test_scaled.values
    feature_names = list(train_fe.columns)
    print(f"  Features: {len(feature_names)} dims")

    # Run experiments
    total_exp = len(EXPERIMENTS)
    print("\n[3] Running experiments...")
    all_results = []
    all_trained = {}
    for idx, exp in enumerate(EXPERIMENTS):
        now = datetime.now().strftime('%H:%M:%S')
        print(f"\n{'='*60}")
        print(f"  [{idx+1}/{total_exp}] {exp['name']}: {exp['desc']}  ({now})")
        print(f"{'='*60}")
        result, trained = run_experiment(exp, X_train, y_train, X_val, y_val, feature_names)
        all_results.append(result)
        all_trained[exp['name']] = trained
        print(f"  >>> Done [{idx+1}/{total_exp}] | AUC-PR={result['auc_pr']:.4f} F1={result['f1']:.4f} ({result['time']:.0f}s)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - Val Set Results")
    print("=" * 70)
    summary = pd.DataFrame(all_results)
    summary = summary[['name', 'auc_pr', 'f1', 'threshold', 'val_anomalies', 'time']]
    print(summary.to_string(index=False))

    summary.to_csv(f'{OUTPUT_DIR}/ablation_results.csv', index=False)
    print(f"\n  Saved to {OUTPUT_DIR}/ablation_results.csv")

    # Best experiment (by AUC-PR)
    best = max(all_results, key=lambda r: r['auc_pr'])
    print(f"\n  Best: {best['name']} (AUC-PR={best['auc_pr']:.4f}, F1={best['f1']:.4f})")

    # === FINAL: train best config on Train+Val, predict Test, then test_simple/complex ===
    print("\n" + "=" * 70)
    print(f"FINAL - Retrain '{best['name']}' on Train+Val, predict Test + test sets")
    print("=" * 70)

    full_train_raw = pd.concat([train_raw, val_raw])
    full_y = full_train_raw['y'].values

    lof_model_f, scaler_lof_f, pca_model_f, pca_scaler_f = fit_lof_pca(full_train_raw, feature_cols)
    full_fe = create_features(full_train_raw.drop(columns=['y']), feature_cols,
                              lof_model_f, scaler_lof_f, pca_model_f, pca_scaler_f)
    full_fe = full_fe[common_cols]
    full_scaled, full_scaler = preprocess(full_fe, fit_scaler=True)
    X_full = full_scaled.values

    # Prepare test sets
    test_simple_fe = create_features(test_simple_df, feature_cols,
                                     lof_model_f, scaler_lof_f, pca_model_f, pca_scaler_f)
    test_complex_fe = create_features(test_complex_df, feature_cols,
                                      lof_model_f, scaler_lof_f, pca_model_f, pca_scaler_f)
    for col in common_cols:
        if col not in test_simple_fe.columns:
            test_simple_fe[col] = 0
            test_complex_fe[col] = 0
    test_simple_fe = test_simple_fe[common_cols]
    test_complex_fe = test_complex_fe[common_cols]
    X_test_simple = preprocess(test_simple_fe, scaler=full_scaler).values
    X_test_complex = preprocess(test_complex_fe, scaler=full_scaler).values

    # Retrain best config
    def predict_ensemble_best(exp_name, models, X, X_sel=None):
        if exp_name == 'E8_full_v4':
            base = 0.25 * predict_xgb(models['xgb_std'], X) \
                 + 0.25 * predict_xgb(models['xgb_focal'], X) \
                 + 0.25 * predict_lgb(models['lgb'], X) \
                 + 0.25 * predict_if(models['iforest'], X)
            pred_cascade = cascade_predict(models['cascade'], X)
            if pred_cascade is not None:
                base = 0.7 * base + 0.3 * pred_cascade
            if X_sel is not None:
                pred_xgb_sel = predict_xgb(models['sel_xgb'], X_sel)
                pred_lgb_sel = predict_lgb(models['sel_lgb'], X_sel)
                base = 0.8 * base + 0.2 * (0.5 * pred_xgb_sel + 0.5 * pred_lgb_sel)
            return temporal_smooth(base, window=5)
        else:
            weights = EXPERIMENTS[[e['name'] for e in EXPERIMENTS].index(exp_name)]['weights']
            model_names = EXPERIMENTS[[e['name'] for e in EXPERIMENTS].index(exp_name)]['models']
            scores = None
            for model_name, w in zip(model_names, weights):
                if model_name == 'xgb_std':
                    p = predict_xgb(models['xgb_std'], X)
                elif model_name == 'lgb':
                    p = predict_lgb(models['lgb'], X)
                elif model_name == 'iforest':
                    p = predict_if(models['iforest'], X)
                else:
                    continue
                scores = (w * p) if scores is None else scores + w * p
            return temporal_smooth(scores, window=5)

    final_models = {}
    print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Retraining final model...")
    if 'xgb_std' in [m for e in EXPERIMENTS if e['name'] == best['name'] for m in e['models']] \
       or best['name'] == 'E8_full_v4':
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] XGBoost standard...")
        final_models['xgb_std'] = train_xgb_standard(X_full, full_y, X_full, full_y)
    if 'xgb_focal' in [m for e in EXPERIMENTS if e['name'] == best['name'] for m in e['models']] \
       or best['name'] == 'E8_full_v4':
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] XGBoost focal...")
        final_models['xgb_focal'] = train_xgb_focal(X_full, full_y, X_full, full_y)
    if 'lgb' in [m for e in EXPERIMENTS if e['name'] == best['name'] for m in e['models']] \
       or best['name'] == 'E8_full_v4':
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] LightGBM...")
        final_models['lgb'] = train_lgb(X_full, full_y, X_full, full_y)
    print(f"    [{datetime.now().strftime('%H:%M:%S')}] IForest...")
    final_models['iforest'] = train_iforest(X_full, full_y)
    if 'cascade' in [m for e in EXPERIMENTS if e['name'] == best['name'] for m in e['models']] \
       or best['name'] == 'E8_full_v4':
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Cascade...")
        final_models['cascade'] = train_cascade(X_full, full_y, X_full, full_y)
    if 'sel' in [m for e in EXPERIMENTS if e['name'] == best['name'] for m in e['models']] \
       or best['name'] == 'E8_full_v4':
        print(f"    [{datetime.now().strftime('%H:%M:%S')}] Feature selection...")
        sel_names = select_features_lgb(X_full, full_y, feature_names, max_features=100)
        sel_idx = [feature_names.index(n) for n in sel_names]
        X_full_sel = X_full[:, sel_idx]
        X_test_simple_sel = X_test_simple[:, sel_idx]
        X_test_complex_sel = X_test_complex[:, sel_idx]
        final_models['sel_xgb'] = train_xgb_standard(X_full_sel, full_y, X_full_sel, full_y)
        final_models['sel_lgb'] = train_lgb(X_full_sel, full_y, X_full_sel, full_y)

    # Predict on internal test set
    if 'sel' in [m for e in EXPERIMENTS if e['name'] == best['name'] for m in e['models']] \
       or best['name'] == 'E8_full_v4':
        X_test_sel = X_test[:, sel_idx]
    else:
        X_test_sel = None

    scores_test = predict_ensemble_best(best['name'], final_models, X_test, X_test_sel)
    scores_simple = predict_ensemble_best(best['name'], final_models, X_test_simple, X_test_simple_sel if 'sel' in str(best['name']) else None)
    scores_complex = predict_ensemble_best(best['name'], final_models, X_test_complex, X_test_complex_sel if 'sel' in str(best['name']) else None)

    # Use best threshold
    thresh = best['threshold']
    pred_test = apply_temporal_consistency((scores_test >= thresh).astype(int))
    pred_simple = apply_temporal_consistency((scores_simple >= thresh).astype(int))
    pred_complex = apply_temporal_consistency((scores_complex >= thresh).astype(int))

    test_auc_pr = average_precision_score(y_test, scores_test)
    test_f1 = f1_score(y_test, pred_test)

    print(f"\n  Internal Test: AUC-PR={test_auc_pr:.4f}, F1={test_f1:.4f}")
    print(f"  Task1: {pred_simple.sum()} anomalies ({pred_simple.mean()*100:.2f}%)")
    print(f"  Task2: {pred_complex.sum()} anomalies ({pred_complex.mean()*100:.2f}%)")

    # Save final predictions
    pd.DataFrame({'y_pred': pred_simple}).to_csv(f'{OUTPUT_DIR}/pred_simple.csv', index=False)
    pd.DataFrame({'y_pred': pred_complex}).to_csv(f'{OUTPUT_DIR}/pred_complex.csv', index=False)

    # Save summary
    with open(f'{OUTPUT_DIR}/final_summary.json', 'w') as f:
        json.dump({
            'best_experiment': best['name'],
            'val_auc_pr': best['auc_pr'],
            'val_f1': best['f1'],
            'test_auc_pr': round(test_auc_pr, 4),
            'test_f1': round(test_f1, 4),
            'threshold': thresh,
            'task1_anomalies': int(pred_simple.sum()),
            'task2_anomalies': int(pred_complex.sum()),
            'split': {'train': SPLIT_TRAIN, 'val': SPLIT_VAL, 'test': '134545:'},
        }, f, indent=2)

    print(f"\n  All results saved to {OUTPUT_DIR}/")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == '__main__':
    main()
