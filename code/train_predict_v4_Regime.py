"""
Robust Anomaly Detection v4 - Full Implementation
Optimizations: multi-fold CV, focal loss, cascade architecture,
              adaptive threshold, temporal smoothing, feature selection

To run: python train_predict_v4_full.py
Output: submission_v4/pred_simple.csv, submission_v4/pred_complex.csv
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

np.random.seed(42)

# === CONFIG ===
TRAIN_PATH = 'data/train.csv'
TEST_SIMPLE_PATH = 'data/test_simple.csv'
TEST_COMPLEX_PATH = 'data/test_complex.csv'
OUTPUT_DIR = 'submission_v4'
os.makedirs(OUTPUT_DIR, exist_ok=True)
CV_SPLITS = [131680, 134035]

# === REGIME DETECTION ===
def detect_regimes(df):
    f1 = df['f1'].values
    changes = [0]
    for i in range(1, len(f1)):
        if abs(f1[i] - f1[i-1]) > 0.3:
            changes.append(i)
    changes.append(len(f1))
    regime_ids = np.zeros(len(f1), dtype=int)
    for i in range(len(changes) - 1):
        regime_ids[changes[i]:changes[i+1]] = i
    return regime_ids

# === FEATURE ENGINEERING ===
def create_features(df, regime_ids, feature_cols, lof_model=None, scaler_lof=None, pca_model=None, pca_scaler=None):
    features = pd.DataFrame(index=df.index)
    for col in feature_cols:
        features[col] = df[col].values
    # Rolling stats
    for w in [5, 10, 20]:
        for col in feature_cols:
            features[f'{col}_rm{w}'] = df[col].rolling(window=w, min_periods=1).mean().values
            features[f'{col}_rs{w}'] = df[col].rolling(window=w, min_periods=1).std().fillna(0).values
    # Differences
    for col in feature_cols:
        features[f'{col}_d1'] = df[col].diff(1).fillna(0).values
        features[f'{col}_d5'] = df[col].diff(5).fillna(0).values
    # Lags
    for lag in [1, 3]:
        for col in feature_cols[:3]:
            features[f'{col}_l{lag}'] = df[col].shift(lag).bfill().ffill().values
    # Regime ID
    features['regime_id'] = regime_ids
    # Interactions
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(3, len(feature_cols))):
            features[f'i_{i}_{j}'] = (df[feature_cols[i]] * df[feature_cols[j]]).values
    # LOF scores
    if lof_model is not None:
        X_raw = df[feature_cols].values
        X_scaled = scaler_lof.transform(X_raw)
        lof_scores = lof_model.decision_function(X_scaled)
        features['lof_score'] = -lof_scores
    else:
        features['lof_score'] = 0
    # Row stats
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

# === PREPROCESSING ===
def preprocess(df, scaler=None, fit_scaler=False):
    df = df.copy().ffill().bfill()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    exclude = ['y', 'regime_id']
    feature_cols = [c for c in df.columns if c not in exclude]
    if fit_scaler:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df, scaler
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df

# === MODEL TRAINING ===
def train_xgb_standard(X_train, y_train, X_val, y_val):
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
                     early_stopping_rounds=100, verbose_eval=200)

def train_xgb_focal(X_train, y_train, X_val, y_val):
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
                     early_stopping_rounds=100, verbose_eval=200)

def train_lgb(X_train, y_train, X_val, y_val):
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
                     callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

def train_iforest(X_train, y_train):
    X_normal = X_train[y_train == 0]
    contamination = max(0.001, min(0.1, y_train.mean() * 3))
    if len(X_normal) > 50000:
        idx = np.random.choice(len(X_normal), 50000, replace=False)
        X_normal = X_normal[idx]
    return IsolationForest(n_estimators=200, contamination=contamination,
                           random_state=42, n_jobs=-1, max_samples=10000).fit(X_normal)

# === CASCADE ===
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
                      evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=100)
    return {'if_model': if_model, 'refine_model': model}

def cascade_predict(cascade, X):
    if cascade is None: return None
    if_scores = cascade['if_model'].decision_function(X)
    if_probs = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    refine_probs = cascade['refine_model'].predict(xgb.DMatrix(X))
    return if_probs * refine_probs

# === POST-PROCESSING ===
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

def adaptive_threshold(scores, base_threshold, regime_ids):
    adjusted = scores.copy()
    for rid in np.unique(regime_ids):
        mask = regime_ids == rid
        local_median = np.median(scores[mask])
        global_median = np.median(scores)
        adjustment = (local_median - global_median) * 0.3
        adjusted[mask] = scores[mask] - adjustment
    return adjusted

# === FEATURE SELECTION ===
def select_features_lgb(X_train, y_train, feature_names, max_features=100):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary', 'metric': 'average_precision',
        'boosting_type': 'gbdt', 'num_leaves': 31,
        'learning_rate': 0.05, 'is_unbalance': True,
        'seed': 42, 'verbose': -1,
    }
    model = lgb.train(params, train_data, num_boost_round=500)
    importance = model.feature_importance(importance_type='gain')
    ranked = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:max_features]]

# === TRAIN FOLD ===
def train_fold(train_df, train_end, feature_cols, lof_model, scaler_lof, pca_model, pca_scaler):
    train_raw = train_df.iloc[:train_end].copy()
    val_raw = train_df.iloc[train_end:].copy()
    y_train = train_raw['y'].values
    y_val = val_raw['y'].values
    
    regime_train = detect_regimes(train_raw)
    regime_val = detect_regimes(val_raw)
    
    train_fe = create_features(train_raw.drop(columns=['y']), regime_train, feature_cols,
                               lof_model, scaler_lof, pca_model, pca_scaler)
    val_fe = create_features(val_raw.drop(columns=['y']), regime_val, feature_cols,
                             lof_model, scaler_lof, pca_model, pca_scaler)
    
    common_cols = sorted(list(set(train_fe.columns) & set(val_fe.columns)))
    train_fe = train_fe[common_cols]
    val_fe = val_fe[common_cols]
    
    train_scaled, scaler = preprocess(train_fe, fit_scaler=True)
    val_scaled = preprocess(val_fe, scaler=scaler)
    X_train = train_scaled.values
    X_val = val_scaled.values
    
    print(f"  Train: {len(train_raw)} rows, {y_train.sum()} anom | Val: {len(val_raw)} rows, {y_val.sum()} anom")
    
    models = {}
    print("  -> XGB std...")
    models['xgb_std'] = train_xgb_standard(X_train, y_train, X_val, y_val)
    print("  -> XGB focal...")
    models['xgb_focal'] = train_xgb_focal(X_train, y_train, X_val, y_val)
    print("  -> LGBM...")
    models['lgb'] = train_lgb(X_train, y_train, X_val, y_val)
    print("  -> IForest...")
    models['iforest'] = train_iforest(X_train, y_train)
    print("  -> Cascade...")
    models['cascade'] = train_cascade(X_train, y_train, X_val, y_val)
    
    pred_xgb_std = models['xgb_std'].predict(xgb.DMatrix(X_val))
    pred_xgb_focal = models['xgb_focal'].predict(xgb.DMatrix(X_val))
    pred_lgb = models['lgb'].predict(X_val, num_iteration=models['lgb'].best_iteration)
    if_scores = models['iforest'].decision_function(X_val)
    pred_if = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    
    ensemble = 0.25 * pred_xgb_std + 0.25 * pred_xgb_focal + 0.25 * pred_lgb + 0.25 * pred_if
    pred_cascade = cascade_predict(models['cascade'], X_val)
    if pred_cascade is not None:
        ensemble = 0.7 * ensemble + 0.3 * pred_cascade
    
    ensemble_smooth = temporal_smooth(ensemble, window=5)
    precision, recall, thresholds = precision_recall_curve(y_val, ensemble_smooth)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[min(best_idx, len(thresholds)-1)] if best_idx < len(thresholds) else 0.5
    
    val_pred = apply_temporal_consistency((ensemble_smooth >= best_thresh).astype(int))
    f1 = f1_score(y_val, val_pred)
    auc_pr = average_precision_score(y_val, ensemble_smooth)
    auc_roc = roc_auc_score(y_val, ensemble_smooth)
    
    print(f"  AUC-PR={auc_pr:.4f}, F1={f1:.4f}, thresh={best_thresh:.4f}")
    
    models['scaler'] = scaler
    models['feature_cols'] = common_cols
    return models, best_thresh, auc_pr, f1

# === MAIN ===
def main():
    print("=" * 70)
    print("ANOMALY DETECTION PIPELINE v4")
    print("=" * 70)
    
    print("\n[1] Loading...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_simple_df = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex_df = pd.read_csv(TEST_COMPLEX_PATH)
    feature_cols = [c for c in train_df.columns if c.startswith('f')]
    
    medians = train_df[feature_cols].median()
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(medians[col])
        test_simple_df[col] = test_simple_df[col].fillna(medians[col])
        test_complex_df[col] = test_complex_df[col].fillna(medians[col])
    
    print("\n[2] Global LOF & PCA...")
    X_full = train_df[feature_cols].values
    scaler_lof = StandardScaler()
    X_lof = scaler_lof.fit_transform(X_full)
    sample_size = min(15000, len(X_lof))
    idx = np.random.choice(len(X_lof), sample_size, replace=False)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto', n_jobs=-1)
    lof_model.fit(X_lof[idx])
    
    pca_scaler = StandardScaler()
    X_pca = pca_scaler.fit_transform(X_full)
    pca_model = PCA(n_components=5, random_state=42)
    pca_model.fit(X_pca)
    
    print("\n[3] Multi-fold CV...")
    fold_results = []
    for fold, train_end in enumerate(CV_SPLITS):
        print(f"\n  Fold {fold+1}: train_end={train_end}")
        models, thresh, auc_pr, f1 = train_fold(
            train_df, train_end, feature_cols, lof_model, scaler_lof, pca_model, pca_scaler)
        fold_results.append({'models': models, 'threshold': thresh, 'auc_pr': auc_pr, 'f1': f1})
    
    avg_auc_pr = np.mean([r['auc_pr'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    print(f"\n  CV Avg: AUC-PR={avg_auc_pr:.4f}, F1={avg_f1:.4f}")
    
    best_fold = max(fold_results, key=lambda x: x['f1'])
    print(f"  Best: F1={best_fold['f1']:.4f}, AUC-PR={best_fold['auc_pr']:.4f}")
    
    print("\n[4] Final model on full data...")
    FINAL_TRAIN_END = 135046
    final_train = train_df.iloc[:FINAL_TRAIN_END].copy()
    final_y = final_train['y'].values
    regime_final = detect_regimes(final_train)
    
    final_fe = create_features(final_train.drop(columns=['y']), regime_final, feature_cols,
                               lof_model, scaler_lof, pca_model, pca_scaler)
    final_scaled, final_scaler = preprocess(final_fe, fit_scaler=True)
    X_final = final_scaled.values
    
    final_models = {}
    print("  -> XGB std...")
    final_models['xgb_std'] = train_xgb_standard(X_final, final_y, X_final, final_y)
    print("  -> XGB focal...")
    final_models['xgb_focal'] = train_xgb_focal(X_final, final_y, X_final, final_y)
    print("  -> LGBM...")
    final_models['lgb'] = train_lgb(X_final, final_y, X_final, final_y)
    print("  -> IForest...")
    final_models['iforest'] = train_iforest(X_final, final_y)
    print("  -> Cascade...")
    final_models['cascade'] = train_cascade(X_final, final_y, X_final, final_y)
    
    print("\n[5] Feature selection...")
    feature_names = list(final_fe.columns)
    selected_names = select_features_lgb(X_final, final_y, feature_names, max_features=100)
    selected_idx = [feature_names.index(name) for name in selected_names]
    
    print("  Retrain selected...")
    X_final_sel = X_final[:, selected_idx]
    final_models['xgb_std_sel'] = train_xgb_standard(X_final_sel, final_y, X_final_sel, final_y)
    final_models['lgb_sel'] = train_lgb(X_final_sel, final_y, X_final_sel, final_y)
    
    print("\n[6] Test prep...")
    regime_simple = detect_regimes(test_simple_df)
    regime_complex = detect_regimes(test_complex_df)
    
    test_simple_fe = create_features(test_simple_df, regime_simple, feature_cols,
                                      lof_model, scaler_lof, pca_model, pca_scaler)
    test_complex_fe = create_features(test_complex_df, regime_complex, feature_cols,
                                       lof_model, scaler_lof, pca_model, pca_scaler)
    
    final_cols = list(final_fe.columns)
    for col in final_cols:
        if col not in test_simple_fe.columns:
            test_simple_fe[col] = 0
            test_complex_fe[col] = 0
    test_simple_fe = test_simple_fe[final_cols]
    test_complex_fe = test_complex_fe[final_cols]
    
    test_simple_scaled = preprocess(test_simple_fe, scaler=final_scaler)
    test_complex_scaled = preprocess(test_complex_fe, scaler=final_scaler)
    
    X_test_simple = test_simple_scaled.values
    X_test_complex = test_complex_scaled.values
    X_test_simple_sel = X_test_simple[:, selected_idx]
    X_test_complex_sel = X_test_complex[:, selected_idx]
    
    print("\n[7] Predict...")
    def predict_ensemble(models, X, X_sel=None):
        pred_xgb_std = models['xgb_std'].predict(xgb.DMatrix(X))
        pred_xgb_focal = models['xgb_focal'].predict(xgb.DMatrix(X))
        pred_lgb = models['lgb'].predict(X, num_iteration=models['lgb'].best_iteration)
        if_scores = models['iforest'].decision_function(X)
        pred_if = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
        base = 0.25 * pred_xgb_std + 0.25 * pred_xgb_focal + 0.25 * pred_lgb + 0.25 * pred_if
        pred_cascade = cascade_predict(models['cascade'], X)
        if pred_cascade is not None:
            base = 0.7 * base + 0.3 * pred_cascade
        if X_sel is not None:
            pred_xgb_sel = models['xgb_std_sel'].predict(xgb.DMatrix(X_sel))
            pred_lgb_sel = models['lgb_sel'].predict(X_sel, num_iteration=models['lgb_sel'].best_iteration)
            base = 0.8 * base + 0.2 * (0.5 * pred_xgb_sel + 0.5 * pred_lgb_sel)
        return temporal_smooth(base, window=5)
    
    scores_simple = predict_ensemble(final_models, X_test_simple, X_test_simple_sel)
    scores_complex = predict_ensemble(final_models, X_test_complex, X_test_complex_sel)
    
    base_thresh = best_fold['threshold']
    scores_complex_adj = adaptive_threshold(scores_complex, base_thresh, regime_complex)
    
    pred_simple = apply_temporal_consistency((scores_simple >= base_thresh).astype(int))
    pred_complex = apply_temporal_consistency((scores_complex_adj >= base_thresh).astype(int))
    
    print(f"\n  Task1: {pred_simple.sum()} anom ({pred_simple.mean()*100:.2f}%)")
    print(f"  Task2: {pred_complex.sum()} anom ({pred_complex.mean()*100:.2f}%)")
    
    print("\n[8] Saving...")
    pd.DataFrame({'y_pred': pred_simple}).to_csv(f'{OUTPUT_DIR}/pred_simple.csv', index=False)
    pd.DataFrame({'y_pred': pred_complex}).to_csv(f'{OUTPUT_DIR}/pred_complex.csv', index=False)
    
    with open(f'{OUTPUT_DIR}/model.pkl', 'wb') as f:
        pickle.dump({
            'models': final_models, 'scaler': final_scaler,
            'threshold': base_thresh, 'best_fold': best_fold,
            'cv_results': fold_results, 'selected_features': selected_names,
        }, f)
    
    print(f"\n  Saved to {OUTPUT_DIR}/")
    print("=" * 70)
    print(f"DONE | CV AUC-PR={avg_auc_pr:.4f} | CV F1={avg_f1:.4f}")
    print("=" * 70)
    return avg_auc_pr, avg_f1

if __name__ == '__main__':
    main()
