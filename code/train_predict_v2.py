"""
Robust Anomaly Detection in Noisy Time-Series Data - v2
Optimized for: temporal structure, severe class imbalance, concept drift
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = 'data/train.csv'
TEST_SIMPLE_PATH = 'data/test_simple.csv'
TEST_COMPLEX_PATH = 'data/test_complex.csv'
OUTPUT_DIR = 'submission'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Temporal split: train on first N rows, validate on rest
# Key: validation must include anomalies! Anomalies are all in rows 124283+
SPLIT_IDX = 127000  # Include some anomalies in train, keep enough for validation

# Window sizes for temporal features
WINDOW_SIZES = [3, 5, 10, 20, 50]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def create_temporal_features(df):
    """Create rolling window, lag, and difference features."""
    features = df.copy()
    feature_cols = [c for c in features.columns if c.startswith('f')]
    
    # Rolling statistics
    for w in WINDOW_SIZES:
        for col in feature_cols:
            features[f'{col}_rm{w}'] = features[col].rolling(window=w, min_periods=1).mean()
            features[f'{col}_rs{w}'] = features[col].rolling(window=w, min_periods=1).std().fillna(0)
            features[f'{col}_rx{w}'] = features[col].rolling(window=w, min_periods=1).max()
            features[f'{col}_rn{w}'] = features[col].rolling(window=w, min_periods=1).min()
    
    # Difference features
    for col in feature_cols:
        features[f'{col}_d1'] = features[col].diff(1).fillna(0)
        features[f'{col}_d5'] = features[col].diff(5).fillna(0)
        features[f'{col}_d10'] = features[col].diff(10).fillna(0)
    
    # Lag features (only for first 5 features to control dimensionality)
    for lag in [1, 3, 5]:
        for col in feature_cols[:5]:
            features[f'{col}_l{lag}'] = features[col].shift(lag).bfill().ffill()
    
    # Rate of change
    for col in feature_cols[:5]:
        features[f'{col}_roc'] = features[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    
    # Simple interactions
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(3, len(feature_cols))):
            col1, col2 = feature_cols[i], feature_cols[j]
            features[f'i_{col1}_{col2}'] = features[col1] * features[col2]
    
    return features


def preprocess(df, scaler=None, fit_scaler=False):
    """Handle missing values and scale features."""
    df = df.copy()
    
    # Fill missing
    df = df.ffill().bfill()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    feature_cols = [c for c in df.columns if not c.startswith('y')]
    
    if fit_scaler:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df, scaler, feature_cols
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
        return df


# ============================================================================
# MODEL TRAINING
# ============================================================================
def find_best_threshold(y_true, scores):
    """Find threshold maximizing F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = 0.5
    return best_thresh, f1_scores[best_idx]


def train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Train XGBoost with early stopping."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'tree_method': 'hist',
        'min_child_weight': 5,
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight):
    """Train LightGBM with early stopping."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'verbose': -1,
        'min_child_samples': 10,
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    return model


def train_iforest(X_train, y_train):
    """Train Isolation Forest on normal samples."""
    X_normal = X_train[y_train == 0]
    contamination = max(0.001, min(0.1, y_train.mean() * 2))
    
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_normal)
    return model


def ensemble_predict(models, X):
    """Weighted ensemble of XGBoost, LightGBM, and Isolation Forest."""
    pred_xgb = models['xgb'].predict(xgb.DMatrix(X))
    pred_lgb = models['lgb'].predict(X, num_iteration=models['lgb'].best_iteration)
    
    # Isolation Forest: convert scores
    if_scores = models['iforest'].decision_function(X)
    pred_if = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    
    ensemble = 0.50 * pred_xgb + 0.40 * pred_lgb + 0.10 * pred_if
    return ensemble


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 70)
    print("ANOMALY DETECTION PIPELINE v2")
    print("=" * 70)
    
    # --- Load Data ---
    print("\n[1] Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_simple_df = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex_df = pd.read_csv(TEST_COMPLEX_PATH)
    
    print(f"  Train: {train_df.shape}")
    print(f"  Test simple: {test_simple_df.shape}")
    print(f"  Test complex: {test_complex_df.shape}")
    
    n = len(train_df)
    
    # Check anomaly distribution
    y_all = train_df['y'].values
    anomaly_idx = np.where(y_all == 1)[0]
    print(f"  Anomalies: {len(anomaly_idx)} total, first@{anomaly_idx[0]}, last@{anomaly_idx[-1]}")
    
    # --- Temporal Split (respect anomaly distribution) ---
    print(f"\n[2] Temporal split at index {SPLIT_IDX} (~{SPLIT_IDX/n*100:.1f}%)")
    
    train_raw = train_df.iloc[:SPLIT_IDX].copy()
    val_raw = train_df.iloc[SPLIT_IDX:].copy()
    
    y_train = train_raw['y'].values
    y_val = val_raw['y'].values
    
    print(f"  Train: {len(train_raw)} rows, {y_train.sum()} anomalies ({y_train.mean()*100:.3f}%)")
    print(f"  Val:   {len(val_raw)} rows, {y_val.sum()} anomalies ({y_val.mean()*100:.3f}%)")
    
    # --- Feature Engineering ---
    print("\n[3] Feature engineering...")
    train_fe = create_temporal_features(train_raw.drop(columns=['y']))
    val_fe = create_temporal_features(val_raw.drop(columns=['y']))
    test_simple_fe = create_temporal_features(test_simple_df)
    test_complex_fe = create_temporal_features(test_complex_df)
    
    # Align columns
    all_cols = set(train_fe.columns)
    for df in [val_fe, test_simple_fe, test_complex_fe]:
        all_cols = all_cols.intersection(set(df.columns))
    common_cols = sorted(list(all_cols))
    
    train_fe = train_fe[common_cols]
    val_fe = val_fe[common_cols]
    test_simple_fe = test_simple_fe[common_cols]
    test_complex_fe = test_complex_fe[common_cols]
    
    print(f"  Feature dimension: {len(common_cols)}")
    
    # --- Preprocessing ---
    print("\n[4] Preprocessing...")
    train_scaled, scaler, feature_cols = preprocess(train_fe, fit_scaler=True)
    val_scaled = preprocess(val_fe, scaler=scaler)
    test_simple_scaled = preprocess(test_simple_fe, scaler=scaler)
    test_complex_scaled = preprocess(test_complex_fe, scaler=scaler)
    
    X_train = train_scaled.values
    X_val = val_scaled.values
    
    # --- Training ---
    print("\n[5] Training models...")
    pos_ratio = y_train.mean()
    scale_pos_weight = max(1.0, (1 - pos_ratio) / (pos_ratio + 1e-8))
    print(f"  pos_ratio={pos_ratio:.6f}, scale_pos_weight={scale_pos_weight:.2f}")
    
    models = {}
    
    print("\n  -> XGBoost...")
    models['xgb'] = train_xgboost(X_train, y_train, X_val, y_val, scale_pos_weight)
    
    print("\n  -> LightGBM...")
    models['lgb'] = train_lightgbm(X_train, y_train, X_val, y_val, scale_pos_weight)
    
    print("\n  -> Isolation Forest...")
    models['iforest'] = train_iforest(X_train, y_train)
    
    # --- Validation ---
    print("\n[6] Validation...")
    val_scores = ensemble_predict(models, X_val)
    
    auc_pr = average_precision_score(y_val, val_scores)
    auc_roc = roc_auc_score(y_val, val_scores)
    best_thresh, best_f1 = find_best_threshold(y_val, val_scores)
    val_pred = (val_scores >= best_thresh).astype(int)
    f1_val = f1_score(y_val, val_pred)
    
    print(f"  AUC-PR:  {auc_pr:.4f}")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  Best F1: {f1_val:.4f} @ threshold={best_thresh:.4f}")
    print(f"  Predicted anomalies in val: {val_pred.sum()} / {len(val_pred)}")
    
    # --- Test Predictions ---
    print("\n[7] Generating test predictions...")
    
    scores_simple = ensemble_predict(models, test_simple_scaled.values)
    pred_simple = (scores_simple >= best_thresh).astype(int)
    
    scores_complex = ensemble_predict(models, test_complex_scaled.values)
    pred_complex = (scores_complex >= best_thresh).astype(int)
    
    print(f"  Task 1 (simple): {pred_simple.sum()} anomalies ({pred_simple.mean()*100:.2f}%)")
    print(f"  Task 2 (complex): {pred_complex.sum()} anomalies ({pred_complex.mean()*100:.2f}%)")
    
    # --- Save Outputs ---
    print("\n[8] Saving outputs...")
    
    pd.DataFrame({'y_pred': pred_simple}).to_csv(f'{OUTPUT_DIR}/pred_simple.csv', index=False)
    pd.DataFrame({'y_pred': pred_complex}).to_csv(f'{OUTPUT_DIR}/pred_complex.csv', index=False)
    
    with open(f'{OUTPUT_DIR}/model.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'scaler': scaler,
            'threshold': best_thresh,
            'feature_cols': common_cols,
            'auc_pr': auc_pr,
            'f1': f1_val
        }, f)
    
    print(f"\n  Saved: pred_simple.csv, pred_complex.csv, model.pkl")
    
    print("\n" + "=" * 70)
    print(f"COMPLETE | AUC-PR={auc_pr:.4f} | F1={f1_val:.4f}")
    print("=" * 70)
    
    return auc_pr, f1_val


if __name__ == '__main__':
    main()
