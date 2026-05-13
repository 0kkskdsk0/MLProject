"""
Robust Anomaly Detection v3 - Fast Version
Efficient feature engineering with LOF, PCA, and improved ensemble
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

# ============================================================================
# CONFIGURATION
# ============================================================================
TRAIN_PATH = 'data/train.csv'
TEST_SIMPLE_PATH = 'data/test_simple.csv'
TEST_COMPLEX_PATH = 'data/test_complex.csv'
OUTPUT_DIR = 'submission_v3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_END = 134035

# ============================================================================
# FEATURE ENGINEERING (Efficient)
# ============================================================================
def create_features_fast(df, feature_cols, lof_model=None, scaler_lof=None):
    """Fast feature engineering with vectorized operations."""
    features = pd.DataFrame(index=df.index)
    
    # 1. Raw features
    for col in feature_cols:
        features[col] = df[col].values
    
    # 2. Efficient global rolling (with min_periods=1 to avoid NaN)
    for w in [5, 10, 20]:
        for col in feature_cols:
            features[f'{col}_rm{w}'] = df[col].rolling(window=w, min_periods=1).mean().values
            features[f'{col}_rs{w}'] = df[col].rolling(window=w, min_periods=1).std().fillna(0).values
    
    # 3. Difference features
    for col in feature_cols:
        features[f'{col}_d1'] = df[col].diff(1).fillna(0).values
        features[f'{col}_d5'] = df[col].diff(5).fillna(0).values
    
    # 4. Lag features (first 3 features)
    for lag in [1, 3]:
        for col in feature_cols[:3]:
            features[f'{col}_l{lag}'] = df[col].shift(lag).bfill().ffill().values
    
    # 5. Feature interactions (top 3)
    for i in range(min(3, len(feature_cols))):
        for j in range(i+1, min(3, len(feature_cols))):
            col1, col2 = feature_cols[i], feature_cols[j]
            features[f'i_{i}_{j}'] = (df[col1] * df[col2]).values
    
    # 6. LOF scores
    if lof_model is not None:
        X_raw = df[feature_cols].values
        X_scaled = scaler_lof.transform(X_raw)
        lof_scores = lof_model.decision_function(X_scaled)
        features['lof_score'] = -lof_scores  # Higher = more anomalous
    else:
        features['lof_score'] = 0
    
    # 7. Simple statistical aggregates per row
    row_data = df[feature_cols].values
    features['row_mean'] = row_data.mean(axis=1)
    features['row_std'] = row_data.std(axis=1)
    features['row_max'] = row_data.max(axis=1)
    features['row_min'] = row_data.min(axis=1)
    
    return features


def add_pca(features, df, feature_cols, pca_model, pca_scaler):
    """Add PCA components as features."""
    X = df[feature_cols].values
    X_scaled = pca_scaler.transform(X)
    comps = pca_model.transform(X_scaled)
    for i in range(comps.shape[1]):
        features[f'pca_{i}'] = comps[:, i]
    return features


# ============================================================================
# PREPROCESSING
# ============================================================================
def preprocess(df, scaler=None, fit_scaler=False):
    df = df.copy()
    df = df.ffill().bfill()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    exclude = ['y']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    if fit_scaler:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df, scaler
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])
        return df


# ============================================================================
# MODEL TRAINING
# ============================================================================
def train_xgboost(X_train, y_train, X_val, y_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    pos_ratio = y_train.mean()
    scale_pos_weight = max(1.0, (1 - pos_ratio) / (pos_ratio + 1e-8))
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42,
        'tree_method': 'hist',
        'min_child_weight': 3,
    }
    
    model = xgb.train(
        params, dtrain,
        num_boost_round=3000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=200
    )
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'is_unbalance': True,
        'seed': 42,
        'verbose': -1,
        'min_child_samples': 5,
    }
    
    model = lgb.train(
        params, train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    return model


def train_isolation_forest(X_train, y_train):
    X_normal = X_train[y_train == 0]
    contamination = max(0.001, min(0.1, y_train.mean() * 3))
    
    # Subsample for speed
    if len(X_normal) > 50000:
        idx = np.random.choice(len(X_normal), 50000, replace=False)
        X_normal = X_normal[idx]
    
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        max_samples=10000
    )
    model.fit(X_normal)
    return model


def ensemble_predict(models, X):
    pred_xgb = models['xgb'].predict(xgb.DMatrix(X))
    pred_lgb = models['lgb'].predict(X, num_iteration=models['lgb'].best_iteration)
    
    if_scores = models['iforest'].decision_function(X)
    pred_if = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    
    # Higher unsupervised weight for robustness
    ensemble = 0.35 * pred_xgb + 0.35 * pred_lgb + 0.30 * pred_if
    return ensemble


def find_best_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        best_thresh = thresholds[best_idx]
    else:
        best_thresh = 0.5
    return best_thresh, f1_scores[best_idx]


def main():
    print("=" * 70)
    print("ANOMALY DETECTION PIPELINE v3 - Fast")
    print("=" * 70)
    
    # Load
    print("\n[1] Loading...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_simple_df = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex_df = pd.read_csv(TEST_COMPLEX_PATH)
    print(f"  Train: {train_df.shape}, TestS: {test_simple_df.shape}, TestC: {test_complex_df.shape}")
    
    # Fill NaN with medians from training data
    feature_cols = [c for c in train_df.columns if c.startswith('f')]
    medians = train_df[feature_cols].median()
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(medians[col])
        test_simple_df[col] = test_simple_df[col].fillna(medians[col])
        test_complex_df[col] = test_complex_df[col].fillna(medians[col])
    
    # Split
    print(f"\n[2] Split at {TRAIN_END}")
    train_raw = train_df.iloc[:TRAIN_END].copy()
    val_raw = train_df.iloc[TRAIN_END:].copy()
    y_train = train_raw['y'].values
    y_val = val_raw['y'].values
    print(f"  Train: {len(train_raw)} rows, {y_train.sum()} anomalies ({y_train.mean()*100:.3f}%)")
    print(f"  Val:   {len(val_raw)} rows, {y_val.sum()} anomalies ({y_val.mean()*100:.3f}%)")
    
    # Fit LOF
    print("\n[3] Fitting LOF...")
    X_train_sample = train_raw[feature_cols].values
    scaler_lof = StandardScaler()
    X_train_lof = scaler_lof.fit_transform(X_train_sample)
    sample_size = min(15000, len(X_train_lof))
    idx = np.random.choice(len(X_train_lof), sample_size, replace=False)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination='auto', n_jobs=-1)
    lof_model.fit(X_train_lof[idx])
    
    # Fit PCA
    print("[4] Fitting PCA...")
    pca_scaler = StandardScaler()
    X_train_pca = pca_scaler.fit_transform(X_train_sample)
    pca_model = PCA(n_components=5, random_state=42)
    pca_model.fit(X_train_pca)
    
    # Feature engineering
    print("\n[5] Feature engineering...")
    train_fe = create_features_fast(train_raw.drop(columns=['y']), feature_cols, lof_model, scaler_lof)
    val_fe = create_features_fast(val_raw.drop(columns=['y']), feature_cols, lof_model, scaler_lof)
    test_simple_fe = create_features_fast(test_simple_df, feature_cols, lof_model, scaler_lof)
    test_complex_fe = create_features_fast(test_complex_df, feature_cols, lof_model, scaler_lof)
    
    # Add PCA
    train_fe = add_pca(train_fe, train_raw, feature_cols, pca_model, pca_scaler)
    val_fe = add_pca(val_fe, val_raw, feature_cols, pca_model, pca_scaler)
    test_simple_fe = add_pca(test_simple_fe, test_simple_df, feature_cols, pca_model, pca_scaler)
    test_complex_fe = add_pca(test_complex_fe, test_complex_df, feature_cols, pca_model, pca_scaler)
    
    # Align
    all_cols = set(train_fe.columns)
    for df in [val_fe, test_simple_fe, test_complex_fe]:
        all_cols = all_cols.intersection(set(df.columns))
    common_cols = sorted(list(all_cols))
    
    train_fe = train_fe[common_cols]
    val_fe = val_fe[common_cols]
    test_simple_fe = test_simple_fe[common_cols]
    test_complex_fe = test_complex_fe[common_cols]
    print(f"  Features: {len(common_cols)}")
    
    # Scale
    print("\n[6] Scaling...")
    train_scaled, scaler = preprocess(train_fe, fit_scaler=True)
    val_scaled = preprocess(val_fe, scaler=scaler)
    test_simple_scaled = preprocess(test_simple_fe, scaler=scaler)
    test_complex_scaled = preprocess(test_complex_fe, scaler=scaler)
    
    X_train = train_scaled.values
    X_val = val_scaled.values
    
    # Train
    print("\n[7] Training...")
    models = {}
    print("  -> XGBoost...")
    models['xgb'] = train_xgboost(X_train, y_train, X_val, y_val)
    print("\n  -> LightGBM...")
    models['lgb'] = train_lightgbm(X_train, y_train, X_val, y_val)
    print("\n  -> Isolation Forest...")
    models['iforest'] = train_isolation_forest(X_train, y_train)
    
    # Validate
    print("\n[8] Validation...")
    val_scores = ensemble_predict(models, X_val)
    auc_pr = average_precision_score(y_val, val_scores)
    auc_roc = roc_auc_score(y_val, val_scores)
    best_thresh, best_f1 = find_best_threshold(y_val, val_scores)
    val_pred = (val_scores >= best_thresh).astype(int)
    f1_val = f1_score(y_val, val_pred)
    print(f"  AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, F1: {f1_val:.4f} @ {best_thresh:.4f}")
    
    # Predict
    print("\n[9] Predicting...")
    scores_simple = ensemble_predict(models, test_simple_scaled.values)
    pred_simple = (scores_simple >= best_thresh).astype(int)
    scores_complex = ensemble_predict(models, test_complex_scaled.values)
    pred_complex = (scores_complex >= best_thresh).astype(int)
    print(f"  Task1: {pred_simple.sum()} anomalies ({pred_simple.mean()*100:.2f}%)")
    print(f"  Task2: {pred_complex.sum()} anomalies ({pred_complex.mean()*100:.2f}%)")
    
    # Save
    print("\n[10] Saving...")
    pd.DataFrame({'y_pred': pred_simple}).to_csv(f'{OUTPUT_DIR}/pred_simple.csv', index=False)
    pd.DataFrame({'y_pred': pred_complex}).to_csv(f'{OUTPUT_DIR}/pred_complex.csv', index=False)
    with open(f'{OUTPUT_DIR}/model.pkl', 'wb') as f:
        pickle.dump({'models': models, 'scaler': scaler, 'threshold': best_thresh,
                     'feature_cols': common_cols, 'auc_pr': auc_pr, 'f1': f1_val}, f)
    
    print(f"\n  Saved to {OUTPUT_DIR}/")
    print("=" * 70)
    print(f"DONE | AUC-PR={auc_pr:.4f} | F1={f1_val:.4f}")
    print("=" * 70)
    return auc_pr, f1_val


if __name__ == '__main__':
    main()
