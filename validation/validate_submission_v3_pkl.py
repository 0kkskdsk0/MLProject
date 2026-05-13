"""
Validate whether submission_v3/model.pkl can reproduce the v3 handover metrics.

This script intentionally loads the saved pickle and uses its final estimators,
scaler, threshold, and feature-column order. Because the pickle does not contain
the LOF/PCA feature-generation helpers used during training, those helpers are
rebuilt from the training split with the same v3 code path.
"""
from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = 134035


def load_v3_module():
    module_path = ROOT / "code" / "train_predict_v3_fast.py"
    spec = importlib.util.spec_from_file_location("train_predict_v3_fast", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensemble_predict(saved_models: dict, x_val: np.ndarray) -> np.ndarray:
    pred_xgb = saved_models["xgb"].predict(xgb.DMatrix(x_val))
    pred_lgb = saved_models["lgb"].predict(x_val, num_iteration=saved_models["lgb"].best_iteration)

    if_scores = saved_models["iforest"].decision_function(x_val)
    pred_if = 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
    return 0.35 * pred_xgb + 0.35 * pred_lgb + 0.30 * pred_if


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(threshold), float(f1_scores[best_idx])


def build_validation_features(
    v3,
    train_df: pd.DataFrame,
    saved_feature_cols: list[str],
    regime_mode: str = "global",
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_cols = [c for c in train_df.columns if c.startswith("f")]
    medians = train_df[feature_cols].median()
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(medians[col])

    train_raw = train_df.iloc[:TRAIN_END].copy()
    val_raw = train_df.iloc[TRAIN_END:].copy()
    y_val = val_raw["y"].values
    if regime_mode == "global":
        regime_ids_val = v3.detect_regimes(train_df)[TRAIN_END:]
    elif regime_mode == "local":
        regime_ids_val = v3.detect_regimes(val_raw)
    elif regime_mode == "zero":
        regime_ids_val = np.zeros(len(val_raw), dtype=int)
    else:
        raise ValueError(f"Unknown regime_mode: {regime_mode}")

    np.random.seed(42)
    x_train_raw = train_raw[feature_cols].values

    scaler_lof = StandardScaler()
    x_train_lof = scaler_lof.fit_transform(x_train_raw)
    sample_size = min(15000, len(x_train_lof))
    idx = np.random.choice(len(x_train_lof), sample_size, replace=False)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto", n_jobs=-1)
    lof_model.fit(x_train_lof[idx])

    pca_scaler = StandardScaler()
    x_train_pca = pca_scaler.fit_transform(x_train_raw)
    pca_model = PCA(n_components=5, random_state=42)
    pca_model.fit(x_train_pca)

    val_fe = v3.create_features_fast(
        val_raw.drop(columns=["y"]),
        regime_ids_val,
        feature_cols,
        lof_model,
        scaler_lof,
    )
    val_fe = v3.add_pca(val_fe, val_raw, feature_cols, pca_model, pca_scaler)

    missing = sorted(set(saved_feature_cols) - set(val_fe.columns))
    if missing:
        raise RuntimeError(f"Missing expected feature columns: {missing[:10]}")

    return val_fe[saved_feature_cols], y_val


def main() -> None:
    v3 = load_v3_module()
    model_path = ROOT / "submission_v3" / "model.pkl"
    train_path = ROOT / "data" / "train.csv"

    with model_path.open("rb") as f:
        saved = pickle.load(f)

    train_df = pd.read_csv(train_path)
    saved_feature_cols = list(saved["feature_cols"])
    saved_threshold = float(saved["threshold"])
    print("=" * 72)
    print("submission_v3/model.pkl validation replay")
    print("=" * 72)
    print(f"Saved metadata AUC-PR: {saved.get('auc_pr'):.6f}")
    print(f"Saved metadata F1    : {saved.get('f1'):.6f}")
    print(f"Saved threshold      : {saved_threshold:.6f}")
    print("-" * 72)
    print(f"Feature columns      : {len(saved_feature_cols)}")
    print("-" * 72)
    for regime_mode in ["global", "local", "zero"]:
        val_fe, y_val = build_validation_features(v3, train_df.copy(), saved_feature_cols, regime_mode)
        val_scaled = val_fe.copy()
        scaler_cols = list(saved["scaler"].feature_names_in_)
        val_scaled[scaler_cols] = saved["scaler"].transform(val_scaled[scaler_cols])
        x_val = val_scaled.values
        scores = ensemble_predict(saved["models"], x_val)

        pred_saved = (scores >= saved_threshold).astype(int)
        oracle_threshold, oracle_f1 = best_threshold(y_val, scores)
        pred_oracle = (scores >= oracle_threshold).astype(int)

        print(f"Regime mode          : {regime_mode}")
        print(f"Validation rows      : {len(y_val)}")
        print(f"Validation positives : {int(y_val.sum())} ({y_val.mean() * 100:.3f}%)")
        print(f"Replay AUC-PR        : {average_precision_score(y_val, scores):.6f}")
        print(f"Replay AUC-ROC       : {roc_auc_score(y_val, scores):.6f}")
        print(f"Replay F1 saved thr  : {f1_score(y_val, pred_saved):.6f}")
        print(f"Replay anomalies     : {int(pred_saved.sum())} / {len(pred_saved)}")
        print(f"Replay best threshold: {oracle_threshold:.6f}")
        print(f"Replay best F1       : {oracle_f1:.6f}")
        print(f"Best-thr anomalies   : {int(pred_oracle.sum())} / {len(pred_oracle)}")
        print("-" * 72)
    print("=" * 72)


if __name__ == "__main__":
    main()
