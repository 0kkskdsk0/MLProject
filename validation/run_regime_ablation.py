"""
Retrain v3 under different regime_id strategies to test validation leakage.

Variants:
- global_regime: original v3 behavior. Detect regimes on the full labeled
  train.csv first, then slice train/validation ids.
- local_regime: detect regimes separately on train and validation splits.
  This matches how unseen test files are handled more closely.
- no_regime: remove regime_id entirely and retrain.

The split, feature engineering, model parameters, and validation metrics follow
code/train_predict_v3_fast.py unless noted above.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = 134035
RESULTS_PATH = ROOT / "validation" / "regime_ablation_results.csv"


def load_v3_module():
    module_path = ROOT / "code" / "train_predict_v3_fast.py"
    spec = importlib.util.spec_from_file_location("train_predict_v3_fast", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_data() -> tuple[pd.DataFrame, list[str]]:
    train_df = pd.read_csv(ROOT / "data" / "train.csv")
    feature_cols = [c for c in train_df.columns if c.startswith("f")]
    medians = train_df[feature_cols].median()
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(medians[col])
    return train_df, feature_cols


def regime_ids_for_variant(v3, train_df: pd.DataFrame, train_raw: pd.DataFrame, val_raw: pd.DataFrame, variant: str):
    if variant == "global_regime":
        regime_all = v3.detect_regimes(train_df)
        return regime_all[:TRAIN_END], regime_all[TRAIN_END:]
    if variant == "local_regime":
        return v3.detect_regimes(train_raw), v3.detect_regimes(val_raw)
    if variant == "no_regime":
        return np.zeros(len(train_raw), dtype=int), np.zeros(len(val_raw), dtype=int)
    raise ValueError(f"Unknown variant: {variant}")


def build_features(v3, train_df: pd.DataFrame, feature_cols: list[str], variant: str):
    train_raw = train_df.iloc[:TRAIN_END].copy()
    val_raw = train_df.iloc[TRAIN_END:].copy()
    y_train = train_raw["y"].values
    y_val = val_raw["y"].values

    regime_train, regime_val = regime_ids_for_variant(v3, train_df, train_raw, val_raw, variant)

    np.random.seed(42)
    x_train_sample = train_raw[feature_cols].values

    scaler_lof = StandardScaler()
    x_train_lof = scaler_lof.fit_transform(x_train_sample)
    sample_size = min(15000, len(x_train_lof))
    idx = np.random.choice(len(x_train_lof), sample_size, replace=False)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination="auto", n_jobs=-1)
    lof_model.fit(x_train_lof[idx])

    pca_scaler = StandardScaler()
    x_train_pca = pca_scaler.fit_transform(x_train_sample)
    pca_model = PCA(n_components=5, random_state=42)
    pca_model.fit(x_train_pca)

    train_fe = v3.create_features_fast(
        train_raw.drop(columns=["y"]),
        regime_train,
        feature_cols,
        lof_model,
        scaler_lof,
    )
    val_fe = v3.create_features_fast(
        val_raw.drop(columns=["y"]),
        regime_val,
        feature_cols,
        lof_model,
        scaler_lof,
    )

    train_fe = v3.add_pca(train_fe, train_raw, feature_cols, pca_model, pca_scaler)
    val_fe = v3.add_pca(val_fe, val_raw, feature_cols, pca_model, pca_scaler)

    common_cols = sorted(set(train_fe.columns).intersection(val_fe.columns))
    if variant == "no_regime" and "regime_id" in common_cols:
        common_cols.remove("regime_id")

    train_fe = train_fe[common_cols]
    val_fe = val_fe[common_cols]

    train_scaled, scaler = v3.preprocess(train_fe, fit_scaler=True)
    val_scaled = v3.preprocess(val_fe, scaler=scaler)
    return train_scaled.values, y_train, val_scaled.values, y_val, common_cols


def run_variant(v3, train_df: pd.DataFrame, feature_cols: list[str], variant: str) -> dict:
    print("=" * 80, flush=True)
    print(f"Running variant: {variant}", flush=True)
    started = time.time()

    x_train, y_train, x_val, y_val, common_cols = build_features(v3, train_df.copy(), feature_cols, variant)
    print(
        f"Prepared features: train={x_train.shape}, val={x_val.shape}, "
        f"train_pos={int(y_train.sum())}, val_pos={int(y_val.sum())}",
        flush=True,
    )

    np.random.seed(42)
    models = {
        "xgb": v3.train_xgboost(x_train, y_train, x_val, y_val),
        "lgb": v3.train_lightgbm(x_train, y_train, x_val, y_val),
        "iforest": v3.train_isolation_forest(x_train, y_train),
    }

    scores = v3.ensemble_predict(models, x_val)
    best_threshold, best_f1 = v3.find_best_threshold(y_val, scores)
    pred = (scores >= best_threshold).astype(int)

    result = {
        "variant": variant,
        "train_rows": len(y_train),
        "val_rows": len(y_val),
        "train_anomalies": int(y_train.sum()),
        "val_anomalies": int(y_val.sum()),
        "feature_count": len(common_cols),
        "auc_pr": average_precision_score(y_val, scores),
        "auc_roc": roc_auc_score(y_val, scores),
        "best_threshold": best_threshold,
        "f1": f1_score(y_val, pred),
        "predicted_anomalies": int(pred.sum()),
        "runtime_seconds": time.time() - started,
    }
    print(pd.Series(result).to_string(), flush=True)

    del models, x_train, x_val
    gc.collect()
    return result


def main() -> None:
    v3 = load_v3_module()
    train_df, feature_cols = load_data()
    variants = ["global_regime", "local_regime", "no_regime"]
    results = [run_variant(v3, train_df, feature_cols, variant) for variant in variants]
    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULTS_PATH, index=False)
    print("=" * 80)
    print("Summary")
    print(result_df.to_string(index=False))
    print(f"Saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
