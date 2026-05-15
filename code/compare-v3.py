"""
Compare four fixed recipes on v3-2 Scheme C split (same features / split).

Models:
  A) v3 default: XGB/LGB hyperparams from train_predict_v3_fast.py, weights 0.35/0.35/0.30.
  B) v3-2 grid winner (fixed): XGB/LGB from best row of grid_search_val (xi=0,lj=0),
     weights 0.65 / 0.20 / 0.15.
  C) Same trees as B, ensemble 0.75 * XGB + 0.25 * LGB (no IsolationForest).
  D) XGB only: same booster as B (v3-2-opt hyperparameters), no LGB / no IF.

Run from repo root:
    python code/compare-v3.py

Outputs:
    pic/compare_v3_metrics.png

Leakage (see docstring body in main):
  - No test-label leakage: preprocessing + trees + IF use TRAIN only; threshold uses VAL only.
  - Selection caveat: B (and D's XGB, C's trees) share v3-2-opt hyperparameters chosen on VAL
    in the original grid_search_val run; VAL metrics for those rows are optimistic vs nested CV.
"""
from __future__ import annotations

import os
import warnings
from typing import Any

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

np.random.seed(42)

TRAIN_PATH = "../data/train.csv"
TRAIN_END = 130816
VAL_END = 134545
OUTPUT_FIG = os.path.join("pic", "compare_v3_metrics.png")

# --- v3 default tree hyperparameters (train_predict_v3_fast.py) ---
XGB_KW_V3: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 3,
}
LGB_KW_V3: dict[str, Any] = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 5,
}
WEIGHTS_V3 = (0.35, 0.35, 0.30)

# --- v3-2 grid CSV top row (code/submission_v3_2/grid_search_val.csv) ---
XGB_KW_V32_OPT: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_child_weight": 3,
}
LGB_KW_V32_OPT: dict[str, Any] = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 5,
}
WEIGHTS_V32_OPT = (0.65, 0.20, 0.15)
WEIGHTS_XL = (0.75, 0.25)  # XGB, LGB only


def create_features_fast(df, feature_cols, lof_model=None, scaler_lof=None):
    features = pd.DataFrame(index=df.index)
    for col in feature_cols:
        features[col] = df[col].values
    for w in [5, 10, 20]:
        for col in feature_cols:
            features[f"{col}_rm{w}"] = df[col].rolling(window=w, min_periods=1).mean().values
            features[f"{col}_rs{w}"] = (
                df[col].rolling(window=w, min_periods=1).std().fillna(0).values
            )
    for col in feature_cols:
        features[f"{col}_d1"] = df[col].diff(1).fillna(0).values
        features[f"{col}_d5"] = df[col].diff(5).fillna(0).values
    for lag in [1, 3]:
        for col in feature_cols[:3]:
            features[f"{col}_l{lag}"] = df[col].shift(lag).bfill().ffill().values
    for i in range(min(3, len(feature_cols))):
        for j in range(i + 1, min(3, len(feature_cols))):
            col1, col2 = feature_cols[i], feature_cols[j]
            features[f"i_{i}_{j}"] = (df[col1] * df[col2]).values
    if lof_model is not None:
        X_raw = df[feature_cols].values
        X_scaled = scaler_lof.transform(X_raw)
        lof_scores = lof_model.decision_function(X_scaled)
        features["lof_score"] = -lof_scores
    else:
        features["lof_score"] = 0
    row_data = df[feature_cols].values
    features["row_mean"] = row_data.mean(axis=1)
    features["row_std"] = row_data.std(axis=1)
    features["row_max"] = row_data.max(axis=1)
    features["row_min"] = row_data.min(axis=1)
    return features


def add_pca(features, df, feature_cols, pca_model, pca_scaler):
    X = df[feature_cols].values
    X_scaled = pca_scaler.transform(X)
    comps = pca_model.transform(X_scaled)
    for i in range(comps.shape[1]):
        features[f"pca_{i}"] = comps[:, i]
    return features


def preprocess(df, scaler=None, fit_scaler=False):
    df = df.copy()
    df = df.ffill().bfill()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    exclude = ["y"]
    feature_cols = [c for c in df.columns if c not in exclude]
    if fit_scaler:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        return df, scaler
    df[feature_cols] = scaler.transform(df[feature_cols])
    return df


def fit_lof_pca(train_slice: pd.DataFrame, feature_cols: list[str]):
    X_train_sample = train_slice[feature_cols].values
    scaler_lof = StandardScaler()
    X_train_lof = scaler_lof.fit_transform(X_train_sample)
    sample_size = min(15000, len(X_train_lof))
    idx = np.random.choice(len(X_train_lof), sample_size, replace=False)
    lof_model = LocalOutlierFactor(
        n_neighbors=20, novelty=True, contamination="auto", n_jobs=-1
    )
    lof_model.fit(X_train_lof[idx])
    pca_scaler = StandardScaler()
    X_train_pca = pca_scaler.fit_transform(X_train_sample)
    pca_model = PCA(n_components=5, random_state=42)
    pca_model.fit(X_train_pca)
    return lof_model, scaler_lof, pca_model, pca_scaler


def train_xgboost_custom(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    xgb_kw: dict[str, Any],
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    pos_ratio = y_train.mean()
    scale_pos_weight = max(1.0, (1 - pos_ratio) / (pos_ratio + 1e-8))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": int(xgb_kw["max_depth"]),
        "learning_rate": float(xgb_kw["learning_rate"]),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "seed": 42,
        "tree_method": "hist",
        "min_child_weight": int(xgb_kw["min_child_weight"]),
    }
    return xgb.train(
        params,
        dtrain,
        num_boost_round=3000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )


def train_lightgbm_custom(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lgb_kw: dict[str, Any],
) -> lgb.Booster:
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "num_leaves": int(lgb_kw["num_leaves"]),
        "learning_rate": float(lgb_kw["learning_rate"]),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,
        "seed": 42,
        "verbose": -1,
        "min_child_samples": int(lgb_kw["min_child_samples"]),
    }
    return lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )


def train_isolation_forest(X_train: np.ndarray, y_train: np.ndarray) -> IsolationForest:
    X_normal = X_train[y_train == 0]
    contamination = max(0.001, min(0.1, y_train.mean() * 3))
    if len(X_normal) > 50000:
        idx = np.random.choice(len(X_normal), 50000, replace=False)
        X_normal = X_normal[idx]
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        max_samples=10000,
    )
    model.fit(X_normal)
    return model


def if_prob_scores(iforest: IsolationForest, X: np.ndarray) -> np.ndarray:
    if_scores = iforest.decision_function(X)
    return 1 - (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)


def ensemble_weighted(
    models: dict[str, Any],
    X: np.ndarray,
    wx: float,
    wl: float,
    wf: float,
) -> np.ndarray:
    pred_xgb = models["xgb"].predict(xgb.DMatrix(X))
    pred_lgb = models["lgb"].predict(X, num_iteration=models["lgb"].best_iteration)
    pred_if = if_prob_scores(models["iforest"], X)
    return wx * pred_xgb + wl * pred_lgb + wf * pred_if


def ensemble_xgb_lgb_only(models: dict[str, Any], X: np.ndarray, wx: float, wl: float) -> np.ndarray:
    pred_xgb = models["xgb"].predict(xgb.DMatrix(X))
    pred_lgb = models["lgb"].predict(X, num_iteration=models["lgb"].best_iteration)
    return wx * pred_xgb + wl * pred_lgb


def ensemble_xgb_only(models: dict[str, Any], X: np.ndarray) -> np.ndarray:
    return models["xgb"].predict(xgb.DMatrix(X))


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    if best_idx < len(thresholds):
        best_thresh = float(thresholds[best_idx])
    else:
        best_thresh = 0.5
    return best_thresh, float(f1_scores[best_idx])


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def eval_at_threshold(
    y: np.ndarray, scores: np.ndarray, thresh: float
) -> dict[str, float]:
    pred = (scores >= thresh).astype(int)
    return {
        "f1": float(f1_score(y, pred)),
        "auc_pr": float(average_precision_score(y, scores)),
        "auc_roc": safe_roc_auc(y, scores),
    }


def plot_grouped_bars(
    results: dict[str, dict[str, dict[str, float]]],
    out_path: str,
) -> None:
    """results[model_name][split_name][metric]"""
    models = list(results.keys())
    splits = ["train", "val", "test"]
    metrics = ["f1", "auc_pr", "auc_roc"]
    metric_labels = ["F1", "AUC-PR", "AUC-ROC"]

    n_models = len(models)
    width = min(0.22, 0.72 / max(n_models, 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(len(splits))
    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        for i, mname in enumerate(models):
            vals = [results[mname][s][metric] for s in splits]
            offset = (i - (n_models - 1) / 2.0) * width
            ax.bar(x + offset, vals, width, label=mname)
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Scheme C: train / val / test (threshold from val, PR-F1)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print("=" * 72)
    print("compare-v3.py | Scheme C | four models (3 ensembles + XGB-only)")
    print("=" * 72)
    print(
        "\nLeakage check:\n"
        "  - TRAIN [:TRAIN_END): fit LOF, PCA, StandardScaler, XGB, LGB, IF.\n"
        "  - VAL [TRAIN_END:VAL_END): only used to pick score threshold (PR-F1).\n"
        "  - TEST [VAL_END:): labels used only for evaluation after threshold is fixed.\n"
        "  => No TEST information flows into fitting or thresholding.\n"
        "\nSelection caveat:\n"
        "  - Models using v3-2-opt tree HPO (ensemble B, 0.75/0.25 C, XGB-only D) share\n"
        "    hyperparameters chosen on VAL in the original grid_search_val run; their VAL\n"
        "    metrics are optimistic vs nested CV. TEST was not used in that search.\n"
    )

    train_df = pd.read_csv(TRAIN_PATH)
    feature_cols = [c for c in train_df.columns if c.startswith("f")]

    train_slice = train_df.iloc[:TRAIN_END]
    medians = train_slice[feature_cols].median()
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(medians[col])

    train_raw = train_df.iloc[:TRAIN_END].copy()
    val_raw = train_df.iloc[TRAIN_END:VAL_END].copy()
    test_raw = train_df.iloc[VAL_END:].copy()
    y_train = train_raw["y"].values.astype(int)
    y_val = val_raw["y"].values.astype(int)
    y_test = test_raw["y"].values.astype(int)

    print(
        f"Train {len(train_raw)} | Val {len(val_raw)} | Test {len(test_raw)} "
        f"| pos rates {y_train.mean():.4f} / {y_val.mean():.4f} / {y_test.mean():.4f}"
    )

    lof_m, sl_m, pca_m, pca_s = fit_lof_pca(train_raw, feature_cols)

    train_fe = create_features_fast(train_raw.drop(columns=["y"]), feature_cols, lof_m, sl_m)
    val_fe = create_features_fast(val_raw.drop(columns=["y"]), feature_cols, lof_m, sl_m)
    test_fe = create_features_fast(test_raw.drop(columns=["y"]), feature_cols, lof_m, sl_m)

    train_fe = add_pca(train_fe, train_raw, feature_cols, pca_m, pca_s)
    val_fe = add_pca(val_fe, val_raw, feature_cols, pca_m, pca_s)
    test_fe = add_pca(test_fe, test_raw, feature_cols, pca_m, pca_s)

    all_cols = set(train_fe.columns) & set(val_fe.columns) & set(test_fe.columns)
    common_cols = sorted(all_cols)
    train_fe = train_fe[common_cols]
    val_fe = val_fe[common_cols]
    test_fe = test_fe[common_cols]

    train_scaled, scaler = preprocess(train_fe, fit_scaler=True)
    val_scaled = preprocess(val_fe, scaler=scaler)
    test_scaled = preprocess(test_fe, scaler=scaler)

    X_train = train_scaled.values
    X_val = val_scaled.values
    X_test = test_scaled.values

    if_model = train_isolation_forest(X_train, y_train)

    print("\n[1] Train v3 XGB/LGB ...")
    xgb_v3 = train_xgboost_custom(X_train, y_train, X_val, y_val, XGB_KW_V3)
    lgb_v3 = train_lightgbm_custom(X_train, y_train, X_val, y_val, LGB_KW_V3)

    print("[2] Train v3-2-opt XGB/LGB ...")
    xgb_o = train_xgboost_custom(X_train, y_train, X_val, y_val, XGB_KW_V32_OPT)
    lgb_o = train_lightgbm_custom(X_train, y_train, X_val, y_val, LGB_KW_V32_OPT)

    configs: list[tuple[str, dict[str, Any], str]] = [
        (
            "v3 (0.35/0.35/0.30)",
            {"xgb": xgb_v3, "lgb": lgb_v3, "iforest": if_model},
            "xlf",
        ),
        (
            "v3-2-opt (0.65/0.20/0.15)",
            {"xgb": xgb_o, "lgb": lgb_o, "iforest": if_model},
            "xlf",
        ),
        (
            "0.75 XGB + 0.25 LGB (opt trees)",
            {"xgb": xgb_o, "lgb": lgb_o, "iforest": if_model},
            "xl",
        ),
        (
            "XGB only (v3-2-opt HPO)",
            {"xgb": xgb_o},
            "x",
        ),
    ]

    results: dict[str, dict[str, dict[str, float]]] = {}

    for name, models, mode in configs:
        if mode == "xlf":
            wx, wl, wf = WEIGHTS_V3 if name.startswith("v3 (") else WEIGHTS_V32_OPT
            val_scores = ensemble_weighted(models, X_val, wx, wl, wf)
            train_scores = ensemble_weighted(models, X_train, wx, wl, wf)
            test_scores = ensemble_weighted(models, X_test, wx, wl, wf)
        elif mode == "x":
            val_scores = ensemble_xgb_only(models, X_val)
            train_scores = ensemble_xgb_only(models, X_train)
            test_scores = ensemble_xgb_only(models, X_test)
        else:
            wx, wl = WEIGHTS_XL
            val_scores = ensemble_xgb_lgb_only(models, X_val, wx, wl)
            train_scores = ensemble_xgb_lgb_only(models, X_train, wx, wl)
            test_scores = ensemble_xgb_lgb_only(models, X_test, wx, wl)

        thresh, _ = find_best_threshold(y_val, val_scores)
        results[name] = {
            "train": eval_at_threshold(y_train, train_scores, thresh),
            "val": eval_at_threshold(y_val, val_scores, thresh),
            "test": eval_at_threshold(y_test, test_scores, thresh),
        }
        print(f"\n{name}")
        print(f"  threshold (val PR-F1): {thresh:.6f}")
        for split in ("train", "val", "test"):
            m = results[name][split]
            print(
                f"  {split:5s}  F1={m['f1']:.4f}  AUC-PR={m['auc_pr']:.4f}  "
                f"AUC-ROC={m['auc_roc']:.4f}"
            )

    plot_grouped_bars(results, OUTPUT_FIG)
    print(f"\nSaved figure -> {OUTPUT_FIG}")
    print("=" * 72)


if __name__ == "__main__":
    main()
