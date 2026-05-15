"""
v3-2: Scheme C temporal split + validation-set selection of ensemble weights and
XGBoost / LightGBM hyperparameters. Labeled test slice is evaluated once at the end.

Split (Scheme C):
  Train  [:TRAIN_END)
  Val    [TRAIN_END, VAL_END)
  Test   [VAL_END,)

Run from repo root: python code/v3-2.py
Outputs: submission_v3_2/
"""
from __future__ import annotations

import itertools
import os
import pickle
import warnings
from typing import Any

import lightgbm as lgb
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
TEST_SIMPLE_PATH = "../data/test_simple.csv"
TEST_COMPLEX_PATH = "../data/test_complex.csv"
OUTPUT_DIR = "../submission_v3_2"

# Scheme C
TRAIN_END = 130816
VAL_END = 134545

# Search grids (keep runtime modest)
XGB_PARAM_GRID: list[dict[str, Any]] = [
    {"max_depth": 6, "learning_rate": 0.05, "min_child_weight": 3},
    {"max_depth": 6, "learning_rate": 0.04, "min_child_weight": 2},
    {"max_depth": 5, "learning_rate": 0.05, "min_child_weight": 3},
    {"max_depth": 5, "learning_rate": 0.06, "min_child_weight": 4},
]

LGB_PARAM_GRID: list[dict[str, Any]] = [
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 5},
    {"num_leaves": 47, "learning_rate": 0.05, "min_child_samples": 5},
    {"num_leaves": 31, "learning_rate": 0.04, "min_child_samples": 10},
    {"num_leaves": 63, "learning_rate": 0.05, "min_child_samples": 5},
]


def create_features_fast(df, feature_cols, lof_model=None, scaler_lof=None):
    features = pd.DataFrame(index=df.index)
    for col in feature_cols:
        features[col] = df[col].values
    for w in [5, 10, 20]:
        for col in feature_cols:
            features[f"{col}_rm{w}"] = df[col].rolling(window=w, min_periods=1).mean().values
            features[f"{col}_rs{w}"] = df[col].rolling(window=w, min_periods=1).std().fillna(0).values
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
    models: dict,
    X: np.ndarray,
    wx: float,
    wl: float,
    wf: float,
) -> np.ndarray:
    pred_xgb = models["xgb"].predict(xgb.DMatrix(X))
    pred_lgb = models["lgb"].predict(X, num_iteration=models["lgb"].best_iteration)
    pred_if = if_prob_scores(models["iforest"], X)
    return wx * pred_xgb + wl * pred_lgb + wf * pred_if


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    if best_idx < len(thresholds):
        best_thresh = float(thresholds[best_idx])
    else:
        best_thresh = 0.5
    return best_thresh, float(f1_scores[best_idx])


def generate_weight_grid(step: float = 0.05) -> list[tuple[float, float, float]]:
    triples: set[tuple[float, float, float]] = set()
    w = np.arange(0.20, 0.71, step)
    for wx, wl in itertools.product(w, w):
        wf = 1.0 - float(wx) - float(wl)
        if wf < 0.15 - 1e-9:
            continue
        triples.add(
            (round(float(wx), 3), round(float(wl), 3), round(float(wf), 3))
        )
    return sorted(triples, key=lambda t: (t[0], t[1]))


def train_final_xgb(
    X: np.ndarray,
    y: np.ndarray,
    xgb_kw: dict[str, Any],
    num_boost_round: int,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y)
    pos_ratio = y.mean()
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
    return xgb.train(params, dtrain, num_boost_round=int(num_boost_round), verbose_eval=False)


def booster_best_rounds(booster: xgb.Booster) -> int:
    lim = getattr(booster, "best_ntree_limit", None)
    if lim is not None:
        return int(lim)
    bi = getattr(booster, "best_iteration", None)
    if bi is not None:
        return int(bi) + 1
    return int(booster.num_boosted_rounds())


def lgb_best_rounds(model: lgb.Booster) -> int:
    bi = getattr(model, "best_iteration", None)
    if bi is not None:
        return int(bi) + 1
    if hasattr(model, "num_trees"):
        return max(int(model.num_trees()), 50)
    return 500


def train_final_lgb(
    X: np.ndarray,
    y: np.ndarray,
    lgb_kw: dict[str, Any],
    num_boost_round: int,
) -> lgb.Booster:
    train_data = lgb.Dataset(X, label=y)
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
        num_boost_round=int(num_boost_round),
        callbacks=[lgb.log_evaluation(0)],
    )


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 72)
    print("v3-2 | Scheme C split | val-driven weight + hyperparam search")
    print("=" * 72)

    train_df = pd.read_csv(TRAIN_PATH)
    test_simple_df = pd.read_csv(TEST_SIMPLE_PATH)
    test_complex_df = pd.read_csv(TEST_COMPLEX_PATH)
    feature_cols = [c for c in train_df.columns if c.startswith("f")]

    train_slice = train_df.iloc[:TRAIN_END]
    medians = train_slice[feature_cols].median()
    for col in feature_cols:
        train_df[col] = train_df[col].fillna(medians[col])
        test_simple_df[col] = test_simple_df[col].fillna(medians[col])
        test_complex_df[col] = test_complex_df[col].fillna(medians[col])

    train_raw = train_df.iloc[:TRAIN_END].copy()
    val_raw = train_df.iloc[TRAIN_END:VAL_END].copy()
    test_raw = train_df.iloc[VAL_END:].copy()
    y_train = train_raw["y"].values.astype(int)
    y_val = val_raw["y"].values.astype(int)
    y_test = test_raw["y"].values.astype(int)

    print("\n[Split scheme C]")
    print(
        f"  Train [:TRAIN_END)     rows={len(train_raw)} pos={int(y_train.sum())} rate={y_train.mean()*100:.4f}%"
    )
    print(
        f"  Val   [TRAIN_END:VAL)  rows={len(val_raw)} pos={int(y_val.sum())} rate={y_val.mean()*100:.4f}%"
    )
    print(
        f"  Test  [VAL_END:)       rows={len(test_raw)} pos={int(y_test.sum())} rate={y_test.mean()*100:.4f}%"
    )

    print("\n[1] LOF / PCA fit on TRAIN slice only ...")
    lof_model, scaler_lof, pca_model, pca_scaler = fit_lof_pca(train_raw, feature_cols)

    print("[2] Feature engineering (train / val / test / submit tests) ...")
    train_fe = create_features_fast(train_raw.drop(columns=["y"]), feature_cols, lof_model, scaler_lof)
    val_fe = create_features_fast(val_raw.drop(columns=["y"]), feature_cols, lof_model, scaler_lof)
    test_fe = create_features_fast(test_raw.drop(columns=["y"]), feature_cols, lof_model, scaler_lof)
    test_simple_fe = create_features_fast(test_simple_df, feature_cols, lof_model, scaler_lof)
    test_complex_fe = create_features_fast(test_complex_df, feature_cols, lof_model, scaler_lof)

    train_fe = add_pca(train_fe, train_raw, feature_cols, pca_model, pca_scaler)
    val_fe = add_pca(val_fe, val_raw, feature_cols, pca_model, pca_scaler)
    test_fe = add_pca(test_fe, test_raw, feature_cols, pca_model, pca_scaler)
    test_simple_fe = add_pca(test_simple_fe, test_simple_df, feature_cols, pca_model, pca_scaler)
    test_complex_fe = add_pca(test_complex_fe, test_complex_df, feature_cols, pca_model, pca_scaler)

    all_cols = set(train_fe.columns)
    for df in (val_fe, test_fe, test_simple_fe, test_complex_fe):
        all_cols &= set(df.columns)
    common_cols = sorted(all_cols)
    train_fe = train_fe[common_cols]
    val_fe = val_fe[common_cols]
    test_fe = test_fe[common_cols]
    test_simple_fe = test_simple_fe[common_cols]
    test_complex_fe = test_complex_fe[common_cols]

    train_scaled, scaler = preprocess(train_fe, fit_scaler=True)
    val_scaled = preprocess(val_fe, scaler=scaler)
    test_scaled = preprocess(test_fe, scaler=scaler)
    test_simple_scaled = preprocess(test_simple_fe, scaler=scaler)
    test_complex_scaled = preprocess(test_complex_fe, scaler=scaler)

    X_train = train_scaled.values
    X_val = val_scaled.values
    X_test = test_scaled.values

    print("\n[3] Fit IsolationForest once on train ...")
    models_if = {"iforest": train_isolation_forest(X_train, y_train)}

    weight_grid = generate_weight_grid(0.05)
    print(
        f"\n[4] Grid search on VAL | {len(XGB_PARAM_GRID)} xgb x {len(LGB_PARAM_GRID)} lgb x {len(weight_grid)} weights"
    )

    best_val_f1 = -1.0
    best_auc_pr = -1.0
    best_rec: dict[str, Any] | None = None
    rows_scored = 0
    scan_rows: list[dict[str, Any]] = []

    for xi, xgb_kw in enumerate(XGB_PARAM_GRID):
        for lj, lgb_kw in enumerate(LGB_PARAM_GRID):
            print(f"  -> training xgb cfg {xi} lgb cfg {lj} ...", flush=True)
            m_xgb = train_xgboost_custom(X_train, y_train, X_val, y_val, xgb_kw)
            m_lgb = train_lightgbm_custom(X_train, y_train, X_val, y_val, lgb_kw)
            models = {"xgb": m_xgb, "lgb": m_lgb, "iforest": models_if["iforest"]}

            for wx, wl, wf in weight_grid:
                val_scores = ensemble_weighted(models, X_val, wx, wl, wf)
                thresh, _ = find_best_threshold(y_val, val_scores)
                auc_pr = average_precision_score(y_val, val_scores)
                pred_v = (val_scores >= thresh).astype(int)
                f1_v = f1_score(y_val, pred_v)
                rows_scored += 1
                auc_roc_v = roc_auc_score(y_val, val_scores)
                scan_rows.append(
                    {
                        "xi": xi,
                        "lj": lj,
                        "xgb_max_depth": xgb_kw["max_depth"],
                        "xgb_learning_rate": xgb_kw["learning_rate"],
                        "xgb_min_child_weight": xgb_kw["min_child_weight"],
                        "lgb_num_leaves": lgb_kw["num_leaves"],
                        "lgb_learning_rate": lgb_kw["learning_rate"],
                        "lgb_min_child_samples": lgb_kw["min_child_samples"],
                        "wx": wx,
                        "wl": wl,
                        "wf": wf,
                        "val_threshold_f1": thresh,
                        "val_f1": f1_v,
                        "val_auc_pr": auc_pr,
                        "val_auc_roc": auc_roc_v,
                    }
                )
                better = f1_v > best_val_f1 or (
                    np.isclose(f1_v, best_val_f1) and auc_pr > best_auc_pr
                )
                if better:
                    best_val_f1 = f1_v
                    best_auc_pr = auc_pr
                    best_rec = {
                        "xi": xi,
                        "lj": lj,
                        "xgb_kw": dict(xgb_kw),
                        "lgb_kw": dict(lgb_kw),
                        "wx": wx,
                        "wl": wl,
                        "wf": wf,
                        "threshold_search": thresh,
                        "val_f1": f1_v,
                        "val_auc_pr": auc_pr,
                        "val_auc_roc": auc_roc_v,
                    }

    assert best_rec is not None

    grid_csv = os.path.join(OUTPUT_DIR, "grid_search_val.csv")
    pd.DataFrame(scan_rows).sort_values(
        ["val_f1", "val_auc_pr"], ascending=[False, False]
    ).to_csv(grid_csv, index=False)
    print(f"\n  Full scan table -> {grid_csv} ({len(scan_rows)} rows)")
    print(f"\n  Scored {rows_scored} (param_pair, weight) combinations on VAL.")
    print("\n[5] Retrain best XGB/LGB pair once (avoid stale booster handles) ...")
    m_xgb_best = train_xgboost_custom(
        X_train, y_train, X_val, y_val, best_rec["xgb_kw"]
    )
    m_lgb_best = train_lightgbm_custom(
        X_train, y_train, X_val, y_val, best_rec["lgb_kw"]
    )
    best_models_search = {
        "xgb": m_xgb_best,
        "lgb": m_lgb_best,
        "iforest": models_if["iforest"],
    }
    val_scores_best = ensemble_weighted(
        best_models_search,
        X_val,
        best_rec["wx"],
        best_rec["wl"],
        best_rec["wf"],
    )
    best_thresh_val, _ = find_best_threshold(y_val, val_scores_best)
    best_rec["val_f1"] = f1_score(
        y_val, (val_scores_best >= best_thresh_val).astype(int)
    )
    best_rec["val_auc_pr"] = average_precision_score(y_val, val_scores_best)
    best_rec["val_auc_roc"] = roc_auc_score(y_val, val_scores_best)

    print("\n[5b] Best on validation (primary F1, tie-break AUC-PR)")
    for k in (
        "xi",
        "lj",
        "xgb_kw",
        "lgb_kw",
        "wx",
        "wl",
        "wf",
        "threshold_search",
        "val_f1",
        "val_auc_pr",
        "val_auc_roc",
    ):
        print(f"    {k}: {best_rec[k]}")

    # --- Final refit: LOF/PCA + scaler on train+val; trees on full train+val with rounds from search ---
    print("\n[6] Final refit on Train+Val (no test) ...")
    tv_df = train_df.iloc[:VAL_END].copy()
    lof_f, sl_f, pca_f, pca_s_f = fit_lof_pca(tv_df, feature_cols)

    tv_fe = create_features_fast(tv_df.drop(columns=["y"]), feature_cols, lof_f, sl_f)
    tv_fe = add_pca(tv_fe, tv_df, feature_cols, pca_f, pca_s_f)
    tv_fe = tv_fe[common_cols]
    tv_scaled, scaler_f = preprocess(tv_fe, fit_scaler=True)
    X_tv = tv_scaled.values
    y_tv = tv_df["y"].values.astype(int)

    xgb_ntree = booster_best_rounds(best_models_search["xgb"])
    lgb_rounds = lgb_best_rounds(best_models_search["lgb"])

    final_models = {
        "iforest": train_isolation_forest(X_tv, y_tv),
        "xgb": train_final_xgb(
            X_tv, y_tv, best_rec["xgb_kw"], max(50, int(xgb_ntree * 1.15))
        ),
        "lgb": train_final_lgb(
            X_tv, y_tv, best_rec["lgb_kw"], max(50, int(lgb_rounds * 1.15))
        ),
    }

    val_idx = np.arange(TRAIN_END, VAL_END)
    val_fe_f = create_features_fast(
        train_df.iloc[val_idx].drop(columns=["y"]),
        feature_cols,
        lof_f,
        sl_f,
    )
    val_fe_f = add_pca(val_fe_f, train_df.iloc[val_idx], feature_cols, pca_f, pca_s_f)
    val_fe_f = val_fe_f[common_cols]
    val_scaled_f = preprocess(val_fe_f, scaler=scaler_f)
    X_val_f = val_scaled_f.values

    val_scores_cal = ensemble_weighted(
        final_models,
        X_val_f,
        best_rec["wx"],
        best_rec["wl"],
        best_rec["wf"],
    )
    final_thresh, _ = find_best_threshold(y_val, val_scores_cal)
    print(f"    Recalibrated threshold on VAL (final models): {final_thresh:.6f}")

    print("\n[7] Held-out TEST metrics (single pass, no selection on test) ...")
    test_idx = np.arange(VAL_END, len(train_df))
    test_fe_f = create_features_fast(
        train_df.iloc[test_idx].drop(columns=["y"]),
        feature_cols,
        lof_f,
        sl_f,
    )
    test_fe_f = add_pca(test_fe_f, train_df.iloc[test_idx], feature_cols, pca_f, pca_s_f)
    test_fe_f = test_fe_f[common_cols]
    test_scaled_f = preprocess(test_fe_f, scaler=scaler_f)
    X_test_f = test_scaled_f.values

    test_scores = ensemble_weighted(
        final_models,
        X_test_f,
        best_rec["wx"],
        best_rec["wl"],
        best_rec["wf"],
    )
    test_auc_pr = average_precision_score(y_test, test_scores)
    test_auc_roc = roc_auc_score(y_test, test_scores)
    test_pred = (test_scores >= final_thresh).astype(int)
    test_f1 = f1_score(y_test, test_pred)
    print(f"    TEST AUC-PR: {test_auc_pr:.4f} | AUC-ROC: {test_auc_roc:.4f} | F1: {test_f1:.4f}")
    print(f"    TEST positives predicted: {int(test_pred.sum())} / {len(test_pred)}")

    print("\n[8] Predict external test CSVs ...")
    ts_simple = create_features_fast(test_simple_df, feature_cols, lof_f, sl_f)
    ts_complex = create_features_fast(test_complex_df, feature_cols, lof_f, sl_f)
    ts_simple = add_pca(ts_simple, test_simple_df, feature_cols, pca_f, pca_s_f)
    ts_complex = add_pca(ts_complex, test_complex_df, feature_cols, pca_f, pca_s_f)
    ts_simple = ts_simple[common_cols]
    ts_complex = ts_complex[common_cols]
    ts_simple_s = preprocess(ts_simple, scaler=scaler_f)
    ts_complex_s = preprocess(ts_complex, scaler=scaler_f)

    scores_simple = ensemble_weighted(
        final_models,
        ts_simple_s.values,
        best_rec["wx"],
        best_rec["wl"],
        best_rec["wf"],
    )
    scores_complex = ensemble_weighted(
        final_models,
        ts_complex_s.values,
        best_rec["wx"],
        best_rec["wl"],
        best_rec["wf"],
    )
    pred_simple = (scores_simple >= final_thresh).astype(int)
    pred_complex = (scores_complex >= final_thresh).astype(int)
    print(f"    pred_simple: {int(pred_simple.sum())} / {len(pred_simple)}")
    print(f"    pred_complex: {int(pred_complex.sum())} / {len(pred_complex)}")

    pd.DataFrame({"y_pred": pred_simple}).to_csv(f"{OUTPUT_DIR}/pred_simple.csv", index=False)
    pd.DataFrame({"y_pred": pred_complex}).to_csv(f"{OUTPUT_DIR}/pred_complex.csv", index=False)
    with open(f"{OUTPUT_DIR}/model.pkl", "wb") as f:
        pickle.dump(
            {
                "models": final_models,
                "scaler": scaler_f,
                "threshold": final_thresh,
                "ensemble_weights": (best_rec["wx"], best_rec["wl"], best_rec["wf"]),
                "xgb_kw": best_rec["xgb_kw"],
                "lgb_kw": best_rec["lgb_kw"],
                "feature_cols": common_cols,
                "split": {"TRAIN_END": TRAIN_END, "VAL_END": VAL_END},
                "val_metrics": {
                    "f1": best_rec["val_f1"],
                    "auc_pr": best_rec["val_auc_pr"],
                    "auc_roc": best_rec["val_auc_roc"],
                },
                "test_metrics": {
                    "f1": test_f1,
                    "auc_pr": test_auc_pr,
                    "auc_roc": test_auc_roc,
                },
            },
            f,
        )

    print(f"\n  Saved to {OUTPUT_DIR}/")
    print("=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
