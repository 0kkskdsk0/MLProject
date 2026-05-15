# v5 Experiment Report

**Date**: 2026-05-15
**Goal**: Systematic ablation study to find optimal model configuration for anomaly detection

## Data Split

| Set | Rows | Anomalies | Rate |
|-----|------|-----------|------|
| Train | 130,816 | 270 | 0.21% |
| Val | 3,729 | 180 | 4.83% |
| Test | 2,647 | 120 | 4.53% |

## Models

| Code | Model | Key Params |
|------|-------|-----------|
| A | XGBoost std | depth=6, lr=0.05, scale_pos_weight=240, 2000 rounds |
| B | **XGBoost Focal** | depth=5, lr=0.03, scale_pos_weight=480, 1500 rounds |
| C | LightGBM | leaves=31, lr=0.05, is_unbalance=True, 2000 rounds |
| D | XGBoost Selected | same as A, top-100 features |
| E | LightGBM Selected | same as C, top-100 features |

## Top Results (ranked by Val AUC-PR)

| Rank | Config | Val AUC-PR | Test AUC-PR | Test F1 | FP | FN |
|------|--------|:---------:|:-----------:|:-------:|:--:|:--:|
| **1** | **E17 B + nosmooth** | **0.9923** | **0.9974** | **0.9569** | **1** | **9** |
| **2** | **E15 B + smooth3** | **0.9830** | **0.9891** | **0.9356** | **4** | **11** |
| 3 | E2 B single | 0.9756 | 0.9825 | 0.9333 | 8 | 8 |

## Key Findings

1. **XGBoost Focal single model is optimal** — no ensemble, no feature selection needed
2. **smooth3 is most stable** — consistent metrics across train/val/test
3. **LightGBM overfits** — Val AUC-PR only 0.55
4. **Multi-model ensemble is useless** — all combos worse than single B

## Recommendation

**Primary: E15 — XGBoost Focal + smooth3** (Test AUC-PR=0.9891, F1=0.9356)
**Alternative: E17 — no smoothing** (Test AUC-PR=0.9974, F1=0.9569, FP=1)
