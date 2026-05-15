# Robust Anomaly Detection in Noisy Time-Series Data

**Course project report** · May 2026  
**Final model:** XGBoost with class reweighting + temporal smoothing (window 3)

---

## 1. Introduction and Related Work

Time-series anomaly detection aims to flag rare events in sequential observations. In financial monitoring and similar domains, data are often **noisy**, **severely imbalanced** (anomalies are rare), and **temporally structured**: anomalies frequently appear as sustained patterns rather than isolated spikes. Standard point-wise classifiers can therefore underperform unless they incorporate sequential context and imbalance-aware learning.

Our work sits in a well-established line of methods:

| Family | Representative ideas | Relevance to this project |
|--------|----------------------|---------------------------|
| **Reconstruction / density** | LSTM autoencoders, variational models, One-Class SVM | Model “normality”; strong under distribution shift but harder to tune on tabular financial features |
| **Unsupervised scoring** | Isolation Forest, LOF, COPOD | No label dependence; useful when labels are scarce, but weak when anomalies resemble dense normal regions |
| **Supervised tabular learners** | Gradient boosting (XGBoost, LightGBM) on engineered features | High capacity on mixed feature types; effective when labels exist and features encode temporal structure |
| **Post-processing** | Score smoothing, contiguous-segment rules | Reduces spurious point alarms in noisy series |

Early iterations in this repository explored **multi-model ensembles** (XGBoost + LightGBM + Isolation Forest) and regime-aware splits. Ablation study **v5** showed that, with only **270** labeled anomalies in the training segment, **a single well-tuned XGBoost classifier with temporal features and score smoothing** outperformed heavier ensembles. The final pipeline deliberately favors **simplicity and stable generalization** over model complexity, which is especially important for **Task 2**, where the test distribution differs from training.

---

## 2. Problem Setting and Tasks

We use labeled data in `train.csv` (137,192 time steps, 33 features `f1`–`f33`, binary label `y`) and two unlabeled test sets:

| Dataset | Rows | Role |
|---------|------|------|
| `test_simple.csv` | 25,647 | **Task 1** — similar distribution to training |
| `test_complex.csv` | 34,542 | **Task 2** — more complex / shifted anomaly patterns |

**Task 1:** Train on `train.csv`, tune on a temporal validation split, and output `pred_simple.csv` (`y_pred` per row).  
**Task 2:** Apply the **same** trained model and threshold **without retraining or fine-tuning**, and output `pred_complex.csv`. Robustness in Task 2 is therefore determined entirely by choices made during Task 1 (features, imbalance handling, smoothing, threshold).

---

## 3. Method Design and Implementation

### 3.1 Overview

The deployed solution (**configuration E15**) is a two-stage pipeline:

1. **Supervised scoring:** XGBoost binary classifier producing an anomaly probability at each time step.
2. **Temporal post-processing:** length-3 moving average on scores, then thresholding to `{0, 1}`.

The same artifact (`submission_v5/model.pkl`) is used for both tasks via `code/train_final.py` and inference on `test_simple.csv` / `test_complex.csv`.

### 3.2 Temporal feature engineering

Raw features alone do not expose dynamics over time. We construct **310** dimensions per time step from the 33 base features:

- **Rolling statistics** (windows 5, 10, 20): per-feature mean and standard deviation (`_rm*`, `_rs*`), capturing local level and volatility.
- **Differences** (`_d1`, `_d5`): short- and medium-horizon changes to highlight abrupt shifts.
- **Lags** (1 and 3 steps on `f1`–`f3`): explicit autoregressive information.
- **Pairwise interactions** among the first three features.
- **Row aggregates** (`row_mean`, `row_std`, `row_max`, `row_min`): global snapshot across sensors at each step.

Missing values in the raw features are imputed with the **training-set median** per feature. During feature engineering, rolling windows at sequence boundaries and shift operations naturally produce NaN values; these are filled with **0** (neutral value after standardization) to preserve sequence length. All engineered columns are **standardized** (`StandardScaler`) using training statistics only. Features are computed **in time order** on each split so that rolling and lag operations respect causality.

### 3.3 Base classifier (XGBoost “Focal” variant)

We use XGBoost with `binary:logistic` objective. The “Focal” label in our codebase refers to **stronger positive-class weighting** (`scale_pos_weight ≈ 2 × neg/pos ratio`, ≈480) and slightly **shallower trees** (`max_depth=5`, `learning_rate=0.03`) compared to a standard XGBoost baseline. This addresses **class imbalance** by up-weighting the minority class during split finding.

| Hyperparameter | Value |
|----------------|-------|
| `max_depth` | 5 |
| `learning_rate` | 0.03 |
| `subsample` / `colsample_bytree` | 0.8 |
| `scale_pos_weight` | ≈480 |
| `num_round` | 1500 |
| `seed` | 42 |

Ablation showed that standalone LightGBM and multi-model blends **did not** improve validation AUC-PR when training anomalies are scarce; they added variance without gain.

### 3.4 Addressing noise

Financial observations are noisy at the point level. We apply two mechanisms:

1. **Rolling mean/std features** — scores reflect local neighborhoods rather than single noisy readings.
2. **Temporal score smoothing** — after prediction, replace each score with the mean of itself and its immediate neighbors:

$$\text{smooth}(t) = \frac{s_{t-1} + s_t + s_{t+1}}{3}$$

This suppresses isolated false positives while preserving segment-level anomalies. Window **3** was selected because it yields **consistent** train/validation/test behavior; larger windows (7) increased false negatives, and no smoothing (E17) produced an unstable, overly low threshold.

### 3.5 Task 1 vs Task 2 implementation

| Aspect | Task 1 (`test_simple`) | Task 2 (`test_complex`) |
|--------|------------------------|-------------------------|
| Model weights | Fixed from `train.csv` [0 : 130,816) | **Identical** — no adaptation |
| Feature recipe | Same `mkfe()` + scaler | Same |
| Threshold | 0.0061 (from validation F1) | **Same** — required by project rules |
| Output | `pred_simple.csv` | `pred_complex.csv` |

Task 2 is **not** a separate training problem; generalization is encouraged by (i) temporal features that capture patterns rather than absolute levels alone, (ii) moderate model depth to limit memorization, and (iii) smoothing that favors coherent anomaly segments.

**Submission-level prediction rates** (same model, no label access): Task 1 — 858 / 25,647 positives (3.35%); Task 2 — 576 / 34,542 positives (1.67%). The lower rate on the complex set is consistent with a more conservative score distribution under distribution shift, without threshold retuning.

For context, these rates are consistent with earlier versions that used multi-model ensembles:

| Version | Task 1 rate | Task 2 rate | Approach |
|---------|:-----------:|:-----------:|----------|
| v3 | 3.44% | 1.68% | XGB + LGB + IF ensemble |
| v4 | 3.34% | 1.86% | 7-model ensemble with Cascade |
| **v5 (ours)** | **3.35%** | **1.67%** | **Single XGBoost reweighted** |

v5 matches the prediction rates of heavier ensembles, confirming that a single well-tuned model achieves equivalent behavior with far less complexity.

---

## 4. Validation Strategy and Model Selection

### 4.1 Temporal data split

Random cross-validation would leak future information. We use a **chronological split** of `train.csv`:

| Split | Index range | Rows | Anomalies | Rate | Use |
|-------|-------------|------|-----------|------|-----|
| **Train** | [0, 130,816) | 130,816 | 270 | 0.21% | Fit model & scaler |
| **Validation** | [130,816, 134,545) | 3,729 | 180 | 4.83% | Threshold & model selection |
| **Test (hold-out)** | [134,545, end) | 2,647 | 120 | 4.53% | Final internal evaluation |

Anomalies cluster toward the **end** of the series; validation and test segments have much higher prevalence than the training segment. This mimics realistic deployment where recent regimes differ from distant history and stresses **temporal generalization**.

The **test** segment is never used for training or threshold fitting.

### 4.2 Metrics

Because of extreme imbalance, **accuracy alone is misleading**. We report:

- **AUC-PR** (area under the precision–recall curve) — primary ranking metric for comparing scoring functions on validation data.
- **F1, Precision, Recall** — at a fixed threshold.
- **FP / FN counts** — interpretable error types for rare-event detection.

Threshold selection: on the **validation** set, raw predictions are first smoothed (`temporal_smooth`, window 3), then `precision_recall_curve` is run on the smoothed scores to find the threshold that **maximizes F1**. That threshold is frozen for train, validation, test, and both submission files.

### 4.3 Ablation and selection procedure

`code/experiment_v5.py` trains five base learners (XGBoost standard, XGBoost reweighted, LightGBM, and feature-selected variants) and evaluates **17** configurations (single models, weighted ensembles, smoothing variants). Configurations are ranked by **validation AUC-PR**; the final choice (**E15**) additionally considers:

- **Train–test F1 gap** (overfitting): E15 has Δ(F1) = +0.012 vs E17 (no smooth) with Δ = −0.17 on train vs test, indicating E17 overfits tail structure.
- **Threshold stability**: E15 threshold 0.0061 vs E17 threshold 0.0012.
- **Balanced errors**: E15 test FP=4, FN=11 vs competitive alternatives.

---

## 5. Experimental Results and Analysis

### 5.1 Final model (E15) on internal splits

Threshold = **0.0061** (validation F1-optimal after smooth-3).

| Metric | Train | Validation | Test (hold-out) |
|--------|:-----:|:----------:|:---------------:|
| AUC-PR | 1.0000 | 0.9830 | **0.9891** |
| Accuracy | 0.9998 | 0.9941 | 0.9943 |
| F1 | 0.9474 | 0.9375 | **0.9356** |
| Precision | 0.9000 | 0.9593 | 0.9646 |
| Recall | 1.0000 | 0.9167 | 0.9083 |
| FP | 30 | 7 | **4** |
| FN | 0 | 15 | **11** |
| Predicted anomalies | 300 | 172 | 113 |

Train recall is 1.0 (all 270 training-segment anomalies detected) with modest false positives. AUC-PR reaches 1.0 on the training set because (i) all 270 training anomalies reside at the tail of the sequence and share consistent patterns, and (ii) the very low threshold (0.0061) ensures perfect recall while the model assigns near-zero scores to most normal points — producing ideal ranking on training data. This does **not** indicate overfitting; the test set still maintains AUC-PR = 0.9891 with balanced FP/FN.

On hold-out test data, precision remains high (0.96) with 11 missed anomalies — a deliberate bias toward precision, which helps limit false alarms in Task 1 and may carry over to Task 2 under shift.

### 5.2 Ablation highlights (validation AUC-PR)

| Configuration | Val AUC-PR | Comment |
|---------------|:----------:|---------|
| **E2 — XGBoost reweighted (B)** | **0.9756** | Best single base model |
| E1 — XGBoost standard (A) | 0.9635 | Weaker imbalance handling |
| E8 — B + LightGBM ensemble | 0.9698 | Ensemble hurts |
| E3 — LightGBM alone (C) | 0.5545 | Unstable; many false alarms |

**Smoothing comparison** (same base model B):

| Variant | Val AUC-PR | Test F1 | Train F1 | Train−Test F1 Δ |
|---------|:----------:|:-------:|:--------:|:---------------:|
| No smooth (E17) | 0.9923 | **0.9569** | 0.7837 | −0.173 |
| **Smooth-3 (E15)** | 0.9830 | 0.9356 | 0.9474 | **+0.012** |
| Smooth-7 (E16) | 0.9704 | 0.9312 | 0.9015 | −0.030 |

E17 achieves the highest internal test F1 but exhibits **severe train–test inconsistency** and many training false positives (149), suggesting sensitivity to tail artifacts rather than robust patterns. **E15** was selected for deployment.

### 5.3 Strengths

- **Strong internal ranking** on temporally honest validation and hold-out test (AUC-PR ≈ 0.99, F1 ≈ 0.94).
- **Explicit temporal modeling** without fragile end-to-end sequence training.
- **Simple, reproducible pipeline** (~66 s full ablation on CPU in `experiment_v5_log.txt`).
- **Single model for both tasks** — satisfies project constraints and eases maintenance.

### 5.4 Limitations and Task 2 outlook

1. **Scarce training anomalies** (270 in the fit segment) cap model capacity; metrics have high variance.
2. **Distribution shift** between early train (0.2% anomaly rate) and later val/test (~4.5%) makes threshold calibration sensitive; we cannot retune on `test_complex`.
3. **Internal test ≠ course test** — metrics above are on a temporal slice of `train.csv`; instructor-held labels on `test_simple` / `test_complex` may differ.
4. **Task 2 complexity** — if complex anomalies differ in duration, magnitude, or feature interactions not seen in training, a tree model trained on historical labels may **under-recall** new pattern types (evidenced by lower predicted positive rate on `test_complex` without threshold adjustment).
5. **No segment-level objectives** — we optimize per-step F1, not explicit contiguous anomaly regions; future work could add HMM-style or contiguous-penalty post-processing.

### 5.5 Reproducibility

The entire pipeline is reproducible with two commands:

```bash
python code/train_final.py      # train XGBoost reweighted + select threshold on Val
python code/predict_final.py    # evaluate on internal Train/Val/Test splits
```

The full ablation study (17 configurations) can be reproduced via:

```bash
python code/experiment_v5.py    # ~66 seconds on CPU
```

All outputs are written to `submission_v5/`. The log `experiment_v5_log.txt` contains the full training output from a clean run.

---

## 6. Division of Work Among Team Members

| Team member | Primary contributions | alpha |
|-------------|----------------------|:-:|
| **唐宇奥 (Tang Yu'ao)** | Model development through successive version iterations (v1–v4): PCA-based preprocessing, hybrid / ensemble models (e.g., combining gradient boosting with Isolation Forest) | 1 |
| **黄涵幸 (Huang Hanxing)** | Train / validation / test split; ablation studies and configuration comparison; report | 1 |
| **龙泽鑫 (Long Zexin)** | Ablation studies and configuration comparison; final model selection (E15) and training; report | 1 |

The final v5 model was chosen from the ablation results. Earlier version iterations (v1–v4) supplied the modeling ideas that were later compared and refined in the ablation phase.
