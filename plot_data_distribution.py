"""
Plot dataset summaries for train / test_simple / test_complex.
Run from repository root: python plot_data_distribution.py
Outputs PNGs under pic/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PIC_DIR = ROOT / "pic"


def main() -> None:
    PIC_DIR.mkdir(parents=True, exist_ok=True)

    train_path = DATA_DIR / "train.csv"
    simple_path = DATA_DIR / "test_simple.csv"
    complex_path = DATA_DIR / "test_complex.csv"

    print("Loading CSV...")
    train_df = pd.read_csv(train_path)
    simple_df = pd.read_csv(simple_path)
    complex_df = pd.read_csv(complex_path)

    feature_cols = [c for c in train_df.columns if c.startswith("f")]
    n_train, n_simple, n_complex = len(train_df), len(simple_df), len(complex_df)

    # --- 1) Row counts ---
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["train", "test_simple", "test_complex"]
    counts = [n_train, n_simple, n_complex]
    ax.bar(names, counts, color=["#4477aa", "#66ccee", "#ee8866"])
    ax.set_ylabel("Rows")
    ax.set_title("Dataset sizes")
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PIC_DIR / "dataset_row_counts.png", dpi=150)
    plt.close(fig)

    # --- 2) Label distribution (train) ---
    y = train_df["y"].astype(int)
    vc = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["y=0 (normal)", "y=1 (anomaly)"], [vc.get(0, 0), vc.get(1, 0)], color=["#44aa99", "#cc6677"])
    ax.set_ylabel("Count")
    ax.set_title("Train label distribution")
    for i, lab in enumerate([vc.get(0, 0), vc.get(1, 0)]):
        ax.text(i, lab, f"{lab:,}\n({100 * lab / n_train:.3f}%)", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PIC_DIR / "train_label_distribution.png", dpi=150)
    plt.close(fig)

    # --- 3) Anomaly positions along time index ---
    idx = np.arange(n_train)
    anomaly_idx = idx[y.values == 1]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(anomaly_idx, np.ones_like(anomaly_idx), s=8, c="#cc6677", alpha=0.6, label="anomaly")
    ax.set_xlim(0, n_train)
    ax.set_yticks([])
    ax.set_xlabel("Row index (time order)")
    ax.set_title("Train anomalies along index")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(PIC_DIR / "train_anomaly_index.png", dpi=150)
    plt.close(fig)

    # --- 4) Rolling positive rate (smoothed) ---
    window = 5000
    roll = y.rolling(window=window, min_periods=500).mean()
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(idx, roll.values, color="#332288", lw=1)
    ax.set_xlabel("Row index")
    ax.set_ylabel(f"Rolling mean(y), window={window}")
    ax.set_title("Local anomaly rate along train (time-ordered)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(PIC_DIR / "train_rolling_anomaly_rate.png", dpi=150)
    plt.close(fig)

    # --- 5) Missing value rate per column (train) ---
    miss = train_df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(miss) + 1)))
    if len(miss):
        miss.plot(kind="barh", ax=ax, color="#6699cc")
        ax.set_xlabel("Fraction NaN")
        ax.set_title("Train columns with missing values")
    else:
        ax.text(0.5, 0.5, "No missing values in train", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(PIC_DIR / "train_missing_rate.png", dpi=150)
    plt.close(fig)

    # --- 6) Example feature f1: normal vs anomaly ---
    f1 = train_df["f1"].astype(float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(f1[y == 0], bins=80, density=True, alpha=0.55, label="y=0", color="#44aa99")
    ax.hist(f1[y == 1], bins=40, density=True, alpha=0.65, label="y=1", color="#cc6677")
    ax.set_xlabel("f1")
    ax.set_ylabel("Density")
    ax.set_title("Train f1 distribution by label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PIC_DIR / "train_f1_by_label.png", dpi=150)
    plt.close(fig)

    # --- 7) Correlation heatmap of a small feature subset (optional, compact) ---
    subset = feature_cols[::7][:8]
    corr = train_df[subset].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(subset)))
    ax.set_yticks(range(len(subset)))
    ax.set_xticklabels(subset, rotation=45, ha="right")
    ax.set_yticklabels(subset)
    ax.set_title("Correlation (sample of f-columns)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PIC_DIR / "train_feature_corr_sample.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures to {PIC_DIR.resolve()}")
    for p in sorted(PIC_DIR.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
