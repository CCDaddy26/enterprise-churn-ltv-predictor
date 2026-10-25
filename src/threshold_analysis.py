"""Classification threshold analysis and trade-off simulations for Phase 3."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.config import FIGURES_DIR, TABLES_DIR

logger = logging.getLogger(__name__)


def compute_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray = None,
) -> pd.DataFrame:
    """Compute precision, recall, F1, and confusion matrix counts across threshold spectrum."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 81)

    records = []
    total_customers = len(y_true)

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        flagged = int(tp + fp)

        records.append(
            {
                "threshold": round(float(tau), 3),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "customers_flagged": flagged,
                "pct_flagged": round(float(flagged / total_customers * 100), 2),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
            }
        )

    return pd.DataFrame(records)


def plot_threshold_tradeoffs(
    threshold_df: pd.DataFrame,
    selected_thresholds: List[float] = [0.30, 0.40, 0.50, 0.60],
    save_path: Path = None,
) -> Path:
    """Plot threshold trade-off curves (Precision, Recall, F1, Customer Volume Flagged)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Precision, Recall, and F1 curves
    axes[0].plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#1b9e77", lw=2)
    axes[0].plot(threshold_df["threshold"], threshold_df["recall"], label="Recall (Sensitivity)", color="#d95f02", lw=2)
    axes[0].plot(threshold_df["threshold"], threshold_df["f1"], label="F1-Score", color="#2b5c8f", lw=2, linestyle="--")

    for tau in selected_thresholds:
        axes[0].axvline(x=tau, color="grey", linestyle=":", alpha=0.6)

    axes[0].set_title("Precision-Recall Trade-off vs. Threshold", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Decision Threshold (tau)", fontsize=10)
    axes[0].set_ylabel("Score", fontsize=10)
    axes[0].set_xlim(0.1, 0.9)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower left")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Operational Volume Flagged & Error Composition
    axes[1].plot(
        threshold_df["threshold"],
        threshold_df["customers_flagged"],
        label="Total Customers Flagged",
        color="#333333",
        lw=2.5,
    )
    axes[1].plot(threshold_df["threshold"], threshold_df["true_positives"], label="True Churners (TP)", color="#2b5c8f", lw=1.8)
    axes[1].plot(threshold_df["threshold"], threshold_df["false_positives"], label="False Alarms (FP)", color="#e41a1c", lw=1.8)

    axes[1].set_title("Operational Outreach Volume Flagged vs. Threshold", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Decision Threshold (tau)", fontsize=10)
    axes[1].set_ylabel("Customer Count", fontsize=10)
    axes[1].set_xlim(0.1, 0.9)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Operational Threshold Diagnostics (Development Split, N=5,634)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "13_threshold_tradeoffs.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
