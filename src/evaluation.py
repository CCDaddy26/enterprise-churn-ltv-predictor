"""Evaluation, metric calculation, and visualization routines for Phase 2."""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import FIGURES_DIR, TABLES_DIR

logger = logging.getLogger(__name__)


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics including confusion matrix and AUC scores."""
    y_t = np.array(y_true)
    y_p = (np.array(y_prob) >= threshold).astype(int) if y_prob is not None else np.array(y_pred)

    cm = confusion_matrix(y_t, y_p)
    tn, fp, fn, tp = cm.ravel()

    try:
        roc_auc = float(roc_auc_score(y_t, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, y_prob))
    except Exception:
        pr_auc = float(y_t.mean())

    return {
        "threshold": round(float(threshold), 3),
        "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
        "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
        "f1": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision_pr_auc": round(pr_auc, 4),
        "accuracy": round(float((tp + tn) / len(y_t)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def plot_model_cv_comparison(
    cv_summary_df: pd.DataFrame,
    save_path: Path = None,
) -> Path:
    """Generate high-resolution comparison chart of cross-validation metrics across models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    models = cv_summary_df.index.tolist()
    x = np.arange(len(models))

    # 1. PR-AUC and ROC-AUC
    axes[0].bar(
        x - 0.2,
        cv_summary_df["pr_auc_mean"],
        yerr=cv_summary_df["pr_auc_std"],
        width=0.38,
        label="Average Precision (PR-AUC)",
        color="#2b5c8f",
        capsize=4,
        edgecolor="black",
    )
    axes[0].bar(
        x + 0.2,
        cv_summary_df["roc_auc_mean"],
        yerr=cv_summary_df["roc_auc_std"],
        width=0.38,
        label="ROC-AUC",
        color="#d95f02",
        capsize=4,
        edgecolor="black",
    )
    axes[0].set_title("Cross-Validation Ranking Metrics (5-Fold CV)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Score", fontsize=10)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Precision and Recall
    axes[1].bar(
        x - 0.2,
        cv_summary_df["recall_mean"],
        yerr=cv_summary_df["recall_std"],
        width=0.38,
        label="Recall (Sensitivity)",
        color="#7570b3",
        capsize=4,
        edgecolor="black",
    )
    axes[1].bar(
        x + 0.2,
        cv_summary_df["precision_mean"],
        yerr=cv_summary_df["precision_std"],
        width=0.38,
        label="Precision",
        color="#1b9e77",
        capsize=4,
        edgecolor="black",
    )
    axes[1].set_title("Cross-Validation Precision & Recall (5-Fold CV)", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Score", fontsize=10)
    axes[1].set_ylim(0, 1.0)
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Model Family Comparison on Development Partition (N=5,634)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "06_model_cv_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_precision_recall_curves(
    y_test: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
    save_path: Path = None,
) -> Path:
    """Plot Precision-Recall curves on held-out test predictions."""
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # Prior baseline prevalence (no skill line)
    baseline_prev = y_test.mean()
    ax.axhline(
        y=baseline_prev,
        color="grey",
        linestyle="--",
        label=f"Random Chance (Prevalence = {baseline_prev:.2%})",
    )

    palette = {
        "dummy_most_frequent": "#999999",
        "logistic_regression": "#2b5c8f",
        "random_forest": "#1b9e77",
        "xgboost_champion": "#d95f02",
    }

    for name, y_prob in model_probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        color = palette.get(name, "#333333")
        ax.plot(
            recall,
            precision,
            color=color,
            lw=2,
            label=f"{name} (Avg Prec = {ap:.4f})",
        )

    ax.set_title("Precision-Recall Curves on Held-Out Test Split (N=1,409)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, labelpad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "07_precision_recall_curves.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
