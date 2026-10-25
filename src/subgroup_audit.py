"""Subgroup performance audit and fairness diagnostics for Phase 3."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR, TABLES_DIR

logger = logging.getLogger(__name__)


def audit_subgroup_performance(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    subgroup_cols: List[str],
    decision_threshold: float = 0.5,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Calculate granular subgroup metrics including sample sizes, prevalence, precision, recall, and AUCs.

    Ensures that groups with small sample sizes or single-class distributions are labeled unstable.
    """
    records = []

    for col in subgroup_cols:
        if col not in df.columns:
            continue

        for val, group_df in df.groupby(col, observed=False):
            n_samples = len(group_df)
            y_t = group_df[y_true_col].to_numpy()
            y_p = group_df[y_prob_col].to_numpy()
            y_pred = (y_p >= decision_threshold).astype(int)

            n_churners = int(np.sum(y_t))
            prevalence = round(float(n_churners / n_samples), 4) if n_samples > 0 else 0.0

            # Guard against tiny groups
            if n_samples < min_samples:
                records.append(
                    {
                        "Feature": col,
                        "Subgroup": str(val),
                        "Total_Customers": n_samples,
                        "Actual_Churners": n_churners,
                        "Churn_Prevalence": prevalence,
                        "Status": "Sample size < 30 (Unstable)",
                        "Precision": None,
                        "Recall": None,
                        "F1_Score": None,
                        "ROC_AUC": None,
                        "PR_AUC": None,
                    }
                )
                continue

            # Standard metrics
            cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            prec = float(precision_score(y_t, y_pred, zero_division=0))
            rec = float(recall_score(y_t, y_pred, zero_division=0))
            f1 = float(f1_score(y_t, y_pred, zero_division=0))

            # AUC metrics only defined if both classes exist in subgroup
            if len(np.unique(y_t)) > 1:
                roc_auc = round(float(roc_auc_score(y_t, y_p)), 4)
                pr_auc = round(float(average_precision_score(y_t, y_p)), 4)
                status = "Evaluated"
            else:
                roc_auc = None
                pr_auc = None
                status = "Single Class Only (AUC undefined)"

            records.append(
                {
                    "Feature": col,
                    "Subgroup": str(val),
                    "Total_Customers": n_samples,
                    "Actual_Churners": n_churners,
                    "Churn_Prevalence": prevalence,
                    "Status": status,
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "F1_Score": round(f1, 4),
                    "ROC_AUC": roc_auc,
                    "PR_AUC": pr_auc,
                }
            )

    return pd.DataFrame(records)


def plot_subgroup_audit_summary(
    subgroup_df: pd.DataFrame,
    features_to_plot: List[str] = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"],
    save_path: Path = None,
) -> Path:
    """Generate visual subgroup performance audit chart across key segments."""
    plot_df = subgroup_df[
        (subgroup_df["Feature"].isin(features_to_plot)) & (subgroup_df["Status"] == "Evaluated")
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    # 1. Recall across subgroups
    bars1 = axes[0].barh(
        plot_df["Subgroup"],
        plot_df["Recall"],
        color="#2b5c8f",
        edgecolor="black",
        height=0.55,
    )
    for bar, (_, row) in zip(bars1, plot_df.iterrows()):
        w = bar.get_width()
        axes[0].text(
            w + 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.2f} (n={row['Actual_Churners']}/{row['Total_Customers']})",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    axes[0].set_title("Recall (Sensitivity) Across Customer Subgroups", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Recall Rate", fontsize=10)
    axes[0].set_xlim(0, 1.25)
    axes[0].grid(axis="x", linestyle="--", alpha=0.6)

    # 2. Precision across subgroups
    bars2 = axes[1].barh(
        plot_df["Subgroup"],
        plot_df["Precision"],
        color="#d95f02",
        edgecolor="black",
        height=0.55,
    )
    for bar, (_, row) in zip(bars2, plot_df.iterrows()):
        w = bar.get_width()
        axes[1].text(
            w + 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.2f}",
            ha="left",
            va="center",
            fontsize=8.5,
        )

    axes[1].set_title("Precision Across Customer Subgroups", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Precision Rate", fontsize=10)
    axes[1].set_xlim(0, 1.15)
    axes[1].grid(axis="x", linestyle="--", alpha=0.6)

    plt.suptitle("Subgroup Performance & Demographic Parity Audit (Threshold tau=0.50)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "14_subgroup_performance_audit.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
