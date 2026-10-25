"""Error analysis and model failure mode diagnostics for Phase 3."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import FIGURES_DIR, TABLES_DIR

logger = logging.getLogger(__name__)


def categorize_prediction_errors(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Categorize predictions into True Positive, False Positive, True Negative, False Negative."""
    df_err = df.copy()
    y_t = df_err[y_true_col].to_numpy()
    y_p = df_err[y_prob_col].to_numpy()
    y_pred = (y_p >= threshold).astype(int)

    conditions = [
        (y_t == 1) & (y_pred == 1),
        (y_t == 0) & (y_pred == 1),
        (y_t == 0) & (y_pred == 0),
        (y_t == 1) & (y_pred == 0),
    ]
    choices = ["True Positive", "False Positive", "True Negative", "False Negative"]

    df_err["Error_Category"] = np.select(conditions, choices, default="Unknown")
    df_err["Prediction_Confidence_Margin"] = np.abs(y_p - threshold)
    return df_err


def summarize_error_distributions(
    df_err: pd.DataFrame,
    numeric_features: List[str] = ["tenure", "MonthlyCharges", "TotalServices"],
    categorical_features: List[str] = ["Contract", "InternetService", "PaymentMethod"],
) -> Dict[str, Any]:
    """Calculate comparative aggregates for error archetypes (FP vs TP, FN vs TN)."""
    # Summary of numerics by error category
    grouped = df_err.groupby("Error_Category")[numeric_features].agg(["mean", "median"])
    num_summary = {}
    for col in numeric_features:
        num_summary[col] = {
            "mean": {str(k): round(float(v), 2) for k, v in grouped[(col, "mean")].items()},
            "median": {str(k): round(float(v), 2) for k, v in grouped[(col, "median")].items()},
        }

    # Percentage distribution of categorical features across error categories
    cat_summary = {}
    for col in categorical_features:
        if col in df_err.columns:
            ct = pd.crosstab(df_err["Error_Category"], df_err[col], normalize="index") * 100
            cat_summary[col] = {str(k): {str(subk): round(float(subv), 2) for subk, subv in v.items()} for k, v in ct.to_dict().items()}

    return {
        "numeric_summary": num_summary,
        "categorical_summary": cat_summary,
    }


def plot_error_diagnostics(
    df_err: pd.DataFrame,
    save_path: Path = None,
) -> Path:
    """Generate diagnostic chart showing error concentration across tenure and monthly charges."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    palette = {
        "True Positive": "#2b5c8f",
        "False Positive": "#e41a1c",
        "True Negative": "#4daf4a",
        "False Negative": "#ff7f00",
    }

    # 1. Tenure Boxplots across Error Categories
    sns.boxplot(
        data=df_err,
        x="Error_Category",
        y="tenure",
        order=["True Positive", "False Positive", "True Negative", "False Negative"],
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Customer Tenure Across Classification Categories", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Prediction Outcome", fontsize=10)
    axes[0].set_ylabel("Tenure (Months)", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # 2. Probability Distribution for False Alarms vs. Missed Churners
    errors_only = df_err[df_err["Error_Category"].isin(["False Positive", "False Negative"])]
    sns.kdeplot(
        data=errors_only,
        x="churn_probability",
        hue="Error_Category",
        palette={"False Positive": "#e41a1c", "False Negative": "#ff7f00"},
        fill=True,
        alpha=0.3,
        common_norm=False,
        ax=axes[1],
    )
    axes[1].axvline(0.5, color="black", linestyle="--", alpha=0.7, label="Threshold tau=0.50")
    axes[1].set_title("Prediction Probability Density for Model Errors", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Churn Probability", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Model Error Diagnostics & Failure Mode Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "15_error_diagnostics.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
