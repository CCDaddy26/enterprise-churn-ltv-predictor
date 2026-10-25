"""Robustness checks, feature ablation, and sensitivity analysis for Phase 3."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import FIGURES_DIR, RANDOM_SEED, TABLES_DIR
from src.preprocessing import build_feature_preprocessor

logger = logging.getLogger(__name__)


def run_feature_ablation_study(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    random_state: int = RANDOM_SEED,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare cross-validation performance across feature ablation variants:

    1. Full Model (All Features + Engineered: TotalServices, TenureCohort, HasPartnerAndDependents, ChargesRatio)
    2. Ablation: Without ChargesRatio
    3. Ablation: Without TenureCohort
    4. Baseline Variant: No Engineered Features (Raw Schema Only)
    """
    logger.info("Executing feature ablation experiments on Development partition...")

    variants = {
        "Full Engineered Model": build_feature_preprocessor(
            scale_numeric=False, include_engineered_features=True, include_ratios=True
        ),
        "Ablation: No ChargesRatio": build_feature_preprocessor(
            scale_numeric=False, include_engineered_features=True, include_ratios=False
        ),
        "Baseline (Raw Features Only)": build_feature_preprocessor(
            scale_numeric=False, include_engineered_features=False
        ),
    }

    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=2.5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    results = []

    for name, prep in variants.items():
        logger.info(f"Evaluating ablation variant: '{name}'...")
        pipe = Pipeline(steps=[("prep", prep), ("classifier", base_xgb)])

        scores = cross_validate(
            pipe,
            X_dev,
            y_dev,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,
        )

        results.append(
            {
                "Variant": name,
                "PR_AUC_Mean": round(float(np.mean(scores["test_pr_auc"])), 4),
                "PR_AUC_Std": round(float(np.std(scores["test_pr_auc"])), 4),
                "ROC_AUC_Mean": round(float(np.mean(scores["test_roc_auc"])), 4),
                "ROC_AUC_Std": round(float(np.std(scores["test_roc_auc"])), 4),
                "F1_Mean": round(float(np.mean(scores["test_f1"])), 4),
                "Recall_Mean": round(float(np.mean(scores["test_recall"])), 4),
                "Precision_Mean": round(float(np.mean(scores["test_precision"])), 4),
            }
        )

    ablation_df = pd.DataFrame(results).set_index("Variant")
    return ablation_df


def plot_ablation_comparison(
    ablation_df: pd.DataFrame,
    save_path: Path = None,
) -> Path:
    """Generate visual comparison chart for feature ablation variants."""
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

    variants = ablation_df.index.tolist()
    y_pos = np.arange(len(variants))

    bars = ax.barh(
        y_pos,
        ablation_df["PR_AUC_Mean"],
        xerr=ablation_df["PR_AUC_Std"],
        color="#2b5c8f",
        capsize=4,
        edgecolor="black",
        height=0.5,
    )

    for bar, val in zip(bars, ablation_df["PR_AUC_Mean"]):
        ax.text(
            val + 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.4f}",
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variants, fontsize=10)
    ax.set_title("Feature Ablation Study: Impact on Average Precision (5-Fold CV)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Average Precision (PR-AUC)", fontsize=10)
    ax.set_xlim(0, 0.85)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "16_feature_ablation_comparison.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
