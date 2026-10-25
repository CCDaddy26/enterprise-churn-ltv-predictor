"""Probability calibration assessment, cross-fit calibrator training, and reliability diagnostics."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import FIGURES_DIR, PROJECT_ROOT, RANDOM_SEED, TABLES_DIR

logger = logging.getLogger(__name__)

CALIBRATOR_PATH = PROJECT_ROOT / "models" / "sigmoid_calibrator.joblib"


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) via uniform probability binning."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_prob = np.mean(y_prob[in_bin])
            avg_true = np.mean(y_true[in_bin])
            ece += (bin_size / n) * np.abs(avg_prob - avg_true)

    return round(float(ece), 4)


def evaluate_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Calculate Brier score, Log Loss, and ECE for predicted probabilities."""
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "model": model_name,
        "sample_size": len(y_true),
        "brier_score": round(brier, 4),
        "log_loss": round(ll, 4),
        "expected_calibration_error": round(ece, 4),
    }


def train_and_evaluate_development_calibrator(
    pipeline,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    cv_splits: int = 5,
    random_seed: int = RANDOM_SEED,
    save_artifact_path: Optional[Path] = CALIBRATOR_PATH,
) -> Tuple[LogisticRegression, pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit and strictly evaluate a sigmoid calibrator using Development data ONLY.

    Leakage-Conscious Evaluation Protocol:
    1. Uses 5-Fold Stratified Cross-Validation on Development accounts (N=5,634) to produce
       out-of-fold (OOF) uncalibrated model scores: each prediction is made without having seen
       that customer's target label during model fitting.
    2. Uses a secondary 5-fold cross-fitting procedure on the OOF scores to evaluate the
       calibrator: each customer's calibrated score is obtained from a calibrator fitted without
       using that customer's label.
    3. Fits the final deployment calibrator on the full Development OOF score vector and saves
       the calibrator artifact for inference/dashboard consumption.
    4. The held-out test set is NEVER used to fit, tune, or select the calibrator.

    Returns:
        Tuple of (fitted_calibrator, comparison_metrics_df, oof_uncal_scores, oof_cal_probs).
    """
    logger.info(f"Training leakage-conscious calibrator on {len(X_dev)} development accounts...")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)

    # 1. Generate OOF uncalibrated predictions from champion pipeline
    oof_uncal = cross_val_predict(
        pipeline,
        X_dev,
        y_dev,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # 2. Strict Cross-Fit Calibration Evaluation (No customer label seen by calibrator)
    oof_cal = np.zeros_like(oof_uncal)
    y_dev_arr = y_dev.to_numpy() if isinstance(y_dev, pd.Series) else y_dev

    for train_idx, val_idx in cv.split(X_dev, y_dev):
        fold_calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=random_seed)
        fold_calibrator.fit(oof_uncal[train_idx].reshape(-1, 1), y_dev_arr[train_idx])
        oof_cal[val_idx] = fold_calibrator.predict_proba(oof_uncal[val_idx].reshape(-1, 1))[:, 1]

    # Evaluate metrics
    metrics_uncal = evaluate_calibration_metrics(
        y_dev_arr, oof_uncal, model_name="uncalibrated_xgboost_dev_oof"
    )
    metrics_cal = evaluate_calibration_metrics(
        y_dev_arr, oof_cal, model_name="sigmoid_calibrated_dev_crossfitted"
    )

    metrics_df = pd.DataFrame([metrics_uncal, metrics_cal]).set_index("model")
    logger.info(f"Cross-fit calibration evaluation results:\n{metrics_df}")

    # 3. Fit final production calibrator mapping raw score -> calibrated probability
    final_calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=random_seed)
    final_calibrator.fit(oof_uncal.reshape(-1, 1), y_dev_arr)

    if save_artifact_path:
        save_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_calibrator, save_artifact_path)
        logger.info(f"Saved calibrator artifact to {save_artifact_path}")

    return final_calibrator, metrics_df, oof_uncal, oof_cal


def plot_reliability_diagram(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    n_bins: int = 10,
    save_path: Path = None,
) -> Path:
    """Plot reliability diagram comparing observed frequency vs. mean predicted probability."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Calibration Curves
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")

    colors = {
        "uncalibrated": "#2b5c8f",
        "uncalibrated_dev_oof": "#2b5c8f",
        "sigmoid_calibrated": "#d95f02",
        "sigmoid_calibrated_dev_crossfitted": "#d95f02",
        "prevalence_baseline": "grey",
    }

    for name, probs in prob_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
        axes[0].plot(
            prob_pred,
            prob_true,
            marker="o",
            label=f"{name}",
            color=colors.get(name, "#7570b3"),
            lw=2,
        )

    axes[0].set_title("Reliability Diagram (Observed vs Predicted Churn)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Mean Predicted Probability / Score", fontsize=10)
    axes[0].set_ylabel("Empirical Churn Prevalence", fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Probability Distribution Histogram
    for name, probs in prob_dict.items():
        if name != "prevalence_baseline":
            axes[1].hist(
                probs,
                bins=20,
                alpha=0.4,
                label=f"{name}",
                color=colors.get(name, "#333333"),
                edgecolor="black",
            )

    axes[1].set_title("Predicted Score / Probability Distribution", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Value", fontsize=10)
    axes[1].set_ylabel("Customer Count", fontsize=10)
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    plt.suptitle("Probability Calibration Analysis (Development Split, N=5,634)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "12_calibration_reliability_diagram.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
