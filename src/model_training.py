"""Model training, candidate definition, and cross-validation execution for Phase 2."""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_SEED
from src.preprocessing import build_feature_preprocessor

logger = logging.getLogger(__name__)


def get_model_candidates(
    random_state: int = RANDOM_SEED,
    include_engineered_features: bool = True,
) -> Dict[str, Pipeline]:
    """Instantiate candidate model pipelines with full preprocessing encapsulation.

    Models:
    1. dummy_most_frequent: Reference baseline (predicts most frequent class).
    2. dummy_stratified: Reference baseline (random class sampling with prior prevalence).
    3. logistic_regression: Linear benchmark with StandardScaler & class_weight='balanced'.
    4. random_forest: Non-linear bagging ensemble (no scaling required).
    5. xgboost: Gradient boosted tree ensemble (no scaling required).
    """
    candidates = {}

    # 1. Dummy Classifiers
    prep_dummy = build_feature_preprocessor(
        scale_numeric=False,
        include_engineered_features=include_engineered_features,
    )
    candidates["dummy_most_frequent"] = Pipeline(
        steps=[
            ("prep", prep_dummy),
            ("classifier", DummyClassifier(strategy="most_frequent")),
        ]
    )

    candidates["dummy_stratified"] = Pipeline(
        steps=[
            ("prep", prep_dummy),
            ("classifier", DummyClassifier(strategy="stratified", random_state=random_state)),
        ]
    )

    # 2. Logistic Regression (Requires Standard Scaling)
    prep_scaled = build_feature_preprocessor(
        scale_numeric=True,
        include_engineered_features=include_engineered_features,
    )
    candidates["logistic_regression"] = Pipeline(
        steps=[
            ("prep", prep_scaled),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )

    # 3. Random Forest Classifier
    prep_unscaled = build_feature_preprocessor(
        scale_numeric=False,
        include_engineered_features=include_engineered_features,
    )
    candidates["random_forest"] = Pipeline(
        steps=[
            ("prep", prep_unscaled),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 4. XGBoost Classifier
    # Compute scale_pos_weight roughly for class imbalance: (5174 / 1869 ≈ 2.76)
    candidates["xgboost"] = Pipeline(
        steps=[
            ("prep", prep_unscaled),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=2.76,
                    random_state=random_state,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return candidates


def evaluate_candidates_cv(
    candidates: Dict[str, Pipeline],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Stratified K-Fold Cross-Validation across candidate models on development data.

    Returns:
    - summary_df: Mean and standard deviation of key metrics across folds.
    - fold_scores_df: Per-fold detailed performance metrics.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    summary_rows = []
    fold_rows = []

    for name, pipeline in candidates.items():
        logger.info(f"Running {n_splits}-fold CV for '{name}'...")
        scores = cross_validate(
            pipeline,
            X_dev,
            y_dev,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=1,  # Single-process outer loop to prevent nested thread contention
        )

        # Record per-fold scores
        for fold_idx in range(n_splits):
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold_idx + 1,
                    "pr_auc": round(float(scores["test_pr_auc"][fold_idx]), 4),
                    "roc_auc": round(float(scores["test_roc_auc"][fold_idx]), 4),
                    "f1": round(float(scores["test_f1"][fold_idx]), 4),
                    "precision": round(float(scores["test_precision"][fold_idx]), 4),
                    "recall": round(float(scores["test_recall"][fold_idx]), 4),
                }
            )

        # Record summary stats
        summary_rows.append(
            {
                "model": name,
                "pr_auc_mean": round(float(np.mean(scores["test_pr_auc"])), 4),
                "pr_auc_std": round(float(np.std(scores["test_pr_auc"])), 4),
                "roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 4),
                "roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 4),
                "f1_mean": round(float(np.mean(scores["test_f1"])), 4),
                "f1_std": round(float(np.std(scores["test_f1"])), 4),
                "recall_mean": round(float(np.mean(scores["test_recall"])), 4),
                "recall_std": round(float(np.std(scores["test_recall"])), 4),
                "precision_mean": round(float(np.mean(scores["test_precision"])), 4),
                "precision_std": round(float(np.std(scores["test_precision"])), 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows).set_index("model")
    fold_scores_df = pd.DataFrame(fold_rows)

    return summary_df, fold_scores_df
