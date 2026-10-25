"""Hyperparameter tuning and model optimization module for Phase 2."""

import logging
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_SEED
from src.preprocessing import build_feature_preprocessor

logger = logging.getLogger(__name__)


def tune_xgboost_hyperparameters(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_splits: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[Pipeline, Dict[str, Any], pd.DataFrame]:
    """Perform bounded hyperparameter tuning on XGBoost using Development partition ONLY.

    Optimizes for Average Precision (PR-AUC) using StratifiedKFold.
    Restricts search space to computationally bounded, high-impact parameters.
    """
    logger.info("Initializing bounded hyperparameter tuning for XGBoost...")

    prep = build_feature_preprocessor(
        scale_numeric=False,
        include_engineered_features=True,
    )

    base_xgb = XGBClassifier(
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("prep", prep),
            ("classifier", base_xgb),
        ]
    )

    # Bounded grid search space (12 combinations total)
    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [3, 4],
        "classifier__learning_rate": [0.03, 0.08],
        "classifier__scale_pos_weight": [2.5, 2.76],
        "classifier__subsample": [0.8],
        "classifier__colsample_bytree": [0.8],
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=1,  # Single-process outer loop to prevent nested thread issues
        refit=True,
        verbose=1,
    )

    grid_search.fit(X_dev, y_dev)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    logger.info(f"Optimal XGBoost Average Precision (CV): {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")

    cv_results_df = pd.DataFrame(grid_search.cv_results_)

    metadata = {
        "best_cv_average_precision": round(best_score, 4),
        "best_parameters": best_params,
        "n_candidates_evaluated": len(cv_results_df),
        "search_scoring_metric": "average_precision",
        "n_splits": n_splits,
    }

    return best_pipeline, metadata, cv_results_df
