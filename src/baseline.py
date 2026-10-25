"""Baseline modeling module for Northstar SaaS Churn & LTV Predictor.

Strictly adheres to data-leakage prevention:
- Features X and target y are isolated prior to modeling.
- Customer IDs are excluded from features.
- Train and test partitions are stratified and locked using RANDOM_SEED.
- Imputation, scaling, and categorical encoding are fitted solely on the training partition.
- Cross-validation fits pipelines strictly within each fold.
- Evaluation metrics: Precision, Recall, F1, ROC-AUC, PR-AUC (Average Precision), and Confusion Matrix.
- Compares against a DummyClassifier (most frequent / stratified).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    RANDOM_SEED,
    TABLES_DIR,
    TARGET_COLUMN,
    TARGET_POSITIVE_VALUE,
)
from src.data_validation import separate_features_and_target

logger = logging.getLogger(__name__)


class NumericCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to safely convert string numeric values (like TotalCharges)

    to floats with whitespace handled, fitting nothing on the training set to prevent leakage.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            if X_df[col].dtype == object:
                X_df[col] = (
                    pd.to_numeric(X_df[col].astype(str).str.strip(), errors="coerce")
                    .fillna(self.fill_value)
                )
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(self.fill_value)
        return X_df.to_numpy()


def build_preprocessor(
    numeric_features: list,
    categorical_features: list,
) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for preprocessing.

    All imputers, encoders, and scalers are defined here to be fitted strictly
    on training splits.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("cleaner", NumericCleaner(fill_value=0.0)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def create_baseline_pipeline(
    preprocessor: ColumnTransformer,
    model_type: str = "logistic_regression",
    random_state: int = RANDOM_SEED,
) -> Pipeline:
    """Create a complete scikit-learn Pipeline with preprocessing and model.

    Args:
        preprocessor: Fitted or unfitted ColumnTransformer.
        model_type: 'dummy_most_frequent', 'dummy_stratified', or 'logistic_regression'.
        random_state: Seed for reproducibility.

    Returns:
        Unfitted scikit-learn Pipeline.
    """
    if model_type == "dummy_most_frequent":
        classifier = DummyClassifier(strategy="most_frequent")
    elif model_type == "dummy_stratified":
        classifier = DummyClassifier(strategy="stratified", random_state=random_state)
    elif model_type == "logistic_regression":
        classifier = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics for binary churn."""
    # Ensure binary 0/1
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    cm = confusion_matrix(y_t, y_p)
    tn, fp, fn, tp = cm.ravel()

    # ROC AUC and PR AUC
    # If prob is constant or single class, handle gracefully
    try:
        roc_auc = float(roc_auc_score(y_t, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_t, y_prob))
    except ValueError:
        pr_auc = float(y_t.mean())

    return {
        "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
        "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
        "f1": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "accuracy": round(float((tp + tn) / len(y_t)), 4),
    }


def run_baseline_benchmarking(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute end-to-end baseline benchmarking comparing Dummy and Logistic Regression models.

    Steps:
    1. Separate features X and target y (excluding customerID).
    2. Convert target y to binary 1/0.
    3. Stratified train/test split.
    4. Run 5-fold cross-validation on Training fold ONLY.
    5. Evaluate once on the held-out test split as the formal baseline reference.
    """
    X, y_raw = separate_features_and_target(df, exclude_id=True)
    y = (y_raw == TARGET_POSITIVE_VALUE).astype(int)

    # Determine present numeric and categorical columns
    num_cols = [c for c in NUMERIC_COLUMNS if c in X.columns]
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in X.columns]

    # Stratified Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info(
        f"Train size: {len(X_train)} ({y_train.mean():.2%} positive), "
        f"Test size: {len(X_test)} ({y_test.mean():.2%} positive)"
    )

    # Define candidate baselines
    models = {
        "dummy_most_frequent": create_baseline_pipeline(
            build_preprocessor(num_cols, cat_cols), "dummy_most_frequent", random_state
        ),
        "dummy_stratified": create_baseline_pipeline(
            build_preprocessor(num_cols, cat_cols), "dummy_stratified", random_state
        ),
        "logistic_regression": create_baseline_pipeline(
            build_preprocessor(num_cols, cat_cols), "logistic_regression", random_state
        ),
    }

    cv_results = {}
    test_results = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, pipeline in models.items():
        logger.info(f"Running 5-fold CV for {name} on training partition...")
        # Cross validation fitting preprocessor strictly inside folds
        scoring = ["precision", "recall", "f1", "roc_auc", "average_precision"]
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

        cv_results[name] = {
            "cv_f1_mean": round(float(np.mean(scores["test_f1"])), 4),
            "cv_f1_std": round(float(np.std(scores["test_f1"])), 4),
            "cv_roc_auc_mean": round(float(np.mean(scores["test_roc_auc"])), 4),
            "cv_roc_auc_std": round(float(np.std(scores["test_roc_auc"])), 4),
            "cv_pr_auc_mean": round(float(np.mean(scores["test_average_precision"])), 4),
            "cv_pr_auc_std": round(float(np.std(scores["test_average_precision"])), 4),
            "cv_recall_mean": round(float(np.mean(scores["test_recall"])), 4),
            "cv_precision_mean": round(float(np.mean(scores["test_precision"])), 4),
        }

        # Train on full X_train, evaluate on held-out X_test
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Get probability of churn (class 1)
        if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
            y_test_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_test_prob = np.zeros(len(y_test))

        test_results[name] = evaluate_binary_classifier(y_test, y_test_pred, y_test_prob)

    # Save summary table
    summary_df = pd.DataFrame(test_results).T
    summary_df.index.name = "Model"
    summary_df.to_csv(TABLES_DIR / "baseline_model_evaluation.csv")

    cv_df = pd.DataFrame(cv_results).T
    cv_df.index.name = "Model"
    cv_df.to_csv(TABLES_DIR / "baseline_cv_scores.csv")

    return {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_churn_rate": round(float(y_train.mean()), 4),
        "test_churn_rate": round(float(y_test.mean()), 4),
        "cv_results": cv_results,
        "test_results": test_results,
    }
