"""Reusable, modular preprocessing pipelines for Northstar SaaS Churn & LTV Predictor."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)
from src.feature_engineering import ChurnFeatureEngineer


class SafeNumericCleaner(BaseEstimator, TransformerMixin):
    """Transformer that safely parses string-numeric columns (like TotalCharges)

    and converts non-numeric / whitespace strings into float values with zero leakage.
    """

    def __init__(self, default_fill: float = 0.0):
        self.default_fill = default_fill

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            if X_df[col].dtype == object:
                X_df[col] = (
                    pd.to_numeric(X_df[col].astype(str).str.strip(), errors="coerce")
                    .fillna(self.default_fill)
                )
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(self.default_fill)
        return X_df.to_numpy()


def build_feature_preprocessor(
    scale_numeric: bool = True,
    include_engineered_features: bool = True,
    include_ratios: bool = True,
) -> Pipeline:
    """Build a comprehensive scikit-learn preprocessing pipeline.

    Includes optional ChurnFeatureEngineer transformer followed by a ColumnTransformer
    with median imputation, scaling (if requested), and one-hot encoding.

    Args:
        scale_numeric: Whether to apply StandardScaler to numeric features (needed for LR).
        include_engineered_features: Whether to generate engineered features.
        include_ratios: Whether to include ratio features like ChargesRatio.

    Returns:
        Complete, unfitted scikit-learn Pipeline.
    """
    # Define columns
    num_cols = list(NUMERIC_COLUMNS)
    cat_cols = list(CATEGORICAL_COLUMNS)

    if include_engineered_features:
        num_cols.extend(["TotalServices", "HasPartnerAndDependents"])
        if include_ratios:
            num_cols.append("ChargesRatio")
        cat_cols.append("TenureCohort")

    # Numeric sub-pipeline
    num_steps = [
        ("cleaner", SafeNumericCleaner(default_fill=0.0)),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    # Categorical sub-pipeline
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
    )

    if include_engineered_features:
        pipeline = Pipeline(
            steps=[
                ("feature_engineer", ChurnFeatureEngineer(include_ratios=include_ratios)),
                ("preprocessor", column_preprocessor),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", column_preprocessor),
            ]
        )

    return pipeline
