"""Unit tests for Phase 2 modular preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocessing import SafeNumericCleaner, build_feature_preprocessor


@pytest.fixture
def synthetic_preproc_df():
    """Return synthetic dataset with mixed datatypes and edge cases."""
    return pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3", "C4"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 0],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [1, 12, 0, 48],
            "PhoneService": ["Yes", "No", "Yes", "Yes"],
            "MultipleLines": ["No", "No phone service", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "Yes"],
            "OnlineBackup": ["No", "Yes", "No internet service", "No"],
            "DeviceProtection": ["No", "No", "No internet service", "Yes"],
            "TechSupport": ["No", "No", "No internet service", "Yes"],
            "StreamingTV": ["No", "Yes", "No internet service", "Yes"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 80.50, 19.90, 65.40],
            "TotalCharges": ["29.85", "966.00", " ", "3139.20"],
        }
    )


def test_safe_numeric_cleaner():
    """Test string whitespace parsing to numeric array without errors."""
    cleaner = SafeNumericCleaner(default_fill=0.0)
    raw = pd.DataFrame({"charges": ["12.5", " ", "45.0", None]})
    cleaned = cleaner.transform(raw)

    assert isinstance(cleaned, np.ndarray)
    assert cleaned.shape == (4, 1)
    assert cleaned[0, 0] == 12.5
    assert cleaned[1, 0] == 0.0
    assert cleaned[2, 0] == 45.0
    assert cleaned[3, 0] == 0.0


def test_build_feature_preprocessor_fit_transform(synthetic_preproc_df):
    """Test that build_feature_preprocessor runs inside Pipeline and outputs numeric matrix."""
    pipeline = build_feature_preprocessor(
        scale_numeric=True,
        include_engineered_features=True,
    )

    # Exclude customerID
    X = synthetic_preproc_df.drop(columns=["customerID"])

    # Fit and transform
    X_trans = pipeline.fit_transform(X)
    assert isinstance(X_trans, np.ndarray)
    assert X_trans.shape[0] == len(X)
    assert not np.isnan(X_trans).any()


def test_preprocessor_unseen_categories(synthetic_preproc_df):
    """Test that preprocessor gracefully handles previously unseen categorical levels at test time."""
    pipeline = build_feature_preprocessor(
        scale_numeric=True,
        include_engineered_features=True,
    )

    X_train = synthetic_preproc_df.iloc[0:3].drop(columns=["customerID"])
    pipeline.fit(X_train)

    # Test set contains category not seen in first 3 rows
    X_test = synthetic_preproc_df.iloc[3:4].drop(columns=["customerID"]).copy()
    X_test.loc[3, "PaymentMethod"] = "Crypto currency (unseen)"

    # Should not throw exception; handle_unknown='ignore'
    X_test_trans = pipeline.transform(X_test)
    assert X_test_trans.shape[0] == 1
    assert not np.isnan(X_test_trans).any()
