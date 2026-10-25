"""Tests for data validation and schema auditing."""

import pandas as pd
import pytest

from src.data_validation import (
    audit_dataset,
    clean_dataset_for_exploration,
    separate_features_and_target,
    validate_raw_schema,
)


@pytest.fixture
def synthetic_valid_df():
    """Return a small, clean synthetic dataset adhering to schema."""
    return pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3", "C4"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 0, 1, 0],
            "tenure": [1, 24, 0, 60],
            "MonthlyCharges": [29.85, 56.95, 20.00, 99.50],
            "TotalCharges": ["29.85", "1366.80", " ", "5970.00"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Month-to-month",
                "Two year",
            ],
            "Churn": ["No", "No", "No", "Yes"],
        }
    )


def test_validate_raw_schema_empty():
    """Test schema validation fails on empty dataframe."""
    with pytest.raises(ValueError, match="Cannot validate an empty"):
        validate_raw_schema(pd.DataFrame())


def test_validate_raw_schema_missing_target(synthetic_valid_df):
    """Test schema validation fails when target column is missing."""
    df_no_target = synthetic_valid_df.drop(columns=["Churn"])
    with pytest.raises(ValueError, match="Mandatory target column 'Churn' is missing"):
        validate_raw_schema(df_no_target)


def test_validate_raw_schema_unexpected_target_values(synthetic_valid_df):
    """Test schema validation fails with invalid target categories."""
    df_bad_target = synthetic_valid_df.copy()
    df_bad_target.loc[0, "Churn"] = "Maybe"
    with pytest.raises(ValueError, match="contains unexpected classes"):
        validate_raw_schema(df_bad_target)


def test_validate_raw_schema_success(synthetic_valid_df):
    """Test successful schema validation and whitespace detection."""
    result = validate_raw_schema(synthetic_valid_df)
    assert result["is_valid"] is True
    assert result["n_rows"] == 4
    assert result["id_unique"] is True
    assert result["blank_total_charges_count"] == 1


def test_clean_dataset_for_exploration_blank_handling(synthetic_valid_df):
    """Test that blank TotalCharges strings are converted to 0.0 without mutating original."""
    cleaned = clean_dataset_for_exploration(synthetic_valid_df)
    # Original should be untouched
    assert synthetic_valid_df.loc[2, "TotalCharges"] == " "
    # Cleaned should have numeric 0.0
    assert cleaned.loc[2, "TotalCharges"] == 0.0
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    # Binary indicator column added
    assert "Churn_Binary" in cleaned.columns
    assert list(cleaned["Churn_Binary"]) == [0, 0, 0, 1]


def test_separate_features_and_target_excludes_id(synthetic_valid_df):
    """Test that customerID is strictly excluded from predictor matrix X."""
    X, y = separate_features_and_target(synthetic_valid_df, exclude_id=True)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert "Churn_Binary" not in X.columns
    assert len(X) == 4
    assert len(y) == 4
    assert list(y) == ["No", "No", "No", "Yes"]


def test_audit_dataset_metrics(synthetic_valid_df):
    """Test audit_dataset produces correct summary dictionaries."""
    audit = audit_dataset(synthetic_valid_df)
    assert audit["total_rows"] == 4
    assert audit["total_columns"] == 8
    assert audit["id_duplicates"] == 0
    assert audit["target_counts"]["No"] == 3
    assert audit["target_counts"]["Yes"] == 1
    assert "tenure" in audit["columns"]
    assert audit["columns"]["TotalCharges"]["whitespace_count"] == 1
