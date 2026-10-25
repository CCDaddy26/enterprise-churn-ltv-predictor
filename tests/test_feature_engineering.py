"""Unit tests for Phase 2 feature engineering transformer."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import ChurnFeatureEngineer


@pytest.fixture
def sample_feature_df():
    """Return synthetic customer DataFrame for feature engineering tests."""
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003"],
            "tenure": [2, 24, 60],
            "MonthlyCharges": [50.0, 70.0, 100.0],
            "TotalCharges": ["100.0", "1680.0", "6000.0"],
            "Partner": ["Yes", "No", "Yes"],
            "Dependents": ["Yes", "No", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["No", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "No", "No internet service"],
        }
    )


def test_feature_engineer_transform_columns(sample_feature_df):
    """Test that all expected engineered columns are added to output."""
    fe = ChurnFeatureEngineer(include_ratios=True)
    transformed = fe.transform(sample_feature_df)

    assert "TotalServices" in transformed.columns
    assert "TenureCohort" in transformed.columns
    assert "HasPartnerAndDependents" in transformed.columns
    assert "ChargesRatio" in transformed.columns


def test_feature_engineer_total_services_count(sample_feature_df):
    """Test that TotalServices correctly sums 'Yes' responses."""
    fe = ChurnFeatureEngineer()
    transformed = fe.transform(sample_feature_df)

    # C001: OnlineSecurity='Yes', OnlineBackup='Yes' -> TotalServices = 2
    assert transformed.loc[0, "TotalServices"] == 2
    # C002: DeviceProtection='Yes', StreamingTV='Yes' -> TotalServices = 2
    assert transformed.loc[1, "TotalServices"] == 2
    # C003: All 'No internet service' -> TotalServices = 0
    assert transformed.loc[2, "TotalServices"] == 0


def test_feature_engineer_tenure_cohort_bins(sample_feature_df):
    """Test that tenure brackets match expected cohort categories."""
    fe = ChurnFeatureEngineer()
    transformed = fe.transform(sample_feature_df)

    # 2 months -> 0-6m
    assert transformed.loc[0, "TenureCohort"] == "0-6m"
    # 24 months -> 13-24m
    assert transformed.loc[1, "TenureCohort"] == "13-24m"
    # 60 months -> 49m+
    assert transformed.loc[2, "TenureCohort"] == "49m+"


def test_feature_engineer_partner_and_dependents(sample_feature_df):
    """Test logical flag for dual partner and dependents."""
    fe = ChurnFeatureEngineer()
    transformed = fe.transform(sample_feature_df)

    # C001: Partner='Yes', Dependents='Yes' -> 1
    assert transformed.loc[0, "HasPartnerAndDependents"] == 1
    # C002: Partner='No', Dependents='No' -> 0
    assert transformed.loc[1, "HasPartnerAndDependents"] == 0
    # C003: Partner='Yes', Dependents='No' -> 0
    assert transformed.loc[2, "HasPartnerAndDependents"] == 0


def test_feature_engineer_charges_ratio_safeguards():
    """Test that zero tenure or missing TotalCharges handled safely without divide-by-zero."""
    df_edge = pd.DataFrame(
        {
            "tenure": [0, 10],
            "MonthlyCharges": [30.0, 50.0],
            "TotalCharges": [" ", "500.0"],
            "Partner": ["No", "No"],
            "Dependents": ["No", "No"],
        }
    )

    fe = ChurnFeatureEngineer(include_ratios=True)
    transformed = fe.transform(df_edge)

    assert not np.isnan(transformed["ChargesRatio"]).any()
    assert not np.isinf(transformed["ChargesRatio"]).any()
    # tenure=0 should yield default 1.0
    assert transformed.loc[0, "ChargesRatio"] == 1.0
    # 500 / (10 * 50) = 1.0
    assert transformed.loc[1, "ChargesRatio"] == pytest.approx(1.0)
