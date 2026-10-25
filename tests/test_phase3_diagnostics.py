"""Unit tests for Phase 3 explainability, calibration, thresholds, and audits."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.calibration import compute_expected_calibration_error, evaluate_calibration_metrics
from src.error_analysis import categorize_prediction_errors, summarize_error_distributions
from src.explainability import extract_transformed_feature_names
from src.preprocessing import build_feature_preprocessor
from src.subgroup_audit import audit_subgroup_performance
from src.threshold_analysis import compute_threshold_curve


@pytest.fixture
def synthetic_diagnostic_fixture():
    """Create a synthetic dataset for testing diagnostic metrics."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "customerID": [f"ID_{i:03d}" for i in range(n)],
            "tenure": np.random.randint(1, 72, size=n),
            "MonthlyCharges": np.random.uniform(20.0, 100.0, size=n),
            "TotalCharges": [str(round(np.random.uniform(50.0, 3000.0), 2)) for _ in range(n)],
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
            "PaymentMethod": np.random.choice(["Electronic check", "Mailed check"], size=n),
            "SeniorCitizen": np.random.choice([0, 1], size=n),
            "gender": np.random.choice(["Male", "Female"], size=n),
            "Actual_Churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
            "churn_probability": np.random.uniform(0.05, 0.95, size=n),
        }
    )


def test_compute_expected_calibration_error():
    """Test Expected Calibration Error calculation on deterministic synthetic inputs."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_evaluate_calibration_metrics():
    """Test Brier score and log loss evaluation dictionary."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = evaluate_calibration_metrics(y_true, y_prob, "test_model")
    assert metrics["model"] == "test_model"
    assert "brier_score" in metrics
    assert "log_loss" in metrics
    assert "expected_calibration_error" in metrics
    assert metrics["brier_score"] < 0.2


def test_compute_threshold_curve():
    """Test threshold curve generation across spectrum."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    thresholds = np.array([0.3, 0.5, 0.7])

    curve_df = compute_threshold_curve(y_true, y_prob, thresholds=thresholds)
    assert len(curve_df) == 3
    assert list(curve_df["threshold"]) == [0.3, 0.5, 0.7]
    assert "precision" in curve_df.columns
    assert "recall" in curve_df.columns
    assert "customers_flagged" in curve_df.columns
    # At tau=0.3, 3 customers flagged (0.4, 0.6, 0.8)
    assert curve_df.loc[0, "customers_flagged"] == 3


def test_audit_subgroup_performance(synthetic_diagnostic_fixture):
    """Test subgroup metrics calculation and minimum sample size guard."""
    subgroup_cols = ["Contract", "InternetService"]
    subgroup_df = audit_subgroup_performance(
        synthetic_diagnostic_fixture,
        y_true_col="Actual_Churn",
        y_prob_col="churn_probability",
        subgroup_cols=subgroup_cols,
        decision_threshold=0.5,
        min_samples=10,
    )

    assert not subgroup_df.empty
    assert "Subgroup" in subgroup_df.columns
    assert "Churn_Prevalence" in subgroup_df.columns
    assert "Precision" in subgroup_df.columns
    assert "Recall" in subgroup_df.columns


def test_categorize_prediction_errors(synthetic_diagnostic_fixture):
    """Test error categorization into TP, FP, TN, FN."""
    df_err = categorize_prediction_errors(
        synthetic_diagnostic_fixture,
        y_true_col="Actual_Churn",
        y_prob_col="churn_probability",
        threshold=0.5,
    )

    assert "Error_Category" in df_err.columns
    assert "Prediction_Confidence_Margin" in df_err.columns
    categories = set(df_err["Error_Category"].unique())
    assert categories.issubset({"True Positive", "False Positive", "True Negative", "False Negative"})


def test_extract_transformed_feature_names():
    """Test extracting feature names from fitted preprocessing pipeline."""
    prep = build_feature_preprocessor(scale_numeric=False, include_engineered_features=True)
    df_sample = pd.DataFrame(
        {
            "tenure": [1, 2],
            "MonthlyCharges": [30.0, 40.0],
            "TotalCharges": ["30.0", "80.0"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "PhoneService": ["Yes", "Yes"],
            "MultipleLines": ["No", "Yes"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No"],
            "OnlineBackup": ["No", "Yes"],
            "DeviceProtection": ["No", "No"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
        }
    )
    prep.fit(df_sample)
    feature_names = extract_transformed_feature_names(prep)

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert "tenure" in feature_names
    assert "TotalServices" in feature_names
