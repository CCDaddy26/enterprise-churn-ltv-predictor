"""Unit tests for Phase 2 candidate model training and Stratified CV."""

import numpy as np
import pandas as pd
import pytest

from src.model_training import evaluate_candidates_cv, get_model_candidates


@pytest.fixture
def synthetic_training_fixture():
    """Create a balanced synthetic dataset with 50 samples."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "customerID": [f"ID_{i}" for i in range(n)],
            "gender": np.random.choice(["Male", "Female"], size=n),
            "SeniorCitizen": np.random.choice([0, 1], size=n),
            "Partner": np.random.choice(["Yes", "No"], size=n),
            "Dependents": np.random.choice(["Yes", "No"], size=n),
            "tenure": np.random.randint(0, 72, size=n),
            "PhoneService": np.random.choice(["Yes", "No"], size=n),
            "MultipleLines": np.random.choice(["Yes", "No"], size=n),
            "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
            "OnlineSecurity": np.random.choice(["Yes", "No"], size=n),
            "OnlineBackup": np.random.choice(["Yes", "No"], size=n),
            "DeviceProtection": np.random.choice(["Yes", "No"], size=n),
            "TechSupport": np.random.choice(["Yes", "No"], size=n),
            "StreamingTV": np.random.choice(["Yes", "No"], size=n),
            "StreamingMovies": np.random.choice(["Yes", "No"], size=n),
            "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
            "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
            "PaymentMethod": np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)"], size=n
            ),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
            "TotalCharges": [str(round(np.random.uniform(50.0, 3000.0), 2)) for _ in range(n)],
            "Churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        }
    )


def test_get_model_candidates_keys():
    """Verify that all required candidate models are present in dict."""
    candidates = get_model_candidates(random_state=42)
    expected_models = [
        "dummy_most_frequent",
        "dummy_stratified",
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]
    for m in expected_models:
        assert m in candidates
        assert hasattr(candidates[m], "fit")
        assert hasattr(candidates[m], "predict")


def test_evaluate_candidates_cv(synthetic_training_fixture):
    """Test running 3-fold CV across candidates on synthetic data."""
    X = synthetic_training_fixture.drop(columns=["customerID", "Churn"])
    y = synthetic_training_fixture["Churn"]

    # Select lightweight candidates to keep unit tests fast
    all_candidates = get_model_candidates(random_state=42)
    test_candidates = {
        "dummy": all_candidates["dummy_most_frequent"],
        "lr": all_candidates["logistic_regression"],
    }

    summary_df, fold_df = evaluate_candidates_cv(
        test_candidates,
        X,
        y,
        n_splits=3,
        random_state=42,
    )

    assert "dummy" in summary_df.index
    assert "lr" in summary_df.index
    assert "pr_auc_mean" in summary_df.columns
    assert len(fold_df) == 6  # 2 models * 3 folds
