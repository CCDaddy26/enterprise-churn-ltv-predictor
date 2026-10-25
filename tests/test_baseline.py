"""Tests for baseline pipeline construction, fitting, and leakage prevention."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.baseline import (
    build_preprocessor,
    create_baseline_pipeline,
    evaluate_binary_classifier,
    run_baseline_benchmarking,
)


@pytest.fixture
def synthetic_training_data():
    """Create a synthetic dataset with numeric and categorical features."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "customerID": [f"ID_{i:03d}" for i in range(n)],
            "tenure": np.random.randint(0, 72, size=n),
            "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
            "TotalCharges": [
                " " if i % 20 == 0 else str(round(np.random.uniform(50.0, 3000.0), 2))
                for i in range(n)
            ],
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], size=n
            ),
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"], size=n
            ),
            "Churn": np.random.choice(["No", "Yes"], size=n, p=[0.75, 0.25]),
        }
    )


def test_build_preprocessor_structure():
    """Test building preprocessor with column transformers."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["Contract", "InternetService"]
    prep = build_preprocessor(num_cols, cat_cols)

    assert hasattr(prep, "transformers")
    transformer_names = [t[0] for t in prep.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_baseline_pipeline_fit_predict(synthetic_training_data):
    """Test that baseline pipeline fits and transforms training data without error."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = ["Contract", "InternetService"]
    prep = build_preprocessor(num_cols, cat_cols)
    pipeline = create_baseline_pipeline(prep, model_type="logistic_regression", random_state=42)

    X = synthetic_training_data.drop(columns=["customerID", "Churn"])
    y = (synthetic_training_data["Churn"] == "Yes").astype(int)

    # Fit pipeline
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert set(np.unique(preds)).issubset({0, 1})


def test_evaluate_binary_classifier():
    """Test evaluation metrics helper with known synthetic predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.7, 0.4, 0.9])

    metrics = evaluate_binary_classifier(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.5
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_negatives"] == 1
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_run_baseline_benchmarking_reproducibility(synthetic_training_data):
    """Test that running baseline benchmark twice with fixed seed yields exact identical results."""
    res1 = run_baseline_benchmarking(synthetic_training_data, test_size=0.25, random_state=42)
    res2 = run_baseline_benchmarking(synthetic_training_data, test_size=0.25, random_state=42)

    assert res1["test_results"]["logistic_regression"]["roc_auc"] == pytest.approx(
        res2["test_results"]["logistic_regression"]["roc_auc"]
    )
    assert res1["test_results"]["logistic_regression"]["f1"] == pytest.approx(
        res2["test_results"]["logistic_regression"]["f1"]
    )
    assert res1["train_samples"] == res2["train_samples"]
    assert res1["test_samples"] == res2["test_samples"]
