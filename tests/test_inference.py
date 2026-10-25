"""Unit tests for Phase 2 model evaluation and inference engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from src.evaluation import evaluate_model_performance
from src.inference import (
    ChurnInferenceEngine,
    load_pipeline_artifact,
    save_pipeline_artifact,
)


def test_evaluate_model_performance():
    """Test performance metric calculation on known binary arrays."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.4, 0.9])

    metrics = evaluate_model_performance(y_true, y_pred, y_prob, threshold=0.5)
    assert metrics["threshold"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert metrics["false_negatives"] == 1
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["average_precision_pr_auc"] <= 1.0


def test_serialization_and_inference_engine(tmp_path: Path):
    """Test saving pipeline artifact and executing ChurnInferenceEngine."""
    p_path = tmp_path / "model.joblib"
    m_path = tmp_path / "meta.json"

    # Minimal dummy pipeline
    pipe = Pipeline([("dummy", DummyClassifier(strategy="constant", constant=1))])
    X_train = pd.DataFrame({"feat1": [1, 2, 3]})
    y_train = pd.Series([1, 1, 1])
    pipe.fit(X_train, y_train)

    meta = {"model_name": "test_dummy", "version": "1.0"}
    save_pipeline_artifact(pipe, meta, p_path, m_path)

    assert p_path.exists()
    assert m_path.exists()

    # Load into inference engine
    engine = ChurnInferenceEngine.from_artifact(p_path, m_path, decision_threshold=0.5)
    test_input = pd.DataFrame({"customerID": ["C100", "C101"], "feat1": [5, 6]})
    preds_df = engine.predict(test_input)

    assert "customerID" in preds_df.columns
    assert "churn_probability" in preds_df.columns
    assert "is_churn_risk" in preds_df.columns
    assert "risk_tier" in preds_df.columns
    assert len(preds_df) == 2
    assert list(preds_df["customerID"]) == ["C100", "C101"]
