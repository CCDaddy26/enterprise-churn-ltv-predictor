"""Inference interface and pipeline serialization for Northstar SaaS Churn Predictor."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import ID_COLUMN, PROJECT_ROOT, TARGET_COLUMN

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_PIPELINE_PATH = DEFAULT_MODEL_DIR / "champion_pipeline.joblib"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "model_metadata.json"


def save_pipeline_artifact(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    pipeline_path: Union[str, Path] = DEFAULT_PIPELINE_PATH,
    metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
) -> Tuple[Path, Path]:
    """Serialize the trained pipeline and save model metadata to disk."""
    p_path = Path(pipeline_path)
    m_path = Path(metadata_path)

    p_path.parent.mkdir(parents=True, exist_ok=True)
    m_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, p_path)
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved pipeline artifact to {p_path}")
    logger.info(f"Saved metadata to {m_path}")
    return p_path, m_path


def load_pipeline_artifact(
    pipeline_path: Union[str, Path] = DEFAULT_PIPELINE_PATH,
    metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load a locally saved pipeline artifact and its metadata."""
    p_path = Path(pipeline_path)
    m_path = Path(metadata_path)

    if not p_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {p_path}")

    pipeline = joblib.load(p_path)

    metadata = {}
    if m_path.exists():
        with open(m_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, metadata


class ChurnInferenceEngine:
    """Production inference wrapper for scoring customer churn risk.

    Validates schema, handles missing values safely, and outputs calibrated probabilities
    along with thresholded decisions.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        decision_threshold: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.pipeline = pipeline
        self.decision_threshold = decision_threshold
        self.metadata = metadata or {}

    @classmethod
    def from_artifact(
        cls,
        pipeline_path: Union[str, Path] = DEFAULT_PIPELINE_PATH,
        metadata_path: Union[str, Path] = DEFAULT_METADATA_PATH,
        decision_threshold: float = 0.5,
    ) -> "ChurnInferenceEngine":
        """Instantiate inference engine from serialized local artifacts."""
        pipeline, metadata = load_pipeline_artifact(pipeline_path, metadata_path)
        return cls(pipeline, decision_threshold=decision_threshold, metadata=metadata)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score customer DataFrame and return probabilities, binary risk flags, and customer IDs."""
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")

        # Isolate ID if present
        customer_ids = df[ID_COLUMN].tolist() if ID_COLUMN in df.columns else [f"Cust_{i}" for i in range(len(df))]

        # Drop ID and target columns if present in inference input
        drop_cols = [c for c in [ID_COLUMN, TARGET_COLUMN, "Churn_Binary"] if c in df.columns]
        X = df.drop(columns=drop_cols)

        # Get probabilities
        try:
            proba_arr = self.pipeline.predict_proba(X)
            if proba_arr.ndim == 2 and proba_arr.shape[1] > 1:
                probs = proba_arr[:, 1]
            else:
                probs = proba_arr.ravel()
        except AttributeError:
            # Fallback if estimator has no predict_proba
            preds = self.pipeline.predict(X)
            probs = preds.astype(float)

        preds = (probs >= self.decision_threshold).astype(int)

        results = pd.DataFrame(
            {
                "customerID": customer_ids,
                "churn_probability": np.round(probs, 4),
                "is_churn_risk": preds,
                "risk_tier": pd.cut(
                    probs,
                    bins=[-0.01, 0.25, 0.50, 0.75, 1.0],
                    labels=["Low", "Moderate", "Elevated", "High Risk"],
                ),
            }
        )

        return results
