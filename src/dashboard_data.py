"""Data loading, caching, and scenario preparation routines for Streamlit dashboard."""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from src.config import PROJECT_ROOT, RAW_DATA_PATH
from src.data_loader import load_raw_dataset
from src.inference import ChurnInferenceEngine
from src.ltv_engine import compute_customer_value_table
from src.calibration import CALIBRATOR_PATH


def load_and_score_dashboard_data(
    contribution_margin: float = 0.70,
    value_horizon_months: int = 12,
    pipeline_path: Path = PROJECT_ROOT / "models" / "champion_pipeline.joblib",
    metadata_path: Path = PROJECT_ROOT / "models" / "model_metadata.json",
    calibrator_path: Optional[Path] = CALIBRATOR_PATH,
    use_calibration: bool = True,
) -> pd.DataFrame:
    """Load raw customer dataset, score with champion pipeline, apply cross-fitted calibrator,
    and compute financial scenario columns with explicit probability semantics.

    IMPORTANT ARCHITECTURAL RULE:
    Does NOT refit, retrain, or mutate the champion model artifact.
    """
    df_raw = load_raw_dataset(RAW_DATA_PATH)

    # Score customers via inference engine
    engine = ChurnInferenceEngine.from_artifact(pipeline_path, metadata_path)
    scored_df = engine.predict(df_raw)

    merged = df_raw.copy()
    raw_scores = scored_df["churn_probability"].to_numpy()
    merged["churn_risk_score"] = raw_scores
    merged["is_churn_risk"] = scored_df["is_churn_risk"]
    merged["risk_tier"] = scored_df["risk_tier"]

    # Calibrate probabilities if calibrator artifact is available
    is_calibrated = False
    calibrator_file = Path(calibrator_path) if calibrator_path else CALIBRATOR_PATH
    if use_calibration and calibrator_file.exists():
        try:
            calibrator = joblib.load(calibrator_file)
            cal_probs = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
            merged["calibrated_churn_probability"] = np.round(cal_probs, 4)
            merged["churn_probability"] = merged["calibrated_churn_probability"]
            is_calibrated = True
        except Exception:
            merged["calibrated_churn_probability"] = raw_scores
            merged["churn_probability"] = raw_scores
    else:
        merged["calibrated_churn_probability"] = raw_scores
        merged["churn_probability"] = raw_scores

    # Compute financial proxy and value at risk / risk-weighted proxy
    eval_col = "calibrated_churn_probability" if is_calibrated else "churn_risk_score"
    enriched_df = compute_customer_value_table(
        merged,
        prob_col=eval_col,
        monthly_charges_col="MonthlyCharges",
        contribution_margin=contribution_margin,
        value_horizon_months=value_horizon_months,
        is_calibrated=is_calibrated,
    )
    enriched_df["is_calibrated_mode"] = is_calibrated

    return enriched_df
