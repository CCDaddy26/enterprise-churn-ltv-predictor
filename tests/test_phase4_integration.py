"""Integration tests for Phase 4: Data scoring, calibration verification, non-mutating model checks, and export schemas."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import PROJECT_ROOT, RAW_DATA_PATH
from src.dashboard_data import load_and_score_dashboard_data
from src.calibration import CALIBRATOR_PATH


def test_dashboard_data_loading_and_scoring_calibrated():
    """Test that load_and_score_dashboard_data scores customers and applies calibrated probabilities."""
    df_scored = load_and_score_dashboard_data(
        contribution_margin=0.70,
        value_horizon_months=12,
        use_calibration=True,
    )

    assert not df_scored.empty
    assert len(df_scored) == 7043
    assert "churn_risk_score" in df_scored.columns
    assert "calibrated_churn_probability" in df_scored.columns
    assert "Estimated_Remaining_Contribution" in df_scored.columns
    assert "Estimated_Value_at_Risk" in df_scored.columns
    assert "Balanced_Priority_Score" in df_scored.columns
    assert bool(df_scored["is_calibrated_mode"].iloc[0]) is True
    assert df_scored["calibrated_churn_probability"].between(0.0, 1.0).all()
    # Check that calibrated probabilities are lower on average than uncalibrated scores (due to scale_pos_weight=2.5)
    assert df_scored["calibrated_churn_probability"].mean() < df_scored["churn_risk_score"].mean()


def test_dashboard_data_loading_uncalibrated_mode():
    """Test uncalibrated mode loads with risk proxy semantics."""
    df_scored = load_and_score_dashboard_data(
        contribution_margin=0.70,
        value_horizon_months=12,
        use_calibration=False,
    )

    assert not df_scored.empty
    assert bool(df_scored["is_calibrated_mode"].iloc[0]) is False
    assert "Risk_Weighted_Value_Proxy" in df_scored.columns
    assert df_scored.loc[0, "Exposure_Metric_Name"] == "Risk_Weighted_Value_Proxy"


def test_model_artifact_not_overwritten():
    """Test that running scoring does NOT modify or retrain the champion pipeline on disk."""
    pipeline_path = PROJECT_ROOT / "models" / "champion_pipeline.joblib"
    mtime_before = pipeline_path.stat().st_mtime

    # Run data scoring
    _ = load_and_score_dashboard_data()

    mtime_after = pipeline_path.stat().st_mtime
    assert mtime_before == mtime_after, "Model artifact was modified during scoring!"
