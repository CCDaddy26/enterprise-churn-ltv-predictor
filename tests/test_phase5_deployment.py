"""Deployment-focused tests for Phase 5: Packaging, configuration, readiness, and artifact stability."""

from pathlib import Path
import json
import pytest
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, RAW_DATA_PATH
from src.dashboard_data import load_and_score_dashboard_data
from src.inference import ChurnInferenceEngine
from src.run_phase5_check import run_deployment_readiness_checks


def test_deployment_readiness_checks_pass():
    """Test that the comprehensive Phase 5 operational diagnostics report returns PASS."""
    report = run_deployment_readiness_checks()
    assert report["status"] == "PASS"
    assert report["checks_failed"] == 0
    assert report["checks_passed"] >= 8


def test_model_and_calibrator_artifacts_exist():
    """Verify all runtime artifacts exist and are non-empty."""
    model_path = PROJECT_ROOT / "models" / "champion_pipeline.joblib"
    cal_path = PROJECT_ROOT / "models" / "sigmoid_calibrator.joblib"
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"

    assert model_path.exists() and model_path.stat().st_size > 0
    assert cal_path.exists() and cal_path.stat().st_size > 0
    assert meta_path.exists() and meta_path.stat().st_size > 0


def test_readiness_summary_json_valid():
    """Verify that phase5_readiness_summary.json is written and valid JSON."""
    summary_path = PROJECT_ROOT / "reports" / "phase5_readiness_summary.json"
    assert summary_path.exists()
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "PASS"
    assert "Raw Dataset File" in data["details"]
    assert "Champion Pipeline Artifact" in data["details"]


def test_public_export_anonymization():
    """Verify that committed customer sample exports do not contain real customer identifiers."""
    sample_path = PROJECT_ROOT / "reports" / "tables" / "phase4_customer_sample_anonymized.csv"
    assert sample_path.exists()
    df_sample = pd.read_csv(sample_path)
    # Check that display_customer_id starts with ANON_CUST_
    assert "display_customer_id" in df_sample.columns
    assert df_sample["display_customer_id"].str.startswith("ANON_CUST_").all()
    # Confirm raw Telco customerID column is not present
    assert "customerID" not in df_sample.columns
