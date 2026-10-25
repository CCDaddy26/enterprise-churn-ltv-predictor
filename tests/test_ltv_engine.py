"""Unit tests for Phase 4 Customer Lifetime Value (LTV) Engine."""

import pytest
import numpy as np
import pandas as pd

from src.ltv_engine import (
    calculate_customer_ltv_proxy,
    calculate_value_at_risk,
    compute_customer_value_table,
)


def test_calculate_customer_ltv_proxy_standard():
    """Test standard contribution value formula: MonthlyCharges * margin * horizon."""
    # Monthly: $100, Margin: 0.80, Horizon: 12 months -> $100 * 0.8 * 12 = $960.00
    val = calculate_customer_ltv_proxy(100.0, contribution_margin=0.80, value_horizon_months=12)
    assert val == 960.00


def test_calculate_customer_ltv_proxy_series():
    """Test pandas Series calculation with zero and missing values."""
    s = pd.Series([50.0, 0.0, np.nan, -20.0])
    res = calculate_customer_ltv_proxy(s, contribution_margin=0.75, value_horizon_months=10)
    # 50 * 0.75 * 10 = 375.0, others should be 0.0
    assert res.iloc[0] == 375.0
    assert res.iloc[1] == 0.0
    assert res.iloc[2] == 0.0
    assert res.iloc[3] == 0.0


def test_calculate_customer_ltv_proxy_invalid_margin():
    """Test error handling when contribution margin is out of (0.0, 1.0] bounds."""
    with pytest.raises(ValueError, match="Contribution margin must be strictly between"):
        calculate_customer_ltv_proxy(100.0, contribution_margin=0.0)
    with pytest.raises(ValueError, match="Contribution margin must be strictly between"):
        calculate_customer_ltv_proxy(100.0, contribution_margin=1.5)


def test_calculate_customer_ltv_proxy_invalid_horizon():
    """Test error handling when horizon is less than 1 month."""
    with pytest.raises(ValueError, match="Value horizon must be at least 1 month"):
        calculate_customer_ltv_proxy(100.0, contribution_margin=0.8, value_horizon_months=0)


def test_calculate_value_at_risk_standard():
    """Test Value at Risk formula: Churn Probability * Remaining Contribution."""
    # Probability: 0.60, Contribution: $1000 -> $600.00
    var = calculate_value_at_risk(0.60, 1000.0)
    assert var == 600.00


def test_calculate_value_at_risk_invalid_prob():
    """Test error handling when churn probability is outside [0.0, 1.0]."""
    with pytest.raises(ValueError, match="Churn probability must be strictly between"):
        calculate_value_at_risk(1.2, 500.0)
    with pytest.raises(ValueError, match="Churn probability must be strictly between"):
        calculate_value_at_risk(-0.1, 500.0)


def test_compute_customer_value_table_calibrated():
    """Test calibrated probability mode in compute_customer_value_table."""
    df = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "MonthlyCharges": [100.0, 50.0],
            "calibrated_churn_probability": [0.80, 0.20],
        }
    )

    enriched = compute_customer_value_table(
        df,
        prob_col="calibrated_churn_probability",
        contribution_margin=0.75,
        value_horizon_months=12,
        is_calibrated=True,
    )

    assert "Estimated_Remaining_Contribution" in enriched.columns
    assert "Estimated_Value_at_Risk" in enriched.columns
    assert "Balanced_Priority_Score" in enriched.columns
    assert enriched.loc[0, "Exposure_Metric_Name"] == "Estimated_Value_at_Risk"
    assert enriched.loc[0, "Estimated_Value_at_Risk"] == 720.0


def test_compute_customer_value_table_uncalibrated_risk_proxy():
    """Test uncalibrated mode creates Risk_Weighted_Value_Proxy with clear semantics."""
    df = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "MonthlyCharges": [100.0, 50.0],
            "churn_risk_score": [0.85, 0.25],
        }
    )

    enriched = compute_customer_value_table(
        df,
        prob_col="churn_risk_score",
        contribution_margin=0.75,
        value_horizon_months=12,
        is_calibrated=False,
    )

    assert "Risk_Weighted_Value_Proxy" in enriched.columns
    assert enriched.loc[0, "Exposure_Metric_Name"] == "Risk_Weighted_Value_Proxy"
    # 100 * 0.75 * 12 = 900.0; Risk Proxy = 0.85 * 900 = 765.0
    assert enriched.loc[0, "Risk_Weighted_Value_Proxy"] == 765.0
