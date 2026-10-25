"""Unit tests for Phase 4 scenario sensitivity and break-even calculations."""

import pytest
import numpy as np
import pandas as pd

from src.scenario_analysis import (
    calculate_break_even_save_rate,
    compute_sensitivity_heatmap_data,
    generate_scenario_matrix,
)


@pytest.fixture
def sample_test_cohort():
    """Return synthetic customer cohort for sensitivity tests."""
    return pd.DataFrame(
        {
            "customerID": [f"Cust_{i}" for i in range(20)],
            "calibrated_churn_probability": [0.85] * 10 + [0.30] * 10,
            "churn_probability": [0.85] * 10 + [0.30] * 10,
            "Estimated_Remaining_Contribution": [1200.0] * 20,
        }
    )


def test_calculate_break_even_save_rate_valid(sample_test_cohort):
    """Test mathematically valid break-even save rate calculation."""
    be_rate = calculate_break_even_save_rate(
        sample_test_cohort,
        contact_cost=20.0,
        incentive_cost=40.0,
        reach_rate=0.80,
        response_rate=0.50,
        incentive_trigger="responders_only",
    )
    assert be_rate is not None
    assert 0.0 < be_rate <= 1.0


def test_calculate_break_even_triggers(sample_test_cohort):
    """Test break-even save rate consistency across different cost triggers."""
    # Triggers: 'all_contacted' has highest cost, then 'responders_only', then 'saved_only'
    be_all = calculate_break_even_save_rate(
        sample_test_cohort,
        contact_cost=20.0,
        incentive_cost=40.0,
        reach_rate=0.80,
        response_rate=0.50,
        incentive_trigger="all_contacted",
    )
    be_resp = calculate_break_even_save_rate(
        sample_test_cohort,
        contact_cost=20.0,
        incentive_cost=40.0,
        reach_rate=0.80,
        response_rate=0.50,
        incentive_trigger="responders_only",
    )
    be_saved = calculate_break_even_save_rate(
        sample_test_cohort,
        contact_cost=20.0,
        incentive_cost=40.0,
        reach_rate=0.80,
        response_rate=0.50,
        incentive_trigger="saved_only",
    )

    assert be_all is not None and be_resp is not None and be_saved is not None
    # 'all_contacted' requires higher save rate than 'responders_only', which requires higher than 'saved_only'
    assert be_all > be_resp > be_saved


def test_calculate_break_even_save_rate_unreachable():
    """Test unreachable break-even scenario (>100% save rate needed) returns None."""
    # Tiny value customers ($1 contribution) with high contact cost ($100)
    low_value_cohort = pd.DataFrame(
        {
            "calibrated_churn_probability": [0.10],
            "churn_probability": [0.10],
            "Estimated_Remaining_Contribution": [1.0],
        }
    )
    be_rate = calculate_break_even_save_rate(
        low_value_cohort,
        contact_cost=100.0,
        incentive_cost=50.0,
        reach_rate=0.5,
        response_rate=0.5,
    )
    assert be_rate is None


def test_generate_scenario_matrix(sample_test_cohort):
    """Test generation of Low/Base/High scenario matrix."""
    matrix_df = generate_scenario_matrix(sample_test_cohort)
    assert len(matrix_df) == 3
    scenarios = list(matrix_df["Scenario"])
    assert any("Conservative" in s for s in scenarios)
    assert any("Base" in s for s in scenarios)
    assert any("Optimistic" in s for s in scenarios)
    assert "Expected_ROI_Pct" in matrix_df.columns
    assert "Net_Benefit" in matrix_df.columns
    assert "ROI_Display" in matrix_df.columns


def test_compute_sensitivity_heatmap_data(sample_test_cohort):
    """Test generation of 2D sensitivity grid."""
    heatmap_df = compute_sensitivity_heatmap_data(
        sample_test_cohort,
        save_rates=[0.15, 0.25],
        response_rates=[0.30, 0.50],
    )
    assert heatmap_df.shape == (2, 2)
    assert not heatmap_df.isna().any().any()
