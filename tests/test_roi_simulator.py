"""Unit tests for Phase 4 Retention Campaign ROI Simulator and Strategy Comparator."""

import pytest
import numpy as np
import pandas as pd

from src.roi_simulator import RetentionCampaignSimulator


@pytest.fixture
def sample_campaign_cohort():
    """Return synthetic customer DataFrame for campaign testing."""
    return pd.DataFrame(
        {
            "customerID": [f"Cust_{i}" for i in range(10)],
            "calibrated_churn_probability": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            "churn_probability": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            "Estimated_Remaining_Contribution": [1000.0] * 10,
            "Estimated_Value_at_Risk": [900, 800, 700, 600, 500, 400, 300, 200, 100, 50],
            "Balanced_Priority_Score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        }
    )


def test_single_customer_economics():
    """Test single customer prevented churn and cost without double-counting."""
    sim = RetentionCampaignSimulator(
        contact_cost=20.0,
        incentive_cost=50.0,
        reach_rate=0.80,
        response_rate=0.50,
        save_rate=0.25,
        incentive_trigger="responders_only",
    )
    # p_reach = 0.80, p_response = 0.40, p_save = 0.10, p_prevented = 0.80 * 0.10 = 0.08
    # cost = 20 + 50 * (0.8 * 0.5) = 20 + 20 = 40.0
    # gross_benefit = 0.08 * 1000 = 80.0
    # net_benefit = 80 - 40 = 40.0
    res = sim.compute_single_customer_economics(churn_prob=0.80, remaining_contribution=1000.0)
    assert res["p_prevented_churn"] == pytest.approx(0.08)
    assert res["expected_cost"] == pytest.approx(40.0)
    assert res["expected_gross_benefit"] == pytest.approx(80.0)
    assert res["expected_net_benefit"] == pytest.approx(40.0)


def test_simulate_cohort_campaign_empty():
    """Test campaign simulation on empty DataFrame handles zero safely and ROI is None."""
    sim = RetentionCampaignSimulator()
    empty_df = pd.DataFrame(columns=["calibrated_churn_probability", "Estimated_Remaining_Contribution"])
    res = sim.simulate_cohort_campaign(empty_df)
    assert res["customers_targeted"] == 0
    assert res["expected_campaign_cost"] == 0.0
    assert res["expected_net_benefit"] == 0.0
    assert res["expected_roi_pct"] is None
    assert "no campaign expenditure" in res["roi_display"]


def test_no_campaign_baseline_zero_cost_roi_none(sample_campaign_cohort):
    """Test that the No-Campaign baseline reports None/N/A ROI rather than 0%."""
    sim = RetentionCampaignSimulator()
    strategies_df = sim.compare_targeting_strategies(sample_campaign_cohort, target_size=5, mode="like_for_like")
    base_row = strategies_df[strategies_df["Strategy"].str.contains("No Campaign")].iloc[0]
    assert base_row["Campaign_Cost"] == 0.0
    assert base_row["Gross_Benefit"] == 0.0
    assert base_row["Net_Benefit"] == 0.0
    assert base_row["Targeted_Customers"] == 0
    assert pd.isna(base_row["ROI_Pct"]) or base_row["ROI_Pct"] is None
    assert "N/A — no campaign expenditure" in base_row["ROI_Display"]


def test_like_for_like_cohort_counts(sample_campaign_cohort):
    """Mode A: Test that all active targeting strategies have the exact same target count."""
    sim = RetentionCampaignSimulator(contact_cost=10.0, incentive_cost=20.0)
    target_count = 4
    df_mode_a = sim.compare_targeting_strategies(sample_campaign_cohort, target_size=target_count, mode="like_for_like")

    active_rows = df_mode_a[~df_mode_a["Strategy"].str.contains("No Campaign")]
    assert len(active_rows) == 3
    for _, row in active_rows.iterrows():
        assert row["Targeted_Customers"] == target_count


def test_budget_maximized_targeting(sample_campaign_cohort):
    """Mode B: Test that budget cap and unit cost dynamically determine cohort size."""
    # unit cost = 20 + 40 * (0.8 * 0.5) = 36.0
    # max_budget = 150.0 -> max_cust = 150 // 36 = 4 customers
    sim = RetentionCampaignSimulator(
        contact_cost=20.0,
        incentive_cost=40.0,
        reach_rate=0.80,
        response_rate=0.50,
        save_rate=0.25,
        max_budget=150.0,
        incentive_trigger="responders_only",
    )
    df_mode_b = sim.compare_targeting_strategies(sample_campaign_cohort, mode="budget_maximized")
    active_rows = df_mode_b[~df_mode_b["Strategy"].str.contains("No Campaign")]
    for _, row in active_rows.iterrows():
        assert row["Targeted_Customers"] == 4
        assert row["Campaign_Cost"] <= 150.0
