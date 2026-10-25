"""Sensitivity analysis, break-even calculations, and scenario comparators for Phase 4."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.roi_simulator import RetentionCampaignSimulator


def calculate_break_even_save_rate(
    targeted_df: pd.DataFrame,
    contact_cost: float = 150.0,
    incentive_cost: float = 500.0,
    reach_rate: float = 0.85,
    response_rate: float = 0.40,
    churn_prob_col: str = "calibrated_churn_probability",
    remaining_contrib_col: str = "Estimated_Remaining_Contribution",
    incentive_trigger: str = "responders_only",
) -> Optional[float]:
    """Calculate the minimal save rate required for Net Benefit to equal zero,
    rigorously accounting for the specific incentive cost trigger.

    Mathematical Derivation by Cost Trigger:
    Let:
        p_c = mean(P(Churn) * RemainingContribution)
        N = len(targeted_df)

    1. 'all_contacted':
        Total Cost = N * (contact_cost + incentive_cost)
        Gross Benefit = N * reach_rate * response_rate * SaveRate * p_c
        SaveRate_BE = (contact_cost + incentive_cost) / (reach_rate * response_rate * p_c)

    2. 'responders_only':
        Total Cost = N * (contact_cost + incentive_cost * reach_rate * response_rate)
        SaveRate_BE = (contact_cost + incentive_cost * reach_rate * response_rate) / (reach_rate * response_rate * p_c)

    3. 'saved_only':
        Total Cost = N * contact_cost + incentive_cost * N * reach_rate * response_rate * SaveRate * mean(P(Churn))
        Gross Benefit = N * reach_rate * response_rate * SaveRate * p_c
        Setting Net Benefit = 0:
        SaveRate_BE * (reach_rate * response_rate * (p_c - incentive_cost * mean(P(Churn)))) = contact_cost
        SaveRate_BE = contact_cost / (reach_rate * response_rate * (p_c - incentive_cost * mean(P(Churn))))

    Returns:
        Break-even save rate (0.0 < rate <= 1.0) or None if unreachable (> 1.0 or mathematically undefined).
    """
    if churn_prob_col not in targeted_df.columns and "churn_probability" in targeted_df.columns:
        churn_prob_col = "churn_probability"
    if remaining_contrib_col not in targeted_df.columns and "Estimated_Remaining_Contribution" in targeted_df.columns:
        remaining_contrib_col = "Estimated_Remaining_Contribution"

    n = len(targeted_df)
    if n == 0 or reach_rate <= 0 or response_rate <= 0:
        return None

    probs = targeted_df[churn_prob_col].to_numpy()
    contribs = targeted_df[remaining_contrib_col].to_numpy()

    p_churn_val_mean = float(np.mean(probs * contribs))
    if p_churn_val_mean <= 0:
        return None

    if incentive_trigger == "all_contacted":
        unit_cost = contact_cost + incentive_cost
        denom = reach_rate * response_rate * p_churn_val_mean
        if denom <= 0:
            return None
        be_save_rate = unit_cost / denom

    elif incentive_trigger == "responders_only":
        unit_cost = contact_cost + incentive_cost * (reach_rate * response_rate)
        denom = reach_rate * response_rate * p_churn_val_mean
        if denom <= 0:
            return None
        be_save_rate = unit_cost / denom

    elif incentive_trigger == "saved_only":
        mean_p_churn = float(np.mean(probs))
        net_contrib_per_save = p_churn_val_mean - (incentive_cost * mean_p_churn)
        if net_contrib_per_save <= 0:
            return None  # Incentive exceeds gross saved contribution
        denom = reach_rate * response_rate * net_contrib_per_save
        if denom <= 0:
            return None
        be_save_rate = contact_cost / denom

    else:
        # Default fallback to responders_only
        unit_cost = contact_cost + incentive_cost * (reach_rate * response_rate)
        denom = reach_rate * response_rate * p_churn_val_mean
        if denom <= 0:
            return None
        be_save_rate = unit_cost / denom

    # Return None if save rate exceeds physical 100% boundary or is negative
    if be_save_rate > 1.0 or be_save_rate < 0.0:
        return None

    return round(float(be_save_rate), 4)


def generate_scenario_matrix(
    targeted_df: pd.DataFrame,
    base_contact_cost: float = 150.0,
    base_incentive_cost: float = 500.0,
    max_budget: Optional[float] = 200000.0,
    incentive_trigger: str = "responders_only",
    churn_prob_col: str = "calibrated_churn_probability",
) -> pd.DataFrame:
    """Evaluate Low (Conservative), Base, and High (Optimistic) financial scenarios."""
    scenarios = [
        {
            "Scenario": "Conservative (Pessimistic)",
            "Reach_Rate": 0.70,
            "Response_Rate": 0.25,
            "Save_Rate": 0.15,
            "Description": "Low engagement, high customer skepticism, weak offer response",
        },
        {
            "Scenario": "Base Case (Realistic Plan)",
            "Reach_Rate": 0.85,
            "Response_Rate": 0.40,
            "Save_Rate": 0.25,
            "Description": "Standard customer success cadence and balanced concession",
        },
        {
            "Scenario": "Optimistic (Upside Potential)",
            "Reach_Rate": 0.95,
            "Response_Rate": 0.55,
            "Save_Rate": 0.35,
            "Description": "High white-glove executive engagement and compelling incentive",
        },
    ]

    records = []
    for sc in scenarios:
        sim = RetentionCampaignSimulator(
            contact_cost=base_contact_cost,
            incentive_cost=base_incentive_cost,
            reach_rate=sc["Reach_Rate"],
            response_rate=sc["Response_Rate"],
            save_rate=sc["Save_Rate"],
            max_budget=max_budget,
            incentive_trigger=incentive_trigger,
        )
        res = sim.simulate_cohort_campaign(
            targeted_df,
            churn_prob_col=churn_prob_col,
        )
        be_rate = calculate_break_even_save_rate(
            targeted_df,
            contact_cost=base_contact_cost,
            incentive_cost=base_incentive_cost,
            reach_rate=sc["Reach_Rate"],
            response_rate=sc["Response_Rate"],
            churn_prob_col=churn_prob_col,
            incentive_trigger=incentive_trigger,
        )

        records.append(
            {
                "Scenario": sc["Scenario"],
                "Reach_Rate": sc["Reach_Rate"],
                "Response_Rate": sc["Response_Rate"],
                "Assumed_Save_Rate": sc["Save_Rate"],
                "Break_Even_Save_Rate": be_rate if be_rate is not None else "Unreachable (>100%)",
                "Targeted_Customers": res["customers_targeted"],
                "Expected_Churn_Prevented": res["expected_churn_prevented"],
                "Campaign_Cost": res["expected_campaign_cost"],
                "Gross_Benefit": res["expected_gross_benefit"],
                "Net_Benefit": res["expected_net_benefit"],
                "Expected_ROI_Pct": res["expected_roi_pct"],
                "ROI_Display": res["roi_display"],
                "Description": sc["Description"],
            }
        )

    return pd.DataFrame(records)


def compute_sensitivity_heatmap_data(
    targeted_df: pd.DataFrame,
    contact_cost: float = 150.0,
    incentive_cost: float = 500.0,
    reach_rate: float = 0.85,
    save_rates: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    response_rates: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60],
    incentive_trigger: str = "responders_only",
    churn_prob_col: str = "calibrated_churn_probability",
) -> pd.DataFrame:
    """Compute 2D grid of Expected Net Benefit across varying response and save rates."""
    grid = np.zeros((len(response_rates), len(save_rates)))

    for i, resp in enumerate(response_rates):
        for j, save in enumerate(save_rates):
            sim = RetentionCampaignSimulator(
                contact_cost=contact_cost,
                incentive_cost=incentive_cost,
                reach_rate=reach_rate,
                response_rate=resp,
                save_rate=save,
                incentive_trigger=incentive_trigger,
            )
            res = sim.simulate_cohort_campaign(
                targeted_df,
                churn_prob_col=churn_prob_col,
            )
            grid[i, j] = res["expected_net_benefit"]

    df_grid = pd.DataFrame(
        grid,
        index=[f"{int(r*100)}% Response" for r in response_rates],
        columns=[f"{int(s*100)}% Save" for s in save_rates],
    )
    return df_grid
