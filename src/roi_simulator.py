"""Retention campaign financial simulator and multi-strategy targeting comparator.

Provides modular, tested calculations for:
- Campaign Cost (without double-counting) across triggers ('responders_only', 'saved_only', 'all_contacted')
- Expected Prevented Churn Events
- Expected Gross Benefit, Net Benefit, and ROI
- Correct Zero-Cost Baseline ROI Handling: When campaign cost is 0, ROI is undefined (None / 'N/A')
- Strategy Comparisons:
    Mode A: Like-For-Like Cohort Size (Identical target count across all active outreach strategies)
    Mode B: Budget-Maximized Targeting (Target count determined by available budget and unit economics)

IMPORTANT GOVERNANCE & TRUTHFULNESS RULES:
- The public IBM Telco dataset does not contain observed campaign outcomes or treatment effects.
- All reach, response, and save rates are user-configurable assumptions, NOT experimentally proven facts.
- No-campaign baseline has exactly 0 cost, 0 campaign benefit, and undefined ROI (None).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class RetentionCampaignSimulator:
    """Configurable simulation engine for retention marketing campaigns."""

    def __init__(
        self,
        contact_cost: float = 150.0,
        incentive_cost: float = 500.0,
        reach_rate: float = 0.85,
        response_rate: float = 0.40,
        save_rate: float = 0.25,
        max_budget: Optional[float] = 200000.0,
        incentive_trigger: str = "responders_only",  # 'all_contacted', 'responders_only', or 'saved_only'
    ):
        """Initialize campaign simulator with transparent assumptions.

        Args:
            contact_cost: Direct outreach cost per customer (labor, call, SMS).
            incentive_cost: Value of concession/discount offered.
            reach_rate: Probability customer is reached (0.0 <= r <= 1.0).
            response_rate: Probability reached customer engages (0.0 <= r <= 1.0).
            save_rate: Probability responding customer cancels their churn decision (0.0 <= r <= 1.0).
            max_budget: Maximum campaign budget constraint.
            incentive_trigger: When incentive cost is incurred ('all_contacted', 'responders_only', 'saved_only').
        """
        if contact_cost < 0.0 or incentive_cost < 0.0:
            raise ValueError("Campaign unit costs cannot be negative.")
        for rate_name, rate in [("reach_rate", reach_rate), ("response_rate", response_rate), ("save_rate", save_rate)]:
            if rate < 0.0 or rate > 1.0:
                raise ValueError(f"{rate_name} must be strictly between 0.0 and 1.0 (got {rate}).")

        valid_triggers = ["all_contacted", "responders_only", "saved_only"]
        if incentive_trigger not in valid_triggers:
            raise ValueError(f"incentive_trigger must be one of {valid_triggers} (got '{incentive_trigger}').")

        self.contact_cost = float(contact_cost)
        self.incentive_cost = float(incentive_cost)
        self.reach_rate = float(reach_rate)
        self.response_rate = float(response_rate)
        self.save_rate = float(save_rate)
        self.max_budget = float(max_budget) if max_budget is not None else None
        self.incentive_trigger = incentive_trigger

    def compute_single_customer_economics(
        self,
        churn_prob: float,
        remaining_contribution: float,
    ) -> Dict[str, float]:
        """Calculate expected costs and benefit for a single customer.

        Args:
            churn_prob: Model churn risk probability or score (0.0 <= p <= 1.0).
            remaining_contribution: Estimated remaining contribution value.

        Returns:
            Dictionary containing customer-level expected funnel metrics.
        """
        if churn_prob < 0.0 or churn_prob > 1.0:
            raise ValueError("Churn probability must be strictly between 0.0 and 1.0.")

        p_reach = self.reach_rate
        p_response = p_reach * self.response_rate
        p_save = p_response * self.save_rate
        p_prevented_churn = churn_prob * p_save

        if self.incentive_trigger == "all_contacted":
            expected_incentive = self.incentive_cost
        elif self.incentive_trigger == "responders_only":
            expected_incentive = self.incentive_cost * p_response
        elif self.incentive_trigger == "saved_only":
            expected_incentive = self.incentive_cost * p_prevented_churn
        else:
            expected_incentive = self.incentive_cost * p_response

        expected_cost = self.contact_cost + expected_incentive
        expected_gross_benefit = p_prevented_churn * remaining_contribution
        expected_net_benefit = expected_gross_benefit - expected_cost

        return {
            "p_reach": round(p_reach, 4),
            "p_response": round(p_response, 4),
            "p_prevented_churn": round(p_prevented_churn, 4),
            "expected_cost": round(expected_cost, 2),
            "expected_gross_benefit": round(expected_gross_benefit, 2),
            "expected_net_benefit": round(expected_net_benefit, 2),
        }

    def compute_expected_unit_cost(self, mean_p_prevented: float = 0.0) -> float:
        """Compute average cost incurred per customer targeted based on the incentive trigger."""
        if self.incentive_trigger == "all_contacted":
            unit_incentive = self.incentive_cost
        elif self.incentive_trigger == "responders_only":
            unit_incentive = self.incentive_cost * (self.reach_rate * self.response_rate)
        elif self.incentive_trigger == "saved_only":
            unit_incentive = self.incentive_cost * mean_p_prevented
        else:
            unit_incentive = self.incentive_cost * (self.reach_rate * self.response_rate)
        return float(self.contact_cost + unit_incentive)

    def simulate_cohort_campaign(
        self,
        targeted_df: pd.DataFrame,
        churn_prob_col: str = "calibrated_churn_probability",
        remaining_contrib_col: str = "Estimated_Remaining_Contribution",
    ) -> Dict[str, Any]:
        """Simulate campaign economics across an entire targeted customer cohort.

        Returns zero expenditure and None ROI if cohort is empty.
        """
        if churn_prob_col not in targeted_df.columns and "churn_probability" in targeted_df.columns:
            churn_prob_col = "churn_probability"
        if remaining_contrib_col not in targeted_df.columns and "Estimated_Remaining_Contribution" in targeted_df.columns:
            remaining_contrib_col = "Estimated_Remaining_Contribution"

        n_selected = len(targeted_df)
        if n_selected == 0:
            return {
                "customers_targeted": 0,
                "expected_reached": 0.0,
                "expected_responders": 0.0,
                "expected_churn_prevented": 0.0,
                "expected_campaign_cost": 0.0,
                "expected_gross_benefit": 0.0,
                "expected_net_benefit": 0.0,
                "expected_roi_pct": None,
                "roi_display": "N/A — no campaign expenditure",
                "budget_utilized": 0.0,
                "budget_utilization_pct": 0.0 if self.max_budget else None,
            }

        probs = targeted_df[churn_prob_col].to_numpy()
        contribs = targeted_df[remaining_contrib_col].to_numpy()

        p_prevented_arr = probs * self.reach_rate * self.response_rate * self.save_rate
        expected_prevented_total = float(np.sum(p_prevented_arr))

        expected_reached_total = float(n_selected * self.reach_rate)
        expected_responders_total = float(expected_reached_total * self.response_rate)

        # Incentive cost based on configured trigger
        if self.incentive_trigger == "all_contacted":
            unit_incentive = self.incentive_cost
        elif self.incentive_trigger == "responders_only":
            unit_incentive = self.incentive_cost * (self.reach_rate * self.response_rate)
        elif self.incentive_trigger == "saved_only":
            unit_incentive = self.incentive_cost * float(np.mean(p_prevented_arr))
        else:
            unit_incentive = self.incentive_cost * (self.reach_rate * self.response_rate)

        unit_cost = self.contact_cost + unit_incentive
        total_campaign_cost = float(n_selected * unit_cost)

        total_gross_benefit = float(np.sum(p_prevented_arr * contribs))
        total_net_benefit = float(total_gross_benefit - total_campaign_cost)

        # Zero cost handling: ROI is undefined when cost is zero
        if total_campaign_cost > 0.0:
            roi_pct = round(float(total_net_benefit / total_campaign_cost * 100), 2)
            roi_display = f"{roi_pct:+.1f}%"
        else:
            roi_pct = None
            roi_display = "N/A — no campaign expenditure"

        budget_util_pct = (
            min(100.0, round(total_campaign_cost / self.max_budget * 100, 2))
            if self.max_budget and self.max_budget > 0
            else None
        )

        return {
            "customers_targeted": int(n_selected),
            "expected_reached": round(expected_reached_total, 1),
            "expected_responders": round(expected_responders_total, 1),
            "expected_churn_prevented": round(expected_prevented_total, 1),
            "expected_campaign_cost": round(total_campaign_cost, 2),
            "expected_gross_benefit": round(total_gross_benefit, 2),
            "expected_net_benefit": round(total_net_benefit, 2),
            "expected_roi_pct": roi_pct,
            "roi_display": roi_display,
            "budget_utilized": round(total_campaign_cost, 2),
            "budget_utilization_pct": budget_util_pct,
        }

    def compare_targeting_strategies(
        self,
        df_customers: pd.DataFrame,
        churn_prob_col: str = "calibrated_churn_probability",
        var_col: str = "Estimated_Value_at_Risk",
        balanced_score_col: str = "Balanced_Priority_Score",
        target_size: int = 500,
        mode: str = "like_for_like",  # 'like_for_like' (Mode A) or 'budget_maximized' (Mode B)
    ) -> pd.DataFrame:
        """Compare targeting strategies on the identical eligible customer population.

        Supports two strict evaluation modes:
        - Mode A ('like_for_like'): All active strategies target the EXACT same count of customers.
        - Mode B ('budget_maximized'): Target count is dynamically determined by max_budget // unit_cost.

        No-campaign baseline always has 0 outreach, 0 cost, 0 gross benefit, 0 net benefit, and undefined ROI (None / 'N/A').
        """
        if churn_prob_col not in df_customers.columns and "churn_probability" in df_customers.columns:
            churn_prob_col = "churn_probability"
        if var_col not in df_customers.columns and "Risk_Weighted_Value_Proxy" in df_customers.columns:
            var_col = "Risk_Weighted_Value_Proxy"

        n_eligible = len(df_customers)
        unit_cost = self.compute_expected_unit_cost()

        # Baseline: No Campaign
        no_campaign = {
            "Strategy": "1. No Campaign (Baseline)",
            "Evaluation_Mode": mode,
            "Ranking_Rule": "No outreach conducted",
            "Targeted_Customers": 0,
            "Eligible_Customers": n_eligible,
            "Expected_Prevented_Churn": 0.0,
            "Campaign_Cost": 0.0,
            "Gross_Benefit": 0.0,
            "Net_Benefit": 0.0,
            "ROI_Pct": None,
            "ROI_Display": "N/A — no campaign expenditure",
            "Budget_Utilization_Pct": 0.0,
            "Budget_Remaining": round(self.max_budget, 2) if self.max_budget else None,
        }

        # Determine target count based on mode
        if mode == "like_for_like":
            cohort_size = min(int(target_size), n_eligible)
            strategies = [
                ("2. Highest Churn Risk First", churn_prob_col, cohort_size, f"Top {cohort_size} accounts by churn risk"),
                ("3. Highest Value at Risk First", var_col, cohort_size, f"Top {cohort_size} accounts by Value at Risk / Risk Proxy"),
                ("4. Balanced Risk & Value", balanced_score_col, cohort_size, f"Top {cohort_size} accounts by 50% Risk + 50% VaR"),
            ]
        elif mode == "budget_maximized":
            if self.max_budget and unit_cost > 0:
                max_cust_budget = int(self.max_budget // unit_cost)
                cohort_size = min(max_cust_budget, n_eligible)
            else:
                cohort_size = min(int(target_size), n_eligible)

            strategies = [
                ("2. Highest Churn Risk (Budget-Maximized)", churn_prob_col, cohort_size, f"Budget-capped capacity (n={cohort_size}) by risk"),
                ("3. Highest Value at Risk (Budget-Maximized)", var_col, cohort_size, f"Budget-capped capacity (n={cohort_size}) by VaR"),
                ("4. Balanced Strategy (Budget-Maximized)", balanced_score_col, cohort_size, f"Budget-capped capacity (n={cohort_size}) by Balanced Score"),
            ]
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose 'like_for_like' or 'budget_maximized'.")

        rows = [no_campaign]

        for name, sort_col, count, rule in strategies:
            if sort_col in df_customers.columns:
                sub_df = df_customers.sort_values(by=sort_col, ascending=False).head(count)
                sim = self.simulate_cohort_campaign(
                    sub_df,
                    churn_prob_col=churn_prob_col,
                    remaining_contrib_col="Estimated_Remaining_Contribution",
                )
                rem_budget = (
                    round(self.max_budget - sim["expected_campaign_cost"], 2)
                    if self.max_budget is not None
                    else None
                )
                rows.append(
                    {
                        "Strategy": name,
                        "Evaluation_Mode": mode,
                        "Ranking_Rule": rule,
                        "Targeted_Customers": sim["customers_targeted"],
                        "Eligible_Customers": n_eligible,
                        "Expected_Prevented_Churn": sim["expected_churn_prevented"],
                        "Campaign_Cost": sim["expected_campaign_cost"],
                        "Gross_Benefit": sim["expected_gross_benefit"],
                        "Net_Benefit": sim["expected_net_benefit"],
                        "ROI_Pct": sim["expected_roi_pct"],
                        "ROI_Display": sim["roi_display"],
                        "Budget_Utilization_Pct": sim["budget_utilization_pct"],
                        "Budget_Remaining": rem_budget,
                    }
                )

        return pd.DataFrame(rows)
