"""Phase 4.1 Execution Runner: Generates scenario matrices, strategy comparisons (Modes A & B),
and exports reproducible tables for executive reporting with calibrated probabilities and zero-cost ROI handling.
"""

from pathlib import Path
import logging
import pandas as pd

from src.dashboard_data import load_and_score_dashboard_data
from src.roi_simulator import RetentionCampaignSimulator
from src.scenario_analysis import (
    generate_scenario_matrix,
    compute_sensitivity_heatmap_data,
    calculate_break_even_save_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_phase4_pipeline():
    logger.info("Starting Phase 4.1 execution and table generation...")
    data_path = Path("data/raw/Telco-Customer-Churn.csv")
    model_path = Path("models/champion_pipeline.joblib")
    reports_tables_dir = Path("reports/tables")
    reports_tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load, score data, apply calibrated probability mapping, and enrich with financial fields
    df_value = load_and_score_dashboard_data(
        contribution_margin=0.70,
        value_horizon_months=12,
        pipeline_path=model_path,
        use_calibration=True,
    )
    logger.info(f"Loaded and scored {len(df_value)} customer records (Calibrated Mode: {df_value['is_calibrated_mode'].iloc[0]}).")

    # Anonymize customer ID for committed public exports
    df_export_customers = df_value.copy()
    df_export_customers["display_customer_id"] = [
        f"ANON_CUST_{i:05d}" for i in range(1, len(df_export_customers) + 1)
    ]
    export_cols = [
        "display_customer_id",
        "churn_risk_score",
        "calibrated_churn_probability",
        "MonthlyCharges",
        "Estimated_Remaining_Contribution",
        "Estimated_Value_at_Risk",
        "Balanced_Priority_Score",
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]
    sample_export = df_export_customers[export_cols].head(50)
    sample_export.to_csv(reports_tables_dir / "phase4_customer_sample_anonymized.csv", index=False)
    logger.info("Saved anonymized customer sample to reports/tables/phase4_customer_sample_anonymized.csv")

    # 2. Simulate Campaign Strategies in Mode A (Like-for-like: 500 accounts)
    simulator = RetentionCampaignSimulator(
        contact_cost=150.0,
        incentive_cost=500.0,
        reach_rate=0.85,
        response_rate=0.40,
        save_rate=0.25,
        max_budget=200000.0,
        incentive_trigger="responders_only",
    )
    comparison_mode_a = simulator.compare_targeting_strategies(
        df_value,
        churn_prob_col="calibrated_churn_probability",
        var_col="Estimated_Value_at_Risk",
        balanced_score_col="Balanced_Priority_Score",
        target_size=500,
        mode="like_for_like",
    )
    comparison_mode_a.to_csv(reports_tables_dir / "phase4_strategy_comparison.csv", index=False)
    logger.info("Saved Mode A strategy comparison to reports/tables/phase4_strategy_comparison.csv")

    # 3. Simulate Campaign Strategies in Mode B (Budget-Maximized: budget // unit_cost)
    comparison_mode_b = simulator.compare_targeting_strategies(
        df_value,
        churn_prob_col="calibrated_churn_probability",
        var_col="Estimated_Value_at_Risk",
        balanced_score_col="Balanced_Priority_Score",
        target_size=500,
        mode="budget_maximized",
    )
    comparison_mode_b.to_csv(reports_tables_dir / "phase4_strategy_comparison_budget_maximized.csv", index=False)
    logger.info("Saved Mode B strategy comparison to reports/tables/phase4_strategy_comparison_budget_maximized.csv")

    # 4. Generate Scenario Matrix
    targeted_top_var = df_value.sort_values(by="Estimated_Value_at_Risk", ascending=False).head(500)
    scenario_df = generate_scenario_matrix(
        targeted_top_var,
        base_contact_cost=150.0,
        base_incentive_cost=500.0,
        max_budget=200000.0,
        incentive_trigger="responders_only",
        churn_prob_col="calibrated_churn_probability",
    )
    scenario_df.to_csv(reports_tables_dir / "phase4_scenario_matrix.csv", index=False)
    logger.info("Saved scenario matrix to reports/tables/phase4_scenario_matrix.csv")

    # 5. Generate Sensitivity Grid
    save_rates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    response_rates = [0.20, 0.30, 0.40, 0.50, 0.60]
    heatmap_df = compute_sensitivity_heatmap_data(
        targeted_top_var,
        contact_cost=150.0,
        incentive_cost=500.0,
        reach_rate=0.85,
        save_rates=save_rates,
        response_rates=response_rates,
        incentive_trigger="responders_only",
        churn_prob_col="calibrated_churn_probability",
    )
    heatmap_df.to_csv(reports_tables_dir / "phase4_sensitivity_heatmap.csv")
    logger.info("Saved sensitivity heatmap grid to reports/tables/phase4_sensitivity_heatmap.csv")

    # 6. Break-Even Analysis Summary across triggers
    break_even_results = []
    triggers = ["responders_only", "saved_only", "all_contacted"]
    for trg in triggers:
        sim_trg = RetentionCampaignSimulator(
            contact_cost=150.0,
            incentive_cost=500.0,
            reach_rate=0.85,
            response_rate=0.40,
            save_rate=0.25,
            max_budget=200000.0,
            incentive_trigger=trg,
        )
        sim_res = sim_trg.simulate_cohort_campaign(
            targeted_top_var,
            churn_prob_col="calibrated_churn_probability",
        )
        be_rate = calculate_break_even_save_rate(
            targeted_top_var,
            contact_cost=150.0,
            incentive_cost=500.0,
            reach_rate=0.85,
            response_rate=0.40,
            churn_prob_col="calibrated_churn_probability",
            incentive_trigger=trg,
        )
        break_even_results.append(
            {
                "Trigger": trg,
                "Targeted_Customers": sim_res["customers_targeted"],
                "Expected_Campaign_Cost": sim_res["expected_campaign_cost"],
                "Expected_Gross_Benefit": sim_res["expected_gross_benefit"],
                "Expected_Net_Benefit": sim_res["expected_net_benefit"],
                "Break_Even_Save_Rate": be_rate if be_rate is not None else "Unreachable (>100%)",
                "Assumed_Save_Rate": 0.25,
                "Margin_of_Safety_Pts": round((0.25 - be_rate) * 100, 2) if be_rate is not None else "N/A",
            }
        )

    be_df = pd.DataFrame(break_even_results)
    be_df.to_csv(reports_tables_dir / "phase4_break_even_summary.csv", index=False)
    logger.info("Saved break-even summary to reports/tables/phase4_break_even_summary.csv")

    logger.info("Phase 4.1 table generation complete.")


if __name__ == "__main__":
    run_phase4_pipeline()
