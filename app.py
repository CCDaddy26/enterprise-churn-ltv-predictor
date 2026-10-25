"""Streamlit Executive Dashboard: Phase 4 Customer Lifetime Value (LTV) Engine
& Retention Campaign ROI Simulator for Northstar SaaS.

EDUCATIONAL DISCLAIMER:
Northstar SaaS is a fictional subscription software scenario built upon the public IBM
Telco Customer Churn educational dataset. All LTV calculations, remaining contribution values,
campaign costs, and retention ROI projections are illustrative scenario estimates based on
explicit user-configurable assumptions, NOT verified company financial forecasts or proven
experimental causal outcomes.
"""

from pathlib import Path
from datetime import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np

from src.dashboard_data import load_and_score_dashboard_data
from src.ltv_engine import compute_customer_value_table
from src.roi_simulator import RetentionCampaignSimulator
from src.scenario_analysis import (
    calculate_break_even_save_rate,
    generate_scenario_matrix,
    compute_sensitivity_heatmap_data,
)

# -----------------------------------------------------------------------------
# 1. PAGE SETUP & THEME CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Northstar SaaS | Retention ROI Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Executive CSS Design
st.markdown(
    """
    <style>
    /* Clean, modern typography and cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 20px;
        color: #f8fafc;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-subtext {
        font-size: 0.75rem;
        color: #cbd5e1;
        margin-top: 4px;
    }
    .disclaimer-banner {
        background-color: #1e1b4b;
        border-left: 4px solid #6366f1;
        border-radius: 4px;
        padding: 12px 16px;
        color: #e0e7ff;
        font-size: 0.85rem;
        margin-bottom: 20px;
    }
    .status-badge-cal {
        background-color: #065f46;
        color: #a7f3d0;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    .status-badge-uncal {
        background-color: #7f1d1d;
        color: #fecaca;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 2. SIDEBAR ASSUMPTIONS & CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.image(
    "https://img.shields.io/badge/Northstar%20SaaS-Executive%20Suite-0284c7?style=for-the-badge",
    use_container_width=True,
)
st.sidebar.title("Simulation Controls")

# Currency selection
currency = st.sidebar.selectbox(
    "Display Currency",
    options=["INR (₹)", "USD ($)", "EUR (€)", "GBP (£)"],
    index=0,
    help="Default educational portfolio currency is INR. Data source remains public educational IBM dataset.",
)
curr_sym = currency.split("(")[-1].replace(")", "").strip()

# Probability Calibration Toggle & Semantics
st.sidebar.markdown("---")
st.sidebar.subheader("Probability Calibration Mode")
use_calibration = st.sidebar.toggle(
    "Use Defensible Calibrated Probabilities",
    value=True,
    help="When enabled, uses the cross-fitted Sigmoid Calibrator (ECE=0.0110). When disabled, uses raw model scores as relative risk proxies.",
)

if use_calibration:
    st.sidebar.markdown('<span class="status-badge-cal">✓ Calibrated Mode (ECE=0.0110)</span>', unsafe_allow_html=True)
    prob_col_name = "calibrated_churn_probability"
    exposure_label = "Estimated Value at Risk"
    prob_label = "Calibrated Churn Probability"
else:
    st.sidebar.markdown('<span class="status-badge-uncal">⚠️ Uncalibrated Risk Score Mode</span>', unsafe_allow_html=True)
    prob_col_name = "churn_risk_score"
    exposure_label = "Risk-Weighted Value Proxy"
    prob_label = "Model Churn Risk Score"

st.sidebar.markdown("---")
st.sidebar.subheader("1. LTV & Value Proxy")
margin = st.sidebar.slider(
    "Assumed Contribution Margin",
    min_value=0.10,
    max_value=0.95,
    value=0.70,
    step=0.05,
    format="%d%%",
    help="Fraction of monthly charges representing variable gross contribution after service delivery costs.",
)
horizon = st.sidebar.slider(
    "Remaining Value Horizon (Months)",
    min_value=1,
    max_value=36,
    value=12,
    step=1,
    help="Number of future months evaluated for churn-associated value loss.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Campaign Unit Economics")
contact_cost = st.sidebar.number_input(
    f"Contact Cost per Customer ({curr_sym})",
    min_value=0.0,
    max_value=1000.0,
    value=150.0,
    step=10.0,
    help="Direct outreach cost (call center, personalized email, SMS, account manager time).",
)
incentive_cost = st.sidebar.number_input(
    f"Incentive / Discount Concession ({curr_sym})",
    min_value=0.0,
    max_value=5000.0,
    value=500.0,
    step=50.0,
    help="Financial concession or gift value offered to retain the customer.",
)
incentive_trigger = st.sidebar.selectbox(
    "Incentive Incurred On",
    options=["responders_only", "saved_only", "all_contacted"],
    index=0,
    format_func=lambda x: {
        "responders_only": "Responding Customers Only",
        "saved_only": "Retained / Saved Customers Only",
        "all_contacted": "All Targeted Customers (Upfront Gift)",
    }[x],
    help="Governs exactly when incentive expenditure is realized to prevent double-counting.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Campaign Effectiveness Assumptions")
reach_rate = st.sidebar.slider(
    "Campaign Reach Rate",
    min_value=0.10,
    max_value=1.00,
    value=0.85,
    step=0.05,
    format="%d%%",
    help="Probability that targeted customer receives outreach.",
)
response_rate = st.sidebar.slider(
    "Response Rate (Given Reach)",
    min_value=0.05,
    max_value=1.00,
    value=0.40,
    step=0.05,
    format="%d%%",
    help="Probability reached customer opens communication and evaluates the offer.",
)
save_rate = st.sidebar.slider(
    "Save Rate (Given Response)",
    min_value=0.01,
    max_value=1.00,
    value=0.25,
    step=0.01,
    format="%d%%",
    help="Assumed fraction of responding at-risk customers who rescind cancellation.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("4. Budget & Strategy Controls")
max_budget = st.sidebar.number_input(
    f"Total Campaign Budget Limit ({curr_sym})",
    min_value=10000.0,
    max_value=2000000.0,
    value=200000.0,
    step=10000.0,
    help="Maximum available funding for the retention outreach cohort.",
)
target_size = st.sidebar.slider(
    "Standard Cohort Size (Mode A: Like-for-Like)",
    min_value=50,
    max_value=2000,
    value=500,
    step=50,
    help="Number of prioritized accounts targeted when evaluating equal-size strategies.",
)

# -----------------------------------------------------------------------------
# 3. DATA LOADING & ENRICHMENT (CACHED)
# -----------------------------------------------------------------------------
@st.cache_data
def get_dashboard_data(margin_val: float, horizon_val: int, cal_toggle: bool):
    return load_and_score_dashboard_data(
        contribution_margin=margin_val,
        value_horizon_months=horizon_val,
        use_calibration=cal_toggle,
    )

df_all = get_dashboard_data(margin, horizon, use_calibration)
n_total = len(df_all)
historical_churn_rate = (df_all["Churn"] == "Yes").mean() if "Churn" in df_all.columns else 0.2654

# Global Persistent Disclaimer
st.markdown(
    f"""
    <div class="disclaimer-banner">
        <strong>⚠️ MANDATORY EXECUTIVE DISCLAIMER:</strong> All Lifetime Values, Remaining Contribution Values,
        and Campaign ROI projections are <strong>Scenario Estimates</strong> based on user-supplied assumptions
        (Margin: {int(margin*100)}%, Horizon: {horizon}m, Reach: {int(reach_rate*100)}%, Response: {int(response_rate*100)}%, Save: {int(save_rate*100)}%).
        Evaluation Metric: <strong>{exposure_label}</strong> using <strong>{prob_label}</strong>.
        These figures are illustrative models derived from the public IBM Telco dataset and <strong>NOT</strong> validated financial forecasts or experimentally verified causal outcomes.
    </div>
    """,
    unsafe_allow_html=True,
)

# Multi-Page Navigation
tabs = st.tabs([
    "📊 1. Executive Overview",
    "👥 2. Customer Risk & Value Explorer",
    "🎯 3. Campaign ROI Simulator",
    "⚖️ 4. Scenario & Break-Even Analysis",
    "📋 5. Model & Data Limitations",
])

# -----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW
# -----------------------------------------------------------------------------
with tabs[0]:
    st.header(f"Executive Summary: {exposure_label} & Campaign Economics")
    st.caption("Strategic high-level view connecting machine learning risk scoring with balance-sheet exposure.")

    col1, col2, col3, col4 = st.columns(4)

    total_exp = df_all["Estimated_Value_at_Risk"].sum()
    mean_exp = df_all["Estimated_Value_at_Risk"].mean()

    top_cohort = df_all.sort_values(by="Estimated_Value_at_Risk", ascending=False).head(target_size)
    quick_sim = RetentionCampaignSimulator(
        contact_cost=contact_cost,
        incentive_cost=incentive_cost,
        reach_rate=reach_rate,
        response_rate=response_rate,
        save_rate=save_rate,
        max_budget=max_budget,
        incentive_trigger=incentive_trigger,
    )
    quick_res = quick_sim.simulate_cohort_campaign(
        top_cohort,
        churn_prob_col=prob_col_name,
    )

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Portfolio Accounts</div>
                <div class="metric-value">{n_total:,}</div>
                <div class="metric-subtext">Historical Churn: <strong>{historical_churn_rate*100:.1f}%</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Portfolio {exposure_label}</div>
                <div class="metric-value">{curr_sym} {total_exp:,.0f}</div>
                <div class="metric-subtext">Avg Exposure: <strong>{curr_sym} {mean_exp:,.1f}/acct</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        roi_color = '#ef4444' if (quick_res['expected_roi_pct'] or 0) < 0 else '#10b981'
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Targeted Outreach ROI ({target_size} Accts)</div>
                <div class="metric-value" style="color: {roi_color};">{quick_res['roi_display']}</div>
                <div class="metric-subtext">Net Benefit: <strong>{curr_sym} {quick_res['expected_net_benefit']:+,.0f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Campaign Budget Exposure</div>
                <div class="metric-value">{curr_sym} {quick_res['expected_campaign_cost']:,.0f}</div>
                <div class="metric-subtext">Utilizing <strong>{quick_res['budget_utilization_pct']}%</strong> of {curr_sym} {max_budget:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 🔍 Risk Probability vs. Financial Exposure")
    st.info(
        f"""
        **Executive Distinction:**
        - **{prob_label}:** Statistical likelihood of account cancellation.
        - **{exposure_label}:** Currency exposure ($P(\\text{{Churn}}) \\times \\text{{Remaining Contribution}}$).
        Focusing solely on defection risk without factoring in contract value misallocates capital toward low-margin accounts while leaving high-value enterprise subscriptions unprotected.
        """
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader(f"Contract Type vs. {exposure_label}")
        contract_summary = df_all.groupby("Contract")["Estimated_Value_at_Risk"].agg(["sum", "mean", "count"]).reset_index()
        contract_summary.columns = ["Contract", "Total_Exposure", "Mean_Exposure", "Accounts"]
        st.dataframe(
            contract_summary.style.format({
                "Total_Exposure": f"{curr_sym} {{:,.0f}}",
                "Mean_Exposure": f"{curr_sym} {{:,.1f}}",
                "Accounts": "{:,}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(contract_summary.set_index("Contract")["Total_Exposure"])

    with chart_col2:
        st.subheader(f"Internet Service Tier vs. {exposure_label}")
        service_summary = df_all.groupby("InternetService")["Estimated_Value_at_Risk"].agg(["sum", "mean", "count"]).reset_index()
        service_summary.columns = ["Internet_Service", "Total_Exposure", "Mean_Exposure", "Accounts"]
        st.dataframe(
            service_summary.style.format({
                "Total_Exposure": f"{curr_sym} {{:,.0f}}",
                "Mean_Exposure": f"{curr_sym} {{:,.1f}}",
                "Accounts": "{:,}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(service_summary.set_index("Internet_Service")["Total_Exposure"])

# -----------------------------------------------------------------------------
# PAGE 2: CUSTOMER RISK & VALUE EXPLORER
# -----------------------------------------------------------------------------
with tabs[1]:
    st.header("Customer Cohort Risk & Value Explorer")
    st.caption("Filter, segment, and inspect individual accounts across model scores and financial proxies.")

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        sel_contract = st.multiselect("Contract Type", options=df_all["Contract"].unique(), default=df_all["Contract"].unique())
    with f_col2:
        sel_service = st.multiselect("Internet Service", options=df_all["InternetService"].unique(), default=df_all["InternetService"].unique())
    with f_col3:
        min_risk, max_risk = st.slider(f"{prob_label} Range", 0.0, 1.0, (0.0, 1.0), 0.05)
    with f_col4:
        sort_by = st.selectbox(
            "Sort Cohort By",
            options=["Estimated_Value_at_Risk", prob_col_name, "Balanced_Priority_Score", "MonthlyCharges"],
            format_func=lambda x: {
                "Estimated_Value_at_Risk": f"{exposure_label} (Descending)",
                prob_col_name: f"{prob_label} (Descending)",
                "Balanced_Priority_Score": "Balanced Priority Score (Descending)",
                "MonthlyCharges": "Monthly Charges (Descending)",
            }[x],
        )

    filtered_df = df_all[
        (df_all["Contract"].isin(sel_contract)) &
        (df_all["InternetService"].isin(sel_service)) &
        (df_all[prob_col_name] >= min_risk) &
        (df_all[prob_col_name] <= max_risk)
    ].sort_values(by=sort_by, ascending=False)

    st.write(f"Showing **{len(filtered_df):,}** matching customer accounts (out of {n_total:,}).")

    display_df = filtered_df.copy()
    display_df["display_id"] = [f"ANON_{idx:05d}" for idx in range(1, len(display_df) + 1)]
    table_cols = [
        "display_id",
        prob_col_name,
        "MonthlyCharges",
        "Estimated_Remaining_Contribution",
        "Estimated_Value_at_Risk",
        "Balanced_Priority_Score",
        "tenure",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]

    st.dataframe(
        display_df[table_cols].head(100).style.format({
            prob_col_name: "{:.2%}",
            "MonthlyCharges": f"{curr_sym} {{:,.2f}}",
            "Estimated_Remaining_Contribution": f"{curr_sym} {{:,.2f}}",
            "Estimated_Value_at_Risk": f"{curr_sym} {{:,.2f}}",
            "Balanced_Priority_Score": "{:.3f}",
            "tenure": "{} mos",
        }),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = display_df[table_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"📥 Download Current Filtered Cohort ({len(display_df):,} Records)",
        data=csv_data,
        file_name=f"northstar_customer_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# PAGE 3: RETENTION CAMPAIGN SIMULATOR
# -----------------------------------------------------------------------------
with tabs[2]:
    st.header("Retention Campaign ROI Simulator")
    st.caption("Model multi-stage outreach funnel economics across distinct targeting strategies.")

    sim = RetentionCampaignSimulator(
        contact_cost=contact_cost,
        incentive_cost=incentive_cost,
        reach_rate=reach_rate,
        response_rate=response_rate,
        save_rate=save_rate,
        max_budget=max_budget,
        incentive_trigger=incentive_trigger,
    )

    eval_mode = st.radio(
        "Select Targeting Evaluation Mode",
        options=["Mode A: Like-for-Like Cohort Size", "Mode B: Budget-Maximized Targeting"],
        horizontal=True,
        help="Mode A holds targeted account counts identical across active strategies. Mode B caps each strategy dynamically by available budget.",
    )
    selected_mode = "like_for_like" if "Mode A" in eval_mode else "budget_maximized"

    comparison_df = sim.compare_targeting_strategies(
        df_all,
        churn_prob_col=prob_col_name,
        var_col="Estimated_Value_at_Risk",
        balanced_score_col="Balanced_Priority_Score",
        target_size=target_size,
        mode=selected_mode,
    )

    st.subheader(f"Strategy Comparison Table ({eval_mode})")
    st.dataframe(
        comparison_df.style.format({
            "Targeted_Customers": "{:,}",
            "Eligible_Customers": "{:,}",
            "Expected_Prevented_Churn": "{:,.1f}",
            "Campaign_Cost": f"{curr_sym} {{:,.0f}}",
            "Gross_Benefit": f"{curr_sym} {{:,.0f}}",
            "Net_Benefit": f"{curr_sym} {{:+,.0f}}",
            "ROI_Display": "{}",
            "Budget_Utilization_Pct": "{:.1f}%",
            "Budget_Remaining": f"{curr_sym} {{:,.0f}}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Strategy Net Benefit Comparison")
    chart_df = comparison_df.set_index("Strategy")[["Net_Benefit"]]
    st.bar_chart(chart_df)

    st.markdown("---")
    st.markdown("### 🧮 Mathematical Funnel Formulations")
    st.latex(r"E[\text{Prevented Churn}] = \sum_{i \in \text{Cohort}} P(\text{Churn}_i) \times P(\text{Reach}) \times P(\text{Response}) \times P(\text{Save})")
    st.latex(r"\text{Gross Benefit} = \sum_{i \in \text{Cohort}} E[\text{Prevented Churn}_i] \times \text{Remaining Contribution}_i")
    st.latex(r"\text{Total Cost} = N \times (\text{Contact Cost} + \text{Incentive Cost} \times P(\text{Trigger}))")
    st.latex(r"\text{Net Benefit} = \text{Gross Benefit} - \text{Total Cost}")

    strat_csv = comparison_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Campaign Strategy Results (CSV)",
        data=strat_csv,
        file_name=f"northstar_strategy_comparison_{selected_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# -----------------------------------------------------------------------------
# PAGE 4: SCENARIO & BREAK-EVEN ANALYSIS
# -----------------------------------------------------------------------------
with tabs[3]:
    st.header("Scenario & Break-Even Sensitivity Analysis")
    st.caption("Stress-test assumptions against uncertain future retention dynamics.")

    target_cohort = df_all.sort_values(by="Estimated_Value_at_Risk", ascending=False).head(target_size)

    st.subheader("1. Three-Point Scenario Matrix (Low, Base, High)")
    matrix_df = generate_scenario_matrix(
        target_cohort,
        base_contact_cost=contact_cost,
        base_incentive_cost=incentive_cost,
        max_budget=max_budget,
        incentive_trigger=incentive_trigger,
        churn_prob_col=prob_col_name,
    )
    st.dataframe(
        matrix_df.style.format({
            "Reach_Rate": "{:.0%}",
            "Response_Rate": "{:.0%}",
            "Assumed_Save_Rate": "{:.0%}",
            "Targeted_Customers": "{:,}",
            "Expected_Churn_Prevented": "{:,.1f}",
            "Campaign_Cost": f"{curr_sym} {{:,.0f}}",
            "Gross_Benefit": f"{curr_sym} {{:,.0f}}",
            "Net_Benefit": f"{curr_sym} {{:+,.0f}}",
            "ROI_Display": "{}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")
    st.subheader(f"2. Break-Even Save Rate Analysis (Trigger: {incentive_trigger})")
    be_rate = calculate_break_even_save_rate(
        target_cohort,
        contact_cost=contact_cost,
        incentive_cost=incentive_cost,
        reach_rate=reach_rate,
        response_rate=response_rate,
        churn_prob_col=prob_col_name,
        incentive_trigger=incentive_trigger,
    )

    be_col1, be_col2 = st.columns(2)
    with be_col1:
        if be_rate is not None:
            st.metric(
                label="Required Break-Even Save Rate",
                value=f"{be_rate*100:.1f}%",
                delta=f"{(save_rate - be_rate)*100:+.1f}% vs Assumed ({save_rate*100:.1f}%)",
                help="Minimum save rate among responding customers required for Net Benefit >= 0.",
            )
            if save_rate >= be_rate:
                st.success(f"✅ The assumed save rate ({save_rate*100:.1f}%) exceeds the break-even threshold ({be_rate*100:.1f}%). The scenario generates positive expected net value.")
            else:
                st.error(f"❌ The assumed save rate ({save_rate*100:.1f}%) is below break-even ({be_rate*100:.1f}%). This scenario operates at a financial deficit.")
        else:
            st.metric(label="Required Break-Even Save Rate", value="Unreachable (>100%)")
            st.error("⚠️ Under current contact costs and margins, break-even is mathematically impossible even with a 100% save rate! Reduce contact/incentive costs or extend the value horizon.")

    with be_col2:
        unit_c = sim.compute_expected_unit_cost()
        st.markdown(
            f"""
            **Break-Even Context:**
            - Configured Incentive Trigger: **`{incentive_trigger}`**
            - Effective Unit Cost per targeted customer: **{curr_sym} {unit_c:,.2f}**.
            - Evaluated on top **{len(target_cohort):,}** accounts sorted by {exposure_label}.
            """
        )

    st.markdown("---")
    st.subheader("3. Sensitivity Heatmap (Response Rate vs. Save Rate)")
    s_rates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    r_rates = [0.20, 0.30, 0.40, 0.50, 0.60]
    heat_df = compute_sensitivity_heatmap_data(
        target_cohort,
        contact_cost=contact_cost,
        incentive_cost=incentive_cost,
        reach_rate=reach_rate,
        save_rates=s_rates,
        response_rates=r_rates,
        incentive_trigger=incentive_trigger,
        churn_prob_col=prob_col_name,
    )

    st.dataframe(
        heat_df.style.format(f"{curr_sym} {{:+,.0f}}").background_gradient(cmap="RdYlGn"),
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# PAGE 5: MODEL & DATA LIMITATIONS
# -----------------------------------------------------------------------------
with tabs[4]:
    st.header("Model Audits, Calibration & Data Limitations")
    st.caption("Transparent disclosure of data boundaries, model diagnostics, and prerequisites for deployment.")

    st.markdown(
        """
        ### 1. Dataset Provenance & Retrospective Nature
        - **Source:** Publicly available IBM Telco Customer Churn dataset (7,043 customer accounts).
        - **Retrospective Cross-Section:** The dataset represents a historical snapshot without longitudinal event logs or timestamps.
        - **No Experimental Treatment Data:** There are **zero** observed retention interventions, randomized control trials (RCTs), discount receipts, or actual treatment uplift figures in this data.
        """
    )

    st.markdown(
        """
        ### 2. Probability Calibration Audit (Phase 4.1 Correction)
        - **Champion Architecture:** XGBoost Classifier with 5-fold stratified cross-validation ($F_1 = 0.6300 \\pm 0.0210$, $\\text{PR-AUC} = 0.6589 \\pm 0.0175$).
        - **Class Imbalance Reweighting:** The champion uses `scale_pos_weight = 2.5`, pushing uncalibrated model scores higher than empirical probabilities (Brier Score $\\approx 0.1542$, ECE $\\approx 0.1152$).
        - **Strict Cross-Fit Calibrator:** Built on Development data ONLY ($N=5,634$) using secondary 5-fold cross-fitting where no evaluated customer's label is used to fit their calibrator mapping. This improves Expected Calibration Error (ECE) from **0.1152 to 0.0110** (a 90% error reduction).
        - **Generalization:** Evaluated on historical held-out test data, achieving Brier Score `0.1363` and ECE `0.0275`.
        """
    )

    st.markdown(
        """
        ### 3. Association vs. Causation Warning
        - **Feature Attributions (SHAP):** High monthly charges and month-to-month contracts are strongly *associated* with churn. They are **not** proven causal levers.
        - **Intervention Risks:** Lowering a customer's charges or offering an incentive does not guarantee they will behave like an organic low-charge customer. In real campaigns, incentives may cause churn contagion or attract deal-seekers.
        """
    )

    st.markdown(
        """
        ### 4. What Real Enterprise Data Is Required Before Production Deployment?
        Before Northstar SaaS (or any enterprise) deploys real capital based on this simulator, the following verified business data must be integrated:
        1. **Verified Ledger Unit Margins:** Actual cost-to-serve per account (infrastructure, bandwidth, support tickets).
        2. **Randomized A/B Retention Trials:** Holdout-based uplift experiments measuring true incremental save rates $\\tau_i = E[Y_1 - Y_0 | X]$.
        3. **Actual Concession Redemption Costs:** Tracked discount uptake, credit card processor fees, and retention agent commission.
        4. **Cohort Survival Modeling:** Transitioning from static proxies to continuous Cox Proportional Hazards or Weibull survival curves.
        """
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Project 01/30 Phase 4.1 | Northstar SaaS Suite")
st.sidebar.caption("Built for Educational & Portfolio Evaluation")
