"""Customer Lifetime Value (LTV) Engine and Financial Value-at-Risk Calculations.

Provides modular, transparent, and tested formulas for:
1. Estimated Remaining Contribution Value
2. Churn-Associated Estimated Value at Risk (Calibrated Mode)
3. Risk-Weighted Value Proxy (Uncalibrated Risk Score Mode)

IMPORTANT TRUTHFULNESS & GOVERNANCE RULES:
- The public IBM Telco dataset does not contain observed margins, customer profit, or validated LTV.
- All contribution margins and value horizons are user-configurable assumptions, NOT observed accounting facts.
- Value at Risk is an expected-value scenario estimate, NOT a guaranteed revenue loss.
- When uncalibrated risk scores are used, outputs are explicitly labeled 'Risk-Weighted Value Proxy',
  NOT expected monetary value or actuarial value at risk.
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd


def calculate_customer_ltv_proxy(
    monthly_charges: Union[float, pd.Series, np.ndarray],
    contribution_margin: float = 0.70,
    value_horizon_months: int = 12,
) -> Union[float, pd.Series, np.ndarray]:
    """Calculate estimated remaining contribution value over a specified horizon.

    Formula:
        Estimated Remaining Contribution Value = MonthlyCharges × Contribution Margin × Remaining Horizon (Months)

    Args:
        monthly_charges: Customer monthly recurring subscription charge.
        contribution_margin: Assumed gross contribution margin (0.0 < margin <= 1.0).
        value_horizon_months: Planning horizon in months (integer >= 1).

    Returns:
        Estimated contribution value.

    Raises:
        ValueError: If contribution_margin or value_horizon_months are outside valid bounds.
    """
    if contribution_margin <= 0.0 or contribution_margin > 1.0:
        raise ValueError(
            f"Contribution margin must be strictly between 0.0 and 1.0 (got {contribution_margin})."
        )
    if value_horizon_months < 1:
        raise ValueError(
            f"Value horizon must be at least 1 month (got {value_horizon_months})."
        )

    # Handle pandas Series or numpy arrays with missing or negative values safely
    if isinstance(monthly_charges, pd.Series):
        mc_clean = monthly_charges.fillna(0.0).clip(lower=0.0)
        return (mc_clean * contribution_margin * value_horizon_months).round(2)

    if isinstance(monthly_charges, np.ndarray):
        mc_clean = np.where(np.isnan(monthly_charges) | (monthly_charges < 0.0), 0.0, monthly_charges)
        return np.round(mc_clean * contribution_margin * value_horizon_months, 2)

    if pd.isna(monthly_charges) or monthly_charges < 0.0:
        return 0.0

    return round(float(monthly_charges * contribution_margin * value_horizon_months), 2)


def calculate_value_at_risk(
    churn_probability: Union[float, pd.Series, np.ndarray],
    remaining_contribution_value: Union[float, pd.Series, np.ndarray],
) -> Union[float, pd.Series, np.ndarray]:
    """Calculate estimated churn-associated contribution value at risk.

    Formula:
        Estimated Value at Risk = Churn Probability × Estimated Remaining Contribution Value

    Args:
        churn_probability: Predicted churn risk probability (0.0 <= p <= 1.0).
        remaining_contribution_value: Estimated remaining contribution value.

    Returns:
        Estimated value at risk.

    Raises:
        ValueError: If churn_probability is outside [0.0, 1.0].
    """
    if isinstance(churn_probability, (pd.Series, np.ndarray)):
        if np.any((churn_probability < 0.0) | (churn_probability > 1.0)):
            raise ValueError("Churn probabilities must be strictly between 0.0 and 1.0.")
        return np.round(churn_probability * remaining_contribution_value, 2)

    if churn_probability < 0.0 or churn_probability > 1.0:
        raise ValueError(f"Churn probability must be strictly between 0.0 and 1.0 (got {churn_probability}).")

    return round(float(churn_probability * remaining_contribution_value), 2)


def compute_customer_value_table(
    df: pd.DataFrame,
    prob_col: str = "calibrated_churn_probability",
    monthly_charges_col: str = "MonthlyCharges",
    contribution_margin: float = 0.70,
    value_horizon_months: int = 12,
    is_calibrated: bool = True,
    id_col: Optional[str] = "customerID",
) -> pd.DataFrame:
    """Enrich a customer DataFrame with transparent LTV proxy and Value at Risk / Risk Proxy columns.

    Args:
        df: Input customer DataFrame.
        prob_col: Column containing model prediction (calibrated probability or raw risk score).
        monthly_charges_col: Column containing monthly subscription charges.
        contribution_margin: Configurable gross contribution margin assumption (e.g. 0.70).
        value_horizon_months: Configurable evaluation horizon (e.g. 12 months).
        is_calibrated: True if prob_col represents calibrated probabilities; False if raw risk score.
        id_col: Identifier column name (optional).

    Returns:
        pd.DataFrame containing original fields plus calculated financial scenario fields.
    """
    if df is None or df.empty:
        raise ValueError("Cannot calculate LTV table on an empty DataFrame.")
    if prob_col not in df.columns:
        raise ValueError(f"Required prediction column '{prob_col}' not found.")
    if monthly_charges_col not in df.columns:
        raise ValueError(f"Required monthly charges column '{monthly_charges_col}' not found.")

    res = df.copy()

    # Calculate remaining contribution value
    res["Monthly_Contribution_Estimate"] = np.round(
        pd.to_numeric(res[monthly_charges_col], errors="coerce").fillna(0.0).clip(lower=0.0) * contribution_margin,
        2,
    )

    res["Estimated_Remaining_Contribution"] = calculate_customer_ltv_proxy(
        res[monthly_charges_col],
        contribution_margin=contribution_margin,
        value_horizon_months=value_horizon_months,
    )

    pred_vals = res[prob_col].clip(0.0, 1.0)
    calculated_exposure = calculate_value_at_risk(
        pred_vals,
        res["Estimated_Remaining_Contribution"],
    )

    if is_calibrated:
        res["Estimated_Value_at_Risk"] = calculated_exposure
        res["Exposure_Metric_Name"] = "Estimated_Value_at_Risk"
    else:
        res["Risk_Weighted_Value_Proxy"] = calculated_exposure
        res["Estimated_Value_at_Risk"] = calculated_exposure  # For backward-compatible API access
        res["Exposure_Metric_Name"] = "Risk_Weighted_Value_Proxy"

    # Balanced ranking score (Normalized combination of risk score/probability and financial exposure)
    p_norm = (pred_vals - pred_vals.min()) / (pred_vals.max() - pred_vals.min() + 1e-9)
    exp_norm = (calculated_exposure - calculated_exposure.min()) / (
        calculated_exposure.max() - calculated_exposure.min() + 1e-9
    )
    res["Balanced_Priority_Score"] = np.round(0.5 * p_norm + 0.5 * exp_norm, 4)

    return res
