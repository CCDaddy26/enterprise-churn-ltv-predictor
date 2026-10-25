"""Feature Engineering Transformer for Northstar SaaS Churn & LTV Predictor.

Strictly adheres to data-leakage prevention:
- All features are computed strictly from available row-level predictors at prediction time.
- Zero reliance on the target column (y / Churn).
- Fully reproducible in scikit-learn Pipeline environments.
- Accommodates missing, zero, and unexpected values safely.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineers domain-relevant subscription and service features for churn prediction.

    Features generated:
    1. TotalServices: Count of active security, backup, protection, support, and streaming add-ons.
    2. TenureCohort: Binned tenure brackets ('0-6m', '7-12m', '13-24m', '25-48m', '49m+').
    3. HasPartnerAndDependents: Binary indicator of dual household support.
    4. ChargesRatio: Historical average monthly billing relative to current monthly charge:
       TotalCharges / (tenure * MonthlyCharges + 1e-6).
    """

    SERVICE_COLUMNS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    TENURE_BINS = [-1, 6, 12, 24, 48, 120]
    TENURE_LABELS = ["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]

    def __init__(self, include_ratios: bool = True):
        self.include_ratios = include_ratios

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless transformer: no parameters are fitted from training data to leak
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations to input DataFrame."""
        X_df = X.copy()

        # 1. TotalServices: count number of add-ons where value is 'Yes'
        available_services = [col for col in self.SERVICE_COLUMNS if col in X_df.columns]
        if available_services:
            X_df["TotalServices"] = (X_df[available_services] == "Yes").sum(axis=1)
        else:
            X_df["TotalServices"] = 0

        # 2. TenureCohort: non-linear tenure cohort categories
        if "tenure" in X_df.columns:
            tenure_clean = pd.to_numeric(X_df["tenure"], errors="coerce").fillna(0)
            X_df["TenureCohort"] = pd.cut(
                tenure_clean,
                bins=self.TENURE_BINS,
                labels=self.TENURE_LABELS,
                right=True,
            ).astype(str)
        else:
            X_df["TenureCohort"] = "0-6m"

        # 3. HasPartnerAndDependents: household stability indicator
        if "Partner" in X_df.columns and "Dependents" in X_df.columns:
            X_df["HasPartnerAndDependents"] = (
                (X_df["Partner"] == "Yes") & (X_df["Dependents"] == "Yes")
            ).astype(int)
        else:
            X_df["HasPartnerAndDependents"] = 0

        # 4. ChargesRatio: Historical billing ratio (TotalCharges vs expected cumulative)
        if self.include_ratios and "TotalCharges" in X_df.columns and "MonthlyCharges" in X_df.columns:
            tc = pd.to_numeric(
                X_df["TotalCharges"].astype(str).str.strip(), errors="coerce"
            ).fillna(0.0)
            mc = pd.to_numeric(X_df["MonthlyCharges"], errors="coerce").fillna(0.0)
            tenure = pd.to_numeric(X_df["tenure"], errors="coerce").fillna(0.0)

            # Avoid division by zero
            denom = tenure * mc
            ratio = np.where(denom > 0, tc / denom, 1.0)
            # Clip extreme anomalies safely between 0.1 and 3.0
            X_df["ChargesRatio"] = np.clip(ratio, 0.1, 3.0)
        else:
            X_df["ChargesRatio"] = 1.0

        return X_df
