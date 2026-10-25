"""Data validation, audit, and clean pipeline preparation for Northstar SaaS Churn & LTV Predictor."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    ID_COLUMN,
    TARGET_COLUMN,
    TARGET_POSITIVE_VALUE,
    TARGET_NEGATIVE_VALUE,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
)

logger = logging.getLogger(__name__)


def validate_raw_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate that the DataFrame adheres to required core schema elements.

    Args:
        df: Input DataFrame to check.

    Returns:
        Dictionary with validation results and metrics.

    Raises:
        ValueError: If mandatory columns or target values are invalid.
    """
    if df is None or df.empty:
        raise ValueError("Cannot validate an empty or None DataFrame.")

    # 1. Target Column Existence
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Mandatory target column '{TARGET_COLUMN}' is missing from DataFrame.")

    # 2. Target Column Values
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    expected_targets = {TARGET_POSITIVE_VALUE, TARGET_NEGATIVE_VALUE}
    if not unique_targets.issubset(expected_targets):
        unexpected = unique_targets - expected_targets
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' contains unexpected classes: {unexpected}. "
            f"Expected only {expected_targets}."
        )

    # 3. Customer ID uniqueness check (if present)
    id_unique = True
    duplicate_ids_count = 0
    if ID_COLUMN in df.columns:
        duplicate_ids_count = int(df.duplicated(subset=[ID_COLUMN]).sum())
        id_unique = (duplicate_ids_count == 0)

    # 4. Total Charges whitespace check (if present)
    blank_total_charges = 0
    if "TotalCharges" in df.columns:
        if df["TotalCharges"].dtype == object:
            blank_total_charges = int((df["TotalCharges"].astype(str).str.strip() == "").sum())

    return {
        "is_valid": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "id_column_present": ID_COLUMN in df.columns,
        "id_unique": id_unique,
        "duplicate_ids_count": duplicate_ids_count,
        "target_classes": sorted(list(unique_targets)),
        "blank_total_charges_count": blank_total_charges,
    }


def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform a comprehensive programmatic audit of the dataset.

    Returns audit findings for missing values, whitespace anomalies, data types,
    cardinality, outliers, and class imbalance.

    Args:
        df: DataFrame to audit.

    Returns:
        Structured dictionary of audit metrics.
    """
    total_rows, total_cols = df.shape

    # Duplicates
    full_row_duplicates = int(df.duplicated().sum())
    id_duplicates = int(df.duplicated(subset=[ID_COLUMN]).sum()) if ID_COLUMN in df.columns else 0

    # Target stats
    target_counts = {}
    target_pcts = {}
    if TARGET_COLUMN in df.columns:
        val_counts = df[TARGET_COLUMN].value_counts(dropna=False)
        target_counts = {str(k): int(v) for k, v in val_counts.items()}
        target_pcts = {str(k): round(float(v / total_rows * 100), 2) for k, v in val_counts.items()}

    # Column-level audit
    column_audit = {}
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        n_whitespace = 0
        if s.dtype == object:
            # Count empty string or whitespace-only
            n_whitespace = int((s.astype(str).str.strip() == "").sum())

        n_unique = int(s.nunique(dropna=False))
        dtype_str = str(s.dtype)

        col_info = {
            "dtype": dtype_str,
            "missing_count": n_missing,
            "whitespace_count": n_whitespace,
            "unique_count": n_unique,
            "sample_values": s.dropna().unique()[:5].tolist(),
        }

        # Check numeric characteristics if numeric
        if pd.api.types.is_numeric_dtype(s):
            col_info.update({
                "min": float(s.min()) if not s.dropna().empty else None,
                "median": float(s.median()) if not s.dropna().empty else None,
                "max": float(s.max()) if not s.dropna().empty else None,
                "mean": float(s.mean()) if not s.dropna().empty else None,
                "std": float(s.std()) if not s.dropna().empty else None,
            })

        column_audit[col] = col_info

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "full_row_duplicates": full_row_duplicates,
        "id_duplicates": id_duplicates,
        "target_counts": target_counts,
        "target_percentages": target_pcts,
        "columns": column_audit,
    }


def clean_dataset_for_exploration(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a typed, exploration-ready DataFrame without dropping rows.

    Important:
    - Does NOT mutate the input DataFrame.
    - Resolves string-typed numeric columns (e.g. TotalCharges with blanks).
    - Blanks in TotalCharges (representing brand-new customers with tenure=0) are converted to 0.0.
    - Encodes binary target into Churn_Binary (1 for Yes, 0 for No).
    - Preserves all original rows and raw columns so all downstream steps are auditable.

    Args:
        df: Input DataFrame.

    Returns:
        Cleaned copy of the DataFrame.
    """
    df_clean = df.copy()

    # Convert TotalCharges if present as string
    if "TotalCharges" in df_clean.columns:
        # Convert whitespace to NaN first, then fill with 0.0 for tenure==0 or median
        tc_numeric = pd.to_numeric(df_clean["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # For customers where tenure is 0, TotalCharges is naturally 0.0
        tc_numeric = tc_numeric.fillna(0.0)
        df_clean["TotalCharges"] = tc_numeric

    # Map target to binary indicator 1/0 for statistical aggregations
    if TARGET_COLUMN in df_clean.columns:
        df_clean["Churn_Binary"] = (df_clean[TARGET_COLUMN] == TARGET_POSITIVE_VALUE).astype(int)

    return df_clean


def separate_features_and_target(
    df: pd.DataFrame,
    exclude_id: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Cleanly separate feature matrix X from target vector y.

    Ensures that customer identifiers are excluded from predictors to prevent
    spurious feature memorization or leakage.

    Args:
        df: Input DataFrame.
        exclude_id: Whether to exclude the ID column from features X. Defaults to True.

    Returns:
        Tuple of (X, y) where X is a DataFrame and y is a Series.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not present in DataFrame.")

    cols_to_drop = [TARGET_COLUMN]
    if "Churn_Binary" in df.columns:
        cols_to_drop.append("Churn_Binary")

    if exclude_id and ID_COLUMN in df.columns:
        cols_to_drop.append(ID_COLUMN)

    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COLUMN].copy()

    return X, y
