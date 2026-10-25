"""Exploratory Data Analysis (EDA) and reporting routines for Northstar SaaS."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURES_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    TARGET_POSITIVE_VALUE,
    TARGET_NEGATIVE_VALUE,
)
from src.data_validation import clean_dataset_for_exploration

logger = logging.getLogger(__name__)

# Apply clean professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"No": "#2b5c8f", "Yes": "#d95f02"}
ACCENT_COLOR = "#1f77b4"


def compute_target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute exact counts and percentages of the target variable."""
    counts = df[TARGET_COLUMN].value_counts()
    pcts = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    summary = pd.DataFrame({
        "Count": counts,
        "Percentage": pcts.round(2)
    })
    summary.index.name = "Churn Status"
    return summary


def compute_segment_churn_rates(
    df: pd.DataFrame,
    segment_cols: List[str]
) -> pd.DataFrame:
    """Calculate churn rates alongside group denominators for multiple segments.

    Ensures that tiny groups are easily identified and avoids misleading conclusions.
    """
    df_clean = clean_dataset_for_exploration(df)
    results = []

    for col in segment_cols:
        if col not in df_clean.columns:
            continue
        grouped = df_clean.groupby(col).agg(
            Total_Customers=("Churn_Binary", "count"),
            Churned_Customers=("Churn_Binary", "sum"),
            Churn_Rate_Pct=("Churn_Binary", lambda s: round(s.mean() * 100, 2)),
        ).reset_index()
        grouped.rename(columns={col: "Segment_Value"}, inplace=True)
        grouped.insert(0, "Feature", col)
        results.append(grouped)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def plot_target_distribution(df: pd.DataFrame, save_path: Path = None) -> Path:
    """Generate and save target class distribution chart."""
    dist = compute_target_distribution(df)
    
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    bars = ax.bar(
        dist.index,
        dist["Count"],
        color=[PALETTE.get(k, "#333333") for k in dist.index],
        edgecolor="black",
        linewidth=0.8,
        width=0.55
    )
    
    total = dist["Count"].sum()
    for bar in bars:
        h = bar.get_height()
        pct = (h / total) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 50,
            f"{int(h):,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title("Customer Churn Distribution (Baseline Target)", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel("Churn Status", fontsize=11, labelpad=8)
    ax.set_ylabel("Number of Customers", fontsize=11, labelpad=8)
    ax.set_ylim(0, total * 0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "01_target_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_churn_by_contract_and_internet(df: pd.DataFrame, save_path: Path = None) -> Path:
    """Generate and save churn rate across Contract Type and Internet Service."""
    df_clean = clean_dataset_for_exploration(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Contract Type
    contract_df = df_clean.groupby("Contract")["Churn_Binary"].agg(
        Total="count",
        Churned="sum",
        Rate=lambda s: s.mean() * 100
    ).reset_index()

    bars1 = axes[0].bar(
        contract_df["Contract"],
        contract_df["Rate"],
        color="#2b5c8f",
        edgecolor="black",
        width=0.5
    )
    axes[0].set_title("Churn Rate by Contract Type", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[0].set_ylim(0, 55)
    for bar, (_, row) in zip(bars1, contract_df.iterrows()):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            f"{row['Rate']:.1f}%\n(n={int(row['Total']):,})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # 2. Internet Service
    internet_df = df_clean.groupby("InternetService")["Churn_Binary"].agg(
        Total="count",
        Churned="sum",
        Rate=lambda s: s.mean() * 100
    ).reset_index()

    bars2 = axes[1].bar(
        internet_df["InternetService"],
        internet_df["Rate"],
        color="#d95f02",
        edgecolor="black",
        width=0.5
    )
    axes[1].set_title("Churn Rate by Internet Service Type", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Churn Rate (%)", fontsize=10)
    axes[1].set_ylim(0, 55)
    for bar, (_, row) in zip(bars2, internet_df.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.2,
            f"{row['Rate']:.1f}%\n(n={int(row['Total']):,})",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.suptitle("Core Product & Contract Churn Variations (With Group Denominators)", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "02_churn_by_contract_and_service.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_numeric_distributions_by_churn(df: pd.DataFrame, save_path: Path = None) -> Path:
    """Generate distributions of tenure and MonthlyCharges segmented by Churn."""
    df_clean = clean_dataset_for_exploration(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # 1. Tenure Kernel Density / Histogram
    sns.histplot(
        data=df_clean,
        x="tenure",
        hue=TARGET_COLUMN,
        palette=PALETTE,
        multiple="dodge",
        bins=24,
        ax=axes[0],
        kde=True
    )
    axes[0].set_title("Customer Tenure (Months) by Churn Status", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Tenure (Months)", fontsize=10)
    axes[0].set_ylabel("Customer Count", fontsize=10)

    # 2. Monthly Charges Boxplot & KDE
    sns.kdeplot(
        data=df_clean,
        x="MonthlyCharges",
        hue=TARGET_COLUMN,
        palette=PALETTE,
        common_norm=False,
        fill=True,
        alpha=0.3,
        ax=axes[1]
    )
    axes[1].set_title("Monthly Charges Density by Churn Status", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Monthly Charges ($)", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)

    plt.suptitle("Tenure and Billing Distributions by Churn Status", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()

    out_path = save_path or (FIGURES_DIR / "03_numeric_distributions_by_churn.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_missing_data_summary(df: pd.DataFrame, save_path: Path = None) -> Path:
    """Generate and save audit of missing and whitespace values across features."""
    total = len(df)
    missing_stats = []

    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        whitespace_count = int((s.astype(str).str.strip() == "").sum()) if s.dtype == object else 0
        total_missing = null_count + whitespace_count
        missing_stats.append({
            "Column": col,
            "Null_Count": null_count,
            "Whitespace_Count": whitespace_count,
            "Total_Missing": total_missing,
            "Pct_Missing": round((total_missing / total) * 100, 3)
        })

    mdf = pd.DataFrame(missing_stats)
    mdf_filtered = mdf[mdf["Total_Missing"] > 0]

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    if mdf_filtered.empty:
        ax.text(
            0.5, 0.5,
            "No Standard Missing Values Detected\n(All 21 columns have 0 NaN values)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="green"
        )
        ax.set_axis_off()
    else:
        bars = ax.barh(
            mdf_filtered["Column"],
            mdf_filtered["Total_Missing"],
            color="#e41a1c",
            edgecolor="black",
            height=0.4
        )
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + 0.3,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(w)} blank rows ({(w/total)*100:.2f}%)",
                ha="left",
                va="center",
                fontsize=9
            )
        ax.set_title("Missing / Blank Values by Column", fontsize=12, fontweight="bold")
        ax.set_xlabel("Count of Incomplete Records", fontsize=10)
        ax.set_xlim(0, max(mdf_filtered["Total_Missing"]) * 1.6)

    plt.tight_layout()
    out_path = save_path or (FIGURES_DIR / "04_missing_data_summary.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_tenure_cohort_business_inquiry(df: pd.DataFrame, save_path: Path = None) -> Path:
    """Generate a chart translating a data pattern into a business decision question.

    Specifically: Examine churn risk across tenure cohorts (0-6m, 6-12m, 12-24m, 24-48m, 48m+)
    coupled with contract types to question onboarding retention interventions.
    """
    df_clean = clean_dataset_for_exploration(df)

    # Bin tenure into cohorts
    bins = [-1, 6, 12, 24, 48, 100]
    labels = ["0-6 mos", "7-12 mos", "1-2 yrs", "2-4 yrs", "4+ yrs"]
    df_clean["Tenure_Cohort"] = pd.cut(df_clean["tenure"], bins=bins, labels=labels)

    cohort_churn = df_clean.groupby(["Tenure_Cohort", "Contract"], observed=False)["Churn_Binary"].agg(
        Total="count",
        ChurnRate=lambda s: round(s.mean() * 100, 1)
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(
        data=cohort_churn,
        x="Tenure_Cohort",
        y="ChurnRate",
        hue="Contract",
        palette=["#e41a1c", "#377eb8", "#4daf4a"],
        ax=ax,
        edgecolor="black"
    )

    ax.set_title(
        "Business Decision Inquiry: The 'Cliff Effect' in Month-to-Month Onboarding",
        fontsize=13,
        fontweight="bold",
        pad=15
    )
    ax.set_xlabel("Customer Tenure Cohort", fontsize=11, labelpad=8)
    ax.set_ylabel("Observed Churn Rate (%)", fontsize=11, labelpad=8)
    ax.set_ylim(0, 70)

    # Add descriptive annotation
    ax.annotate(
        "Over 50% churn in first 6 months\nfor Month-to-Month contracts!\nBusiness Question: Can early onboarding\ninterventions flatten this cliff?",
        xy=(0.1, 53),
        xytext=(0.5, 58),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
        backgroundcolor="#fff8dc"
    )

    plt.tight_layout()
    out_path = save_path or (FIGURES_DIR / "05_business_inquiry_tenure_cohorts.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_full_eda(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute all EDA visualizations, tables, and metric summaries."""
    logger.info("Computing target distribution...")
    target_dist = compute_target_distribution(df)
    target_dist.to_csv(TABLES_DIR / "target_distribution.csv")

    logger.info("Computing segment churn rates...")
    segment_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling", "SeniorCitizen", "Partner", "Dependents"]
    segment_rates = compute_segment_churn_rates(df, segment_cols)
    segment_rates.to_csv(TABLES_DIR / "segment_churn_rates.csv", index=False)

    logger.info("Generating EDA figures...")
    p1 = plot_target_distribution(df)
    p2 = plot_churn_by_contract_and_internet(df)
    p3 = plot_numeric_distributions_by_churn(df)
    p4 = plot_missing_data_summary(df)
    p5 = plot_tenure_cohort_business_inquiry(df)

    return {
        "target_distribution": target_dist.to_dict(),
        "figure_paths": [str(p) for p in [p1, p2, p3, p4, p5]],
        "tables_saved": [
            str(TABLES_DIR / "target_distribution.csv"),
            str(TABLES_DIR / "segment_churn_rates.csv"),
        ]
    }
