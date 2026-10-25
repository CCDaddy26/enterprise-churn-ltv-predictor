"""Configuration constants and paths for Northstar SaaS Churn & LTV Predictor."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

# Default Raw Dataset
RAW_DATA_FILENAME = "Telco-Customer-Churn.csv"
RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "telco_churn_clean.parquet"

# Dataset Remote Source
REMOTE_DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/"
    "data/Telco-Customer-Churn.csv"
)

# Reproducibility Seed
RANDOM_SEED = 42

# Column Schema Definitions
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TARGET_POSITIVE_VALUE = "Yes"
TARGET_NEGATIVE_VALUE = "No"

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",  # Binary 0/1 integer, treated as categorical factor
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Ensure required runtime directories exist
def ensure_directories():
    """Ensure all expected project directories exist."""
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        DOCS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
