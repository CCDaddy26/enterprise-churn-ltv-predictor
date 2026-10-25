# Phase 5: Environment & Dependency Audit

## 1. Runtime Environment Specification
- **Primary Tested Python Version:** `Python 3.11.6` (64-bit on Windows 10/11 x86_64).
- **Target Container Base Image:** `python:3.11-slim` (official Debian-based minimal image).

---

## 2. Pinned Direct Dependencies & Audit Rationale

| Package | Tested Version | Direct Role in Project | Verification Notes |
| :--- | :---: | :--- | :--- |
| **`streamlit`** | `1.58.0` | Executive Dashboard UI, widgets, caching, tables | Tested via local HTTP 200 smoke test and browser rendering. |
| **`pandas`** | `2.3.2` | Dataframe manipulation, feature joins, CSV exports | Handles mixed dtypes, blank cleaning, and cohort segmentations. |
| **`numpy`** | `2.3.2` | Numeric calculations, probability vectors, arrays | Vectorized expected-value formulas and ECE binning. |
| **`scikit-learn`** | `1.7.1` | Preprocessing, Pipeline, StratifiedKFold, Calibrator | Fits Sigmoid Calibrator; executes ColumnTransformers and OneHotEncoders. |
| **`xgboost`** | `3.2.0` | Champion gradient boosting classifier | Serialized champion model execution; high-precision decision trees. |
| **`shap`** | `0.51.0` | Explainable AI & TreeExplainer attributions | Global beeswarm, dependence, and waterfall explanations. |
| **`joblib`** | `1.5.2` | Model serialization and deserialization | Loads `champion_pipeline.joblib` and `sigmoid_calibrator.joblib`. |
| **`matplotlib`** | `3.10.5` | Diagnostic figure rendering | Generates reliability diagrams, threshold curves, and SHAP figures. |
| **`seaborn`** | `0.13.2` | Statistical plotting aesthetics | Subgroup audits and confusion matrix visualizations. |
| **`python-dotenv`**| `1.0.0` | Environment configuration loading | Parses optional `.env` files for runtime directory overrides. |
| **`pyarrow`** | `>=14.0.0` | Parquet file storage support | Reads and writes `data/processed/telco_churn_clean.parquet`. |
| **`pytest`** | `9.1.1` | Automated testing & regression framework | Executes 54+ tests across unit and integration suites. |

---

## 3. Fresh Environment Setup Procedure (Windows PowerShell)

```powershell
# 1. Create a fresh isolated Python 3.11 virtual environment
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install all pinned dependencies
pip install -r requirements.txt

# 5. Verify installed packages match manifest
pip list
```

---

## 4. Container Installation Strategy
In `Dockerfile`, dependencies are installed into the container's system environment without cache clutter:
```dockerfile
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
```
This minimizes image layer size while ensuring reproducible builds.
