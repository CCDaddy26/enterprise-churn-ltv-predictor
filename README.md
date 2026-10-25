# Leak-Free Enterprise Churn & LTV Predictor

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Tests: 58 Passed](https://img.shields.io/badge/pytest-58%20passed-brightgreen.svg)]()
[![Model: Tuned XGBoost](https://img.shields.io/badge/champion-XGBoost-orange.svg)]()
[![Dashboard: Streamlit](https://img.shields.io/badge/dashboard-Streamlit-red.svg)]()
[![Economics: LTV & ROI](https://img.shields.io/badge/economics-LTV%20%26%20ROI-green.svg)]()

> A leakage-resistant machine learning and customer lifetime value system developed for an educational case study: **"Northstar SaaS"** (fictional enterprise scenario using the public IBM Telco Customer Churn dataset).

---

## 1. Executive Summary & Business Scenario

Subscription recurring revenue businesses depend heavily on proactive customer retention. Acquiring a new customer typically costs 5x to 7x more than retaining an existing account. In our portfolio case study, **Northstar SaaS** is a fictional B2B subscription software enterprise evaluating customer churn risk and campaign economics.

Executive leadership requires not only accurate probability predictions, but **explainable, calibrated, audited machine learning outputs** and **transparent financial simulations** before committing retention marketing budgets.

### Crucial Educational Disclaimers:
- **Fictional Scenario:** Northstar SaaS is a fictional business context created for portfolio demonstration. All analyses use the public IBM Telco Customer Churn dataset.
- **Assumption-Based Financials:** All customer lifetime values (LTV), value at risk (VaR), and campaign return on investment (ROI) calculations are **scenario simulations** based on user-configurable assumptions (contribution margin, value horizon, contact and incentive costs). They are **not** verified corporate balance-sheet figures or observed accounting facts.
- **Association vs. Causation:** Statistical feature importance (such as SHAP values) and predicted churn risks reflect observational correlations. They do **not** prove that offering discounts or reaching out will causally prevent customer cancellation.

---

## 2. Multi-Phase Progress & Key Metrics

| Phase | Core Objective | Key Deliverables & Verified Metrics |
| :--- | :--- | :--- |
| **Phase 1: Data Foundation** | Leakage-free split, schema validation, baseline benchmark | Audited all 7,043 rows (26.54% churn prevalence). Cleaned whitespace anomalies with zero row deletion. Baseline Logistic Regression: PR-AUC 0.6331, Recall 78.34%. |
| **Phase 2: Pipelines & Models** | Feature engineering & cross-validated benchmarking | Built domain transformers (`TotalServices`, `TenureCohort`, `ChargesRatio`). 5-fold CV across Dummy, Logistic Regression, Random Forest, and Tuned XGBoost. Tuned XGBoost champion: **PR-AUC 0.6571, ROC-AUC 0.8452, F1 0.6284** on held-out test data. |
| **Phase 3: SHAP & Diagnostics** | Explainable AI, calibration, threshold & subgroup audits | SHAP TreeExplainer attributions (Contract type and tenure dominate). Decision threshold analysis ($\tau = 0.40$ captures 83.42% of churners). Demographic and commercial fairness audits. |
| **Phase 4 & 4.1: Economics** | Defensible calibration, LTV proxy & ROI simulator | Cross-fitted Platt Sigmoid Calibrator (reduced Expected Calibration Error from 0.1152 to 0.0110). Mode A (Like-for-Like) and Mode B (Budget-Maximized) targeting comparisons. Trigger-consistent break-even math. Undefined ROI (`N/A`) for zero-cost baseline. |
| **Phase 5: Release Hardening** | Packaging, operational diagnostics, documentation | Pinned dependencies (`requirements.txt`), operational readiness health-check suite (`src/run_phase5_check.py`), 58 automated tests passing. |

---

## 3. System Architecture

```text
+-----------------------------------------------------------------------------+
|               ENTERPRISE CHURN & LTV PREDICTOR ARCHITECTURE                 |
+-----------------------------------------------------------------------------+
|                                                                             |
|  [DATA INGESTION]          [FEATURE PIPELINES]        [DIAGNOSTICS & SHAP]  |
|  - Raw CSV Schema Audit    - Domain Transformers      - SHAP TreeExplainer  |
|  - Blank Cleaning          - Scalers & OneHotEncoders - Reliability Curves  |
|  - Leakage-Free Stratified - 5-Fold Stratified CV     - Threshold Tradeoffs |
|    Train/Test Split (80/20)- Tuned XGBoost Champion   - Subgroup Audits     |
|                                                              |              |
|                                                              v              |
|  [STREAMLIT DASHBOARD] <--- [FINANCIAL ENGINE] <--- [CALIBRATION]           |
|  - Executive Overview       - Remaining Contribution - OOF Cross-Fitting    |
|  - Cohort Risk Explorer     - Value at Risk (VaR)      (ECE 0.1152->0.0110) |
|  - Campaign ROI Simulator   - 3-Stage Outreach Funnel- Uncalibrated /       |
|  - Scenario Sensitivity     - Cost Trigger Logic       Calibrated Modes     |
|  - Limitations & Audits     - Mode A/B Comparisons                          |
+-----------------------------------------------------------------------------+
```

---

## 4. What the Project Does

### 1. Exploratory Data Analysis & Integrity
- Analyzes customer tenure, subscription contracts, billing charges, and churn distribution.
- Handles whitespace anomalies and zero-tenure records without deleting rows.

### 2. Leakage-Free Machine Learning Pipeline
- Quarantines the final 20% test split ($n=1,409$) prior to preprocessing.
- All imputation, feature engineering, and encoding parameters fit strictly inside CV training folds.

### 3. Model Benchmark & Champion Performance
- Evaluates Dummy, Logistic Regression, Random Forest, and Tuned XGBoost.
- Verified test-set performance for Tuned XGBoost:
  - **PR-AUC:** `0.6571` (vs. Dummy baseline `0.2654`)
  - **ROC-AUC:** `0.8452`
  - **F1 Score:** `0.6284` (at default threshold $\tau=0.50$)

### 4. SHAP Explainability & Global Importance
- Global TreeExplainer reveals that **Contract duration** (Two-year vs. Month-to-month) and **tenure** have the strongest protective impact against churn.
- Higher monthly charges and Fiber Optic service correlate with elevated churn risk.

### 5. Defensible Calibration & Risk Semantics
- Addresses the `scale_pos_weight = 2.5` imbalance correction by offering two explicit modes:
  - **Calibrated Mode:** Uses a cross-fitted Platt calibrator (ECE = 0.0110) to generate empirical probabilities and **Estimated Value at Risk**.
  - **Uncalibrated Mode:** Uses raw model scores as a relative **churn risk score** and produces a **risk-weighted value proxy**, explicitly warning users that it is not an expected monetary loss.

### 6. Transparent Customer Value Proxy
- Estimates remaining contribution value over a configurable horizon:
  $$\text{Remaining Contribution Value} = \text{MonthlyCharges} \times \text{Contribution Margin} \times \text{Remaining Horizon (Months)}$$

### 7. Multi-Stage Campaign ROI Simulation
- Models outreach across a 3-stage funnel: $\text{Reach Rate} \times \text{Response Rate} \times \text{Save Rate}$.
- Supports three non-overlapping cost triggers: `responders_only`, `saved_only`, and `all_contacted`.
- Implements two comparison architectures:
  - **Mode A (Like-for-Like):** Evaluates all active targeting strategies on the identical customer batch size ($N=500$).
  - **Mode B (Budget-Maximized):** Evaluates strategies capped at maximum budget capacity ($\lfloor \frac{\text{Budget}}{\text{Unit Cost}} \rfloor$).
- Properly reports undefined ROI as `N/A — no campaign expenditure` when campaign cost is zero.

### 8. Interactive Executive Streamlit Dashboard
- Multi-page application with responsive executive cards, cohort filter/export, interactive ROI controls, scenario break-even analysis, and data limitations disclosure.

---

## 5. Targeting Strategy Comparison (Mode A: Like-for-Like)

Evaluated on 500 targeted accounts from the 7,043 customer population under baseline assumptions:
- Contact Cost: ₹150 | Concession: ₹500 (responders only) | Reach: 85% | Response: 40% | Save: 25% | Margin: 70% | Horizon: 12 months

| Strategy | Targeted Accounts | Expected Prevented Churn | Campaign Cost | Expected Gross Benefit | Expected Net Benefit | Expected ROI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No Campaign Baseline** | 0 | 0.0 | ₹0 | ₹0 | ₹0 | **N/A — no expenditure** |
| **2. Highest Churn Risk First** | 500 | 31.7 | ₹160,000 | ₹21,815 | -₹138,185 | -86.4% |
| **3. Highest Value at Risk First** | 500 | 30.1 | ₹160,000 | ₹23,775 | -₹136,225 | -85.1% |
| **4. Balanced Risk & Value** | 500 | 31.3 | ₹160,000 | ₹23,348 | -₹136,652 | -85.4% |

**Key Business Finding:** Targeting by **Value at Risk** preserves **+₹1,960 (+9.0%)** more revenue than targeting purely by highest churn probability, proving that balance-sheet exposure is a more effective prioritization metric than defection risk alone.

---

## 6. Local Setup & Execution Guide (Windows PowerShell)

### Step 1: Clone Repository & Create Environment
```powershell
# Open Windows PowerShell in your project folder
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

### Step 2: Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Full Automated Quality Gate (58 Tests)
```powershell
python -m pytest -v
```

### Step 4: Run Operational Readiness Check
```powershell
python -m src.run_phase5_check
```

### Step 5: Launch Streamlit Dashboard
```powershell
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

## 7. Model Artifact & Dataset Setup

### Dataset Setup:
The raw dataset (`Telco-Customer-Churn.csv`) is automatically downloaded on-demand from the official public IBM repository when running pipelines:
```powershell
python -c "from src.data_loader import download_raw_dataset; download_raw_dataset()"
```

### Model Artifact Generation:
If model artifacts are not present, they can be generated deterministically and reproducibly using the project's pipeline runners:
```powershell
# Generate Phase 2 Champion Pipeline
python src/run_phase2.py

# Generate Phase 3 & 4.1 Sigmoid Calibrator
python src/run_phase3.py

# Generate Scenario Tables & Anonymized Samples
python src/run_phase4.py
```

---

## 8. Repository Structure

```text
.
├── app.py                     # Streamlit Executive Dashboard entry point
├── requirements.txt           # Pinned production dependencies
├── pytest.ini                 # Test configuration and filters
├── .env.example               # Environment variables template
├── .gitignore                 # Version control exclusion rules
├── src/
│   ├── config.py              # Centralized paths and seed constants
│   ├── data_loader.py         # Dataset ingestion & download utilities
│   ├── data_validation.py      # Schema enforcement & blank cleaning
│   ├── feature_engineering.py # Leakage-free domain feature transformers
│   ├── preprocessing.py       # ColumnTransformers & encoders
│   ├── baseline.py            # Phase 1 Logistic Regression baseline
│   ├── model_training.py      # Stratified CV & candidate benchmarking
│   ├── inference.py           # Production inference wrapper
│   ├── explainability.py      # Phase 3 SHAP TreeExplainer routines
│   ├── calibration.py          # Cross-fitted Platt Sigmoid calibration
│   ├── error_analysis.py      # Error diagnostic classifications
│   ├── subgroup_audit.py      # Demographic & commercial fairness audit
│   ├── robustness.py           # Feature ablation & corruption checks
│   ├── ltv_engine.py          # LTV proxy & Value-at-Risk calculations
│   ├── roi_simulator.py       # Campaign economics & Mode A/B comparator
│   ├── scenario_analysis.py   # Break-even math & sensitivity grids
│   ├── dashboard_data.py      # Non-mutating data scoring interface
│   ├── run_phase1.py ... run_phase4.py
│   └── run_phase5_check.py    # Operational deployment readiness check
├── tests/                     # 58 automated regression and unit tests
├── models/
│   ├── champion_pipeline.joblib # Serialized Tuned XGBoost champion
│   ├── sigmoid_calibrator.joblib# Serialized cross-fit calibrator
│   └── model_metadata.json    # Hyperparameters, contract, seeds
├── reports/
│   ├── figures/               # 16 visual diagnostic figures
│   ├── tables/                # Strategy comparisons & anonymized samples
│   └── phase*_summary.md      # Phase completion reports & audits
└── docs/                      # Methodology, assumptions & guides
```

---

## 9. Limitations & Future Roadmap

1. **Retrospective Dataset:** The IBM Telco dataset is a static snapshot without timestamps, preventing continuous-time survival analysis (such as Cox Proportional Hazards).
2. **Observational Attribution:** Feature attributions (SHAP) and risk predictions reflect historical correlation. True incremental retention requires randomized A/B holdout trials.
3. **Future Work:**
   - Implement continuous-time survival analysis with time-varying covariates.
   - Integrate randomized uplift modeling ($E[Y_1 - Y_0 | X]$) when experimental trial data becomes available.
   - Connect live CRM webhooks (e.g., Salesforce, HubSpot) for real-time risk scoring.

---

## 10. Educational Disclaimer

This project is strictly an educational case study and portfolio demonstration. All financial calculations, customer lifetime values, and campaign return on investment figures are illustrative scenario estimates. They do not represent audited corporate financials, accounting forecasts, or proven commercial outcomes.
