# Phase 1 Summary: Business Discovery & Data Foundation
**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Data Scientist & ML Engineer  
**Date:** 2026-10-23  
**Status:** PASS â€" Complete Data Foundation & Baseline Benchmarking Established  

---

## 1. Executive Summary & Problem Framing

Customer churn in subscription software and telecom services represents a critical drain on enterprise Annual Recurring Revenue (ARR). In collaboration with business leadership at **Northstar SaaS** (a fictional subscription case study based on public benchmark data), our analytics engagement was initiated to:
1. Establish a rigorous, auditable data foundation free of data leakage.
2. Investigate baseline churn rates and segment disparities across customer cohorts.
3. Establish a defensible baseline classification benchmark against which complex machine learning models (e.g. XGBoost, CatBoost, LightGBM) will be evaluated in future phases.
4. Formulate an economically grounded Customer Lifetime Value (LTV) and Retention ROI methodology to prevent misleading financial projections.

All deliverables for Phase 1 have been fully implemented, executed on actual downloaded data, verified through automated unit testing, and documented.

---

## 2. Dataset Attribution & Integrity Audit

The data foundation is built on the authoritative **IBM Telco Customer Churn** benchmark dataset, retrieved directly from IBM's open sample repository:
- **Source URL:** `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **Local Location:** `data/raw/Telco-Customer-Churn.csv` (Size: 970,457 bytes)
- **Shape:** 7,043 rows, 21 columns
- **Identifiers:** `customerID` has 7,043 unique values (0 duplicate customer records).
- **Target Variable:** `Churn` with 5,174 active (`"No"`, 73.46%) and 1,869 churned (`"Yes"`, 26.54%).

### Data Quality Issues Identified & Resolved

| Feature / Issue | Observation | Analytical Consequence | Resolution & Policy |
| :--- | :--- | :--- | :--- |
| `TotalCharges` Data Type & Blanks | Column was stored as string `object`. Exactly 11 rows contained blank whitespace strings (`" "`). | Direct conversion to numeric throws exceptions; silent row deletion would bias early tenure cohorts. | All 11 records have `tenure = 0` (brand-new accounts billed 0 months). Whitespace was cleaned and mapped to `0.0` within `NumericCleaner` transformer. Zero rows deleted. Reversible via raw data preservation. |
| Customer Identifier Memorization | `customerID` is an alphanumeric string unique to each subscriber. | Including IDs in statistical or tree models causes high-cardinality memorization and data leakage. | Strictly excluded from the feature matrix $X$ via `separate_features_and_target()`. |
| Time Dimension Limitations | No timestamped logs, event history, or renewal event dates exist. | Real-time prospective forecasting cannot be validated. | Formally classified as a retrospective classification study in `docs/prediction_contract.md`. |

---

## 3. Exploratory Data Analysis & Strategic Business Findings

All visual assets and underlying tables were generated directly from data and stored under `reports/figures/` and `reports/tables/`.

### Finding 1: Overall Target Class Imbalance
- **Observed Churn Rate:** **26.54%** (1,869 churned out of 7,043 total accounts).
- **Stakeholder Implication:** A naive "majority-class" model predicting that no customer ever churns achieves **73.46% accuracy** while capturing exactly **0% of churners** (Recall = 0.0). Accuracy alone is fundamentally misleading for churn prioritization.
- *Visual Asset:* `reports/figures/01_target_distribution.png`

### Finding 2: Contract Commitment Disparity
- **Month-to-Month Contracts:** Churn rate of **42.71%** ($n = 3,875$).
- **One-Year Contracts:** Churn rate drops to **11.27%** ($n = 1,473$).
- **Two-Year Contracts:** Churn rate drops to **2.83%** ($n = 1,695$).
- **Stakeholder Interpretation & Caveat:** Month-to-month subscribers churn at over **15 times** the rate of two-year subscribers. However, this is an **observational correlation, not proof of causation**. Customers intending to leave naturally select month-to-month contracts. Forcing annual commitments could harm top-of-funnel acquisition without resolving underlying dissatisfaction.
- *Visual Asset:* `reports/figures/02_churn_by_contract_and_service.png`

### Finding 3: Service Architecture Variance
- **Fiber Optic Internet:** Churn rate of **41.89%** ($n = 3,096$).
- **DSL Internet:** Churn rate of **18.96%** ($n = 2,421$).
- **No Internet Service:** Churn rate of **7.40%** ($n = 1,826$).
- **Stakeholder Interpretation:** Fiber optic customers exhibit substantially higher churn despite paying premium monthly fees. This indicates potential friction in network reliability, competitive pricing pressure, or customer onboarding expectations that require product/engineering investigation.

### Finding 4: The Onboarding "Cliff Effect" (Business Decision Inquiry)
- Cross-tabulating customer tenure cohorts against contract types reveals that **over 53% of month-to-month customers churn within their first 6 months**.
- For customers who survive beyond 24 months, month-to-month churn drops below 30%.
- **Actionable Business Question:** Rather than spreading retention discounts evenly across all customers, can Northstar design a targeted **0â€"90 day white-glove onboarding intervention** to flatten this early attrition cliff?
- *Visual Asset:* `reports/figures/05_business_inquiry_tenure_cohorts.png`

---

## 4. Leakage-Resistant Baseline Model Evaluation

To establish honest reference performance, we isolated 20% of the dataset as a stratified held-out test set ($n = 1,409$, exactly 26.54% positive class), locking the random seed at `42`.

Imputation, categorical encoding (`OneHotEncoder`), and standard scaling (`StandardScaler`) were configured inside scikit-learn `Pipeline` architectures. 5-fold cross-validation was conducted exclusively within the training split ($n = 5,634$).

### Empirical Evaluation on Held-Out Test Partition ($n = 1,409$)

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | Accuracy | Confusion Matrix (TP / FP / TN / FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 | 0.7346 | $0 \;/\; 0 \;/\; 1035 \;/\; 374$ |
| **Dummy (Stratified)** | 0.2891 | 0.2914 | 0.2903 | 0.5163 | 0.2723 | 0.6217 | $109 \;/\; 268 \;/\; 767 \;/\; 265$ |
| **Logistic Regression (Balanced)** | **0.5043** | **0.7834** | **0.6136** | **0.8417** | **0.6331** | **0.7381** | **$293 \;/\; 288 \;/\; 747 \;/\; 81$** |

### Key Analytical Insights from Baseline
1. **Recall vs. Precision Trade-off:** The balanced Logistic Regression baseline captures **78.34% of all churners** (293 out of 374 in the test partition), missing only 81 churners.
2. **False Positive Burden:** The model produces 288 false positives ($50.43\%$ precision). In an operational workflow where CSM outreach costs \$25, this trade-off is economically advantageous compared to losing an average customer's annual recurring revenue.
3. **PR-AUC Baseline:** The model achieves a PR-AUC (Average Precision) of **0.6331**, substantially higher than the random baseline of **0.2654**. In accordance with scientific honesty guidelines, future Phase 3 gradient boosted models will be judged against this real benchmark without unsubstantiated performance promises.

---

## 5. Verification & Automated Testing

The modular implementation under `src/` was tested using `pytest`.
- **Test Command:** `python -m pytest -v`
- **Results:** **15 passed, 0 failed, 0 skipped** (Execution time: 10.93s).
- **Coverage Areas:** Missing file handling, empty file rejection, synthetic schema compliance, whitespace imputation, customer ID exclusion, and deterministic reproducibility across repeated runs.

---

## 6. Recommendations & Handoff to Phase 2

Phase 1 provides a clean, modular, and leak-free foundation. 

### Ready for Review:
1. Source data preserved in `data/raw/Telco-Customer-Churn.csv`.
2. Core modules in `src/` (`config.py`, `data_loader.py`, `data_validation.py`, `eda.py`, `baseline.py`, `run_phase1.py`).
3. Governance documents in `docs/` (`data_sources.md`, `data_dictionary.md`, `data_leakage_policy.md`, `prediction_contract.md`, `ltv_and_roi_methodology.md`).
4. Generated charts and tables in `reports/figures/` and `reports/tables/`.
5. Automated test suite in `tests/`.

**Recommended Next Step:** Await stakeholder review and sign-off before proceeding to **Phase 2: Feature Engineering & Preprocessing Pipelines**.
