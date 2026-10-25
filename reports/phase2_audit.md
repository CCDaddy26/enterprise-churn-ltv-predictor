# Phase 1 Code Audit & Governance Findings

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Auditor:** Senior ML Engineer & Applied Data Scientist  
**Date:** 2026-10-23  
**Status:** AUDIT COMPLETE â€" Phase 1 Verified, Enhanced Model Protocol Established

---

## 1. Scope and Purpose of Audit

Prior to beginning Phase 2 feature engineering and advanced model development, an in-depth code, data integrity, and leakage audit of Phase 1 was conducted.

The audit examined:
- Feature/Target isolation and customer ID handling
- Raw data preservation and data type conversions
- Imputation, scaling, and categorical encoding encapsulation
- Cross-validation independence inside each fold
- Temporal and post-event information risks
- Reproducibility of reported baseline metrics
- Test-set governance and test-set reuse risks

---

## 2. Item-by-Item Verification Findings

### A. Target Exclusion from Predictors
- **Location:** `src/data_validation.py` (`separate_features_and_target`) and `src/baseline.py`.
- **Finding:** Verified. `TARGET_COLUMN` (`"Churn"`) and `Churn_Binary` are explicitly dropped from the feature matrix `X` before returning.
- **Test Coverage:** Verified by unit tests in `tests/test_data_validation.py` and `tests/test_baseline.py`.

### B. Customer Identifier Exclusion
- **Location:** `src/data_validation.py` (`separate_features_and_target`).
- **Finding:** Verified. `ID_COLUMN` (`"customerID"`) is dropped from predictors.
- **Audit Note:** The dataset has 7,043 unique `customerID` values and 0 duplicate rows. Excluded from predictors to avoid spurious memorization.

### C. Raw Data Mutability Check
- **Location:** `src/data_loader.py` and `data/raw/Telco-Customer-Churn.csv`.
- **Finding:** Verified. `load_raw_dataset()` opens the file with read-only pandas calls (`pd.read_csv`). No write operations occur on `data/raw/`. The raw file size remains exactly 970,457 bytes.

### D. Transformation Scope & Leakage
- **Location:** `src/baseline.py` (`build_preprocessor`).
- **Finding:** Verified. `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` are packaged in a scikit-learn `ColumnTransformer` inside an unfitted `Pipeline`.
- **Finding on Cross-Validation:** In `run_baseline_benchmarking()`, `cross_validate()` receives the unfitted `Pipeline`, ensuring that statistics (medians, standard deviations, one-hot category dictionaries) are computed strictly from the training slice of each of the 5 folds.

### E. Temporal Validity
- **Finding:** The dataset is a retrospective cross-sectional snapshot without event timestamps.
- **Finding on Features:** All features (contract, service add-ons, monthly charges, tenure) represent the account state at the observation point. No post-cancellation features (e.g. churn surveys, deactivation timestamps) are present.

### F. Reproducibility of Phase 1 Reported Results
- **Finding:** Verified. Running `python src/run_phase1.py` reproduces the historical baseline test results:
  - Logistic Regression (Balanced): Precision 0.5043, Recall 0.7834, F1 0.6136, ROC-AUC 0.8417, PR-AUC (Average Precision) 0.6331, Accuracy 0.7381.

---

## 3. Test-Set Governance & Historical Reference Rule

### The Risk of Repeated Test-Set Querying
In Phase 1, an 80/20 train/test split with `RANDOM_SEED = 42` was created, and baseline models were evaluated on the held-out test split ($n = 1,409$).

In real-world machine learning development, if researchers develop new features, try multiple algorithms (Random Forest, XGBoost), tune hyperparameters, and repeatedly check performance against that same held-out test set, **information from the test set inevitably leaks into engineering decisions via researcher degrees of freedom**.

### Phase 2 Governance Protocol: Development Partition vs. Final Evaluation

To adhere to rigorous ML validation standards:
1. **Preserve Phase 1 Baseline as Historical Reference:** The Phase 1 metrics on the held-out test split ($n = 1,409$, seed 42) remain documented as historical baselines.
2. **Dedicated Development Split for Phase 2:**
   - We utilize a structured 3-way partition: **Development Train (70%, $n = 4,930$)**, **Development Validation (15%, $n = 1,056$)**, and **Final Test (15%, $n = 1,057$)**, or conduct **Stratified 5-Fold Cross-Validation strictly within the Development partition (80%, $n = 5,634$)**.
   - All feature decisions, scaling choices, algorithm comparisons (Dummy, Logistic Regression, Random Forest, XGBoost), and hyperparameter searches (`GridSearchCV` / `RandomizedSearchCV`) are executed **strictly within the Development partition**.
3. **Quarantine of the Final Evaluation Partition:**
   - The final test partition ($n = 1,409$) is strictly quarantined during Phase 2 development.
   - The selected champion model is evaluated **exactly once** on the held-out partition at the conclusion of development for final benchmarking against the Phase 1 reference baseline.
   - No hyperparameter tuning or threshold selection may query the final test partition.

---

## 4. Phase 1 Audit Summary Table

| Audit Area | Phase 1 Status | Potential Leakage Risk | Phase 2 Action Taken |
| :--- | :---: | :---: | :--- |
| **Target Leakage** | PASS | None detected; target isolated | Enforce in `src/data_validation.py` and `src/preprocessing.py` |
| **Identifier Leakage** | PASS | `customerID` dropped | Maintained; unit tested in `tests/test_data_validation.py` |
| **Imputer / Scaler Leakage** | PASS | Enclosed in `Pipeline` | Extended into modular `src/preprocessing.py` |
| **Fold Contamination** | PASS | CV fits inside fold | Kept strict in cross-validation runner |
| **Test Set Governance** | PASS | Single evaluation performed | Formalized development vs. quarantined test protocol |
| **Library Availability** | PARTIAL | `xgboost` not pre-installed | Installed `xgboost 3.2.0`; added to `requirements.txt` |
| **Model Serialization** | NOT IMPLEMENTED | N/A (Phase 1 scope) | Implemented via `joblib` in `src/inference.py` |
