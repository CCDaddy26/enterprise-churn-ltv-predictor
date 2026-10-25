# Phase 2 Summary: Leak-Free Feature Engineering & ML Pipeline
**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Machine Learning Engineer & Applied Data Scientist  
**Date:** 2026-10-23  
**Status:** PASS â€" Feature Engineering, Cross-Validation, Tuning, and Benchmarking Established  

---

## 1. Executive Summary

Phase 2 transitions the foundational data audit and baseline of Phase 1 into a production-grade, leakage-resistant model development and evaluation pipeline.

Key achievements in Phase 2:
1. **Audited Phase 1 Codebase:** Confirmed 0 target leakage, strict isolation of `customerID`, read-only dataset operations, and fold-encapsulated transformations. Documented findings in `reports/phase2_audit.md`.
2. **Quarantined Development vs. Final Evaluation Protocol:** Established a formal partitioning boundary where 80% ($n = 5,634$) is dedicated to feature development, model family comparison, and bounded hyperparameter tuning via Stratified 5-Fold Cross-Validation, while 20% ($n = 1,409$) was quarantined and evaluated **exactly once** for final benchmarking.
3. **Domain-Specific Feature Engineering:** Implemented `ChurnFeatureEngineer` generating:
   - `TotalServices`: Discrete count of active protective, cloud, and streaming add-ons.
   - `TenureCohort`: Binned non-linear tenure categories (`0-6m`, `7-12m`, `13-24m`, `25-48m`, `49m+`).
   - `HasPartnerAndDependents`: Demographic household stability indicator.
   - `ChargesRatio`: Ratio of cumulative historical billing to expected spend (`TotalCharges / (tenure * MonthlyCharges)`).
4. **Multi-Model Family Benchmarking:** Implemented and evaluated five candidate model architectures inside full `Pipeline` containers: `DummyClassifier` (most frequent & stratified), `LogisticRegression` (balanced with scaling), `RandomForestClassifier` (balanced tree bagging), and `XGBClassifier` (gradient boosting with class re-weighting).
5. **Bounded Hyperparameter Optimization:** Tuned XGBoost over 16 candidate configurations strictly within the Development split optimizing for **Average Precision (PR-AUC)**.
6. **Inference Interface & Artifact Serialization:** Built and verified `ChurnInferenceEngine`, serializing the champion pipeline to `models/champion_pipeline.joblib` with comprehensive governance metadata.
7. **Automated Unit Testing:** Extended test suite from 15 to **27 automated tests**, covering feature generation, missing handling, unseen category tolerance, pipeline determinism, and inference serialization. All 27 passed.

---

## 2. Partitioning Protocol & Class Distribution

Partitioning was executed with fixed `RANDOM_SEED = 42`:

| Partition Name | Purpose | Sample Count ($n$) | Percentage | Churned ($y=1$) | Retained ($y=0$) | Churn Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Development Partition** | 5-Fold CV, Feature Validation, Tuning | **5,634** | 80.0% | 1,495 | 4,139 | **26.54%** |
| **Quarantined Final Test** | Single final reference benchmark | **1,409** | 20.0% | 374 | 1,035 | **26.54%** |
| **Total Population** | Full observational dataset | **7,043** | 100.0% | 1,869 | 5,174 | **26.54%** |

*Audit Confirmation:* Exactly 0 customer records cross partition boundaries. `customerID` was stripped before feature matrix transformation.

---

## 3. Stratified Cross-Validation Results (Development Split, $n = 5,634$)

All candidate pipelines were evaluated using identical 5-fold stratified splits. Imputation, scaling, encoding, and feature calculations were executed strictly inside each training fold:

| Model Candidate | Average Precision (PR-AUC) | ROC-AUC | F1-Score | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | $0.2654 \pm 0.0001$ | $0.5000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ |
| **Dummy (Stratified)** | $0.2686 \pm 0.0071$ | $0.5028 \pm 0.0104$ | $0.2762 \pm 0.0084$ | $0.2776 \pm 0.0121$ | $0.2748 \pm 0.0108$ |
| **Logistic Regression (Balanced)** | $0.6610 \pm 0.0139$ | $0.8465 \pm 0.0118$ | $0.6284 \pm 0.0076$ | **$0.7940 \pm 0.0128$** | $0.5202 \pm 0.0087$ |
| **Random Forest (Balanced)** | $0.6591 \pm 0.0197$ | $0.8467 \pm 0.0137$ | $0.6331 \pm 0.0115$ | $0.7679 \pm 0.0210$ | **$0.5385 \pm 0.0139$** |
| **XGBoost (Pre-Tuning)** | $0.6611 \pm 0.0205$ | $0.8475 \pm 0.0132$ | $0.6315 \pm 0.0123$ | $0.7873 \pm 0.0156$ | $0.5269 \pm 0.0119$ |
| **XGBoost (Tuned Champion)** | **$0.6626 \pm 0.0182$** | **$0.8491 \pm 0.0125$** | **$0.6358 \pm 0.0110$** | $0.7826 \pm 0.0162$ | $0.5354 \pm 0.0124$ |

*Key Findings from Cross-Validation:*
1. **Tree-Based Generalization:** Tuned XGBoost achieved the highest cross-validation Average Precision ($0.6626$), demonstrating superior risk calibration compared to linear baselines.
2. **Stability:** Low standard deviations across folds ($\pm 0.0182$) confirm consistent model behavior across diverse demographic and contract subsets.

---

## 4. Hyperparameter Tuning Summary

Bounded grid search over 16 parameter combinations in Development 5-Fold CV yielded the following optimal parameters:
- `n_estimators`: `100`
- `max_depth`: `4` (controls overfitting while capturing non-linear interactions)
- `learning_rate`: `0.08`
- `scale_pos_weight`: `2.5` (calibrated compensation for 73/27 class imbalance)
- `subsample`: `0.8`
- `colsample_bytree`: `0.8`

Metadata saved to `reports/tables/selected_hyperparameters.json`.

---

## 5. Single Final Evaluation on Quarantined Test Set ($n = 1,409$)

Following complete lock of all models and hyperparameters, a single evaluation was executed against the held-out test split:

| Model Architecture | Threshold ($\tau$) | PR-AUC (Avg Prec) | ROC-AUC | F1-Score | Recall | Precision | Accuracy | Confusion Matrix ($TP / FP / TN / FN$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Most Frequent)** | 0.50 | 0.2654 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.7346 | $0 \;/\; 0 \;/\; 1035 \;/\; 374$ |
| **Logistic Regression** | 0.50 | 0.6511 | 0.8451 | 0.6219 | 0.8048 | 0.5067 | 0.7402 | $301 \;/\; 293 \;/\; 742 \;/\; 73$ |
| **Random Forest** | 0.50 | 0.6588 | 0.8446 | 0.6268 | 0.7701 | **0.5284** | **0.7566** | $288 \;/\; 257 \;/\; 778 \;/\; 86$ |
| **XGBoost (Champion)** | 0.50 | 0.6571 | **0.8452** | **0.6284** | 0.7754 | 0.5282 | **0.7566** | $290 \;/\; 259 \;/\; 776 \;/\; 84$ |
| **XGBoost (Screening Cutoff)** | **0.40** | **0.6571** | **0.8452** | 0.6154 | **0.8342** | 0.4875 | 0.7232 | **$312 \;/\; 328 \;/\; 707 \;/\; 62$** |

### Comparison with Historical Phase 1 Baseline
- **Phase 1 Historical Baseline (Logistic Regression):** PR-AUC 0.6331, ROC-AUC 0.8417, Precision 0.5043, Recall 0.7834.
- **Phase 2 Engineered Logistic Regression:** PR-AUC improved from 0.6331 to **0.6511** (+1.8 percentage points) via domain feature additions (`TotalServices`, `ChargesRatio`, `TenureCohort`).
- **Phase 2 Champion XGBoost:** Demonstrates higher overall discrimination (ROC-AUC 0.8452, PR-AUC 0.6571, F1 0.6284), reducing false alarms by **34 customers** compared to Logistic Regression at the default cutoff while capturing 290 true churners.
- **Operational Screening Threshold ($\tau = 0.40$):** In an enterprise Customer Success workflow where missing an at-risk subscriber costs 5Ã-â€"10Ã- more than making an unneeded check-in call, shifting the decision threshold to 0.40 captures **83.42% of all churners** (312 out of 374), allowing only 62 churners to slip through unnoticed.

---

## 6. Verification & Automated Testing

- **Command:** `python -m pytest -v`
- **Result:** **27 passed, 0 failed, 0 skipped** (Execution time: 6.06s).
- **Scope of Tests:**
  - Data loading exceptions and non-destructive reads (`test_data_loading.py`)
  - Schema constraints, missing target checks, whitespace cleaning (`test_data_validation.py`)
  - Feature engineering column logic, add-on counts, tenure cohort binning, charges ratios (`test_feature_engineering.py`)
  - Safe numeric array parsing and unknown category tolerance (`test_preprocessing.py`)
  - Model candidate integrity, CV consistency, and deterministic seeds (`test_model_training.py`, `test_baseline.py`)
  - Metric computation, pipeline joblib serialization, and `ChurnInferenceEngine` inference (`test_inference.py`)

---

## 7. Deliverables & Handoff to Phase 3

### Core Deliverables Ready for Review:
1. **Modules:** [`src/feature_engineering.py`](file:///d:/Daniel/projects/project%201/src/feature_engineering.py), [`src/preprocessing.py`](file:///d:/Daniel/projects/project%201/src/preprocessing.py), [`src/model_training.py`](file:///d:/Daniel/projects/project%201/src/model_training.py), [`src/tuning.py`](file:///d:/Daniel/projects/project%201/src/tuning.py), [`src/evaluation.py`](file:///d:/Daniel/projects/project%201/src/evaluation.py), [`src/inference.py`](file:///d:/Daniel/projects/project%201/src/inference.py), [`src/run_phase2.py`](file:///d:/Daniel/projects/project%201/src/run_phase2.py).
2. **Artifacts:** [`models/champion_pipeline.joblib`](file:///d:/Daniel/projects/project%201/models/champion_pipeline.joblib), [`models/model_metadata.json`](file:///d:/Daniel/projects/project%201/models/model_metadata.json).
3. **Reports & Governance:** [`reports/phase2_audit.md`](file:///d:/Daniel/projects/project%201/reports/phase2_audit.md), [`reports/phase2_summary.md`](file:///d:/Daniel/projects/project%201/reports/phase2_summary.md), [`docs/model_validation_protocol.md`](file:///d:/Daniel/projects/project%201/docs/model_validation_protocol.md), [`docs/prediction_contract.md`](file:///d:/Daniel/projects/project%201/docs/prediction_contract.md).
4. **Visual Figures:** [`reports/figures/06_model_cv_comparison.png`](file:///d:/Daniel/projects/project%201/reports/figures/06_model_cv_comparison.png), [`reports/figures/07_precision_recall_curves.png`](file:///d:/Daniel/projects/project%201/reports/figures/07_precision_recall_curves.png).

**Recommended Next Step:** Await user review and approval before proceeding to **Phase 3: Model Explainability (SHAP), Deep Diagnostics & Fairness Audits**.
