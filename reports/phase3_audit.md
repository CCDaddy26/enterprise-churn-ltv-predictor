# Phase 3 Audit: Model Artifact, Feature Pipeline & Governance Review

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Auditor:** Senior Applied Data Scientist & ML Auditor  
**Date:** 2026-10-23  
**Status:** AUDIT PASSED â€" Model Artifact & Governance Verified

---

## 1. Scope & Objective of Audit

Before implementing SHAP explainability, probability calibration, threshold optimization, and subgroup audits, this audit thoroughly inspected:
- The serialized model artifact (`models/champion_pipeline.joblib`) and metadata (`models/model_metadata.json`).
- Consistency of feature transformations between training and inference.
- Mathematical validity, stability, and prediction-time availability of engineered features (`TotalServices`, `TenureCohort`, `HasPartnerAndDependents`, `ChargesRatio`).
- Integrity of data partitioning and test-set quarantine.
- Test suite status (all 27 baseline unit tests passing).

---

## 2. Model Artifact & Pipeline Verification

### A. Artifact Verification
- **Serialized Model File:** `models/champion_pipeline.joblib` (Size: ~367 KB).
- **Classifier Class:** `xgboost.sklearn.XGBClassifier`.
- **Pipeline Architecture:**
  1. `prep`: `Pipeline(steps=[('feature_engineer', ChurnFeatureEngineer), ('preprocessor', ColumnTransformer)])`
  2. `classifier`: `XGBClassifier` with hyperparameters matching `models/model_metadata.json`:
     - `n_estimators = 100`
     - `max_depth = 4`
     - `learning_rate = 0.08`
     - `scale_pos_weight = 2.5`
     - `subsample = 0.8`
     - `colsample_bytree = 0.8`
     - `eval_metric = 'logloss'`
- **Transformed Feature Space:** 37 distinct features generated from 6 continuous/discrete numeric variables and 31 one-hot encoded indicator variables.

---

## 3. Deep-Dive Audit of Engineered Features

| Feature Name | Formulation | Prediction-Time Availability | Numerical Stability & Zero-Tenure Check | Information Value / Collinearity Audit |
| :--- | :--- | :--- | :--- | :--- |
| **`TotalServices`** | $\sum \mathbb{I}(\text{Service} == \text{'Yes'})$ across 6 add-ons | Fully available at $t_0$ from active account flags | Completely stable; discrete integer $0 \le k \le 6$ | Highly informative; measures service adoption and switching barrier. |
| **`TenureCohort`** | Binned brackets (`0-6m`, `7-12m`, `13-24m`, `25-48m`, `49m+`) | Fully available at $t_0$ | Fully stable; predetermined static bins | Captures the steep early onboarding hazard cliff. |
| **`HasPartnerAndDependents`** | Binary boolean $(\text{Partner}==\text{'Yes'}) \land (\text{Dependents}==\text{'Yes'})$ | Available at signup | Fully stable binary flag (0 or 1) | Moderately informative household stability indicator. |
| **`ChargesRatio`** | $\frac{\text{TotalCharges}}{\text{tenure} \times \text{MonthlyCharges} + \epsilon}$ | TotalCharges and MonthlyCharges on file | Safely bounded: when tenure is 0, defaults to $1.0$; clipped to $[0.1, 3.0]$ | **Audit Finding:** Captures cumulative billing deviations (promotions, expansion). Moderately collinear with tenure. Subjected to feature ablation in Phase 3. |

### ChargeRatio Numerical Stability Findings:
- At `tenure = 0`, `TotalCharges` is 0.0 and expected billing is 0.0. The transformer explicitly intercepts zero-tenure rows and assigns `ChargesRatio = 1.0`, avoiding NaN or division-by-zero artifacts.
- Extreme ratio values resulting from billing adjustments are constrained between $0.1$ and $3.0$.

---

## 4. Test-Set Governance Confirmation

- **Development Partition ($n = 5,634$, 80%):** All SHAP explainer fitting, out-of-fold calibration curves, threshold simulations, subgroup audits, and sensitivity checks are performed **strictly within the Development partition**.
- **Quarantined Final Test Set ($n = 1,409$, 20%):** Kept strictly quarantined and untouched during Phase 3 diagnostic analysis.
- **Customer Identifier Protection:** Customer IDs are excluded from analytical outputs, SHAP tables, and customer explanation examples to prevent personal data exposure.
