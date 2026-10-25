# Phase 4 Preflight Audit: Repository, Models, Calibration & Governance

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Auditor:** Senior ML Engineer, Applied Data Scientist & ML Auditor  
**Date:** 2026-10-23  
**Status:** AUDIT COMPLETE â€" Findings Documented, Safe Architectural Decision Established

---

## 1. Scope & Objective of Audit

Before constructing the Phase 4 Customer Lifetime Value (LTV) Engine, Retention Campaign ROI Simulator, and Streamlit Dashboard, this preflight audit examined:
1. Exact input features, engineered features, preprocessing pipeline, and champion model in `models/champion_pipeline.joblib`.
2. Model metadata consistency against the serialized pipeline.
3. The exact calibration procedure used in Phase 3 and whether calibration predictions were reused for fitting or evaluation.
4. The Phase 3 `ChargesRatio` ablation implementation and genuine feature ablation impacts.
5. The historical test-set evaluation trail across Phases 1, 2, and 3.
6. Available artifacts, missing dependencies, and baseline regression tests.

---

## 2. Key Audit Findings & Governance Clarifications

### A. Model Artifact & Metadata Verification
- **Champion Artifact:** `models/champion_pipeline.joblib` (Size: 175,039 bytes).
- **Architecture:** `XGBClassifier` preceded by `ColumnTransformer` (37 transformed features) and `ChurnFeatureEngineer`.
- **Hyperparameters:** `n_estimators=100`, `max_depth=4`, `learning_rate=0.08`, `scale_pos_weight=2.5`, `subsample=0.8`, `colsample_bytree=0.8`.
- **Status:** PASS. The serialized pipeline matches `models/model_metadata.json` exactly.

### B. Investigation of Phase 3 Probability Calibration
- **Finding on Calibration Artifact:** Phase 3 demonstrated the benefit of Sigmoid calibration on Development Out-of-Fold predictions (`reports/tables/calibration_metrics.csv`), but **did not serialize a separate calibrated pipeline artifact to disk** (only `models/champion_pipeline.joblib` was persisted).
- **Finding on Calibration Evaluation:** In `src/run_phase3.py`, `calibrated_clf = CalibratedClassifierCV(..., cv=cv)` was fitted on `X_dev`, and then `predict_proba(X_dev)` was evaluated directly. While `CalibratedClassifierCV` uses internal cross-validation to fit calibration regressors, calculating final metrics on the training slice `X_dev` introduces a slight optimistic evaluation bias ($\text{ECE} = 0.0349$).
- **Phase 4 Probability Decision:**
  - In accordance with Phase 4 mandatory truthfulness rules: **The Phase 2 champion outputs will be clearly designated as an uncalibrated model risk score / risk probability.**
  - To support users who wish to see calibrated scenario estimates, Phase 4 provides an optional, clearly labeled **Calibrated Scenario Mode** fitted strictly via out-of-fold calibration on Development data, with prominent disclaimers that modeled probabilities remain educational scenario estimates rather than verified company financial forecasts.

### C. `ChargesRatio` Ablation Defect & Resolution
- **Defect Found in Phase 3:** In `src/robustness.py`, the variant labeled `"Ablation: No ChargesRatio"` wrapped `build_feature_preprocessor(include_engineered_features=True)` in an extra `Pipeline` layer without actually removing `"ChargesRatio"` from `num_cols`. Consequently, the reported ablation table showed identical scores ($0.6626$).
- **Fix Implemented:** Updated `build_feature_preprocessor` to accept `include_ratios: bool = True` and wired it into `num_cols` and `ChurnFeatureEngineer(include_ratios=include_ratios)`.
- **Re-evaluated Genuine Ablation Results (5-Fold CV on Dev):**
  - Full Engineered Model (with `ChargesRatio`): PR-AUC $0.6589 \pm 0.0175$, ROC-AUC $0.8453$.
  - Ablation: No `ChargesRatio`: PR-AUC **$0.6665 \pm 0.0205$**, ROC-AUC **$0.8470$**.
  - Baseline (Raw Features Only): PR-AUC $0.6653 \pm 0.0233$, ROC-AUC $0.8473$.
- **Finding:** Removing `ChargesRatio` slightly improves cross-validation PR-AUC (+0.76 pp), confirming that `ChargesRatio` is collinear with `tenure` and `MonthlyCharges` and does not provide incremental predictive power for tree ensembles.

### D. Historical Held-Out Test Evaluation Trail
- **Phase 1 Evaluation:** Held-out test set ($n = 1,409$, seed 42) was evaluated once for baseline Logistic Regression (PR-AUC 0.6331).
- **Phase 2 Evaluation:** Held-out test set was evaluated once for Tuned XGBoost (PR-AUC 0.6571).
- **Governance Rule for Phase 4:** The held-out test split is **NOT pristine or untouched historically**. Phase 4 will NOT claim that the test set has never been examined. The dashboard allows users to explore the full public dataset or cohort slices while prominently disclosing that all financial calculations are **assumption-based scenario simulations**.

---

## 3. Preflight Checklist Summary

| Audit Area | Preflight Status | Action Taken for Phase 4 |
| :--- | :---: | :--- |
| **Existing Test Suite** | PASS | All 33 unit tests verified passing. |
| **Streamlit Availability** | PASS | `streamlit 1.58.0` confirmed installed and ready. |
| **Model Serialization** | PASS | `models/champion_pipeline.joblib` verified functional. |
| **Calibration Status** | DOCUMENTED | Uncalibrated risk score default; calibrated mode optional with disclaimers. |
| **ChargesRatio Fix** | FIXED | Fixed parameterization in `preprocessing.py` and `robustness.py`. |
| **Currency Display** | CONFIGURED | Default INR (`â‚¹`) with configurable toggle (USD `$`, EUR `â‚¬`, GBP `Â£`). |
