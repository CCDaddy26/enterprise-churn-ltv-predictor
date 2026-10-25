# Probability Calibration Assessment & Reliability Report

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Applied Data Scientist & ML Auditor  
**Date:** 2026-10-23  

---

## 1. Executive Summary & Calibration Fundamentals

In commercial retention operations, a classification model is not used simply to make hard binary yes/no predictions. Instead, model probabilities directly power:
- Customer Lifetime Value (LTV) weighting ($\text{Expected Loss} = \hat{p} \times \text{LTV}$)
- Retention budget allocation
- Campaign ROI simulations

A model can have strong discrimination (high ROC-AUC) while having severely distorted probabilities (e.g. predicting $0.80$ for an account cohort whose true empirical churn rate is only $40\%$). Evaluating and addressing probability calibration is essential before financial simulation in Phase 4.

---

## 2. Methodology & Leakage Prevention Protocol

1. **Development-Only Assessment:** Probability calibration was assessed using **Out-of-Fold (OOF) cross-validation predictions** strictly on the Development partition ($n = 5,634$, 5-Fold Stratified CV).
2. **Zero Test-Set Contamination:** The quarantined final test partition ($n = 1,409$) was never queried to evaluate, select, or fit probability calibrators.
3. **Metrics Evaluated:**
   - **Brier Score:** Mean squared difference between predicted probability and binary outcome ($y \in \{0, 1\}$). Lower is better.
   - **Log Loss:** Cross-entropy penalizing confident incorrect predictions. Lower is better.
   - **Expected Calibration Error (ECE):** Sample-weighted average gap between mean confidence and empirical accuracy across 10 probability bins.

---

## 3. Empirical Calibration Findings

| Estimator Variant | Brier Score | Log Loss | Expected Calibration Error (ECE) |
| :--- | :---: | :---: | :---: |
| **Uncalibrated XGBoost Champion** | 0.1542 | 0.4634 | **0.1152** (11.5% average miscalibration) |
| **Sigmoid Calibrated XGBoost** | **0.1164** | **0.3650** | **0.0349** (3.5% average miscalibration) |

### Key Diagnostic Observations:
1. **Uncalibrated Model Overconfidence:** As shown in `reports/figures/12_calibration_reliability_diagram.png`, the uncalibrated XGBoost model exhibits an overconfidence skew in middle-to-high risk bins. Accounts predicted with $\hat{p} \approx 0.60$ exhibit an empirical churn rate of approximately $48\%$. This behavior stems from the class re-weighting parameter (`scale_pos_weight = 2.5`), which intentionally boosts positive log-odds to penalize false negatives during training.
2. **Impact of Sigmoid Calibration:** Applying Platt's Sigmoid scaling via `CalibratedClassifierCV` reduces the Expected Calibration Error from **0.1152 to 0.0349** (a **70% reduction** in probability error), aligning predicted probabilities closely with observed churn frequencies.
3. **Recommendation for Phase 4:** While the uncalibrated champion maintains excellent risk-ranking fidelity (PR-AUC 0.6626), economic LTV and retention ROI simulations in Phase 4 should use the sigmoid-calibrated probability engine to ensure financial dollar estimates remain grounded in reality.
