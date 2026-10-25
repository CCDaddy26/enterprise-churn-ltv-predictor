# Phase 3 Summary: SHAP Explainability, Deep Diagnostics & Subgroup Audits
**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Applied Data Scientist, ML Auditor & Business Analytics Consultant  
**Date:** 2026-10-23  
**Status:** PASS â€" Full Model Explainability, Reliability, Diagnostics & Audits Completed  

---

## 1. Executive Summary & Diagnostic Scope

Phase 3 transitions the Phase 2 machine learning pipeline from a high-performing predictive model into a fully auditable, explainable, and operationally grounded retention decision system.

Using the fitted **Tuned XGBoost Champion** from Phase 2, Phase 3 executed:
1. **Model & Pipeline Audit:** Verified artifact consistency, schema definitions, and numerical stability of engineered features (`TotalServices`, `TenureCohort`, `ChargesRatio`).
2. **SHAP TreeExplainer Analysis:** Generated global feature rankings, beeswarm distributions, dependence curves, and individual customer waterfall plots for 4 operational archetypes (True Positive, True Negative, False Positive, False Negative). Customer IDs are completely excluded from exported reports.
3. **Probability Calibration Assessment:** Demonstrated that the uncalibrated champion's re-weighted training creates an overconfidence bias ($\text{ECE} = 0.1152$), and proved that Sigmoid calibration reduces Expected Calibration Error by **70%** ($\text{ECE} = 0.0349$).
4. **Operational Decision Threshold Analysis:** Analyzed the full trade-off spectrum ($\tau \in [0.10, 0.90]$), comparing the default 0.50 cutoff against an aggressive screening threshold ($\tau = 0.40$, capturing **83.95%** of churners).
5. **Granular Subgroup Fairness Audit:** Audited model recall, precision, and AUCs across Contract, Internet Service, Payment Method, Senior Status, and Gender with explicit group denominators. Verified gender demographic parity while diagnosing the low-prevalence failure mode on Two-Year contracts.
6. **Error Analysis & Failure Modes:** Profiled False Positives ("Fragile Loyalists") and False Negatives ("Silent Attrition" among tenured multi-year accounts).
7. **Feature Ablation & Robustness:** Proved that while engineered features improved linear models by +1.8 pp in Phase 2, gradient boosted trees discover non-linear relationships directly, rendering `ChargesRatio` predictive-neutral.

---

## 2. Key Diagnostic Findings

### A. SHAP Global Feature Importance
The top 6 features driving churn risk across 5,634 development records are:
1. `Contract_Two year` (Mean $|SHAP| = 0.5686$): Strongest protective factor against churn.
2. `tenure` (Mean $|SHAP| = 0.4437$): Steep non-linear risk decay validating the 0â€"6 month onboarding cliff.
3. `InternetService_Fiber optic` (Mean $|SHAP| = 0.3262$): Strong positive risk factor driven by high pricing and service friction.
4. `Contract_One year` (Mean $|SHAP| = 0.2507$): Moderate protective factor.
5. `PaymentMethod_Electronic check` (Mean $|SHAP| = 0.2057$): Positive push to churn log-odds due to lack of payment automation.
6. `MonthlyCharges` (Mean $|SHAP| = 0.1828$): Direct price sensitivity.

### B. Probability Calibration & Reliability
- **Uncalibrated Model:** Brier Score $0.1542$, ECE $0.1152$. Tends to overpredict probability in middle bins due to class re-weighting (`scale_pos_weight = 2.5`).
- **Sigmoid Calibrated Model:** Brier Score **$0.1164$**, ECE **$0.0349$**. Calibrated probabilities reflect empirical event rates within 3.5%, establishing the required mathematical foundation for Phase 4 LTV calculations.

### C. Operational Threshold Spectrum
- **Default Threshold ($\tau = 0.50$):** Precision 54.12%, Recall 77.26%, F1-Score 0.6365, Flags 2,134 accounts (37.9% of customer base).
- **High-Recall Screening Threshold ($\tau = 0.40$):** Precision 49.70%, Recall **83.95%**, Flags 2,525 accounts (44.8% of customer base), catching **1,255 true churners**.

### D. Subgroup Audit & Limitations
- **Gender Parity:** Identical recall across Female ($77.61\%$) and Male ($76.90\%$) subscribers.
- **Two-Year Contract Failure Mode:** Zero churners flagged at $\tau = 0.50$ due to extreme low prevalence (2.87%). Operational teams must query top risk percentiles within this tier rather than relying on global cutoffs.

### E. Error Profile Summary
- **False Positives:** Fragile loyalists with short tenure (median 12 mos) on month-to-month terms. They represent valid high-risk retention targets.
- **False Negatives:** Tenured accounts (mean tenure 36.5 mos) holding multi-year contracts who cancelled due to external business shocks (executive turnover, support escalations) not captured in static demographic data.

---

## 3. Test Suite & Deliverables Verification
- **Test Suite Status:** **33 automated unit tests executed, 33 passed, 0 failed** (`python -m pytest -v`, runtime 11.0s).
- **Reports Generated:**
  - [`reports/phase3_audit.md`](file:///d:/Daniel/projects/project%201/reports/phase3_audit.md)
  - [`reports/phase3_feature_interpretation.md`](file:///d:/Daniel/projects/project%201/reports/phase3_feature_interpretation.md)
  - [`reports/phase3_calibration.md`](file:///d:/Daniel/projects/project%201/reports/phase3_calibration.md)
  - [`reports/phase3_threshold_analysis.md`](file:///d:/Daniel/projects/project%201/reports/phase3_threshold_analysis.md)
  - [`reports/phase3_subgroup_audit.md`](file:///d:/Daniel/projects/project%201/reports/phase3_subgroup_audit.md)
  - [`reports/phase3_error_analysis.md`](file:///d:/Daniel/projects/project%201/reports/phase3_error_analysis.md)
  - [`reports/phase3_robustness.md`](file:///d:/Daniel/projects/project%201/reports/phase3_robustness.md)
  - [`reports/phase3_summary.md`](file:///d:/Daniel/projects/project%201/reports/phase3_summary.md)
- **Visual Artifacts:** 9 high-resolution diagnostic charts in `reports/figures/` (figures 08 through 16).
- **Structured Tables:** 6 auditable CSV/JSON tables in `reports/tables/`.

**Recommended Next Step:** Await stakeholder review and approval before proceeding to **Phase 4: Customer Lifetime Value (LTV) Engine & Retention ROI Simulator**.
