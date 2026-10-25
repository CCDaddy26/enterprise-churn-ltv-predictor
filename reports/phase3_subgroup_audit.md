# Subgroup Performance & Model Fairness Audit

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Applied Data Scientist & ML Auditor  
**Date:** 2026-10-23  

---

## 1. Purpose & Analytical Guardrails

Machine learning models frequently perform unevenly across customer demographics, contract structures, or infrastructure cohorts. A model that performs well globally can fail silently on specific customer segments.

### Guardrails Against Misleading Conclusions:
1. **Denominators Required on All Metrics:** Sample sizes ($n$) and actual churner counts are reported alongside every percentage.
2. **Prevalence Awareness:** Churn prevalence varies widely by segment (e.g. 42.75% for month-to-month contracts vs. 2.87% for two-year contracts). Group performance differences often reflect base-rate differences rather than algorithmic bias.
3. **No Segment Threshold Manipulation:** Performance is audited using the locked, uniform champion model threshold ($\tau = 0.50$). Artificially varying thresholds to force equal error rates across subgroups masks underlying operational realities.
4. **No Causal Discrimination Inferences:** Disparities across commercial or infrastructure groups do not imply deliberate discrimination or disparate impact under civil rights statutes.

---

## 2. Granular Subgroup Audit Results (Development Split, $n = 5,634$)

Audited using out-of-fold predictions with decision threshold $\tau = 0.50$:

| Feature Category | Subgroup | Total Customers ($n$) | Actual Churners | Base Prevalence | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contract** | Month-to-month | 3,102 | 1,326 | 42.75% | 0.5514 | **0.8582** | 0.6714 | 0.7565 | 0.6928 | Evaluated |
| | One year | 1,173 | 130 | 11.08% | 0.2500 | 0.1308 | 0.1717 | 0.6982 | 0.1890 | Evaluated |
| | Two year | 1,359 | 39 | 2.87% | 0.0000 | 0.0000 | 0.0000 | 0.7234 | 0.1081 | Evaluated (Extreme Imbalance) |
| **Internet Service**| Fiber optic | 2,483 | 1,045 | 42.09% | 0.5809 | **0.8593** | 0.6932 | 0.7915 | 0.7206 | Evaluated |
| | DSL | 1,937 | 362 | 18.69% | 0.4491 | 0.5967 | 0.5125 | 0.8075 | 0.4824 | Evaluated |
| | No Internet | 1,214 | 88 | 7.25% | 0.3832 | 0.4659 | 0.4205 | 0.8258 | 0.3082 | Evaluated |
| **Payment Method** | Electronic check | 1,891 | 865 | 45.74% | 0.5929 | **0.8855** | 0.7102 | 0.7704 | 0.7232 | Evaluated |
| | Mailed check | 1,286 | 248 | 19.28% | 0.4865 | 0.7258 | 0.5825 | 0.8574 | 0.5948 | Evaluated |
| | Bank transfer (auto) | 1,244 | 201 | 16.16% | 0.4542 | 0.5672 | 0.5044 | 0.8310 | 0.5403 | Evaluated |
| | Credit card (auto) | 1,213 | 181 | 14.92% | 0.4299 | 0.5249 | 0.4726 | 0.8166 | 0.4676 | Evaluated |
| **Senior Citizen** | Non-Senior (0) | 4,714 | 1,117 | 23.70% | 0.5269 | 0.7457 | 0.6175 | 0.8483 | 0.6358 | Evaluated |
| | Senior (1) | 920 | 378 | 41.09% | 0.5823 | **0.8519** | 0.6917 | 0.7991 | 0.7175 | Evaluated |
| **Gender** | Female | 2,801 | 746 | 26.63% | 0.5488 | 0.7761 | 0.6430 | 0.8497 | 0.6650 | Evaluated |
| | Male | 2,833 | 749 | 26.44% | 0.5338 | 0.7690 | 0.6302 | 0.8416 | 0.6527 | Evaluated |

*Visual Chart:* `reports/figures/14_subgroup_performance_audit.png`  
*Table Artifact:* `reports/tables/subgroup_performance.csv`

---

## 3. Key Diagnostic Findings

### A. Demographic Parity (Gender & Senior Status)
- **Gender Parity:** The model exhibits virtually identical performance across genders (Female Recall: $77.61\%$, Male Recall: $76.90\%$; ROC-AUC $0.8497$ vs $0.8416$). There is zero evidence of gender-based predictive disparity.
- **Senior Citizen Cohort:** Senior citizens show higher churn prevalence (41.09% vs. 23.70%) and correspondingly higher model recall (85.19% vs. 74.57%), reflecting their concentration in month-to-month contracts.

### B. The Low-Prevalence Contract Failure Mode (Two-Year Contracts)
- **Diagnostic Finding:** For subscribers on Two-Year contracts ($n = 1,359$, but only 39 churners, prevalence = 2.87%), the model flags 0 accounts at $\tau = 0.50$ (Recall = 0.0%).
- **Why This Occurs:** The model's baseline probability output for two-year accounts is pushed strongly downward by `Contract_Two year` ($\text{SHAP} \approx -1.5$). In an extreme low-prevalence setting (2.8%), reaching a 50% probability is mathematically improbable without multiple severe risk signals.
- **Operational Recommendation:** Do not rely on fixed 0.50 cutoffs for two-year contracts. To detect the rare two-year churner, Customer Success must query the **top risk percentile within that contract tier** rather than an absolute global cutoff.
