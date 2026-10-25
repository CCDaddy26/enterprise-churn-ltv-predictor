# Model Error Analysis & Failure Mode Diagnostics

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Machine Learning Engineer & Diagnostic Specialist  
**Date:** 2026-10-23  

---

## 1. Executive Summary & Error Archetypes

In evaluating a machine learning classifier, aggregate summary metrics obscure why and where the model fails. To understand operational vulnerability, this report diagnoses actual errors produced by the champion XGBoost model on the Development partition ($n = 5,634$, 5-Fold OOF predictions at threshold $\tau = 0.50$):

- **True Positives ($TP = 1,155$):** Accurately flagged churners (Capture rate = $77.26\%$).
- **True Negatives ($TN = 3,160$):** Accurately identified stable accounts.
- **False Positives ($FP = 979$):** False alarms (Customers predicted to cancel who remained active).
- **False Negatives ($FN = 340$):** Missed churners (Subscribers who churned without being flagged).

*Visual Diagnostic Asset:* `reports/figures/15_error_diagnostics.png`  
*Underlying Summary Data:* `reports/tables/error_diagnostic_summary.json`

---

## 2. Comparative Profile: False Alarms vs. Missed Churners

| Dimension / Metric | False Positives ($FP = 979$) | False Negatives ($FN = 340$) | True Positives ($TP = 1,155$) | True Negatives ($TN = 3,160$) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Tenure** | **18.55 months** | **36.46 months** | 13.03 months | 43.49 months |
| **Median Tenure** | **12.00 months** | **38.00 months** | 7.00 months | 47.00 months |
| **Mean Monthly Charges** | \$72.91 | \$68.45 | \$76.75 | \$57.76 |
| **Mean Total Services** | 1.74 add-ons | **2.48 add-ons** | 1.60 add-ons | 2.28 add-ons |
| **Month-to-Month Contract** | **94.59%** | 55.29% | 98.53% | 26.90% |
| **One-Year Contract** | 5.21% | **33.24%** | 1.47% | 31.39% |
| **Two-Year Contract** | 0.20% | **11.47%** | 0.00% | 41.71% |
| **Fiber Optic Internet** | **66.19%** | 43.24% | 77.75% | 25.00% |
| **Electronic Check Payment**| **53.73%** | 29.12% | 66.32% | 15.82% |

---

## 3. Detailed Failure Mode Diagnostics

### Failure Mode 1: The "Fragile Loyal" False Alarm (False Positives)
- **Profile:** Median tenure of 12 months, paying via electronic check (53.7%), month-to-month contract (94.6%), and subscribing to Fiber Optic internet (66.2%).
- **Why the Model Errs:** Statistically, these accounts look identical to churners: they have high monthly charges, flexible terms, and frictionless cancellation capability. However, they remained active.
- **Business Interpretation:** These customers are not "safe". They are highly fragile loyalists who share all observable risk characteristics of churners. Contacting them with white-glove onboarding or an annual discount incentive is low-risk and commercially sound.

### Failure Mode 2: The "Silent Attrition" Missed Churner (False Negatives)
- **Profile:** Long-tenure accounts (Mean tenure: **36.5 months**), holding multi-year contracts (44.7% on 1-year or 2-year contracts), with multiple service add-ons (Mean: **2.48 services**).
- **Why the Model Errs:** The model heavily weights tenure and multi-year contracts as protective anchors. When an account with 3 years of tenure and a two-year contract decides to cancel, the model's static features suppress its predicted churn probability ($\hat{p} \le 0.35$).
- **Root Cause & Missing Data:** In enterprise SaaS, tenured accounts rarely cancel due to basic pricing or contract term friction. They cancel due to **sudden external shocks**: executive sponsor turnover, corporate acquisitions, budget freezes, or unresolved high-severity technical support tickets. None of these behavioral telemetry streams are captured in a static cross-sectional dataset.
- **Actionable Recommendation for Business Team:** Do not rely on static churn models to detect tenured enterprise churn. Implement dynamic operational triggers (e.g. executive sponsor email inactivity, support ticket escalations $\ge 3$) to monitor tenured accounts.
