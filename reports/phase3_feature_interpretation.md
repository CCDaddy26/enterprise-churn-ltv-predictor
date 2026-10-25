# Feature Importance & Business Interpretation Report

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Applied Data Scientist & Business Analytics Consultant  
**Date:** 2026-10-23  

---

## 1. Executive Summary & Core Conceptual Foundation

Understanding what drives model risk predictions is essential before retention teams commit customer success budget. However, enterprise leaders frequently confuse **statistical feature association** with **business causality**.

### Crucial Analytical Distinctions:
1. **SHAP Measures Model Reliance, Not Ground Truth:** SHAP (SHapley Additive exPlanations) values quantify how much each feature moves the model's prediction relative to the baseline expectation. They explain the algorithm's decision logic, not physical causal laws.
2. **Correlation Does Not Establish Intervention Lift:** A strong association between Month-to-Month contracts and churn ($r = 0.43$) does not prove that forcing customers into annual contracts will make them stay. Customers planning to leave naturally self-select into flexible agreements.
3. **High Risk $\neq$ Guaranteed Churn:** A customer scored with $\hat{p} = 0.85$ has an elevated statistical likelihood of attrition based on historical patterns, but may remain organically if their immediate friction points resolve.

---

## 2. In-Depth Interpretation of Key Predictive Drivers

Based on SHAP TreeExplainer analysis of the champion XGBoost model on the Development partition ($n = 5,634$), the following 6 features exert the greatest influence on predicted churn risk:

```
+â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€+
| TOP 6 PREDICTIVE DRIVERS BY MEAN ABSOLUTE SHAP IMPACT                       |
+â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€+
| 1. Contract_Two year         : Mean |SHAP| = 0.5686 (Strong negative push)  |
| 2. tenure                    : Mean |SHAP| = 0.4437 (Non-linear decay)      |
| 3. InternetService_Fiber optic: Mean |SHAP| = 0.3262 (Elevated risk push)   |
| 4. Contract_One year         : Mean |SHAP| = 0.2507 (Moderate negative push)|
| 5. PaymentMethod_Electronic  : Mean |SHAP| = 0.2057 (Elevated risk push)   |
| 6. MonthlyCharges            : Mean |SHAP| = 0.1828 (Direct pricing effect) |
+â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€+
```

### Feature 1: `Contract_Two year` & `Contract_One year`
- **Source:** Raw Schema (One-Hot Encoded).
- **Model Usage:** Presence of a two-year contract ($= 1$) delivers the single largest negative push to churn log-odds ($\text{SHAP} \approx -1.2$ to $-1.8$), drastically reducing predicted churn probability.
- **Business Interpretation & Caveat:** Multi-year agreements reflect commitment, budget pre-allocation, or high switching costs. However, customer success leadership must not view contract lock-in as a retention panacea. Mandating long-term commitments for onboarding customers could depress top-of-funnel conversion.
- **Recommended Business Investigation:** Interview month-to-month customers who declined annual discounts to uncover underlying product hesitation or trial-use sentiment.

### Feature 2: `tenure` (Customer Lifetime in Months)
- **Source:** Raw Schema (Continuous).
- **Model Usage:** Non-linear inverse relationship. As visualized in `reports/figures/11_shap_dependence_panels.png`, SHAP values are strongly positive for `tenure < 6` months ($\text{SHAP} \approx +0.6$ to $+0.9$), cross zero at month 15, and become strongly protective ($\text{SHAP} \approx -0.8$) beyond month 36.
- **Business Interpretation:** Validates the **"0â€"6 Month Onboarding Cliff"**. Subscribers who fail to integrate product workflows during early onboarding churn at extreme rates. Once an account reaches maturity (2+ years), retention stabilizes.
- **Recommended Business Investigation:** Audit day 1 to day 60 onboarding friction, time-to-first-value (TTFV), and active feature configuration.

### Feature 3: `InternetService_Fiber optic`
- **Source:** Raw Schema (One-Hot Encoded).
- **Model Usage:** Presence of Fiber Optic service ($= 1$) consistently adds $+0.4$ to $+0.8$ to churn log-odds.
- **Business Interpretation:** Fiber optic accounts represent high-ARPU subscribers (average monthly bill \$91.50 vs \$58.10 for DSL). Their elevated churn propensity indicates dissatisfaction with service reliability, unexpected pricing step-ups, or aggressive competitor poaching.
- **Recommended Business Investigation:** Coordinate with network operations and tier-2 support to investigate fiber line stability, latency complaints, and promo-expiration shock.

### Feature 4: `PaymentMethod_Electronic check`
- **Source:** Raw Schema (One-Hot Encoded).
- **Model Usage:** Customers paying via electronic check receive a substantial positive push to churn probability ($\text{SHAP} \approx +0.3$ to $+0.6$).
- **Business Interpretation:** Unlike automated credit card and bank transfer autopay, manual electronic check payments require active monthly intervention by the subscriber. Each payment cycle acts as an active reconsider-to-cancel moment.
- **Recommended Business Investigation:** Implement customer success workflows to incentivize migration to automated credit card or ACH autopay (e.g. one-time bill credit).

### Feature 5: `MonthlyCharges`
- **Source:** Raw Schema (Continuous).
- **Model Usage:** Positive association with churn, particularly when coupled with low tenure or absence of tech support add-ons.
- **Business Interpretation:** High monthly bills amplify price sensitivity. If value realization does not match the invoice amount, churn risk accelerates.

### Feature 6: `TotalServices` & `ChargesRatio` (Engineered Features)
- **`TotalServices`:** Mean $|SHAP| = 0.08$. Subscribers with 4+ active add-ons (Security, Backup, Support) exhibit lower churn log-odds. Product integration depth increases organizational switching inertia.
- **`ChargesRatio`:** Mean $|SHAP| = 0.12$. Accounts with ratios significantly below $1.0$ (reflecting promotional discounts that expired) or above $1.0$ (unanticipated fees) show elevated risk sensitivity.
