# Machine Learning Prediction Contract & Feature Governance

## 1. Problem Formulation & Operational Scope

This Prediction Contract formally specifies the operational unit of prediction, target semantics, operational prediction timing, business actions, utility/cost trade-offs, and an explicit **Feature Inclusion / Exclusion Decision Table**.

```
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
| PREDICTION CONTRACT SPECIFICATION                                           |
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
| Prediction Unit     : Individual Customer Subscription Account              |
| Target Variable     : Churn (Binary: 1 = Cancelled, 0 = Active/Retained)    |
| Operational Timing  : 14 to 30 days prior to contract/billing renewal       |
| Output Modality     : Calibrated Churn Probability P(Churn = 1 | X)         |
| Primary Action      : Prioritised Customer Success Outreach & Support Check |
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
```

---

## 2. Prediction Unit & Target Definition

- **Prediction Unit:** An active customer account identified by a unique `customerID`.
- **Target Definition:** A binary indicator representing whether the customer terminates their subscription service (`"Yes"` = 1) or remains an active subscriber (`"No"` = 0).
- **Target Mapping:**
  - Positive Class ($y = 1$): Customer churned (`Churn == "Yes"`).
  - Negative Class ($y = 0$): Customer remained active (`Churn == "No"`).
- **Retrospective Snapshot Disclaimer:** The dataset represents a single retrospective cross-sectional slice without sequential timestamp logs. This project is formally designated as a **retrospective churn classification study**, not an empirical out-of-time forecasting system.

---

## 3. Intended Operational Prediction Moment & Feature Availability

In live production operations:
1. **Prediction Moment ($t_0$):** Exactly 14 to 30 days prior to renewal (annual contracts) or 7 days prior to monthly invoice generation (month-to-month contracts).
2. **Information State at $t_0$:**
   - Account tenure up to $t_0$.
   - Contract type, payment method, paperless billing status.
   - Active service modules (Internet, Security, Tech Support, Cloud Backup, Streaming).
   - Invoiced monthly billing charges up to $t_0$.
3. **Data Strictly Prohibited at $t_0$:**
   - Any usage or billing events occurring after $t_0$.
   - Cancellation survey comments or cancellation transaction logs.
   - Post-churn payment attempts or delinquency records resulting from cancellation.

---

## 4. Feature Inclusion / Exclusion Decision Table

Every candidate feature from the raw schema and every proposed engineered feature is evaluated for leakage, prediction-time availability, and business validity:

| Feature Name | Raw Data Type | Proposed Treatment | Business Interpretation | Prediction-Time Availability | Leakage / Validity Concern | Inclusion Decision & Justification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `customerID` | `object` | Excluded | Unique customer alphanumeric identifier | Available | **Severe Leakage Risk:** High-cardinality identifier causes spurious memorization. | **EXCLUDE.** Dropped from predictor matrix $X$. |
| `gender` | `object` | One-Hot Encoded (`drop="first"`) | Customer demographic gender | Available at signup | Demographic parity check needed; no direct target leakage. | **INCLUDE.** Baseline demographic control. |
| `SeniorCitizen` | `int64` | Categorical Factor (One-Hot) | Customer age $\ge 65$ indicator | Available at signup | None; static demographic factor. | **INCLUDE.** Useful demographic segmentation. |
| `Partner` | `object` | One-Hot Encoded (`drop="first"`) | Spouse / partner status | Available at signup | None; static household factor. | **INCLUDE.** Baseline demographic factor. |
| `Dependents` | `object` | One-Hot Encoded (`drop="first"`) | Presence of dependents | Available at signup | None; static household factor. | **INCLUDE.** Baseline demographic factor. |
| `tenure` | `int64` | Continuous (Standard Scaled / Tree split) | Months subscribed to service | Available at $t_0$ | None; historical duration prior to prediction. | **INCLUDE.** Primary continuous retention predictor. |
| `PhoneService` | `object` | One-Hot Encoded (`drop="first"`) | Voice telephony subscription | Available at $t_0$ | None; active subscription status at $t_0$. | **INCLUDE.** Core service indicator. |
| `MultipleLines` | `object` | One-Hot Encoded | Multiple voice lines | Available at $t_0$ | None; active subscription status at $t_0$. | **INCLUDE.** Core service indicator. |
| `InternetService` | `object` | One-Hot Encoded | Internet connection architecture (DSL, Fiber, None) | Available at $t_0$ | None; active infrastructure type at $t_0$. | **INCLUDE.** Strong correlation with churn in EDA; core product feature. |
| `OnlineSecurity` | `object` | One-Hot Encoded | Cyber-security add-on module | Available at $t_0$ | None; active add-on status at $t_0$. | **INCLUDE.** Add-on protection indicator. |
| `OnlineBackup` | `object` | One-Hot Encoded | Cloud backup add-on module | Available at $t_0$ | None; active add-on status at $t_0$. | **INCLUDE.** Add-on protection indicator. |
| `DeviceProtection` | `object` | One-Hot Encoded | Hardware warranty add-on | Available at $t_0$ | None; active add-on status at $t_0$. | **INCLUDE.** Add-on protection indicator. |
| `TechSupport` | `object` | One-Hot Encoded | Dedicated tech support tier | Available at $t_0$ | None; active add-on status at $t_0$. | **INCLUDE.** High business relevance for retention. |
| `StreamingTV` | `object` | One-Hot Encoded | TV streaming bundle | Available at $t_0$ | None; active subscription tier at $t_0$. | **INCLUDE.** Content bundle indicator. |
| `StreamingMovies` | `object` | One-Hot Encoded | Movie streaming bundle | Available at $t_0$ | None; active subscription tier at $t_0$. | **INCLUDE.** Content bundle indicator. |
| `Contract` | `object` | One-Hot Encoded | Contract commitment term (Month-to-month, 1-yr, 2-yr) | Available at $t_0$ | Observational correlation vs. causation: must not be treated as causal. | **INCLUDE.** Major predictive feature for risk ranking. |
| `PaperlessBilling` | `object` | One-Hot Encoded (`drop="first"`) | Electronic invoicing preference | Available at $t_0$ | None; billing configuration active at $t_0$. | **INCLUDE.** Billing preference factor. |
| `PaymentMethod` | `object` | One-Hot Encoded | Payment channel (Electronic check, Mailed, Auto) | Available at $t_0$ | None; payment channel on file at $t_0$. | **INCLUDE.** Operational retention indicator. |
| `MonthlyCharges` | `float64` | Continuous (Standard Scaled) | Current invoiced monthly rate | Available at $t_0$ | None; current pricing plan at $t_0$. | **INCLUDE.** Financial commitment feature. |
| `TotalCharges` | `object` | Numeric Cleaned (Whitespace $\to$ 0.0, Scaled) | Cumulative historical revenue | Available at $t_0$ | Highly collinear with `tenure * MonthlyCharges`; contains 11 blanks for `tenure=0`. | **INCLUDE (with sensitivity check).** Converted to float; blank strings filled with 0.0. |
| **Engineered:** `TotalServices` | `int64` | Discrete Count (Scaled) | Count of active security, backup, protection, support, streaming add-ons | Available at $t_0$ | Computed strictly from row-level service fields; no target leakage. | **INCLUDE.** Captures product stickiness / integration depth. |
| **Engineered:** `TenureCohort` | `category` | One-Hot Encoded | Binned tenure brackets (0-6m, 7-12m, 13-24m, 25-48m, 49m+) | Available at $t_0$ | Fixed predetermined boundaries; no target leakage. | **INCLUDE.** Captures non-linear hazard curves (onboarding cliff). |
| **Engineered:** `HasPartnerAndDependents` | `int64` | Binary (0/1) | Family stability indicator | Available at $t_0$ | Computed strictly from demographic fields; no target leakage. | **INCLUDE.** Household stability feature. |
| **Engineered:** `ChargesRatio` | `float64` | Continuous (Scaled) | `TotalCharges / (tenure * MonthlyCharges + 1e-6)` | Available at $t_0$ | Ratio approximating historical price changes/discounts; no target leakage. | **INCLUDE.** Captures historical billing variance. |

---

## 5. Economic Utility Matrix & False-Positive vs. False-Negative Trade-offs

```
                            ACTUAL CLASS
                     Churn (1)        Retained (0)
                ‚"å‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¨‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"ê
  Predicted (1) ‚"Ç  True Positive   ‚"Ç  False Positive  ‚"Ç
                ‚"Ç     (Benefit)    ‚"Ç  (Cost: $c_1)    ‚"Ç
DECISION        ‚"ú‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"º‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"§
  Predicted (0) ‚"Ç  False Negative  ‚"Ç  True Negative   ‚"Ç
                ‚"Ç  (Cost: $c_2)    ‚"Ç    (Neutral)     ‚"Ç
                ‚""‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¥‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"ò
```

- **False Positive (Type I Error):** Model flags customer as at-risk; customer intended to remain.
  - *Operational Cost ($c_1$):* Direct outreach cost (Customer Success outreach time ~\$25) plus potential margin erosion if unnecessary discounts are offered.
- **False Negative (Type II Error):** Model fails to flag at-risk customer; customer cancels unaddressed.
  - *Operational Cost ($c_2$):* Lost remaining customer lifetime gross margin (average \$64.76/month over remaining lifetime).
- **Asymmetric Objective:** In recurring subscription SaaS, missing an at-risk customer is typically 5√- to 10√- more expensive than reviewing an extra healthy customer. Thus, high recall and calibrated ranking (PR-AUC / Average Precision) are prioritized over pure accuracy.
