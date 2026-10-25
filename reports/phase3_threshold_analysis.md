# Operational Decision Threshold & Error Trade-off Report

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Machine Learning Engineer & Business Analytics Consultant  
**Date:** 2026-10-23  

---

## 1. Why Default Cutoffs ($\tau = 0.50$) Fail in Subscription Businesses

In standard binary classification, practitioners often rely on an arbitrary probability cutoff of $\tau = 0.50$. In commercial subscription retention, this default cutoff is rarely optimal because:
1. **Asymmetric Business Costs:** The financial cost of a False Negative (missing a churner who cancels, losing \$65/month in recurring gross margin) is 5Ã- to 10Ã- greater than the cost of a False Positive (Customer Success outreach time ~\$25).
2. **Team Capacity Constraints:** A retention team with 3 Customer Success Managers can only conduct ~250 customer reviews per month. A threshold that flags 1,500 accounts will overwhelm staff, whereas a threshold flagging 100 accounts leaves bandwidth underutilized.
3. **Class Imbalance:** At 26.54% base churn prevalence, an uncalibrated or re-weighted model shifts probability densities, altering the precision-recall trade-off.

---

## 2. Threshold Spectrum Evaluation (Development Partition, $n = 5,634$)

Using 5-fold cross-validated predictions on the Development partition, we evaluated the operational trade-offs across decision thresholds:

| Threshold ($\tau$) | Precision | Recall | F1-Score | Customers Flagged ($n$) | % of Population Flagged | True Churners Identified ($TP$) | False Alarms ($FP$) | Missed Churners ($FN$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.30** | 0.4555 | **89.36%** | 0.6034 | 2,933 | 52.06% | **1,336** | 1,597 | **159** |
| **0.40** | 0.4970 | **83.95%** | 0.6244 | 2,525 | 44.82% | **1,255** | 1,270 | **240** |
| **0.50 (Default)** | 0.5412 | 77.26% | **0.6365** | 2,134 | 37.88% | 1,155 | 979 | 340 |
| **0.60** | **0.5967** | 66.22% | 0.6278 | 1,659 | 29.45% | 990 | 669 | 505 |
| **0.70** | **0.6586** | 52.51% | 0.5843 | 1,192 | 21.16% | 785 | 407 | 710 |

*Full data available in:* `reports/tables/threshold_tradeoff_metrics.csv`  
*Visual Chart:* `reports/figures/13_threshold_tradeoffs.png`

---

## 3. Operational Persona & Strategy Matching

Rather than prescribing a single arbitrary threshold, Northstar leadership can align thresholds with operational personas:

### Strategy A: Comprehensive Risk Screening ($\tau = 0.40$)
- **Best For:** Automated digital retention workflows, onboarding email sequences, in-app health check prompts, or high-value enterprise accounts where retention ROI is heavily skewed toward saving customers.
- **Performance:** Captures **83.95% of all churners** (1,255 out of 1,495), allowing only 240 churners to slip through.
- **Operational Burden:** Flags 44.8% of the customer base with a precision of 49.7%.

### Strategy B: Balanced Operations ($\tau = 0.50$)
- **Best For:** Standard Customer Success team cadences with moderate staffing.
- **Performance:** Captures **77.26% of churners** while achieving peak F1-score (0.6365) and reducing false alarms by nearly 300 accounts compared to $\tau = 0.40$.

### Strategy C: High-Touch White-Glove Prioritization ($\tau = 0.65$)
- **Best For:** Dedicated executive sponsor outreach or limited Customer Success bandwidth where only highest-certainty risk accounts can receive manual calls.
- **Performance:** Achieves **62.94% precision**, minimizing unneeded staff contact.
