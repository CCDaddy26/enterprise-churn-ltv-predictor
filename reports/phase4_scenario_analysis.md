# Phase 4.1: Scenario & Targeting Strategy Analysis (Corrected)

## 1. Executive Summary
This report presents recalculated, calibrated strategy comparisons and break-even assessments across the 7,043 customer accounts of Northstar SaaS under the Phase 4.1 financial integrity rules.

---

## 2. Targeting Strategy Comparisons

### Mode A: Like-for-Like Cohort Size ($N = 500$ Accounts)
In Mode A, all active strategies evaluate the exact same cohort volume ($N=500$), ensuring direct comparability.
- **Base Assumptions:** Contact Cost = ₹150, Incentive = ₹500 (responders only), Reach = 85%, Response = 40%, Save = 25%, Margin = 70%, Horizon = 12 months.
- **Probability Source:** Cross-fitted Calibrated Churn Probability ($\text{ECE} = 0.0110$).

| Strategy | Targeted Accounts | Expected Prevented Churn | Campaign Cost | Gross Benefit | Net Benefit | Expected ROI | Budget Utilization |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No Campaign Baseline** | 0 | 0.0 | ₹0 | ₹0 | ₹0 | **N/A �" no expenditure** | 0.0% |
| **2. Highest Churn Risk First** | 500 | 31.7 | ₹160,000 | ₹21,815 | -₹138,185 | -86.4% | 80.0% |
| **3. Highest Value at Risk First** | 500 | 30.1 | ₹160,000 | ₹23,775 | -₹136,225 | -85.1% | 80.0% |
| **4. Balanced Risk & Value** | 500 | 31.3 | ₹160,000 | ₹23,348 | -₹136,652 | -85.4% | 80.0% |

#### Analytical Insights from Mode A:
1. **Value-at-Risk Advantage:**
   Targeting by **Highest Value at Risk** produces **+₹1,960 (+9.0%)** higher gross balance-sheet protection than targeting by pure churn probability (₹23,775 vs ₹21,815), confirming the strategic necessity of value-weighting.
2. **Defensible Calibration Difference:**
   Using cross-fitted calibrated probabilities yields 30.1 expected prevented churn events on top VaR accounts (compared to 36.4 with uncalibrated scores), correcting the $17\%$ upward distortion caused by class reweighting.

---

### Mode B: Budget-Maximized Targeting ($N = \lfloor \frac{\text{Budget}}{\text{Unit Cost}} \rfloor = 625$ Accounts)
In Mode B, strategies are allocated maximum accounts permitted under the ₹200,000 budget ceiling:
- Unit Cost: $₹150 + ₹500 \times (0.85 \times 0.40) = ₹320.00$
- Capacity: $\lfloor \frac{200,000}{320} \rfloor = 625$ accounts.

| Strategy | Targeted Accounts | Expected Prevented Churn | Campaign Cost | Gross Benefit | Net Benefit | Expected ROI | Budget Utilization | Remaining Budget |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. No Campaign Baseline** | 0 | 0.0 | ₹0 | ₹0 | ₹0 | **N/A �" no expenditure** | 0.0% | ₹200,000 |
| **2. Highest Churn Risk (Budget-Max)** | 625 | 39.0 | ₹200,000 | ₹26,697 | -₹173,303 | -86.7% | 100.0% | ₹0 |
| **3. Highest Value at Risk (Budget-Max)** | 625 | 37.2 | ₹200,000 | ₹28,911 | -₹171,089 | -85.5% | 100.0% | ₹0 |
| **4. Balanced Strategy (Budget-Max)** | 625 | 38.5 | ₹200,000 | ₹28,500 | -₹171,500 | -85.8% | 100.0% | ₹0 |

---

## 3. Break-Even Analysis Across Cost Triggers
For the 500 accounts targeted under Highest Value at Risk:
- Total Preventable Value at 100% Save Rate: **₹95,099**
- Campaign Costs by Trigger:
  - `all_contacted`: ₹325,000
  - `responders_only`: ₹160,000
  - `saved_only`: ₹90,054

*Conclusion:* In all three triggers, a 12-month horizon cannot achieve break-even under ₹150 contact costs and ₹500 concessions. Break-even requires extending the customer value horizon to $\ge 24$ months or reducing contact/incentive costs to low-touch digital tiers.
