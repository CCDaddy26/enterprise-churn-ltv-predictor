# Phase 4.1: Financial Assumptions & Sensitivity Bounds (Corrected)

## 1. Catalog of Model Assumptions

| Assumption Name | Default Value | Valid Range | Unit / Dimension | Business Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **Probability Mode** | `Calibrated` | `Calibrated / Uncalibrated` | Toggle | Governs whether inputs are cross-fitted probabilities (ECE 0.011) or raw risk scores. |
| **Contribution Margin** | `70%` | `10% �" 95%` | Ratio | Portion of billed revenue remaining after variable infrastructure/support costs. |
| **Value Horizon** | `12` | `1 �" 36` | Months | Period over which churned contribution loss is simulated. |
| **Contact Cost** | `₹150.0` | `₹0.0 �" ₹1000.0` | Currency / Account | Direct outreach cost (labor, personalized phone/email, agent time). |
| **Incentive Concession** | `₹500.0` | `₹0.0 �" ₹5000.0` | Currency / Account | Financial concession, discount voucher, or hardware upgrade offered. |
| **Incentive Trigger** | `responders_only` | `responders / saved / all` | Categorical | Dictates when incentive is paid to prevent double-counting. |
| **Campaign Reach Rate** | `85%` | `10% �" 100%` | Ratio | Probability an outreach attempt successfully reaches the account contact. |
| **Response Rate** | `40%` | `5% �" 100%` | Ratio (given Reach) | Probability a reached customer engages with the customer success team. |
| **Save Rate** | `25%` | `1% �" 100%` | Ratio (given Response) | Assumed fraction of responding accounts that cancel their termination. |
| **Max Campaign Budget** | `₹200,000` | `₹10k �" ₹2M` | Currency | Hard spending ceiling allocated for the retention initiative. |
| **Targeting Cohort Size** | `500` | `50 �" 2000` | Accounts | Fixed account batch evaluated under Mode A (Like-for-Like). |

---

## 2. Low, Base, and High Scenario Presets (Calibrated Mode)

Evaluated on Top 500 Accounts by Estimated Value at Risk:

| Metric | Conservative (Pessimistic) | Base Case (Realistic) | Optimistic (Upside) |
| :--- | :--- | :--- | :--- |
| **Reach Rate** | `70%` | `85%` | `95%` |
| **Response Rate** | `25%` | `40%` | `55%` |
| **Save Rate** | `15%` | `25%` | `35%` |
| **Expected Churn Prevented (n=500)** | `9.3` accounts | `30.1` accounts | `64.9` accounts |
| **Campaign Cost** | `₹118,750` | `₹160,000` | `₹205,625` |
| **Gross Benefit** | `₹7,342` | `₹23,775` | `₹51,170` |
| **Net Benefit** | `-₹111,408` | `-₹136,225` | `-₹154,455` |
| **ROI** | `-93.8%` | `-85.1%` | `-75.1%` |

---

## 3. Break-Even Sensitivity Across Cost Triggers (500 Accounts)

| Incentive Cost Trigger | Unit Cost / Acct | Total Cost (n=500) | Break-Even Save Rate | Status |
| :--- | :---: | :---: | :---: | :---: |
| **`saved_only`** | ₹180.11 | ₹90,054 | **Unreachable (>100%)** | Deficit at 12m horizon |
| **`responders_only`** | ₹320.00 | ₹160,000 | **Unreachable (>100%)** | Deficit at 12m horizon |
| **`all_contacted`** | ₹650.00 | ₹325,000 | **Unreachable (>100%)** | Deficit at 12m horizon |

*Strategic Insight:* Concession costs must be restructured to automated digital vouchers or low-cost tier credits (e.g. ₹50�"₹100) and targeted exclusively at accounts with Monthly Charges $> ₹80$ for campaign economics to yield positive net returns within a 12-month horizon.
