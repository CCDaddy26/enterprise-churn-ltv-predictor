# Phase 4.1: Financial Methodology & Scenario Engine (Corrected)

## 1. Executive Context & Purpose
In Phase 4.1 of **Project 01/30 (Leak-Free Enterprise Churn & LTV Predictor)**, we implement targeted financial integrity corrections to resolve the probability-to-finance boundary, prevent baseline ROI misrepresentation, enforce like-for-like strategy comparisons, and guarantee trigger-consistent break-even mathematics for the fictional enterprise **Northstar SaaS**.

---

## 2. Probability-to-Finance Boundary & Defensible Calibration

### 2.1 The Issue with Raw Champion Scores
The Phase 2 champion XGBoost pipeline was trained with `scale_pos_weight = 2.5` to prioritize recall on minority churners. While this achieves strong ranking ability (PR-AUC 0.6589), it deliberately distorts output probabilities upward:
- Empirical Development Prevalence: **26.54%**
- Mean Raw Model Score: **42.10%**
- Uncalibrated Expected Calibration Error (ECE): **0.1152**

Using uncalibrated scores directly as $P(\text{Churn})$ systematically inflates expected prevented churn and balance-sheet value at risk.

### 2.2 Leakage-Conscious Cross-Fitted Calibration Design
To resolve this without using the historical held-out test set:
1. **Development Data Only ($N=5,634$):** Evaluated strictly on the 80% development split.
2. **Two-Stage Cross-Fitting:**
   - Champion pipeline generates 5-fold Stratified Cross-Validation Out-of-Fold (OOF) raw scores.
   - A secondary 5-fold cross-fitting procedure fits a Platt Sigmoid Calibrator ($y \sim \text{LogisticRegression}(\text{logit}(s))$). Each evaluated customer's calibrated probability is produced by a calibrator that **never saw that customer's true target label**.
3. **Generalization on Held-Out Test Data ($N=1,409$):** Evaluated on the held-out test split, confirming no overfitting.

#### Calibration Performance Results:
| Split & Model | Sample Size | Brier Score | Log Loss | Expected Calibration Error (ECE) |
| :--- | :---: | :---: | :---: | :---: |
| **Dev OOF (Uncalibrated)** | 5,634 | 0.1542 | 0.4634 | 0.1152 |
| **Dev Cross-Fitted (Sigmoid Calibrated)** | 5,634 | **0.1357** | **0.4197** | **0.0110** |
| **Held-Out Test (Uncalibrated)** | 1,409 | 0.1564 | 0.4671 | 0.1190 |
| **Held-Out Test (Sigmoid Calibrated)** | 1,409 | **0.1363** | **0.4213** | **0.0275** |

*Important Caveat:* Improving ECE from 0.1152 to 0.0110 establishes a defensible, empirical probability mapping, but does **not** imply perfect certainty or validate customer accounting margins.

---

## 3. Explicit Financial Semantics

The calculation layer, UI, and exports enforce mutually exclusive semantic modes:

### Mode 1: Calibrated Scenario Mode (`is_calibrated = True`)
- **Prediction Metric:** `calibrated_churn_probability`
- **Financial Exposure Metric:** `Estimated_Value_at_Risk`
- **Interpretation:** Expected contribution value at risk under verified empirical calibration and stated margin assumptions.

### Mode 2: Uncalibrated Risk Score Mode (`is_calibrated = False`)
- **Prediction Metric:** `churn_risk_score`
- **Financial Exposure Metric:** `Risk_Weighted_Value_Proxy`
- **Interpretation:** Relative risk-weighted proxy for account ranking only; explicitly flagged as **not an expected monetary loss**.

---

## 4. Zero-Cost Campaign Baseline ROI Handling

When campaign expenditure is zero ($N=0$ or $\text{Cost}=0$), Return on Investment (ROI) is mathematically undefined:
$$\text{ROI} = \frac{\text{Net Benefit}}{\text{Cost}} = \frac{0}{0} \implies \text{Undefined}$$

- **Calculation Layer:** Returns `None` (`np.nan`).
- **UI & Export Representation:** Formatted explicitly as `"N/A â€" no campaign expenditure"`.
- Never reported as $0.0\%$, preventing false impressions of a zero-yield investment.

---

## 5. Strategy Comparison Architecture: Mode A vs. Mode B

To prevent misleading comparisons where one strategy targets more accounts than another:

### Mode A: Like-for-Like Cohort Size
- All active strategies target the **exact same user-configured batch size** (default $N=500$ accounts).
- Evaluates ranking effectiveness independent of cohort volume.

### Mode B: Budget-Maximized Targeting
- Outreach capacity is determined strictly by the budget ceiling and expected unit economics:
  $$N_{\text{max}} = \left\lfloor \frac{\text{Budget}}{\text{Unit Cost}} \right\rfloor$$
- Explicitly tracks budget utilization and remaining unallocated capital.

---

## 6. Mathematical Formulations & Cost Triggers

### 6.1 Multi-Stage Funnel
$$P(\text{Prevented Churn}_i) = P(\text{Churn}_i) \times P(\text{Reach}) \times P(\text{Response}) \times P(\text{Save})$$

### 6.2 Trigger-Consistent Unit Cost & Break-Even Derivations
Let $p_c = \frac{1}{N}\sum_{i=1}^N P(\text{Churn}_i) \times \text{Remaining Contribution}_i$.

1. **Trigger: `responders_only` (Default)**
   - Unit Cost: $c_{\text{unit}} = c_{\text{contact}} + c_{\text{incentive}} \times (P(\text{Reach}) \times P(\text{Response}))$
   - Break-Even Save Rate:
     $$\text{SaveRate}_{\text{BE}} = \frac{c_{\text{unit}}}{P(\text{Reach}) \times P(\text{Response}) \times p_c}$$

2. **Trigger: `saved_only` (Concession on Retained Accounts Only)**
   - Unit Cost: $c_{\text{unit}} = c_{\text{contact}} + c_{\text{incentive}} \times P(\text{Prevented Churn}_i)$
   - Break-Even Save Rate:
     $$\text{SaveRate}_{\text{BE}} = \frac{c_{\text{contact}}}{P(\text{Reach}) \times P(\text{Response}) \times \left( p_c - c_{\text{incentive}} \times \overline{P(\text{Churn})} \right)}$$

3. **Trigger: `all_contacted` (Upfront Concession to All Targeted)**
   - Unit Cost: $c_{\text{unit}} = c_{\text{contact}} + c_{\text{incentive}}$
   - Break-Even Save Rate:
     $$\text{SaveRate}_{\text{BE}} = \frac{c_{\text{contact}} + c_{\text{incentive}}}{P(\text{Reach}) \times P(\text{Response}) \times p_c}$$
