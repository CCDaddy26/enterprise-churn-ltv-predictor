# Customer Lifetime Value (LTV) and Retention ROI Methodology

## 1. Conceptual Foundations: Distinguishing Revenue from Economic Value

In enterprise subscription analytics, confusing top-line revenue metrics with genuine customer lifetime value leads to severely distorted capital allocation and unprofitable marketing spend.

```
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
| CORE ECONOMIC DEFINITIONS FOR SUBSCRIPTION SAAS                             |
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
| Monthly Recurring Revenue (MRR) : Normalized monthly recurring subscription |
| Annual Recurring Revenue (ARR)  : MRR √- 12                                  |
| Gross Margin (GM %)             : (Revenue - Cost of Goods Sold) / Revenue  |
| Customer Lifetime (T)           : Expected active duration (1 / Churn Rate) |
| Customer Lifetime Value (LTV)   : Discounted net contribution margin over T |
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
```

### Why Churn Probability Is Not LTV
A churn prediction model outputs the conditional probability of termination $\hat{p} = P(\text{Churn} = 1 \mid X)$. 
- A high-churn-risk customer with \$200/month recurring spend and 80% gross margin represents vastly greater lost financial value than a high-risk customer paying \$20/month with 20% margin.
- Optimizing retention purely by sorted churn probability treats all churners as economically equivalent. Efficient retention prioritization requires **Expected Value at Risk**:
$$\text{Value at Risk} = \hat{p} \times \text{Remaining Economic LTV}$$

### Why "Revenue √- Lifetime" Is Not Defensible LTV
Multiplying monthly charges by an assumed lifetime (e.g. `MonthlyCharges * (1 / churn_rate)`) fails basic financial scrutiny because:
1. **Ignores Cost to Serve (COGS):** Hosting, customer support, third-party licensing, and transaction processing must be deducted. Only contribution margin can repay customer acquisition and retention costs.
2. **Ignores Discount Rate / Time Value of Money:** Cash received 4 years from now is worth substantially less than cash today.
3. **Assumes Constant Flat Churn Hazard:** Real churn hazard curves follow bathtub or survival distributions: hazard is high during early onboarding (0‚Ä"6 months) and drops substantially for tenured cohorts (24+ months).

---

## 2. Mathematically Grounded LTV Formulation

For subscription businesses with monthly recurring billing, true discounted customer lifetime value is modeled as:

$$\text{LTV}_i = \sum_{t=1}^{H} \frac{\text{MRR}_{i, t} \times \text{GM}_{i, t} \times S_i(t)}{(1 + d)^t}$$

Where:
- $H$: Economic value planning horizon (e.g. 36 or 60 months).
- $\text{MRR}_{i, t}$: Forecasted monthly recurring revenue for customer $i$ at month $t$.
- $\text{GM}_{i, t}$: Gross margin percentage (typically 70% to 85% for pure software; 50% to 65% for bundled telecom/hardware services).
- $S_i(t)$: Estimated survival probability (probability customer remains active at month $t$, derived from survival models or transition matrices).
- $d$: Monthly discount rate (reflecting Weighted Average Cost of Capital, WACC).

### Missing Fields Required for Ground-Truth LTV Measurement
The public IBM Telco dataset provides `MonthlyCharges`, `TotalCharges`, and `tenure`. To calculate auditable LTV, an enterprise data warehouse must provide:
1. **Direct Customer COGS / Margin:** Server infrastructure consumption, customer support ticket costs.
2. **Historical Upgrade / Downgrade Trajectories:** Expansion revenue, seat expansion, cross-sell.
3. **Discount & Concession History:** Promotional discounts, credits granted.
4. **Acquisition Cost (CAC):** Fully loaded marketing and sales expenses per customer cohort.

---

## 3. Retention Campaign ROI Simulation Framework

Retention interventions are investments. Reporting simulated return without accounting for campaign execution costs and treatment effects is misleading.

```
                           RETENTION CAMPAIGN FLOW
                                      ‚"Ç
                         Eligible Customer Cohort (N)
                                      ‚"Ç
                                      ‚-º
                        Targeted for Intervention (N_contacted)
                        Cost: N_contacted √- Cost_per_Contact
                                      ‚"Ç
                     ‚"å‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¥‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"ê
                     ‚-º                                 ‚-º
             Treated Cohort                    Control Cohort
             (Receives Offer / Outreach)       (Standard Business Cadence)
                     ‚"Ç                                 ‚"Ç
                     ‚-º                                 ‚-º
              Observed Churn                    Observed Churn
              Rate: r_treated                   Rate: r_control
                     ‚"Ç                                 ‚"Ç
                     ‚""‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¨‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"ò
                                      ‚-º
                       Incremental Lift (Œ" Churn) = r_control - r_treated
                                      ‚-º
                 Net Incremental Retained Customers = N_contacted √- Œ" Churn
                                      ‚-º
                       Gross Margin Saved = Retained √- LTV
                                      ‚-º
                       Net ROI = (Gross Margin Saved - Campaign Cost) / Cost
```

### The Observational Confounding Caveat
**Critical Data Science Principle:** Observational data alone *cannot* prove that a customer stayed *because* of an intervention.
- In historical data, customers who received calls or offers may have stayed due to self-selection or preexisting loyalty (confounding).
- Determining true causal treatment effects requires a **Randomized Controlled Trial (A/B Test)** where identical high-risk customers are randomized into:
  - **Treatment Group:** Model-directed retention incentive (discount, white-glove onboarding).
  - **Holdout / Control Group:** Standard experience.
- The incremental lift ($\tau = r_{\text{control}} - r_{\text{treated}}$) is the *only* defensible multiplier for ROI simulation.

---

## 4. Configurable Simulation Parameters for Future Phase 4

When the interactive retention simulator is constructed in Phase 4, the following parameters must be explicitly configurable by the user, with realistic enterprise defaults:

| Parameter Name | Description | Default Value | Plausible Range |
| :--- | :--- | :--- | :--- |
| `N_eligible` | Total customer population eligible for retention review | Total Test Split ($1,409$) | $500 - 100,000$ |
| `Risk_Threshold` ($\tau$) | Model predicted probability cutoff for outreach | $0.50$ (or top risk decile) | $0.20 - 0.80$ |
| `Contact_Cost` ($c_{\text{contact}}$) | Direct cost per contacted customer (CSM labor, email, mailer) | $\$25.00$ | $\$5.00 - \$150.00$ |
| `Incentive_Cost` ($c_{\text{offer}}$) | Cost of discount or credit offered to retained customers | $\$50.00$ | $\$0.00 - \$200.00$ |
| `Incremental_Lift` ($\Delta r$) | Causal reduction in churn probability attributable to intervention | $0.15$ ($15\%$ relative lift) | $0.02 - 0.35$ |
| `Gross_Margin` ($\text{GM}$) | Company gross profit margin on subscription revenue | $75.0\%$ | $40.0\% - 90.0\%$ |
| `Value_Horizon` ($H$) | Future evaluation horizon for saved recurring revenue | $12$ months | $6 - 36$ months |

### Mathematical Net ROI Formula
$$\text{Cost}_{\text{total}} = N_{\text{contacted}} \times c_{\text{contact}} + (\text{Customers Saved}) \times c_{\text{offer}}$$
$$\text{Customers Saved} = N_{\text{contacted}} \times \Delta r$$
$$\text{Value Saved} = \text{Customers Saved} \times \left(\text{MonthlyCharges} \times 12 \times \text{GM}\right)$$
$$\text{Net ROI (\%)} = \frac{\text{Value Saved} - \text{Cost}_{\text{total}}}{\text{Cost}_{\text{total}}} \times 100$$

*Note: Phase 1 establishes this rigorous formula and economic framework. In accordance with project instructions, no fabricated dollar savings will be reported until Phase 4 simulation execution.*
