# Phase 4: Executive Dashboard User Guide

## 1. Application Overview
The Northstar SaaS Retention ROI Simulator is an interactive Streamlit application (`app.py`) built to allow executive leadership to explore machine learning churn predictions, simulate campaign financial scenarios, and evaluate targeting trade-offs under complete assumption visibility.

---

## 2. Launch Instructions (Windows PowerShell)

```powershell
# 1. Activate Python virtual environment (if applicable)
.\venv\Scripts\Activate.ps1

# 2. Run automated verification suite
python -m pytest -v

# 3. Launch Streamlit Executive Dashboard
python -m streamlit run app.py
```
Upon startup, the application runs locally at `http://localhost:8501`.

---

## 3. Dashboard Structure & Navigation

### Page 1 â€" Executive Overview
- **Key Metrics:** Total active portfolio accounts (7,043), historical churn rate (26.5%), aggregate portfolio value at risk, expected scenario ROI, and budget exposure.
- **Exposure Analysis:** Bar charts decomposing total value at risk across Contract Types (Month-to-month vs One-year vs Two-year) and Internet Service Tiers (Fiber optic vs DSL vs No Internet).
- **Core Concept Callout:** Highlights the fundamental difference between churn probability and financial value at risk.

### Page 2 â€" Customer Risk & Value Explorer
- **Multi-Facet Filtering:** Segment accounts by Contract, Internet Service, and Churn Risk probability brackets.
- **Flexible Sorting:** Sort accounts by Value at Risk, Churn Probability, Balanced Priority Score, or Monthly Charges.
- **Privacy Safeguards:** Customer IDs are masked with synthetic analytical identifiers (`ANON_00001`) to protect privacy in executive presentations.
- **Data Export:** Trigger local CSV downloads of any filtered customer cohort.

### Page 3 â€" Retention Campaign Simulator
- **Interactive Controls:** Real-time feedback when adjusting contact costs, concessions, reach rates, response rates, and save rates.
- **5-Strategy Comparison Table:** Compares the No-Campaign baseline against Highest Risk, Highest VaR, Balanced, and Budget-Constrained targeting.
- **Formula Visibility:** Displays LaTeX representations of every expected-value formula.
- **Exportable Results:** One-click CSV export of the strategy comparison table.

### Page 4 â€" Scenario & Break-Even Analysis
- **3-Point Matrix:** Compares Conservative (Pessimistic), Base Case, and Optimistic (Upside) scenarios.
- **Break-Even Analysis:** Real-time calculation of the minimum required save rate $\text{SaveRate}_{\text{BE}}$ with visual success/deficit alerts.
- **Sensitivity Heatmap:** 2D gradient grid displaying Expected Net Benefit across varying response rates (20%â€"60%) and save rates (10%â€"40%).

### Page 5 â€" Model & Data Limitations
- Detailed disclosure of retrospective dataset constraints, class reweighting effects (`scale_pos_weight = 2.5`), lack of causal intervention data, and prerequisites before committing real enterprise capital.
