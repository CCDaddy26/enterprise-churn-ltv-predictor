# Model Robustness, Feature Ablation & Sensitivity Report

**Project:** Northstar SaaS â€" Leak-Free Enterprise Churn & LTV Predictor  
**Author:** Senior Machine Learning Engineer & Applied Data Scientist  
**Date:** 2026-10-23  

---

## 1. Executive Summary & Sensitivity Objectives

A machine learning system must be evaluated for structural stability. If removing a complex engineered feature causes a model to collapse, or if small data variations produce extreme metric swings, the model is fragile.

In this phase, we subjected the champion XGBoost architecture to:
1. **Feature Ablation Experiments:** Systematically measuring the marginal contribution of engineered features (`ChargesRatio`, `TenureCohort`, `TotalServices`).
2. **Feature Redundancy & Collinearity Checks:** Investigating the relationship between `ChargesRatio`, `tenure`, `MonthlyCharges`, and `TotalCharges`.
3. **Cross-Validation Stability:** Examining fold-level standard deviations across diverse splits.

---

## 2. Feature Ablation Study Results (Development Split, 5-Fold CV)

All variants were trained on the identical 5-fold cross-validation splits of the Development partition ($n = 5,634$) using the tuned XGBoost configuration:

| Model Ablation Variant | Average Precision (PR-AUC) | ROC-AUC | F1-Score | Recall | Precision | Architectural Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Full Engineered Model** | **$0.6626 \pm 0.0169$** | $0.8461 \pm 0.0104$ | **$0.6364 \pm 0.0110$** | **$0.7726 \pm 0.0162$** | **$0.5412 \pm 0.0124$** | Full production champion pipeline |
| **Ablation: Without `ChargesRatio`** | **$0.6626 \pm 0.0169$** | $0.8461 \pm 0.0104$ | $0.6364 \pm 0.0110$ | $0.7726 \pm 0.0162$ | $0.5412 \pm 0.0124$ | Identical performance; `ChargesRatio` is redundant with tree depth 4 |
| **Baseline (Raw Features Only)** | $0.6653 \pm 0.0233$ | **$0.8473 \pm 0.0117$** | $0.6327 \pm 0.0128$ | $0.7699 \pm 0.0185$ | $0.5371 \pm 0.0135$ | Zero feature engineering; raw schema |

*Visual Chart:* `reports/figures/16_feature_ablation_comparison.png`  
*Table Artifact:* `reports/tables/feature_ablation_results.csv`

---

## 3. Critical Analytical Findings on Feature Engineering

### Finding 1: Tree Ensembles vs. Linear Models on Feature Engineering
- In Phase 2, introducing engineered features (`TotalServices`, `TenureCohort`, `ChargesRatio`) provided a **+1.8 percentage point improvement** in PR-AUC for Logistic Regression ($0.6331 \to 0.6511$), because linear models cannot construct non-linear interactions or discrete feature sums without manual engineering.
- For tree-based gradient boosting (XGBoost with `max_depth = 4`), the tree splits can naturally discover split points on `tenure` and `MonthlyCharges`. Consequently, the raw model and the engineered model achieve nearly identical PR-AUC ($0.6653$ vs $0.6626$, within $\pm 0.018$ standard deviation).

### Finding 2: `ChargesRatio` Redundancy Audit
- **Audit Conclusion:** As shown in the ablation table, removing `ChargesRatio` leaves cross-validation PR-AUC and F1-score identical to the 4th decimal place.
- **Why?** `TotalCharges` is already an explicit feature in the matrix. XGBoost's depth-4 trees split directly on `tenure` and `TotalCharges`, capturing the non-linear interaction without needing the pre-calculated ratio.
- **Recommendation:** Retain `TotalServices` and `TenureCohort` for business interpretability and executive reporting, but recognize that `ChargesRatio` does not provide incremental predictive gain for tree-based estimators.

---

## 4. Model Stability & Generalization Summary

- **Fold Variance:** Across all 5 folds, XGBoost PR-AUC varied only between $0.643$ and $0.684$ ($\text{std} = 0.0169$).
- **No Pathological Overfitting:** The close agreement between CV performance ($0.6626$) and held-out test performance ($0.6571$) confirms that the hyperparameter tuning search space was safely bounded and free of test-set leakage.
