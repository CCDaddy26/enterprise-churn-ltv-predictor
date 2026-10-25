# Model Validation Protocol & Partitioning Governance

## 1. Principles of Validation Design

In production machine learning, evaluating models on data that directly or indirectly influenced training produces inflated offline performance metrics that degrade upon operational deployment.

To ensure reproducible, scientifically honest evaluation, this protocol establishes a strict three-phase boundary:
1. **Partitioning Governance:** Complete isolation of the Development partition from the quarantined Final Test partition.
2. **Cross-Validation Scope:** All feature engineering, scaling, imputing, algorithm benchmarking, and hyperparameter search must occur strictly within the Development partition.
3. **Quarantined Evaluation:** The final test partition is evaluated **exactly once** after all development choices have been locked.

---

## 2. Partitioning Architecture

```
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
|                         TOTAL DATASET (N = 7,043)                           |
+‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä+
                                       ‚"Ç
                    Stratified Train/Test Split (Seed = 42)
                                       ‚"Ç
        ‚"å‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¥‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"ê
        ‚-º                                                             ‚-º
 [DEVELOPMENT PARTITION]                                    [FINAL TEST PARTITION]
  N = 5,634 (80.0%)                                          N = 1,409 (20.0%)
  Churn Rate: 26.54% (1,495 / 5,634)                         Churn Rate: 26.54% (374 / 1,409)
        ‚"Ç                                                             ‚"Ç
        ‚-º                                                             ‚"Ç
 Stratified 5-Fold Cross-Validation                                   ‚"Ç
 (Preprocessors & Models fitted strictly inside fold)                 ‚"Ç
        ‚"Ç                                                             ‚"Ç
        ‚"ú‚"Ä‚-∫ Algorithm Benchmarking (Dummy, LR, RF, XGB)               ‚"Ç
        ‚"ú‚"Ä‚-∫ Hyperparameter Tuning (Average Precision optimization)    ‚"Ç
        ‚"ú‚"Ä‚-∫ Decision Threshold Calibration                            ‚"Ç
        ‚"Ç                                                             ‚"Ç
        ‚-º                                                             ‚"Ç
 [CHAMPION MODEL SELECTION]                                           ‚"Ç
  Selected based purely on Dev 5-Fold CV metrics                      ‚"Ç
        ‚"Ç                                                             ‚"Ç
        ‚-º                                                             ‚"Ç
 Final Model Refit on Full Dev Partition (N = 5,634)                   ‚"Ç
        ‚"Ç                                                             ‚"Ç
        ‚""‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¨‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"ò
                                       ‚-º
                       [SINGLE FINAL BENCHMARK EVALUATION]
                        Evaluated ONCE on N = 1,409
                        Compared to Phase 1 Reference Baseline
```

---

## 3. Quarantined Test-Set Governance Rules

1. **No Tuning Against Final Test:** The final test partition ($n = 1,409$) must NEVER be queried during feature engineering, model selection, or hyperparameter tuning.
2. **Deterministic Seed:** All partitioning uses `RANDOM_SEED = 42`, guaranteeing exact byte-for-byte reproducibility across operating systems.
3. **No Target Aggregation Across Boundary:** Target statistics (e.g. global mean churn, target encodings) must never be calculated using the full 7,043 rows before splitting.
4. **Machine-Readable Audit:** Every run saves a partition audit file (`reports/tables/partition_summary.json`) documenting record counts, class counts, and random seed.

---

## 4. Cross-Validation Scoring Hierarchy

Model optimization must be guided by the business objective (identifying at-risk subscribers under class imbalance), not simple classification accuracy:

1. **Primary Selection Metric:** **Average Precision (PR-AUC)**  
   Calculates the area under the precision-recall curve via weighted interpolation. Emphasizes finding true churners while penalizing false alarms across all possible probability thresholds.
2. **Secondary Ranking Metric:** **ROC-AUC**  
   Measures global discrimination capability between churners and non-churners across all thresholds.
3. **Operational Threshold Metrics (at calibrated cutoff $\tau$):**  
   - **Recall (Sensitivity):** Proportion of actual churners flagged. Target: $\ge 75\%$.
   - **Precision (Positive Predictive Value):** Proportion of flagged accounts that are true churners. Target: $\ge 45\%$.
   - **F1-Score:** Harmonic mean of precision and recall.
   - **False Positive Burden ($FP$):** Direct count of unneeded Customer Success outreach actions.
