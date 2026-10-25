# Data Leakage Prevention Policy

## 1. Principles and Taxonomy of Data Leakage

In production machine learning and predictive analytics, data leakage occurs when information from outside the training partition or post-event information inappropriately influences the model training process. Leakage yields deceptively optimistic offline metrics (e.g. inflated ROC-AUC or F1-scores) that collapse when deployed into live business operations.

To guarantee trustworthy performance estimates, this project defines and guards against the following six categories of data leakage:

```
                                  DATA LEAKAGE TAXONOMY
                                            ‚"Ç
        ‚"å‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¨‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¥‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"¨‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"Ä‚"ê
        ‚-º                   ‚-º                               ‚-º                   ‚-º
 Target Leakage     Contamination Leakage           Temporal Leakage    Entity/Group Leakage
 (Post-cancellation  (Train/Test Mixing,            (Future-to-Past      (Customer duplicates,
  proxy features)     Global Scalers/Imputers)       Information)         Multi-record overlap)
```

### A. Target Leakage
Using features that directly encode, approximate, or result from the target event itself. For example, including fields such as "account deactivation date", "cancellation refund requested", or "exit survey submitted" in a churn model. 
*Standard:* No feature that is created downstream of a customer's decision to churn may be included in the predictor matrix.

### B. Train / Test Contamination
Allowing information from the evaluation partition (test set) to influence the training pipeline.
*Standard:* The held-out test dataset must remain completely untouched until final evaluation. Under no circumstances may hyperparameter tuning, model family selection, or feature engineering choices be made by querying test-set metrics.

### C. Preprocessing & Feature Engineering Leakage
Fitting transformers‚Ä"such as mean/median imputers, variance scalers (`StandardScaler`, `MinMaxScaler`), target encoders, or dimensionality reducers (PCA)‚Ä"on the entire dataset prior to splitting.
*Standard:* All stateful transformers must be fitted strictly on training data (`fit` / `fit_transform`) and applied out-of-fold to validation and test data (`transform`).

### D. Cross-Validation Contamination
Preprocessing the full training split before splitting folds inside cross-validation.
*Standard:* Every cross-validation fold must encapsulate the full preprocessing pipeline via `sklearn.pipeline.Pipeline`, ensuring that statistics (imputation values, standard deviations, one-hot category vocabularies) are learned solely from the fold's internal training slice.

### E. Temporal Leakage
Using future information to predict past events, or random shuffling across time when data has chronological order.
*Standard:* If timestamped transaction logs exist, splits must be out-of-time (prospective). Because the current dataset lacks timestamps (retrospective snapshot), this limitation is explicitly disclosed, and performance is framed strictly as retrospective classification rather than forward forecasting.

### F. Entity / Repeated-Measurement Leakage
When multiple records belong to the same entity (e.g. repeated monthly subscriptions or multi-device sessions), random splitting places instances of the same customer in both training and test partitions.
*Standard:* The dataset audit verifies customer identifier uniqueness. If repeated observations are present, group-aware partitioning (`GroupKFold` / `StratifiedGroupKFold`) is mandatory.

---

## 2. Mandatory Rules for All Modeling Work

1. **Strict Partitioning:** Any future model evaluation must separate training and test data before fitting transformations.
2. **Local Transformation Fitting:** Imputers, scalers, encoders, feature selectors, and other learned transformations must be fitted on training data only.
3. **Pipeline Encapsulation:** Cross-validation must fit the complete preprocessing pipeline separately within each fold using scikit-learn `Pipeline` and `ColumnTransformer`.
4. **Test Set Quarantine:** The final held-out test set must not be used for model selection, feature selection, threshold tuning, or hyperparameter exploration.
5. **No Global Encodings:** Never compute global target encodings, target-derived aggregates, or global feature statistics using all rows before splitting.
6. **Identifier Exclusion:** Customer identifiers (`customerID`) are permanently excluded from predictors unless a defensible, leakage-free use is demonstrated.
7. **No Post-Event Predictors:** Do not use post-cancellation information to predict a churn event that has already occurred.
8. **Group-Aware Splitting:** If repeated observations per customer are discovered in subsequent datasets, group-aware splitting is mandatory.
9. **Transparent Temporal Caveats:** Since a genuine temporal prediction setup is unavailable in the IBM Telco snapshot, random stratified evaluation is used, but we explicitly document that it does not establish future-period performance.
10. **EDA / Pipeline Separation:** Any exploratory analysis performed on the full dataset is strictly descriptive and must never be referenced by or coupled with learned transformations inside model training pipelines.

---

## 3. Enforcement in Code Architecture

In this repository, leakage protection is architecturally enforced in `src/baseline.py`:
- `separate_features_and_target()` systematically drops `customerID` and target columns.
- `train_test_split(..., stratify=y, random_state=42)` isolates test data at the boundary.
- `build_preprocessor()` configures unfitted `ColumnTransformer` instances with `OneHotEncoder(handle_unknown="ignore")` and `SimpleImputer`.
- `cross_validate()` receives the composite `Pipeline([('preprocessor', ...), ('classifier', ...)])` object, recalculating imputations and scalings strictly within each of the 5 cross-validation folds.
- 15 automated pytest unit tests verify this pipeline behavior deterministically.
