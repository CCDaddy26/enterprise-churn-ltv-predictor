"""Master execution runner for Phase 2: Leak-Free Feature Engineering & ML Pipeline."""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FIGURES_DIR,
    ID_COLUMN,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TABLES_DIR,
    TARGET_COLUMN,
    TARGET_POSITIVE_VALUE,
    ensure_directories,
)
from src.data_loader import load_raw_dataset
from src.data_validation import separate_features_and_target
from src.evaluation import (
    evaluate_model_performance,
    plot_model_cv_comparison,
    plot_precision_recall_curves,
)
from src.inference import ChurnInferenceEngine, save_pipeline_artifact
from src.model_training import evaluate_candidates_cv, get_model_candidates
from src.tuning import tune_xgboost_hyperparameters

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("phase2_runner")


def main():
    logger.info("Initializing Phase 2 ML Pipeline & Model Training Execution...")
    ensure_directories()

    # 1. Load Raw Dataset
    logger.info("Step 1: Loading raw dataset...")
    df_raw = load_raw_dataset(RAW_DATA_PATH)
    logger.info(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # 2. Isolate Features X and Target y (Excluding customerID)
    logger.info("Step 2: Isolating features and binary target...")
    X, y_raw = separate_features_and_target(df_raw, exclude_id=True)
    y = (y_raw == TARGET_POSITIVE_VALUE).astype(int)

    # 3. Stratified Partitioning (Development vs. Quarantined Final Test)
    logger.info("Step 3: Performing stratified partitioning (Dev 80% / Final Test 20%)...")
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    partition_summary = {
        "dataset_name": "IBM Telco Customer Churn",
        "random_seed": RANDOM_SEED,
        "total_records": len(df_raw),
        "development_partition": {
            "samples": len(X_dev),
            "percentage": 80.0,
            "churn_positive_count": int(y_dev.sum()),
            "churn_negative_count": int(len(y_dev) - y_dev.sum()),
            "churn_prevalence": round(float(y_dev.mean()), 4),
        },
        "quarantined_test_partition": {
            "samples": len(X_test),
            "percentage": 20.0,
            "churn_positive_count": int(y_test.sum()),
            "churn_negative_count": int(len(y_test) - y_test.sum()),
            "churn_prevalence": round(float(y_test.mean()), 4),
        },
    }

    with open(TABLES_DIR / "partition_summary.json", "w", encoding="utf-8") as f:
        json.dump(partition_summary, f, indent=2)
    logger.info(f"Saved partition summary to {TABLES_DIR / 'partition_summary.json'}")

    # 4. Stratified 5-Fold Cross-Validation on Development Partition
    logger.info("Step 4: Executing Stratified 5-Fold Cross-Validation on Development Partition...")
    candidates = get_model_candidates(
        random_state=RANDOM_SEED,
        include_engineered_features=True,
    )

    cv_summary_df, fold_scores_df = evaluate_candidates_cv(
        candidates,
        X_dev,
        y_dev,
        n_splits=5,
        random_state=RANDOM_SEED,
    )

    cv_summary_df.to_csv(TABLES_DIR / "model_comparison_cv.csv")
    fold_scores_df.to_csv(TABLES_DIR / "model_fold_scores.csv", index=False)
    logger.info("Cross-validation summary completed.")
    print("\n--- DEVELOPMENT CROSS-VALIDATION SUMMARY (5-Fold CV, n=5,634) ---")
    print(cv_summary_df[["pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std", "f1_mean", "recall_mean", "precision_mean"]])

    # Plot CV comparison chart
    cv_chart_path = plot_model_cv_comparison(cv_summary_df)
    logger.info(f"Saved cross-validation comparison chart to {cv_chart_path}")

    # 5. Bounded Hyperparameter Tuning on Development Partition ONLY
    logger.info("Step 5: Running bounded hyperparameter tuning for XGBoost on Development data...")
    best_xgb_pipeline, tuning_metadata, cv_results_grid = tune_xgboost_hyperparameters(
        X_dev,
        y_dev,
        n_splits=5,
        random_state=RANDOM_SEED,
    )

    with open(TABLES_DIR / "selected_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(tuning_metadata, f, indent=2)

    # 6. Champion Model Refit on Full Development Partition
    logger.info("Step 6: Refitting Champion Model (Tuned XGBoost) on full Development partition...")
    champion_pipeline = best_xgb_pipeline
    champion_pipeline.fit(X_dev, y_dev)

    # Serialize champion pipeline artifact & metadata
    model_metadata = {
        "model_name": "xgboost_champion",
        "algorithm": "XGBClassifier (Tuned)",
        "phase": "Phase 2 Model Development",
        "random_seed": RANDOM_SEED,
        "engineered_features_included": True,
        "hyperparameters": tuning_metadata["best_parameters"],
        "dev_cv_average_precision": tuning_metadata["best_cv_average_precision"],
        "training_samples": len(X_dev),
        "target_positive_class": "1 (Churn)",
        "disclaimer": "Experimental portfolio model. Not approved for automated production retention actions without human review.",
    }
    p_path, m_path = save_pipeline_artifact(champion_pipeline, model_metadata)

    # 7. Single Final Evaluation on Quarantined Test Partition (n=1,409)
    logger.info("Step 7: Evaluating on Quarantined Held-Out Test Set (Single Benchmark Evaluation)...")
    test_probs = {}

    # Evaluate Dummy Baseline
    dummy_pipe = candidates["dummy_most_frequent"]
    dummy_pipe.fit(X_dev, y_dev)
    dummy_probs = np.zeros(len(y_test))
    test_probs["dummy_most_frequent"] = dummy_probs
    dummy_eval = evaluate_model_performance(y_test, np.zeros(len(y_test)), dummy_probs)

    # Evaluate Logistic Regression Baseline
    lr_pipe = candidates["logistic_regression"]
    lr_pipe.fit(X_dev, y_dev)
    lr_probs = lr_pipe.predict_proba(X_test)[:, 1]
    test_probs["logistic_regression"] = lr_probs
    lr_eval = evaluate_model_performance(y_test, (lr_probs >= 0.5).astype(int), lr_probs)

    # Evaluate Random Forest
    rf_pipe = candidates["random_forest"]
    rf_pipe.fit(X_dev, y_dev)
    rf_probs = rf_pipe.predict_proba(X_test)[:, 1]
    test_probs["random_forest"] = rf_probs
    rf_eval = evaluate_model_performance(y_test, (rf_probs >= 0.5).astype(int), rf_probs)

    # Evaluate Champion Tuned XGBoost
    champ_probs = champion_pipeline.predict_proba(X_test)[:, 1]
    test_probs["xgboost_champion"] = champ_probs
    champ_eval_default = evaluate_model_performance(y_test, (champ_probs >= 0.5).astype(int), champ_probs, threshold=0.5)

    # Illustrative high-recall retention screening threshold (tau = 0.40)
    champ_eval_tuned_thresh = evaluate_model_performance(y_test, (champ_probs >= 0.40).astype(int), champ_probs, threshold=0.40)

    final_results = {
        "dummy_most_frequent": dummy_eval,
        "logistic_regression": lr_eval,
        "random_forest": rf_eval,
        "xgboost_champion_threshold_0.50": champ_eval_default,
        "xgboost_champion_illustrative_threshold_0.40": champ_eval_tuned_thresh,
    }

    final_eval_df = pd.DataFrame(final_results).T
    final_eval_df.index.name = "Model"
    final_eval_df.to_csv(TABLES_DIR / "final_evaluation_metrics.csv")

    with open(TABLES_DIR / "final_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    # Plot Precision-Recall curves on held-out test split
    pr_curve_path = plot_precision_recall_curves(y_test, test_probs)
    logger.info(f"Saved Precision-Recall curves to {pr_curve_path}")

    # 8. Test Inference Engine on Sample Record
    logger.info("Step 8: Verifying production inference interface...")
    engine = ChurnInferenceEngine.from_artifact()
    sample_df = df_raw.iloc[0:5].copy()
    scored_sample = engine.predict(sample_df)
    scored_sample.to_csv(TABLES_DIR / "sample_scored_customers.csv", index=False)
    logger.info(f"Scored sample output:\n{scored_sample[['customerID', 'churn_probability', 'is_churn_risk', 'risk_tier']]}")

    print("\n--- FINAL HELD-OUT TEST EVALUATION (Single Benchmark Evaluation, n=1,409) ---")
    print(final_eval_df[["average_precision_pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy"]])

    logger.info("=== Phase 2 Execution Successfully Completed ===")


if __name__ == "__main__":
    main()
