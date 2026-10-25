"""Master execution runner for Phase 3: SHAP Explainability, Diagnostics & Subgroup Audits."""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.calibration import evaluate_calibration_metrics, plot_reliability_diagram
from src.config import (
    FIGURES_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    TABLES_DIR,
    TARGET_COLUMN,
    TARGET_POSITIVE_VALUE,
    ensure_directories,
)
from src.data_loader import load_raw_dataset
from src.data_validation import separate_features_and_target
from src.error_analysis import (
    categorize_prediction_errors,
    plot_error_diagnostics,
    summarize_error_distributions,
)
from src.explainability import ModelExplainer
from src.robustness import plot_ablation_comparison, run_feature_ablation_study
from src.subgroup_audit import audit_subgroup_performance, plot_subgroup_audit_summary
from src.threshold_analysis import compute_threshold_curve, plot_threshold_tradeoffs

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("phase3_runner")


def main():
    logger.info("Initializing Phase 3: Explainability, Diagnostics & Subgroup Audits...")
    ensure_directories()

    # 1. Load Data & Isolate Target
    logger.info("Step 1: Loading raw data and isolating predictors...")
    df_raw = load_raw_dataset(RAW_DATA_PATH)
    X, y_raw = separate_features_and_target(df_raw, exclude_id=True)
    y = (y_raw == TARGET_POSITIVE_VALUE).astype(int)

    # Stratified split to mirror Phase 2 development vs quarantined test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    logger.info(f"Development partition: {len(X_dev)} records. Final Test partition quarantined ({len(X_test)} records).")

    # 2. Load Fitted Champion Model Pipeline
    logger.info("Step 2: Loading fitted champion pipeline artifact...")
    pipeline_path = Path("models/champion_pipeline.joblib")
    if not pipeline_path.exists():
        raise FileNotFoundError("models/champion_pipeline.joblib not found. Please run Phase 2 first.")
    champion_pipeline = joblib.load(pipeline_path)

    # 3. SHAP Explainability Engine
    logger.info("Step 3: Initializing SHAP TreeExplainer and calculating SHAP values...")
    explainer = ModelExplainer(champion_pipeline)
    X_dev_trans, shap_vals, explanation = explainer.explain_dataset(X_dev)

    # Global feature importance
    importance_df = explainer.compute_feature_importance(shap_vals)
    importance_df.to_csv(TABLES_DIR / "shap_feature_importance.csv", index=False)
    logger.info(f"Top 5 most influential features:\n{importance_df.head(5)}")

    # Plot beeswarm & importance bar
    beeswarm_path = explainer.plot_summary_beeswarm(explanation)
    bar_path = explainer.plot_feature_importance_bar(importance_df)
    dependence_path = explainer.plot_dependence_panels(explanation)
    logger.info("SHAP global visual assets saved.")

    # 4. Out-of-Fold Probability Calibration on Development Partition
    logger.info("Step 4: Assessing probability calibration via Development 5-Fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Generate cross-validated uncalibrated probability predictions
    oof_uncal_probs = cross_val_predict(
        champion_pipeline,
        X_dev,
        y_dev,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    # Evaluate Sigmoid Calibrated Classifier via CV on Development
    calibrated_clf = CalibratedClassifierCV(
        estimator=champion_pipeline,
        method="sigmoid",
        cv=cv,
    )
    calibrated_clf.fit(X_dev, y_dev)
    oof_cal_probs = calibrated_clf.predict_proba(X_dev)[:, 1]

    cal_metrics_uncal = evaluate_calibration_metrics(y_dev, oof_uncal_probs, "uncalibrated_xgboost")
    cal_metrics_cal = evaluate_calibration_metrics(y_dev, oof_cal_probs, "sigmoid_calibrated_xgboost")

    cal_summary_df = pd.DataFrame([cal_metrics_uncal, cal_metrics_cal]).set_index("model")
    cal_summary_df.to_csv(TABLES_DIR / "calibration_metrics.csv")
    logger.info(f"Calibration summary:\n{cal_summary_df}")

    # Plot reliability diagram
    prob_dict = {
        "uncalibrated": oof_uncal_probs,
        "sigmoid_calibrated": oof_cal_probs,
    }
    rel_diagram_path = plot_reliability_diagram(y_dev.to_numpy(), prob_dict)
    logger.info(f"Saved reliability diagram to {rel_diagram_path}")

    # 5. Threshold Analysis on Development Out-of-Fold Predictions
    logger.info("Step 5: Running decision threshold trade-off analysis...")
    threshold_df = compute_threshold_curve(y_dev.to_numpy(), oof_uncal_probs)
    threshold_df.to_csv(TABLES_DIR / "threshold_tradeoff_metrics.csv", index=False)

    threshold_plot_path = plot_threshold_tradeoffs(threshold_df)
    logger.info(f"Saved threshold trade-off plot to {threshold_plot_path}")

    # 6. Representative Individual Customer Explanations
    logger.info("Step 6: Selecting representative archetypes and generating waterfall plots...")
    from src.feature_engineering import ChurnFeatureEngineer
    fe = ChurnFeatureEngineer()
    dev_results_df = fe.transform(X_dev.copy())
    dev_results_df["Actual_Churn"] = y_dev
    dev_results_df["churn_probability"] = oof_uncal_probs
    dev_results_df = categorize_prediction_errors(dev_results_df, "Actual_Churn", "churn_probability", threshold=0.5)

    # Pick representative indices for the 4 archetypes (TP, FP, TN, FN)
    tp_idx = np.where((y_dev.to_numpy() == 1) & (oof_uncal_probs >= 0.70))[0][0]
    tn_idx = np.where((y_dev.to_numpy() == 0) & (oof_uncal_probs <= 0.15))[0][0]
    fp_idx = np.where((y_dev.to_numpy() == 0) & (oof_uncal_probs >= 0.65))[0][0]
    fn_idx = np.where((y_dev.to_numpy() == 1) & (oof_uncal_probs <= 0.35))[0][0]

    rep_indices = {
        "True Positive Churner": tp_idx,
        "True Negative Retained": tn_idx,
        "False Positive False Alarm": fp_idx,
        "False Negative Missed Churner": fn_idx,
    }

    waterfall_paths = explainer.plot_individual_waterfalls(explanation, rep_indices)

    # Save customer explanation table without personal IDs
    explanation_examples = []
    for archetype, idx in rep_indices.items():
        row_dict = {
            "Archetype": archetype,
            "Actual_Churn": int(y_dev.iloc[idx]),
            "Predicted_Probability": round(float(oof_uncal_probs[idx]), 4),
            "Tenure_Months": int(X_dev.iloc[idx]["tenure"]),
            "MonthlyCharges": float(X_dev.iloc[idx]["MonthlyCharges"]),
            "Contract": str(X_dev.iloc[idx]["Contract"]),
            "InternetService": str(X_dev.iloc[idx]["InternetService"]),
            "Top_Positive_Push_Feature": explainer.feature_names[np.argmax(shap_vals[idx])],
            "Top_Negative_Push_Feature": explainer.feature_names[np.argmin(shap_vals[idx])],
        }
        explanation_examples.append(row_dict)

    rep_df = pd.DataFrame(explanation_examples)
    rep_df.to_csv(TABLES_DIR / "customer_explanation_examples.csv", index=False)
    logger.info(f"Saved representative customer explanations to {TABLES_DIR / 'customer_explanation_examples.csv'}")

    # 7. Subgroup Performance & Fairness Audit
    logger.info("Step 7: Executing granular subgroup performance audit...")
    subgroup_cols = ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen", "gender"]
    subgroup_audit_df = audit_subgroup_performance(
        dev_results_df,
        y_true_col="Actual_Churn",
        y_prob_col="churn_probability",
        subgroup_cols=subgroup_cols,
        decision_threshold=0.5,
    )
    subgroup_audit_df.to_csv(TABLES_DIR / "subgroup_performance.csv", index=False)
    subgroup_chart_path = plot_subgroup_audit_summary(subgroup_audit_df)
    logger.info(f"Saved subgroup audit chart to {subgroup_chart_path}")

    # 8. Error Analysis & Failure Mode Diagnostics
    logger.info("Step 8: Diagnosing model error distributions (FP vs FN)...")
    error_summary = summarize_error_distributions(dev_results_df)
    with open(TABLES_DIR / "error_diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)

    error_plot_path = plot_error_diagnostics(dev_results_df)
    logger.info(f"Saved error diagnostic visual to {error_plot_path}")

    # 9. Robustness & Feature Ablation Study
    logger.info("Step 9: Running feature ablation study...")
    ablation_df = run_feature_ablation_study(X_dev, y_dev, random_state=RANDOM_SEED)
    ablation_df.to_csv(TABLES_DIR / "feature_ablation_results.csv")
    ablation_chart_path = plot_ablation_comparison(ablation_df)
    logger.info(f"Saved feature ablation chart to {ablation_chart_path}")
    print("\n--- FEATURE ABLATION COMPARISON (5-Fold CV on Dev Partition) ---")
    print(ablation_df[["PR_AUC_Mean", "PR_AUC_Std", "ROC_AUC_Mean", "F1_Mean"]])

    logger.info("=== Phase 3 Execution Successfully Completed ===")


if __name__ == "__main__":
    main()
