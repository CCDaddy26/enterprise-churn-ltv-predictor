"""Master pipeline runner for Phase 1: Business Discovery & Data Foundation."""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ensure_directories, RAW_DATA_PATH, TABLES_DIR
from src.data_loader import download_raw_dataset, load_raw_dataset
from src.data_validation import validate_raw_schema, audit_dataset
from src.eda import run_full_eda
from src.baseline import run_baseline_benchmarking

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("phase1_runner")


def main():
    logger.info("Initializing Phase 1 pipeline execution...")
    ensure_directories()

    # 1. Download / Verify Raw Dataset
    logger.info("Step 1: Checking and obtaining raw dataset...")
    raw_path = download_raw_dataset()
    logger.info(f"Raw dataset confirmed at {raw_path}")

    # 2. Load Raw Dataset
    logger.info("Step 2: Loading raw dataset into memory...")
    df_raw = load_raw_dataset(raw_path)
    logger.info(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # 3. Schema Validation & Audit
    logger.info("Step 3: Validating schema and running audit...")
    validation_info = validate_raw_schema(df_raw)
    logger.info(f"Schema validation passed: {validation_info}")

    audit_info = audit_dataset(df_raw)
    with open(TABLES_DIR / "dataset_audit_summary.json", "w") as f:
        json.dump(audit_info, f, indent=2)
    logger.info(f"Saved dataset audit to {TABLES_DIR / 'dataset_audit_summary.json'}")

    # 4. Exploratory Data Analysis & Visualizations
    logger.info("Step 4: Running Exploratory Data Analysis and generating charts...")
    eda_results = run_full_eda(df_raw)
    logger.info(f"Generated {len(eda_results['figure_paths'])} charts and saved tables.")

    # 5. Baseline Modeling & Evaluation
    logger.info("Step 5: Training and evaluating leakage-resistant baselines...")
    baseline_results = run_baseline_benchmarking(df_raw)
    with open(TABLES_DIR / "baseline_results.json", "w") as f:
        json.dump(baseline_results, f, indent=2)

    logger.info("=== Phase 1 Pipeline Successfully Completed ===")
    print("\n--- BASELINE EVALUATION SUMMARY (Held-Out Test Set, n=1,409) ---")
    for model, metrics in baseline_results["test_results"].items():
        print(f"\nModel: {model}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, TN: {metrics['true_negatives']}, FN: {metrics['false_negatives']}")


if __name__ == "__main__":
    main()
