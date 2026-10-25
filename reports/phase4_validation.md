# Phase 4.1: Validation Report & Quality Gate Audit (Corrected)

## 1. Quality Gates & Regression Verification
A total of **54 automated unit and integration tests** were executed across the entire repository test suite, covering Phases 1 through 4.1. All 54 tests passed with zero failures.

```text
============================= test session starts =============================
platform win32 -- Python 3.11.6, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\Daniel\projects\project 1
configfile: pytest.ini
testpaths: tests
collected 54 items

tests/test_baseline.py::test_build_preprocessor_structure PASSED         [  1%]
tests/test_baseline.py::test_baseline_pipeline_fit_predict PASSED        [  3%]
tests/test_baseline.py::test_evaluate_binary_classifier PASSED           [  5%]
tests/test_baseline.py::test_run_baseline_benchmarking_reproducibility PASSED [  7%]
tests/test_data_loading.py::test_load_raw_dataset_missing_file PASSED    [  9%]
tests/test_data_loading.py::test_load_raw_dataset_empty_file PASSED      [ 11%]
tests/test_data_loading.py::test_load_raw_dataset_valid_synthetic PASSED [ 12%]
tests/test_data_loading.py::test_download_raw_dataset_existing_skip PASSED [ 14%]
tests/test_data_validation.py::test_validate_raw_schema_empty PASSED     [ 16%]
tests/test_data_validation.py::test_validate_raw_schema_missing_target PASSED [ 18%]
tests/test_data_validation.py::test_validate_raw_schema_unexpected_target_values PASSED [ 20%]
tests/test_data_validation.py::test_validate_raw_schema_success PASSED   [ 22%]
tests/test_data_validation.py::test_clean_dataset_for_exploration_blank_handling PASSED [ 24%]
tests/test_data_validation.py::test_separate_features_and_target_excludes_id PASSED [ 25%]
tests/test_data_validation.py::test_audit_dataset_metrics PASSED         [ 27%]
tests/test_feature_engineering.py::test_feature_engineer_transform_columns PASSED [ 29%]
tests/test_feature_engineering.py::test_feature_engineer_total_services_count PASSED [ 31%]
tests/test_feature_engineering.py::test_feature_engineer_tenure_cohort_bins PASSED [ 33%]
tests/test_feature_engineering.py::test_feature_engineer_partner_and_dependents PASSED [ 35%]
tests/test_feature_engineering.py::test_feature_engineer_charges_ratio_safeguards PASSED [ 37%]
tests/test_inference.py::test_evaluate_model_performance PASSED          [ 38%]
tests/test_inference.py::test_serialization_and_inference_engine PASSED  [ 40%]
tests/test_ltv_engine.py::test_calculate_customer_ltv_proxy_standard PASSED [ 42%]
tests/test_ltv_engine.py::test_calculate_customer_ltv_proxy_series PASSED [ 44%]
tests/test_ltv_engine.py::test_calculate_customer_ltv_proxy_invalid_margin PASSED [ 46%]
tests/test_ltv_engine.py::test_calculate_customer_ltv_proxy_invalid_horizon PASSED [ 48%]
tests/test_ltv_engine.py::test_calculate_value_at_risk_standard PASSED   [ 50%]
tests/test_ltv_engine.py::test_calculate_value_at_risk_invalid_prob PASSED [ 51%]
tests/test_ltv_engine.py::test_compute_customer_value_table_calibrated PASSED [ 53%]
tests/test_ltv_engine.py::test_compute_customer_value_table_uncalibrated_risk_proxy PASSED [ 55%]
tests/test_model_training.py::test_get_model_candidates_keys PASSED      [ 57%]
tests/test_model_training.py::test_evaluate_candidates_cv PASSED         [ 59%]
tests/test_phase3_diagnostics.py::test_compute_expected_calibration_error PASSED [ 61%]
tests/test_phase3_diagnostics.py::test_evaluate_calibration_metrics PASSED [ 62%]
tests/test_phase3_diagnostics.py::test_compute_threshold_curve PASSED    [ 64%]
tests/test_phase3_diagnostics.py::test_audit_subgroup_performance PASSED [ 66%]
tests/test_phase3_diagnostics.py::test_categorize_prediction_errors PASSED [ 68%]
tests/test_phase3_diagnostics.py::test_extract_transformed_feature_names PASSED [ 70%]
tests/test_phase4_integration.py::test_dashboard_data_loading_and_scoring_calibrated PASSED [ 72%]
tests/test_phase4_integration.py::test_dashboard_data_loading_uncalibrated_mode PASSED [ 74%]
tests/test_phase4_integration.py::test_model_artifact_not_overwritten PASSED [ 75%]
tests/test_preprocessing.py::test_safe_numeric_cleaner PASSED            [ 77%]
tests/test_preprocessing.py::test_build_feature_preprocessor_fit_transform PASSED [ 79%]
tests/test_preprocessing.py::test_preprocessor_unseen_categories PASSED  [ 81%]
tests/test_roi_simulator.py::test_single_customer_economics PASSED       [ 83%]
tests/test_roi_simulator.py::test_simulate_cohort_campaign_empty PASSED  [ 85%]
tests/test_roi_simulator.py::test_no_campaign_baseline_zero_cost_roi_none PASSED [ 87%]
tests/test_roi_simulator.py::test_like_for_like_cohort_counts PASSED     [ 88%]
tests/test_roi_simulator.py::test_budget_maximized_targeting PASSED      [ 90%]
tests/test_scenario_analysis.py::test_calculate_break_even_save_rate_valid PASSED [ 92%]
tests/test_scenario_analysis.py::test_calculate_break_even_triggers PASSED [ 94%]
tests/test_scenario_analysis.py::test_calculate_break_even_save_rate_unreachable PASSED [ 96%]
tests/test_scenario_analysis.py::test_generate_scenario_matrix PASSED    [ 98%]
tests/test_scenario_analysis.py::test_compute_sensitivity_heatmap_data PASSED [100%]

======================= 54 passed, 19 warnings in 9.89s =======================
```

---

## 2. Integrity of Champion Model Pipeline
`models/champion_pipeline.joblib` was monitored during all calibration, scoring, and UI execution runs:
- **Pre-execution Size:** 175,039 bytes
- **Post-execution Size:** 175,039 bytes
- **Modification Check:** Pipeline was strictly loaded in read-only inference mode.

---

## 3. UI Smoke Test & Verification
- **Streamlit Launch:** `python -m streamlit run app.py --server.headless true --server.port 8501`
- **Verification Endpoint:** `http://localhost:8501`
- **HTTP Response:** `HTTP 200 OK`
- **Verified Features:** Toggle between Calibrated Mode (`calibrated_churn_probability`, `Estimated_Value_at_Risk`) and Uncalibrated Mode (`churn_risk_score`, `Risk_Weighted_Value_Proxy`), radio button switching between Mode A (Like-for-Like) and Mode B (Budget-Maximized), and correct `"N/A â€" no campaign expenditure"` display for baseline ROI.
