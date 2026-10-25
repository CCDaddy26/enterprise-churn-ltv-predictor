# Phase 5: Pre-Deployment Repository Audit

## 1. Executive Context & Scope
This audit documents the repository state, software environment, model artifacts, data governance, test status, and deployment readiness of **Project 01/30 (Leak-Free Enterprise Churn & LTV Predictor)** before Phase 5 containerization and packaging.

---

## 2. Repository Structure
```text
.
â"œâ"€â"€ .env.example                               # Template for configurable runtime paths & seeds
â"œâ"€â"€ .gitignore                                 # Git exclusion rules
â"œâ"€â"€ README.md                                  # Executive summary, methodology, and setup guide
â"œâ"€â"€ app.py                                     # Streamlit executive dashboard entry point
â"œâ"€â"€ pytest.ini                                 # Test discovery & warning filters
â"œâ"€â"€ requirements.txt                           # Top-level dependency manifest
â"œâ"€â"€ data/
â"‚   â"œâ"€â"€ raw/Telco-Customer-Churn.csv           # IBM Telco public dataset (7,043 rows, 21 cols)
â"‚   â"œâ"€â"€ interim/                               # Scratch & intermediate partitions
â"‚   â""â"€â"€ processed/                             # Processed clean parquet
â"œâ"€â"€ docs/
â"‚   â"œâ"€â"€ data_sources.md                        # Dataset provenance & attribution
â"‚   â"œâ"€â"€ data_dictionary.md                     # Feature definitions
â"‚   â"œâ"€â"€ data_leakage_policy.md                 # Strict split & transform rules
â"‚   â"œâ"€â"€ prediction_contract.md                 # Schema input/output interface
â"‚   â"œâ"€â"€ model_validation_protocol.md           # 5-fold CV protocol
â"‚   â"œâ"€â"€ ltv_and_roi_methodology.md             # Theoretical foundations
â"‚   â"œâ"€â"€ phase4_financial_methodology.md        # Formula derivations & cost triggers
â"‚   â"œâ"€â"€ phase4_dashboard_guide.md              # Navigation guide
â"‚   â""â"€â"€ phase4_assumptions.md                  # Assumption catalog
â"œâ"€â"€ models/
â"‚   â"œâ"€â"€ champion_pipeline.joblib               # Serialized Tuned XGBoost champion (175,039 bytes)
â"‚   â"œâ"€â"€ sigmoid_calibrator.joblib              # Cross-fitted Platt Sigmoid calibrator (1,349 bytes)
â"‚   â""â"€â"€ model_metadata.json                    # Model hyperparameters, metrics, and contracts
â"œâ"€â"€ reports/
â"‚   â"œâ"€â"€ figures/                               # 16 visual diagnostics (SHAP, ROC, PR, Calibration)
â"‚   â"œâ"€â"€ tables/                                # Generated CSV/JSON tables (Mode A/B, Break-Even)
â"‚   â"œâ"€â"€ phase1_summary.md ... phase4_summary.md# Prior phase completion reports
â"‚   â"œâ"€â"€ phase4_preflight_audit.md              # Phase 4 preflight checks
â"‚   â""â"€â"€ phase4_validation.md                   # Automated quality gate logs
â"œâ"€â"€ src/                                       # 26 modular Python components
â"‚   â"œâ"€â"€ config.py, data_loader.py, data_validation.py, feature_engineering.py
â"‚   â"œâ"€â"€ preprocessing.py, baseline.py, model_training.py, inference.py
â"‚   â"œâ"€â"€ explainability.py, calibration.py, error_analysis.py, subgroup_audit.py
â"‚   â"œâ"€â"€ ltv_engine.py, roi_simulator.py, scenario_analysis.py, dashboard_data.py
â"‚   â""â"€â"€ run_phase1.py ... run_phase4.py
â""â"€â"€ tests/                                     # 12 test suites (54 automated tests)
```

---

## 3. Existing Application Entry Point & Startup
- **Entry Point:** [`app.py`](file:///d:/Daniel/projects/project%201/app.py)
- **Local Startup Command:**
  ```powershell
  python -m streamlit run app.py
  ```
- **Default Port:** `8501` (bound locally to `http://localhost:8501`)

---

## 4. Model and Calibrator Artifacts

| Artifact | File Path | File Size | Status / Role | Required at Runtime? |
| :--- | :--- | :---: | :--- | :---: |
| **Champion Model Pipeline** | `models/champion_pipeline.joblib` | 175,039 bytes | Tuned XGBoost + Preprocessor | **YES (Mandatory)** |
| **Model Metadata** | `models/model_metadata.json` | 681 bytes | Contract, hyperparameters, seed | **YES (Mandatory)** |
| **Sigmoid Calibrator** | `models/sigmoid_calibrator.joblib` | 1,349 bytes | Cross-fitted Platt Calibrator | **YES (Mandatory for Calibrated Mode)** |

*Artifact Safety:* The application loads these artifacts in read-only inference mode; no training or disk mutation occurs during application startup.

---

## 5. Test Suite Verification & Baseline Results
An immediate audit run of `python -m pytest -v` was executed at the start of Phase 5:
- **Total Tests Collected:** 54
- **Passed:** 54
- **Failed:** 0
- **Execution Time:** 13.76 seconds (Python 3.11.6 on Windows).

---

## 6. Dataset Access, Licensing & Redistribution
- **Source:** IBM Telco Customer Churn public dataset (`Telco-Customer-Churn.csv`).
- **License / Terms:** Public educational dataset provided by IBM Community / ICP4D.
- **Redistribution Policy:** Suitable for educational and non-commercial portfolio demonstrations. Personal identifiers do not exist (only synthetic alphanumeric keys like `7590-VHVEG`).
- **Anonymization in Deliverables:** In all exported report tables (`reports/tables/phase4_customer_sample_anonymized.csv`), synthetic keys are re-masked with generic analytical tokens (`ANON_CUST_00001`).

---

## 7. Security, Secrets & Local Paths
- **Secrets Audit:** Grep search over the repository confirmed **zero** API keys, passwords, bearer tokens, or cloud credentials.
- **Path Portability:** [`src/config.py`](file:///d:/Daniel/projects/project%201/src/config.py) resolves all paths dynamically relative to `PROJECT_ROOT = Path(__file__).resolve().parent.parent`. No hardcoded machine-specific absolute Windows paths (`d:\Daniel\...`) exist in runtime modules.
- **Private Data:** No private proprietary customer databases are accessed.

---

## 8. Dependency Compatibility & Docker Availability
- **Python Environment:** Python 3.11.6.
- **Core Runtime Libraries:** `streamlit 1.58.0`, `pandas 2.3.2`, `numpy 2.3.2`, `scikit-learn 1.7.1`, `xgboost 3.2.0`, `shap 0.51.0`, `joblib 1.5.2`.
- **Docker CLI Availability:** Running `docker --version` returns `CommandNotFoundException` on the host machine.
  - *Governing Rule:* In accordance with Phase 5 instructions, all Docker packaging files (`Dockerfile`, `.dockerignore`, `compose.yaml`) will be authored to strict production standards and verified for syntactical correctness, but container execution will be reported honestly as `NOT RUN â€" Docker unavailable on host system`. No container execution will be fabricated.

---

## 9. Known Technical & Modeling Limitations
1. **Retrospective Dataset:** Cross-sectional without event timestamps; prevents continuous-time hazard modeling.
2. **Observational Attribution:** Predictive scores do not imply causal uplift.
3. **Illustrative Educational Proxy:** LTV and ROI figures are user-scenario estimates, not validated corporate financial statements.
