# Phase 5: Container Validation & Docker Readiness Report

## 1. Docker Runtime Environment Status
- **Host Operating System:** Windows 10/11 x86_64
- **Docker CLI Execution Probe:**
  ```powershell
  docker --version
  ```
  **Result:** `CommandNotFoundException: The term 'docker' is not recognized as the name of a cmdlet, function, script file, or operable program.`
- **Audit Determination:** In strict adherence to Phase 5 truthfulness rules (*"If Docker is unavailable, do not fabricate a build or container test. Complete the Docker configuration, validate the files and startup commands as far as possible, and mark runtime container validation as NOT RUN â€" Docker unavailable"*), runtime container execution is recorded as:
  **`NOT RUN â€" Docker unavailable on host system`**.

---

## 2. Docker Configuration Audit & Verification

Although the Docker daemon is absent from this Windows host, the Docker packaging components were constructed to production specifications and verified:

### 2.1 `Dockerfile` Specification
- **Base Image:** `python:3.11-slim` (official, lightweight, security-patched Debian base).
- **Security:** Creates dedicated non-root user `appuser` (UID 1000) and executes with dropped root privileges.
- **Port Binding:** Binds explicitly to `0.0.0.0:8501`.
- **Reproducibility:** Copies pinned `requirements.txt` and runs `pip install --no-cache-dir`.
- **Healthcheck:** Configured with native `curl -f http://localhost:8501/_stcore/health || exit 1`.

### 2.2 `.dockerignore` Specification
Strictly excludes:
- Local git history (`.git/`, `.gitignore`)
- Virtual environments (`venv/`, `.venv/`, `env/`)
- Test caches and bytecode (`.pytest_cache/`, `__pycache__/`, `*.pyc`)
- Private secrets and environment files (`.env`, `.env.local`, `*.log`)
- Exploratory notebooks and scratch directories (`notebooks/`, `docs/`)

### 2.3 `compose.yaml` Specification
- Standard modern Compose spec (`services.northstar-churn-dashboard`).
- Dynamic port mapping: `"${HOST_PORT:-8501}:8501"`, allowing port conflict overrides.
- No privileged flags, host filesystem pollution, or Docker socket mounting.

---

## 3. Local Operational Diagnostics Equivalent
Because the application is packaged to run identical Python code inside and outside the container, local verification of the operational checklist was conducted via:
```powershell
python -m src.run_phase5_check
```
**Results:**
1. Raw dataset verified (`data/raw/Telco-Customer-Churn.csv`, 970,457 bytes).
2. Champion model artifact verified (`models/champion_pipeline.joblib`, 175,039 bytes).
3. Sigmoid calibrator verified (`models/sigmoid_calibrator.joblib`, 879 bytes).
4. Model metadata contract verified (`models/model_metadata.json`, 681 bytes).
5. Non-mutating load verified (artifact mtime untouched).
6. Synthetic customer prediction executed cleanly (Probability = 0.8709).
7. Calibrator mapping executed cleanly (Calibrated Probability = 0.7142).
8. Financial calculation logic verified (Remaining Contrib = â‚¹840.0, VaR = â‚¹504.0).
9. Zero-cost baseline ROI safety verified (`None` / `"N/A â€" no campaign expenditure"`).
10. Live HTTP probe verified (`HTTP 200 OK` on `http://localhost:8501`).
