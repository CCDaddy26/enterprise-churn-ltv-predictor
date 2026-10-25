# Phase 5: Troubleshooting & Operations Guide

## 1. Port Conflicts (Port 8501 Busy)

### Symptom:
`OSError: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8501)`

### Resolution on Windows:
```powershell
# Identify process listening on port 8501
netstat -ano | findstr :8501

# Launch Streamlit on an alternate port
python -m streamlit run app.py --server.port 8502
```

### Resolution with Docker Compose:
```bash
HOST_PORT=8502 docker compose up -d
```

---

## 2. Missing Model or Calibrator Artifacts

### Symptom:
`FileNotFoundError: Model artifact not found at models/champion_pipeline.joblib` or `sigmoid_calibrator.joblib`

### Resolution:
Regenerate model pipelines and cross-fit calibrators via the automated reproducible pipeline runners:
```powershell
# 1. Regenerate champion model pipeline
python -m src.run_phase2

# 2. Regenerate cross-fit calibrator artifact
python -m src.run_phase3

# 3. Regenerate scenario tables
python -m src.run_phase4

# 4. Verify readiness
python -m src.run_phase5_check
```

---

## 3. Virtual Environment Script Execution Policy Error

### Symptom:
`.\venv\Scripts\Activate.ps1 : File cannot be loaded because running scripts is disabled on this system.`

### Resolution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\venv\Scripts\Activate.ps1
```

---

## 4. Health Check Verification

### Local Probe:
```powershell
python -c "import urllib.request; resp = urllib.request.urlopen('http://localhost:8501'); print('Status:', resp.status)"
```
Expected output: `Status: 200`.

### Container Probe:
```bash
curl -f http://localhost:8501/_stcore/health
```
Expected output: `ok`.
