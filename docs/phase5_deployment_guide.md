# Phase 5: Production Deployment Guide

## 1. Quickstart Deployment Guide

### Option A: Local Execution (Windows PowerShell)

```powershell
# 1. Clone repository and navigate to root
cd "d:\Daniel\projects\project 1"

# 2. Create and activate a fresh virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install pinned dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run automated test suite (58 tests)
python -m pytest -v

# 5. Run pre-deployment operational readiness check
python -m src.run_phase5_check

# 6. Launch the Streamlit Executive Dashboard
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

---

### Option B: Docker Container Deployment

```bash
# 1. Build the Docker container image
docker build -t northstar-churn-dashboard .

# 2. Run the container with standard port mapping
docker run -d --name northstar_saas -p 8501:8501 northstar-churn-dashboard

# 3. Check health and startup logs
docker logs -f northstar_saas

# 4. Open browser to http://localhost:8501

# 5. Stop and clean up container
docker stop northstar_saas && docker rm northstar_saas
```

---

### Option C: Docker Compose Orchestration

```bash
# 1. Start service in detached mode
docker compose up -d --build

# 2. View running service status and health
docker compose ps

# 3. Follow application logs
docker compose logs -f

# 4. Stop service cleanly
docker compose down
```

#### Custom Host Port Override:
If port `8501` is already occupied on your host:
```bash
HOST_PORT=8502 docker compose up -d
```
Then navigate to `http://localhost:8502`.
