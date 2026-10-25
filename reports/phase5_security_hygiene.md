# Phase 5: Security & Repository Hygiene Audit

## 1. Secrets & Credentials Audit
A recursive regex scan was conducted across the workspace for sensitive patterns (API keys, AWS credentials, bearer tokens, database connection strings, and private keys):
- **Findings:** **Zero secrets found.**
- **Environment Template:** `.env.example` contains only benign path pointers and configuration flags. No `.env` with actual production secrets is committed.

---

## 2. Customer Privacy & Data Protection

| Vector | Status | Protective Mechanism |
| :--- | :---: | :--- |
| **Public Exports** | PROTECTED | Committed sample reports (`reports/tables/phase4_customer_sample_anonymized.csv`) mask customer IDs with synthetic tokens (`ANON_CUST_00001`). |
| **Interactive Dashboard** | PROTECTED | Customer Explorer table displays anonymized analytical keys (`ANON_00001`). Raw customer IDs are excluded from presentation layouts. |
| **Telemetry & Analytics** | DISABLED | `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` configured in Dockerfile and Compose to eliminate third-party telemetry calls. |
| **Local-First Architecture** | ENFORCED | Inference and financial calculations run 100% locally in-process without outbound API requests. |

---

## 3. Git Exclusion & Hygiene (.gitignore Review)
The repository `.gitignore` was audited and confirmed to exclude:
- Virtual environments (`venv/`, `.venv/`, `env/`)
- Cache directories (`__pycache__/`, `.pytest_cache/`, `*.pyc`)
- Local log outputs (`*.log`)
- Temporary files and OS thumbnails (`.DS_Store`, `Thumbs.db`)
- Large generated scratch directories (`data/interim/*`, `data/processed/*`)

---

## 4. Container Security Hygiene
- **Least Privilege:** The `Dockerfile` drops root privileges, creating and executing as `appuser` (UID 1000).
- **No Docker Socket Mounting:** `compose.yaml` does not mount `/var/run/docker.sock`.
- **Minimal Image Surface:** Utilizes `python:3.11-slim`, stripping unnecessary compilers after package installation.
- **Port Isolation:** Only port 8501 is exposed.

---

## 5. Public Educational Attribution & Licensing
- **Dataset Attribution:** Publicly sourced from IBM Community Telco Customer Churn.
- **Fictional Enterprise Boundary:** The business context ("Northstar SaaS") is prominently labeled across all documentation, reports, and UI banners as an illustrative educational scenario.
