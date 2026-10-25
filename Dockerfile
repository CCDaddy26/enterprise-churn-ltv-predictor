# =============================================================================
# Dockerfile: Production-Conscious Packaging for Northstar SaaS Dashboard
# Base: Official Python 3.11 Slim (Debian)
# =============================================================================

FROM python:3.11-slim

# Set environment variables for non-interactive Python execution and container port
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install system dependencies needed for compiling C-extensions if required, curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root system user for security hygiene
RUN useradd -u 1000 -m appuser && \
    mkdir -p /app/reports/tables /app/reports/figures && \
    chown -R appuser:appuser /app

# Copy application source code, models, and runtime data
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser app.py /app/app.py
COPY --chown=appuser:appuser .env.example /app/.env.example

# Switch to non-root user
USER appuser

# Expose standard Streamlit port
EXPOSE 8501

# Lightweight HTTP healthcheck using built-in curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit executive dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
