# ============================================================================
# Enterprise RAG Assistant — Dockerfile
#
# Target: Hugging Face Spaces (Docker Space, free CPU tier)
# Base:   python:3.11-slim  (no CUDA, no torch — keep image <2 GB)
# Ports:
#   8000 — FastAPI / uvicorn  (localhost, not exposed publicly)
#   7860 — Streamlit          (HF Spaces' required public port)
#
# Ephemeral disk note:
#   HF free-tier Spaces reset the container filesystem on every restart.
#   data/lancedb/ is therefore NOT committed — start.sh re-seeds it from
#   the committed corpus (data/corpus/) on every boot.
#   eval/reports/ IS committed in the image so the dashboard is always
#   populated without needing a live eval run.
#
# Secret management:
#   GOOGLE_API_KEY is set as an HF Space secret (never in this file or the repo).
#   All other tunables are set as HF public env vars.
# ============================================================================

FROM python:3.11-slim

# ── System dependencies ─────────────────────────────────────────────────────
# git  : used by MLflow tracker to capture commit hash (gracefully degrades)
# curl : optional healthcheck / debugging in the container
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces requirement) ────────────────────────────
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Install Python dependencies (production only — no dev/test groups) ──────
# Copy dependency manifests first to maximise Docker layer cache reuse.
# A change to source code does NOT invalidate this layer.
COPY pyproject.toml ./

# Install pip + the project's *production* dependencies only.
# We explicitly exclude the [dev] group which contains mlflow, ragas,
# pytest, datasets, numpy, langchain-community, langchain, and ruff.
# MLflow graceful degradation (NoOpTracker) means the app runs fine without it.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi>=0.115.0,<1.0" \
        "uvicorn[standard]>=0.32.0,<1.0" \
        "pydantic>=2.10.0,<3.0" \
        "pydantic-settings>=2.6.0,<3.0" \
        "pyyaml>=6.0.2,<7.0" \
        "lancedb>=0.15.0,<0.39" \
        "google-genai>=1.14.0,<3.0" \
        "httpx>=0.28.0,<1.0" \
        "tenacity>=9.0.0,<10.0" \
        "pypdf>=5.0.0,<6.0" \
        "streamlit>=1.66.0" \
        "python-multipart>=0.0.32" \
        "python-dotenv>=1.0.0"

# ── Copy application source ─────────────────────────────────────────────────
# Order: least-changing → most-changing for best cache utilisation
COPY configs/        ./configs/
COPY prompts/        ./prompts/
COPY data/corpus/    ./data/corpus/
COPY eval/reports/   ./eval/reports/
COPY eval/golden/    ./eval/golden/
COPY .streamlit/     ./.streamlit/
COPY app/            ./app/
COPY ui/             ./ui/
COPY scripts/        ./scripts/
COPY start.sh        ./start.sh

# ── Prepare writable runtime directories ────────────────────────────────────
# data/lancedb/ is seeded at runtime (ephemeral); data/uploads/ for any
# future upload endpoint use.
RUN mkdir -p data/lancedb data/uploads /tmp \
    && chown -R appuser:appuser /app /tmp

# ── Streamlit config — dynamic port support ─────────────────────────────────
# Remove static port line so start.sh's dynamic --server.port takes precedence
RUN sed -i '/^port =/d' /app/.streamlit/config.toml

# Make start.sh executable
RUN chmod +x /app/start.sh

# ── Runtime environment defaults ────────────────────────────────────────────
# These are safe defaults. Override with cloud env vars / secrets.
ENV PYTHONPATH="/app" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1" \
    BACKEND_URL="http://localhost:8000" \
    DEMO_MODE="true" \
    DEMO_DAILY_CAP="50" \
    QUOTA_FILE_PATH="/tmp/daily_quota.json" \
    MLFLOW_TRACKING_URI=""

# Switch to non-root user
USER appuser

# Expose common PaaS ports (HF Spaces: 7860, Render: 10000)
EXPOSE 7860 10000

# Healthcheck — Streamlit built-in health endpoint on active port
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/_stcore/health || exit 1

ENTRYPOINT ["/bin/bash", "/app/start.sh"]

