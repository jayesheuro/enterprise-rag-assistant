#!/bin/bash
# start.sh — Container entrypoint for Enterprise RAG Assistant on Hugging Face Spaces.
#
# Execution order:
#   1. Seed the vector store from the bundled corpus (idempotent, skips if already done).
#   2. Launch the FastAPI backend (uvicorn) on port 8000 (localhost only).
#   3. Launch the Streamlit frontend on port 7860 (HF Spaces' expected public port).
#
# HF Spaces ephemeral disk note:
#   The container filesystem is reset on every restart. data/lancedb/ is therefore
#   always empty on boot — the seed step regenerates it from the committed corpus.
#   eval/reports/ is committed in the image and survives restarts untouched.
#
# Environment variables (set as HF Space secrets/vars, never hard-coded here):
#   GOOGLE_API_KEY  — Gemini API key (HF Secret — never logged)
#   DEMO_MODE       — "true" for public demo (set as HF public env var)
#   DEMO_DAILY_CAP  — global daily question cap (default: 50)
#   DEMO_QUERY_LIMIT — per-session cap (default: 10, enforced in UI)
#   BACKEND_URL     — FastAPI base URL seen by the UI (default below)

set -e  # Exit immediately on any error during seed

export BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
export PYTHONPATH="/app:${PYTHONPATH}"

echo "=========================================="
echo " Enterprise RAG Assistant — Starting Up"
echo "=========================================="
echo "DEMO_MODE    : ${DEMO_MODE:-false}"
echo "BACKEND_URL  : ${BACKEND_URL}"
echo "DEMO_DAILY_CAP: ${DEMO_DAILY_CAP:-50}"
echo "=========================================="

# Step 1 — Seed corpus into LanceDB (must complete before backend starts)
echo "[seed] Running demo seed..."
python /app/scripts/seed_demo.py
echo "[seed] Done."

# Step 2 — Start FastAPI backend (background, localhost only)
echo "[backend] Starting uvicorn on port 8000..."
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &
BACKEND_PID=$!

# Give the backend a moment to bind before Streamlit tries to connect
sleep 3

# Step 3 — Start Streamlit frontend (foreground, port 7860 for HF Spaces)
echo "[ui] Starting Streamlit on port 7860..."
streamlit run ui/Home.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.fileWatcherType none &
# Cleanup handler for graceful shutdown on SIGINT/SIGTERM (e.g. Ctrl+C)
cleanup() {
    echo "[start.sh] Termination signal received. Stopping services..."
    kill -TERM $BACKEND_PID $UI_PID 2>/dev/null || true
    wait $BACKEND_PID $UI_PID 2>/dev/null || true
    echo "[start.sh] Clean shutdown complete."
    exit 0
}
trap cleanup SIGINT SIGTERM

# Wait for either process to exit — if either dies, stop the other
echo "[start.sh] Both services running. Backend PID=${BACKEND_PID}, UI PID=${UI_PID}"
wait -n $BACKEND_PID $UI_PID
EXIT_CODE=$?

echo "[start.sh] A service exited with code ${EXIT_CODE}. Shutting down."
cleanup

