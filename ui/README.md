# Meridian Enterprise RAG — Streamlit Frontend

A decoupled, recruiter-demoable multipage web interface for the Meridian Enterprise RAG platform.

---

## 🏛️ Architecture Overview

The frontend lives in `ui/` and communicates **strictly over HTTP** with the FastAPI backend using a typed API client (`ui/api_client.py`). It imports **zero backend modules**, ensuring:
- **Independent Deployment**: The frontend can be hosted on Streamlit Cloud or Hugging Face Spaces while the backend runs on Cloud Run, Kubernetes, or a separate container.
- **Contract-First Design**: The backend exposes an OpenAPI specification (`/docs`) treated as a public product.
- **Demo Safety**: A single environment variable (`DEMO_MODE=true`) locks mutating ingestion actions and rate-limits public user sessions.

---

## 🚀 Running Locally

### Step 1: Launch the FastAPI Backend
In your first terminal:
```powershell
uv run uvicorn app.main:app --reload --port 8000
```
Verify the backend is live at `http://localhost:8000/health`.

### Step 2: Launch the Streamlit Frontend
In a second terminal:
```powershell
uv run streamlit run ui/Home.py
```
*(Note: You can also run `ui/app.py`, but `ui/Home.py` ensures the sidebar navigation displays cleanly as "Home" rather than "app".)*
Streamlit will automatically open the application at `http://localhost:8501`.

---

## ⚙️ Configuration & Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `BACKEND_URL` | `http://localhost:8000` | Target URL of the FastAPI backend service |
| `DEMO_MODE` | `false` | When `true`, disables ingestion uploads and limits chat to 10 queries per session |

### Running in Demo Mode (Portfolio Showcase):
```powershell
$env:DEMO_MODE="true"; uv run streamlit run ui/app.py
```

---

## 📋 Manual Verification & Click-Through Checklist

Follow this checklist to verify all features end-to-end:

### 1. Home Page (`ui/app.py`)
- [ ] Backend status pill shows **🟢 Backend Online**.
- [ ] Telemetry cards show live **Corpus Documents**, **Vector Chunks**, and **Active LLM**.
- [ ] Navigation buttons route smoothly to Chat, Eval Dashboard, and Ingestion.

### 2. Chat Assistant (`pages/1_💬_Chat.py`)
- [ ] Click the starter question: *"What is the deadline for drafting a PIR after an incident?"*
- [ ] Verify the response includes citation markers (e.g. `[1]`).
- [ ] Click the **`[1] 📄 incident-response-plan.md`** expander:
  - Passage text is clearly visible.
  - All 4 scores are populated: **Vector Score**, **FTS Score**, **RRF Score**, and **LLM Rerank Score**.
- [ ] Check the sidebar:
  - Latency breakdown shows **Retrieval**, **Reranking**, **Generation**, and **Total ms**.
  - Token usage counts (in / out) are displayed.
  - Grounded badge **✅ Fully Grounded** is active.
- [ ] Test Refusal Guardrail: Ask *"What is the company dress code policy for casual Fridays?"*
  - Pipeline responds: *"I don't have enough information in the available documents to answer this question."*
  - Sidebar shows **⚠️ Refusal Triggered** badge.

### 3. Evaluation Dashboard (`pages/2_📊_Eval_Dashboard.py`)
- [ ] Run selector lists historical runs (`run_A`, `run_B`, `run_C`).
- [ ] Aggregate metric cards display **Hit Rate**, **MRR**, **NDCG**, and **Refusal Accuracy**.
- [ ] Configuration expander shows parameters for the selected run.
- [ ] Per-Question table:
  - Toggle **🚨 Worst-First Ordering**: table sorts by lowest MRR / Hit Rate.
  - Filter by Question Type (e.g. `factual`, `multi-hop`, `out-of-scope`).
- [ ] Question Drill-Down:
  - Select question `q001` or `q015`.
  - View **Generated Answer** vs **Ground Truth Answer** side-by-side.
  - Expand retrieved context passages.
- [ ] Tab 2 (Multi-Run Comparison):
  - Select `run_A`, `run_B`, and `run_C`.
  - View comparative metrics table and metric bar chart.
- [ ] Tab 3 (Experiment Narrative):
  - Renders the Phase 4 A→D progression markdown document (`experiment_1.md`).

### 4. Ingestion Console (`pages/3_📥_Ingestion.py`)
- [ ] Index telemetry displays Total Chunks Indexed and Document Count from LanceDB.
- [ ] In standard mode (`DEMO_MODE=false`), file uploader allows selecting `.md`, `.txt`, `.pdf`.
- [ ] In demo mode (`DEMO_MODE=true`), shows clear lock notice and disables destructive actions.

### 5. Backend Down Resilience
- [ ] Terminate the uvicorn backend terminal (Ctrl+C).
- [ ] Refresh any page in the Streamlit UI.
- [ ] Verify that every page displays a friendly alert: *"Backend server not reachable at http://localhost:8000. Is the FastAPI service running?"*
- [ ] **Zero Python tracebacks or raw stack traces** are displayed to the user.

