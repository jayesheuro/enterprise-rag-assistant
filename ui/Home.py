"""Meridian Enterprise RAG Assistant — Streamlit Application Entrypoint.

Home page providing navigation, system health telemetry, and architectural overview.
"""

import streamlit as st

from ui.api_client import DEMO_MODE, APIClient, APIConnectionError
from ui.styles import apply_custom_styles

# Page configuration
st.set_page_config(
    page_title="Meridian Enterprise RAG",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

client = APIClient()


def render_sidebar():
    """Render global sidebar telemetry and demo status."""
    st.sidebar.title("🏢 Meridian RAG")
    st.sidebar.caption("Enterprise Knowledge Assistant")

    # Demo Mode Badge
    if DEMO_MODE:
        st.sidebar.warning(
            "🔒 **Demo Mode Active**\n\nRead-only ingestion & rate-limited queries enabled."
        )

    st.sidebar.divider()

    # Health & System Status Check
    try:
        health = client.get_health()
        st.sidebar.success("🟢 **Backend Online**")
        provider = health.get("provider", "unknown").upper()
        model = health.get("model", "unknown")
        st.sidebar.caption(f"**Provider:** `{provider}`")
        st.sidebar.caption(f"**Model:** `{model}`")
    except APIConnectionError:
        st.sidebar.error("🔴 **Backend Offline**")
        st.sidebar.caption("Run `uv run uvicorn app.main:app` in terminal.")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Status check failed: {e}")

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        ### Quick Links
        - [💬 Chat Assistant](Chat)
        - [📊 Evaluation Lab](Eval_Dashboard)
        - [📥 Ingestion Console](Ingestion)
        - [ℹ️ Architecture & About](About)
        """
    )


def main():
    apply_custom_styles()
    render_sidebar()

    st.title("🏢 Meridian Enterprise RAG Assistant")
    st.markdown(
        """
        Welcome to the **Meridian Technologies Document Intelligence Platform**.
        This platform combines hybrid vector search, Reciprocal Rank Fusion (RRF),
        cross-encoder LLM reranking, and continuous evaluation tracking with Ragas and MLflow.
        """
    )

    # Telemetry Status Bar
    col1, col2, col3, col4 = st.columns(4)
    backend_online = False
    stats = None
    config = None

    try:
        health = client.get_health()
        backend_online = health.get("status") == "healthy"
        stats = client.get_ingest_status()
        config = client.get_config()
    except Exception:
        pass

    with col1:
        if backend_online:
            st.metric(label="API Status", value="Healthy", delta="Online")
        else:
            st.metric(label="API Status", value="Offline", delta="Down", delta_color="inverse")

    with col2:
        val = stats.doc_count if stats else "—"
        st.metric(label="Corpus Documents", value=val)

    with col3:
        val = stats.row_count if stats else "—"
        st.metric(label="Vector Chunks", value=val)

    with col4:
        val = config.model if config else "—"
        st.metric(label="Active LLM", value=val)

    st.divider()

    if not backend_online:
        st.error(
            "⚠️ **FastAPI Backend is currently unreachable.**\n\n"
            "Please make sure the backend server is running locally on port 8000:\n"
            "```powershell\n"
            "uv run uvicorn app.main:app --reload\n"
            "```\n"
            "Once started, refresh this page."
        )

    # Feature Cards
    st.subheader("Explore the Platform")
    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown(
            """
            ### 💬 Chat with Citations
            **Explainable, transparent RAG.**
            - Strict policy citations (`[1]`, `[2]`)
            - Real-time chunk score inspection (Vector, FTS, RRF, Rerank)
            - Latency profiling per pipeline stage
            - Guardrail refusal detection
            """
        )
        if st.button("Open Chat Assistant →", key="nav_chat"):
            st.switch_page("pages/1_💬_Chat.py")

    with f_col2:
        st.markdown(
            """
            ### 📊 Evaluation Dashboard
            **Continuous quality & regression tracking.**
            - Multi-run comparative bar & radar charts
            - Ragas metrics (Faithfulness, Relevancy, Precision, Recall)
            - Classical IR metrics (Hit Rate@k, MRR)
            - Phase 4 A→D Hallucination reduction narrative
            """
        )
        if st.button("Open Evaluation Lab →", key="nav_eval"):
            st.switch_page("pages/2_📊_Eval_Dashboard.py")

    with f_col3:
        st.markdown(
            """
            ### 📥 Ingestion Console
            **Vector storage & index management.**
            - LanceDB live index statistics
            - Markdown & document ingestion
            - Embedding dimension tracking
            - Safe demo mode protection
            """
        )
        if st.button("Open Ingestion Console →", key="nav_ingest"):
            st.switch_page("pages/3_📥_Ingestion.py")


if __name__ == "__main__":
    main()

