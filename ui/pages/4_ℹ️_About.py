"""About & Architecture Overview.

Recruiter- and stakeholder-facing overview detailing system architecture,
design decisions, and technical stack badges.
"""

import streamlit as st
import streamlit.components.v1 as components

from ui.styles import apply_custom_styles

st.set_page_config(
    page_title="About & Architecture | Meridian RAG",
    page_icon="ℹ️",
    layout="wide",
)


def render_mermaid(code: str, height: int = 580):
    """Render a visual SVG Mermaid diagram using Mermaid.js."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 10px;
                display: flex;
                justify-content: center;
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .mermaid {{
                width: 100%;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{code}
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'neutral',
                flowchart: {{
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html_content, height=height, scrolling=True)


def main():
    apply_custom_styles()
    st.title("ℹ️ About the Meridian Enterprise RAG Platform")
    st.markdown(
        "A production-grade, transparent Retrieval-Augmented Generation (RAG) platform "
        "designed for enterprise internal policy compliance, auditing, and continuous evaluation."
    )

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: Architecture Topology
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("🏛️ Architecture Topology")
    st.markdown(
        "The system follows a strictly **decoupled architecture**: the Streamlit frontend "
        "interacts with the FastAPI backend exclusively over HTTP, ensuring clean service "
        "boundaries and independent deployment lifecycles."
    )

    mermaid_code = """
flowchart TD
    subgraph Client ["Frontend Layer (Streamlit)"]
        UI_Chat["1_Chat.py<br/>Citation & Score Introspection"]
        UI_Eval["2_Eval_Dashboard.py<br/>Metrics & Experiment Story"]
        UI_Ingest["3_Ingestion.py<br/>Corpus Stats & File Upload"]
        Client_API["api_client.py<br/>Typed HTTP Wrapper"]
        UI_Chat --> Client_API
        UI_Eval --> Client_API
        UI_Ingest --> Client_API
    end

    subgraph API ["Service Layer (FastAPI)"]
        Router_Chat["POST /chat & GET /config"]
        Router_Eval["GET /eval/reports & /eval/narrative"]
        Router_Ingest["GET /ingest/status & POST /ingest/upload"]
    end

    subgraph Pipeline ["RAG Intelligence Core"]
        Retriever["Hybrid Retriever<br/>Vector Search + BM25 FTS"]
        RRF["Reciprocal Rank Fusion<br/>Score Normalization (k=60)"]
        Reranker["Cross-Encoder LLM Reranker<br/>Relevance Scoring (0-10)"]
        RAGChain["RAG Generation Chain<br/>Prompt Builder + Guardrails"]

        Retriever --> RRF --> Reranker --> RAGChain
    end

    subgraph Storage ["Storage & Tracking Layer"]
        LanceDB[("LanceDB<br/>Vector Store & Inverted Index")]
        MLflow[("MLflow<br/>Prompt Registry & Run Tracking")]
        Corpus[("Local Corpus<br/>Policies, Guides, Plans")]
    end

    Client_API -- "HTTP (BACKEND_URL)" --> Router_Chat
    Client_API --> Router_Eval
    Client_API --> Router_Ingest
    Router_Chat --> Retriever
    Router_Ingest --> LanceDB
    Router_Eval --> MLflow
    Retriever --> LanceDB
    Retriever --> Corpus
    RAGChain --> MLflow
"""
    render_mermaid(mermaid_code, height=580)

    with st.expander("🔍 View Raw Diagram Source", expanded=False):
        st.code(mermaid_code, language="mermaid")

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: Technical Stack & Badges
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("🛠️ Technical Stack")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            - **Backend Framework:** FastAPI, Uvicorn, Pydantic v2
            - **Vector Database:** LanceDB (Embedded Serverless Vector DB with native FTS)
            - **Frontend:** Streamlit Multipage Framework
            - **LLM Provider:** Google Gemini (`gemini-3.5-flash`, `gemini-embedding-001`)
            - **Local Fallback:** Native Ollama provider interface
            - **Package Management:** `uv` (Ultra-fast Python package manager)
            """
        )

    with col2:
        st.markdown(
            """
            - **Evaluation Framework:** Ragas (Faithfulness, Relevancy, Precision, Recall)
            - **IR Metrics:** Custom first-principles Hit Rate@k, MRR, NDCG@k
            - **Experiment Tracking:** MLflow (Runs, Metrics, Artifacts, Prompt Registry)
            - **Code Quality:** Ruff (linter & formatter), Pytest (unit & integration tests)
            """
        )

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 3: Key Engineering Decisions
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("🎯 Key Architectural Decisions")

    with st.expander("1. Decoupled UI and Backend (API-as-Product)", expanded=True):
        st.markdown(
            """
            - **Why**: The Streamlit frontend imports **zero backend modules**.
              It communicates strictly over HTTP using `ui/api_client.py`.
            - **Benefit**: The backend can be containerized, deployed, or swapped with
              a production React/TypeScript frontend without touching core RAG logic.
            - **Demo Safety**: Allows enforcing `DEMO_MODE=true` on public spaces while
              preserving full local development capabilities.
            """
        )

    with st.expander("2. Hybrid Search with Reciprocal Rank Fusion", expanded=False):
        st.markdown(
            """
            - **Why**: Dense embeddings excel at semantic similarity but struggle with exact
              alphanumeric codes (`SEV1`, `INC-XXXX`, `48 hours`). BM25 excels at keywords.
            - **Benefit**: Combining both using Reciprocal Rank Fusion (RRF, $k=60$) yielded
              a massive jump in retrieval recall compared to vector search alone.
            """
        )

    with st.expander("3. LLM Cross-Encoder Reranking", expanded=False):
        st.markdown(
            """
            - **Why**: Initial retrieval fetches top-12 candidates to maximize recall,
              but packing 12 chunks into the generator degrades prompt focus.
            - **Benefit**: The LLM reranker scores candidates and filters down to the top-4
              most relevant chunks, boosting answer precision and reducing token costs.
            """
        )

    with st.expander("4. Built-in Evaluation Lab & Regression Testing", expanded=False):
        st.markdown(
            """
            - **Why**: RAG applications fail silently when prompts or chunk sizes change.
            - **Benefit**: A 30-question Golden Dataset evaluated with Ragas LLM-as-a-judge
              provides concrete numerical confidence before any deployment.
            """
        )


if __name__ == "__main__":
    main()
