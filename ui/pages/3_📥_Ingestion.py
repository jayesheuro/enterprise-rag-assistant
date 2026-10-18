"""Document Ingestion Console.

Manage and inspect the enterprise knowledge corpus, view live LanceDB index statistics,
and upload documents for chunking, embedding, and indexing.
"""

import streamlit as st

from ui.api_client import DEMO_MODE, APIClient, APIClientError, APIConnectionError
from ui.styles import apply_custom_styles

st.set_page_config(
    page_title="Ingestion Console | Meridian RAG",
    page_icon="📥",
    layout="wide",
)

client = APIClient()


def render_sidebar():
    """Sidebar for ingestion console."""
    st.sidebar.title("📥 Ingestion Console")
    st.sidebar.caption("Knowledge Base & Index Management")

    if DEMO_MODE:
        st.sidebar.warning(
            "🔒 **Demo Mode Active**\n\n"
            "Document uploads and indexing mutations are disabled in the live demo."
        )

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        ### Supported Formats
        - Markdown (`.md`)
        - Plain Text (`.txt`)
        - PDF Documents (`.pdf`)
        """
    )


def main():
    apply_custom_styles()
    render_sidebar()

    st.title("📥 Document Ingestion Console")
    st.markdown(
        "Inspect the current status of the vector database index "
        "and ingest new enterprise policy documents."
    )

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 1: Live Corpus & Vector Store Telemetry
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("📊 Live Index Telemetry")

    try:
        status = client.get_ingest_status()
        config = client.get_config()

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric(label="Total Chunks Indexed", value=status.row_count)

        with m_col2:
            st.metric(label="Unique Documents", value=status.doc_count)

        with m_col3:
            st.metric(label="Vector Store", value="LanceDB")

        with m_col4:
            st.metric(label="Active Embedder", value=status.embedding_model)

        with st.expander("⚙️ Current Chunking & Ingestion Settings", expanded=False):
            ch = config.chunking
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Chunk Size:** `{ch.get('chunk_size', 1000)} chars`")
            c2.write(f"**Chunk Overlap:** `{ch.get('chunk_overlap', 200)} chars`")
            c3.write(f"**Strategy:** `{ch.get('strategy', 'recursive')}`")

    except APIConnectionError as e:
        st.error(f"❌ {e.message}")
        return
    except APIClientError as e:
        st.warning(f"Could not load index status: {e.message}")

    st.divider()

    # ──────────────────────────────────────────────────────────────────────────
    # SECTION 2: Document Ingestion / Upload
    # ──────────────────────────────────────────────────────────────────────────
    st.subheader("📤 Ingest New Documents")

    if DEMO_MODE:
        st.info(
            "🔒 **Demo Mode Notice**: Document ingestion and re-indexing are locked in this public "
            "environment to prevent unauthorized index alteration and protect API quotas.\n\n"
            "In local dev (`DEMO_MODE=false`), file upload and live indexing are fully unlocked."
        )
        return

    st.markdown(
        "Upload one or more files to run them through document loading, recursive chunking, "
        "embedding generation, and LanceDB upsert."
    )

    uploaded_files = st.file_uploader(
        "Select policy files to ingest",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        help="Upload Markdown, Text, or PDF documents.",
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s) ready for ingestion.")

        if st.button("🚀 Process & Ingest Into Vector Store", type="primary"):
            file_tuples = []
            for f in uploaded_files:
                file_tuples.append((f.name, f.getvalue(), f.type or "application/octet-stream"))

            with st.spinner(f"Ingesting {len(file_tuples)} document(s)..."):
                try:
                    report = client.upload_and_ingest(file_tuples)
                    st.success("✅ Ingestion Completed Successfully!")

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Docs Processed", report.docs_processed)
                    r2.metric("Chunks Created", report.chunks_created)
                    r3.metric("Tokens Embedded", report.tokens_embedded)
                    r4.metric("Wall Time", f"{report.wall_time_seconds:.2f}s")

                    if report.failures:
                        st.warning("Some documents encountered errors:")
                        for fail in report.failures:
                            st.write(f"- `{fail.get('path')}`: {fail.get('reason')}")

                except APIConnectionError as e:
                    st.error(f"❌ {e.message}")
                except APIClientError as e:
                    st.error(f"⚠️ Ingestion failed: {e.message}")
                except Exception as e:
                    st.error(f"⚠️ Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
