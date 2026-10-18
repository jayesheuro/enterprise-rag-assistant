"""Chat with Retrieval Introspection & Citation Transparency.

Core conversational interface allowing users to query enterprise knowledge,
inspect cited chunks, view hybrid search scores, and monitor pipeline latency.
"""

import streamlit as st

from ui.api_client import DEMO_MODE, APIClient, APIClientError, APIConnectionError
from ui.styles import apply_custom_styles

st.set_page_config(
    page_title="Chat | Meridian RAG",
    page_icon="💬",
    layout="wide",
)

client = APIClient()

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "latest_telemetry" not in st.session_state:
    st.session_state.latest_telemetry = None

if "global_quota_exhausted" not in st.session_state:
    st.session_state.global_quota_exhausted = False

DEMO_QUERY_LIMIT = 10


def _fetch_global_quota() -> dict | None:
    """Poll the backend for global daily quota status. Returns None on error."""
    if not DEMO_MODE:
        return None
    try:
        return client._request("GET", "/chat/quota")
    except Exception:
        return None


def render_sidebar():
    """Render chat-specific sidebar with pipeline configuration & live telemetry."""
    st.sidebar.title("💬 Chat Controls")

    # Demo Mode Status
    if DEMO_MODE:
        session_remaining = max(0, DEMO_QUERY_LIMIT - st.session_state.query_count)
        st.sidebar.info(
            f"🔒 **Demo Mode Active**\n\n"
            f"Session queries: `{st.session_state.query_count}/{DEMO_QUERY_LIMIT}` "
            f"({session_remaining} remaining)"
        )
        if session_remaining == 0:
            st.sidebar.error("⚠️ Session query limit reached.")

        # Global daily quota from backend
        global_quota = _fetch_global_quota()
        if global_quota and global_quota.get("cap") is not None:
            g_remaining = global_quota.get("remaining", 0)
            g_count = global_quota.get("count", 0)
            g_cap = global_quota.get("cap", 0)
            if global_quota.get("exhausted"):
                st.sidebar.error(
                    f"🚦 **Daily demo cap reached** ({g_count}/{g_cap}). "
                    "Resets at UTC midnight."
                )
                st.session_state.global_quota_exhausted = True
            else:
                st.sidebar.caption(
                    f"🌐 Global daily cap: `{g_count}/{g_cap}` ({g_remaining} left)"
                )

    st.sidebar.divider()

    # Active Pipeline Configuration
    st.sidebar.subheader("⚙️ Active Pipeline")
    try:
        config = client.get_config()
        st.sidebar.caption(f"**Provider:** `{config.provider.upper()}`")
        st.sidebar.caption(f"**LLM Model:** `{config.model}`")
        st.sidebar.caption(f"**Embedding:** `{config.embedding_model}`")

        retrieval = config.retrieval
        st.sidebar.caption(f"**Search Mode:** `{retrieval.get('mode', 'hybrid')}`")
        st.sidebar.caption(f"**Top-K Candidates:** `{retrieval.get('top_k', 12)}`")
        st.sidebar.caption(f"**Final-K Reranked:** `{retrieval.get('final_k', 4)}`")
        st.sidebar.caption(f"**Reranker:** `{retrieval.get('reranker', 'llm')}`")
    except APIConnectionError:
        st.sidebar.error("🔴 Backend Offline")
    except Exception as e:
        st.sidebar.caption(f"Config unavailable: {e}")

    # Latency & Token Telemetry for Latest Response
    telemetry = st.session_state.latest_telemetry
    if telemetry:
        st.sidebar.divider()
        st.sidebar.subheader("⏱️ Execution Telemetry")

        timings = telemetry.get("timings", {})
        st.sidebar.caption(f"• **Retrieval:** `{timings.get('retrieval_ms', 0):.0f} ms`")
        st.sidebar.caption(f"• **Reranking:** `{timings.get('rerank_ms', 0):.0f} ms`")
        st.sidebar.caption(f"• **Generation:** `{timings.get('generation_ms', 0):.0f} ms`")
        st.sidebar.markdown(f"**Total Latency:** `{timings.get('total_ms', 0):.0f} ms`")

        tokens = telemetry.get("token_usage", {})
        if tokens:
            p_tok = tokens.get("prompt_tokens", 0)
            c_tok = tokens.get("completion_tokens", 0)
            tot_tok = tokens.get("total_tokens", 0)
            st.sidebar.caption(
                f"**Tokens:** `{tot_tok}` (`{p_tok}` in / `{c_tok}` out)"
            )

        # Confidence / Refusal Badges
        if telemetry.get("refusal"):
            st.sidebar.warning("⚠️ **Refusal Triggered**: Out-of-scope or ungrounded question.")
        elif telemetry.get("low_confidence"):
            st.sidebar.warning("⚠️ **Low Confidence**: Potential knowledge boundary reached.")
        elif telemetry.get("grounded"):
            st.sidebar.success("✅ **Fully Grounded**: Verified by cited corpus documents.")

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.latest_telemetry = None
        st.rerun()


def render_source_expander(source, idx: int):
    """Render an individual source expander with chunk details and all score breakdowns."""
    page_info = f" (Page {source.page})" if source.page is not None else ""
    title = f"[{idx}] 📄 {source.source}{page_info}"

    with st.expander(title, expanded=False):
        st.markdown(f"**Document Passage:**\n> {source.text}")

        scores = source.scores or {}
        st.markdown("**Hybrid & Rerank Scores:**")
        sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)

        with sc_col1:
            val = scores.get("vector_score")
            st.metric(label="Vector Score", value=f"{val:.3f}" if val is not None else "N/A")

        with sc_col2:
            val = scores.get("fts_score")
            st.metric(label="FTS Score", value=f"{val:.3f}" if val is not None else "N/A")

        with sc_col3:
            val = scores.get("fused_score")
            st.metric(label="RRF Score", value=f"{val:.4f}" if val is not None else "N/A")

        with sc_col4:
            val = scores.get("rerank_score")
            st.metric(label="LLM Rerank", value=f"{val:.1f}" if val is not None else "N/A")

        if source.chunk_id:
            st.caption(f"Chunk ID: `{source.chunk_id}`")


def main():
    apply_custom_styles()
    render_sidebar()

    st.title("💬 Enterprise Chat Assistant")
    st.markdown(
        "Ask questions grounded strictly in Meridian Technologies policies. "
        "Every factual claim includes verified citations and retrieval score transparency."
    )

    # Empty State & Starters
    if not st.session_state.messages and "pending_query" not in st.session_state:
        st.info("💡 **Select a sample question below or type your own query to begin:**")
        sample_questions = [
            "What is the deadline for drafting a PIR after an incident?",
            "If I am modifying a core database schema, how many approvals do I need?",
            "What is the company dress code policy for casual Fridays?",
            "What is the password minimum length and rotation schedule?",
        ]
        cols = st.columns(2)
        for i, sample in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(sample, key=f"sample_{i}", use_container_width=True):
                    st.session_state.pending_query = sample
                    st.rerun()

    # Render Conversation History
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.markdown(msg["content"])

            # If assistant message has cited sources, display them
            sources = msg.get("sources", [])
            if sources:
                st.markdown("**📚 Cited Sources & Retrieval Scores**")
                for i, s in enumerate(sources, 1):
                    render_source_expander(s, i)

    # Chat Input Box
    session_exhausted = DEMO_MODE and st.session_state.query_count >= DEMO_QUERY_LIMIT
    global_exhausted = st.session_state.global_quota_exhausted
    input_disabled = session_exhausted or global_exhausted

    if global_exhausted:
        placeholder = "🚦 Daily demo quota reached — clone the repo to run locally!"
    elif session_exhausted:
        placeholder = "Demo query limit reached (10/10). Restart session to continue."
    else:
        placeholder = "Ask a question about Meridian internal policies..."

    query_to_run = None
    user_input = st.chat_input(placeholder=placeholder, disabled=input_disabled)
    if user_input:
        query_to_run = user_input
    elif "pending_query" in st.session_state:
        query_to_run = st.session_state.pop("pending_query")

    if query_to_run:
        # Check session demo limit (client-side guard)
        if DEMO_MODE and st.session_state.query_count >= DEMO_QUERY_LIMIT:
            st.error("🔒 Query limit reached for this demo session.")
            return

        # 1. Immediately render and save user message so it appears while searching
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        st.session_state.query_count += 1
        with st.chat_message("user"):
            st.markdown(query_to_run)

        # 2. Render assistant message container with spinner
        with st.chat_message("assistant"):
            spinner_msg = (
                "Searching corpus, reranking candidates, and generating grounded answer..."
            )
            with st.spinner(spinner_msg):
                try:
                    result = client.send_chat(query_to_run)
                    st.markdown(result.answer)

                    if result.sources:
                        st.markdown("**📚 Cited Sources & Retrieval Scores**")
                        for i, s in enumerate(result.sources, 1):
                            render_source_expander(s, i)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                        }
                    )
                    st.session_state.latest_telemetry = {
                        "timings": result.timings,
                        "token_usage": result.token_usage,
                        "grounded": result.grounded,
                        "low_confidence": result.low_confidence,
                        "refusal": result.refusal,
                    }
                except APIConnectionError as e:
                    err_msg = f"❌ **Connection Error:** {e.message}"
                    st.markdown(err_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg, "sources": []}
                    )
                except APIClientError as e:
                    # HTTP 429 = global daily quota exhausted — show friendly CTA
                    if e.status_code == 429:
                        quota_msg = e.message
                        st.info(quota_msg)
                        st.session_state.global_quota_exhausted = True
                        st.session_state.messages.append(
                            {"role": "assistant", "content": quota_msg, "sources": []}
                        )
                    else:
                        err_msg = f"⚠️ **Error:** {e.message}"
                        st.markdown(err_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": err_msg, "sources": []}
                        )
                except Exception as e:
                    err_msg = f"⚠️ **Unexpected Error:** {str(e)}"
                    st.markdown(err_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg, "sources": []}
                    )

        # 3. Rerun to refresh the sidebar telemetry cleanly
        st.rerun()


if __name__ == "__main__":
    main()

