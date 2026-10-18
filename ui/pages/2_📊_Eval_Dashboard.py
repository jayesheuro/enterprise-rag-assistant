"""Evaluation Lab Dashboard.

Visualizes offline evaluation reports, comparative benchmark metrics across runs,
per-question retrieval and generation drill-downs, and the Phase 4 A->D experiment narrative.
"""

import pandas as pd
import streamlit as st

from ui.api_client import APIClient, APIClientError, APIConnectionError
from ui.styles import apply_custom_styles

st.set_page_config(
    page_title="Evaluation Dashboard | Meridian RAG",
    page_icon="📊",
    layout="wide",
)

client = APIClient()


def render_sidebar(report_ids: list[str]):
    """Sidebar controls for report selection."""
    st.sidebar.title("📊 Eval Controls")
    st.sidebar.caption("Benchmark & Quality Telemetry")

    if not report_ids:
        st.sidebar.warning("No evaluation runs found.")
        return None

    selected = st.sidebar.selectbox(
        "Select Active Run",
        options=report_ids,
        index=0,
        help="Select a specific evaluation run to inspect its detailed metrics.",
    )
    return selected


def render_metric_cards(report: dict):
    """Render aggregate metric cards for retrieval and generation."""
    retrieval_metrics = report.get("aggregate_retrieval_metrics", {})
    ragas_metrics = report.get("aggregate_ragas_metrics", {})
    refusal_acc = report.get("aggregate_refusal_accuracy", 0.0)

    st.markdown("#### 🎯 Aggregate Performance Metrics")
    cols = st.columns(6)

    # Hit Rate
    with cols[0]:
        hr = next((v for k, v in retrieval_metrics.items() if "hit_rate" in k), None)
        val = f"{hr * 100:.1f}%" if hr is not None else "N/A"
        st.metric(label="Hit Rate", value=val)

    # MRR
    with cols[1]:
        mrr = retrieval_metrics.get("mrr")
        val = f"{mrr:.3f}" if mrr is not None else "N/A"
        st.metric(label="MRR", value=val)

    # NDCG
    with cols[2]:
        ndcg = next((v for k, v in retrieval_metrics.items() if "ndcg" in k), None)
        val = f"{ndcg:.3f}" if ndcg is not None else "N/A"
        st.metric(label="NDCG", value=val)

    # Refusal Accuracy
    with cols[3]:
        val = f"{refusal_acc * 100:.1f}%" if refusal_acc is not None else "N/A"
        st.metric(label="Refusal Acc", value=val)

    # Faithfulness
    with cols[4]:
        faith = ragas_metrics.get("faithfulness")
        val = f"{faith:.3f}" if faith is not None else "N/A"
        st.metric(label="Faithfulness", value=val)

    # Answer Relevancy
    with cols[5]:
        rel = ragas_metrics.get("answer_relevancy")
        val = f"{rel:.3f}" if rel is not None else "N/A"
        st.metric(label="Answer Relevancy", value=val)


def build_questions_dataframe(results: list[dict]) -> pd.DataFrame:
    """Build a structured pandas dataframe from per-question results."""
    rows = []
    for r in results:
        ret_metrics = r.get("retrieval_metrics", {})
        ragas_m = r.get("ragas_metrics", {})

        hr = next((v for k, v in ret_metrics.items() if "hit_rate" in k), 0.0)
        mrr = ret_metrics.get("mrr", 0.0)
        ndcg = next((v for k, v in ret_metrics.items() if "ndcg" in k), 0.0)

        rows.append(
            {
                "ID": r.get("question_id", ""),
                "Type": r.get("question_type", ""),
                "Question": r.get("question", ""),
                "Hit Rate": hr,
                "MRR": mrr,
                "NDCG": ndcg,
                "Faithfulness": ragas_m.get("faithfulness", 0.0),
                "Relevancy": ragas_m.get("answer_relevancy", 0.0),
                "Refusal Acc": r.get("refusal_accuracy", 0.0),
            }
        )
    return pd.DataFrame(rows)


def main():
    apply_custom_styles()
    st.title("📊 Evaluation Lab & Benchmark Dashboard")
    st.markdown(
        "Inspect offline evaluation runs against the 30-question Golden Dataset, "
        "compare retrieval configurations, and verify guardrail performance."
    )

    # Fetch available report IDs
    try:
        report_ids = client.list_eval_reports()
    except APIConnectionError as e:
        st.error(f"❌ {e.message}")
        return
    except APIClientError as e:
        st.error(f"⚠️ Failed to list reports: {e.message}")
        return

    if not report_ids:
        st.info(
            "ℹ️ No evaluation reports found in `eval/reports/`. "
            "Run `python scripts/run_eval.py` to generate reports."
        )
        return

    selected_run = render_sidebar(report_ids)

    # Load active report
    report_data = None
    if selected_run:
        try:
            report_data = client.get_eval_report(selected_run)
        except Exception as e:
            st.error(f"Failed to load report `{selected_run}`: {e}")
            return

    tab_run, tab_compare, tab_narrative = st.tabs(
        [
            "🔍 Run Deep-Dive",
            "📈 Multi-Run Comparison",
            "📖 Experiment Narrative (Phase 4)",
        ]
    )

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 1: Run Deep-Dive
    # ──────────────────────────────────────────────────────────────────────────
    with tab_run:
        if report_data:
            st.subheader(f"Evaluation Run: `{selected_run}`")
            ts = report_data.get("timestamp", "unknown")
            mode = "Retrieval Only" if report_data.get("retrieval_only") else "End-to-End RAG"
            tot_q = report_data.get("total_questions", 0)
            st.caption(f"**Timestamp:** `{ts}` | **Mode:** `{mode}` | **Questions:** `{tot_q}`")

            # Config snapshot expander
            with st.expander("⚙️ Pipeline Configuration Snapshot for this Run", expanded=False):
                cfg = report_data.get("config_snapshot", {})
                ret = cfg.get("retrieval", {})
                prov = cfg.get("provider_config", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.write(f"**Provider:** `{cfg.get('provider')}`")
                c1.write(f"**Model:** `{prov.get('model')}`")
                c2.write(f"**Retrieval Mode:** `{ret.get('mode')}`")
                c2.write(f"**Reranker:** `{ret.get('reranker')}`")
                c3.write(f"**Top-K:** `{ret.get('top_k')}`")
                c3.write(f"**Final-K:** `{ret.get('final_k')}`")
                prompts = report_data.get("prompt_versions", {})
                c4.write(f"**RAG Prompt:** `{prompts.get('rag_answer', 'v1')}`")
                c4.write(f"**Rerank Prompt:** `{prompts.get('rerank', 'v1')}`")

            # Metric cards
            render_metric_cards(report_data)
            st.divider()

            # Per-Question Table
            st.markdown("#### 📋 Per-Question Performance Table")
            results = report_data.get("results", [])
            if results:
                df = build_questions_dataframe(results)

                ctrl_col1, ctrl_col2 = st.columns([1, 3])
                with ctrl_col1:
                    worst_first = st.toggle(
                        "🚨 Worst-First Ordering",
                        value=False,
                        help="Sort by lowest MRR / Hit Rate to diagnose failure cases.",
                    )
                with ctrl_col2:
                    selected_q_type = st.multiselect(
                        "Filter by Question Type",
                        options=list(df["Type"].unique()),
                        default=list(df["Type"].unique()),
                    )

                filtered_df = df[df["Type"].isin(selected_q_type)]
                if worst_first:
                    filtered_df = filtered_df.sort_values(by=["MRR", "Hit Rate"], ascending=True)

                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Hit Rate": st.column_config.NumberColumn(format="%.2f"),
                        "MRR": st.column_config.NumberColumn(format="%.3f"),
                        "NDCG": st.column_config.NumberColumn(format="%.3f"),
                        "Faithfulness": st.column_config.NumberColumn(format="%.3f"),
                        "Relevancy": st.column_config.NumberColumn(format="%.3f"),
                        "Refusal Acc": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

                # Question Drill-Down
                st.divider()
                st.markdown("#### 🔬 Question Drill-Down")
                question_ids = [r["question_id"] for r in results]
                selected_qid = st.selectbox(
                    "Select a Question ID to inspect contexts and answer:",
                    options=question_ids,
                    index=0,
                )

                q_detail = next((r for r in results if r["question_id"] == selected_qid), None)
                if q_detail:
                    st.markdown(f"**Question:** `{q_detail.get('question')}`")
                    q_type = q_detail.get('question_type')
                    sources = q_detail.get('expected_doc_ids')
                    st.caption(f"Type: `{q_type}` | Expected Sources: `{sources}`")

                    # Answers side-by-side
                    ans_c1, ans_c2 = st.columns(2)
                    with ans_c1:
                        st.markdown("**Generated Answer:**")
                        gen_ans = q_detail.get("answer")
                        if gen_ans:
                            st.info(gen_ans)
                        else:
                            st.caption("(Retrieval-only run — no answer generated)")

                    with ans_c2:
                        st.markdown("**Ground Truth Answer:**")
                        gt = q_detail.get("ground_truth")
                        if gt:
                            st.success(gt)
                        else:
                            st.caption("(Out of scope — expected refusal)")

                    # Retrieved Contexts
                    contexts = q_detail.get("retrieved_contexts", [])
                    st.markdown(f"**Retrieved Context Passages ({len(contexts)} chunks):**")
                    for i, ctx in enumerate(contexts, 1):
                        with st.expander(f"Context [{i}]", expanded=False):
                            st.markdown(f"> {ctx}")

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 2: Multi-Run Comparison
    # ──────────────────────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("📈 Multi-Run Comparative Benchmarks")
        st.markdown(
            "Compare retrieval and generation metrics across runs to observe improvements."
        )

        selected_compare = st.multiselect(
            "Select Runs to Compare:",
            options=report_ids,
            default=report_ids[: min(3, len(report_ids))],
        )

        if selected_compare:
            comparison_rows = []
            for rid in selected_compare:
                try:
                    rep = client.get_eval_report(rid)
                    ret_m = rep.get("aggregate_retrieval_metrics", {})
                    rag_m = rep.get("aggregate_ragas_metrics", {})
                    hr = next((v for k, v in ret_m.items() if "hit_rate" in k), 0.0)
                    mrr = ret_m.get("mrr", 0.0)
                    ndcg = next((v for k, v in ret_m.items() if "ndcg" in k), 0.0)
                    faith = rag_m.get("faithfulness", 0.0)
                    rel = rag_m.get("answer_relevancy", 0.0)
                    refusal = rep.get("aggregate_refusal_accuracy", 0.0)

                    cfg = rep.get("config_snapshot", {}).get("retrieval", {})

                    comparison_rows.append(
                        {
                            "Run ID": rid,
                            "Mode": cfg.get("mode", "—"),
                            "Reranker": cfg.get("reranker", "—"),
                            "Hit Rate@k": hr,
                            "MRR": mrr,
                            "NDCG@k": ndcg,
                            "Refusal Acc": refusal,
                            "Faithfulness": faith,
                            "Answer Relevancy": rel,
                        }
                    )
                except Exception as e:
                    st.warning(f"Failed to load run `{rid}`: {e}")

            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                st.markdown("##### 📊 Metric Comparison Chart")
                chart_df = comp_df.set_index("Run ID")[["Hit Rate@k", "MRR", "Refusal Acc"]]
                st.bar_chart(chart_df)

    # ──────────────────────────────────────────────────────────────────────────
    # TAB 3: Experiment Narrative
    # ──────────────────────────────────────────────────────────────────────────
    with tab_narrative:
        st.subheader("📖 The Phase 4 Experiment Narrative: A → D")
        st.markdown(
            "A structured engineering narrative documenting how we progressed from naive "
            "vector search to hybrid retrieval with RRF, LLM reranking, and guardrails."
        )
        st.divider()

        try:
            narrative_text = client.get_eval_narrative()
            st.markdown(narrative_text)
        except Exception as e:
            st.error(f"Could not load experiment narrative: {e}")


if __name__ == "__main__":
    main()

