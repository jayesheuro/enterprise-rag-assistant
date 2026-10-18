"""Chat API routes.

In DEMO_MODE the global daily quota is enforced before the RAG pipeline runs.
A quota-exhausted request returns HTTP 429 with a friendly message so the UI
can display it without a generic error dialog.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import PROJECT_ROOT, get_settings
from app.core.logging import get_logger
from app.models.chat import ChatRequest, ChatResponse
from app.services.generation.guardrails import InputGuardrails
from app.services.generation.rag_chain import RAGChain
from app.services.providers.factory import create_embedder, create_llm
from app.services.quota.daily_quota import consume_quota
from app.services.retrieval.reranker import get_reranker
from app.services.retrieval.retriever import Retriever
from app.services.retrieval.vector_store import VectorStore

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Run the RAG pipeline for a user query.

    In DEMO_MODE, consumes one global daily quota slot before running.
    Returns HTTP 429 with a quota-exhaustion message when the daily cap
    is reached, so the UI can surface a friendly explanation rather than
    a generic error.
    """
    # ── Global daily quota check (DEMO_MODE only) ────────────────────────
    quota = consume_quota()
    if not quota["allowed"]:
        logger.info("Global demo quota exhausted; rejecting chat request.")
        return JSONResponse(
            status_code=429,
            content={
                "error": "QuotaExhausted",
                "message": quota["message"],
                "quota": {
                    "count": quota["count"],
                    "cap": quota["cap"],
                    "remaining": quota["remaining"],
                },
            },
        )

    settings = get_settings()

    # Input guardrails
    sanitized_query, _warnings = InputGuardrails.check_query(
        request.query, settings.retrieval.max_query_length
    )

    # Pipeline
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    vector_store = VectorStore(
        db_path=PROJECT_ROOT / "data" / "lancedb", table_name="chunks"
    )
    retriever = Retriever(
        embedder=embedder, vector_store=vector_store, config=settings.retrieval
    )
    reranker = get_reranker(settings.retrieval, llm)
    rag_chain = RAGChain(
        llm=llm, retriever=retriever, reranker=reranker, config=settings.retrieval
    )

    response = rag_chain.answer(sanitized_query)

    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        grounded=response.grounded,
        low_confidence=response.low_confidence,
        refusal=response.refusal,
        timings=response.timings,
        token_usage=response.token_usage,
        model_name=response.model_name,
        retrieval_mode=response.retrieval_mode,
        reranker_used=response.reranker_used,
    )


@router.get("/config")
async def get_chat_config():
    """Return active provider, retrieval, and chunking configuration."""
    settings = get_settings()
    return {
        "provider": settings.provider,
        "retrieval": settings.retrieval.model_dump(),
        "chunking": settings.chunking.model_dump(),
    }


@router.get("/quota")
async def get_quota():
    """Return current global demo quota status (useful for UI polling)."""
    from app.services.quota.daily_quota import get_quota_status

    return get_quota_status()
