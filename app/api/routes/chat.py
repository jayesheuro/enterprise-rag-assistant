"""Chat API routes."""

from fastapi import APIRouter
from app.models.chat import ChatRequest, ChatResponse
from app.services.generation.rag_chain import RAGChain
from app.services.retrieval.retriever import Retriever
from app.services.retrieval.reranker import get_reranker
from app.services.retrieval.vector_store import VectorStore
from app.services.providers.factory import create_llm, create_embedder
from app.core.config import get_settings
from app.services.generation.guardrails import InputGuardrails
from app.core.logging import get_logger
from app.core.config import PROJECT_ROOT

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    settings = get_settings()
    
    # Input guardrails
    sanitized_query, warnings = InputGuardrails.check_query(request.query, settings.retrieval.max_query_length)
    
    # Pipeline
    llm = create_llm(settings)
    embedder = create_embedder(settings)
    vector_store = VectorStore(db_path=PROJECT_ROOT / "data" / "lancedb", table_name="chunks")
    retriever = Retriever(embedder=embedder, vector_store=vector_store, config=settings.retrieval)
    reranker = get_reranker(settings.retrieval, llm)
    rag_chain = RAGChain(llm=llm, retriever=retriever, reranker=reranker, config=settings.retrieval)
    
    # Answer
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
        reranker_used=response.reranker_used
    )

@router.get("/config")
async def get_chat_config():
    settings = get_settings()
    return {
        "provider": settings.provider,
        "retrieval": settings.retrieval.model_dump(),
        "chunking": settings.chunking.model_dump()
    }
