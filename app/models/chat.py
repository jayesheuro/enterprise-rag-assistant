from pydantic import BaseModel
from typing import Any

from app.services.providers.base import TokenUsage
from app.services.generation.rag_chain import Source

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    low_confidence: bool
    refusal: bool
    timings: dict
    token_usage: TokenUsage
    model_name: str
    retrieval_mode: str
    reranker_used: str
