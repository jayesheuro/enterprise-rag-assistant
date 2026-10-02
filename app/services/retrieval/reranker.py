"""Reranking strategies for retrieved chunks."""

from abc import ABC, abstractmethod
import json
from pydantic import BaseModel
from typing import Optional

from app.core.config import RetrievalConfig
from app.services.providers.base import BaseLLM, Message
from app.services.retrieval.retriever import RetrievedChunk
from app.core.logging import get_logger

logger = get_logger(__name__)

class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        pass

class NoopReranker(BaseReranker):
    def __init__(self, config: RetrievalConfig):
        self.config = config
        
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        # Return chunks as is but assign final_rank and respect final_k
        for i, chunk in enumerate(chunks):
            chunk.final_rank = i + 1
        return chunks[:self.config.final_k]

class Score(BaseModel):
    index: int
    score: int

class RerankResponse(BaseModel):
    scores: list[Score]

class LLMReranker(BaseReranker):
    def __init__(self, config: RetrievalConfig, llm: BaseLLM):
        self.config = config
        self.llm = llm
        
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
            
        from app.services.generation.prompt_builder import load_prompt, render_prompt
        
        try:
            template = load_prompt("rerank")
        except FileNotFoundError:
            logger.warning("Rerank prompt template not found, falling back to NoopReranker")
            return NoopReranker(self.config).rerank(query, chunks)
            
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            chunks_text += f"[{i}] {chunk.text}\n\n"
            
        prompt = render_prompt(template, query=query, chunks=chunks_text)
        
        messages = [
            Message(role="user", content=prompt)
        ]
        
        try:
            # Temperature 0.0 for determinism
            response = self.llm.generate(messages, temperature=0.0)
            text = response.text
            
            # Find json block if wrapped in markdown
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
                
            parsed = RerankResponse.model_validate_json(text)
            
            score_map = {s.index: s.score for s in parsed.scores}
            
            for i, chunk in enumerate(chunks):
                chunk.rerank_score = float(score_map.get(i, 0.0))
                
            chunks.sort(key=lambda x: x.rerank_score, reverse=True)
            
        except Exception as e:
            logger.warning(f"LLMReranker failed to parse response: {e}. Returning original order.")
            # Default fallback: keep original order
            
        # Assign final_rank and truncate
        for i, chunk in enumerate(chunks):
            chunk.final_rank = i + 1
            
        return chunks[:self.config.final_k]

def get_reranker(config: RetrievalConfig, llm: Optional[BaseLLM] = None) -> BaseReranker:
    if config.reranker == "llm" and llm is not None:
        return LLMReranker(config, llm)
    return NoopReranker(config)
