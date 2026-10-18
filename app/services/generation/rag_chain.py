"""RAG Chain implementation."""

import re
import time

from pydantic import BaseModel

from app.core.config import RetrievalConfig
from app.core.logging import get_logger
from app.services.generation.guardrails import OutputGuardrails
from app.services.generation.prompt_builder import load_prompt, render_prompt
from app.services.providers.base import BaseLLM, Message, TokenUsage
from app.services.retrieval.reranker import BaseReranker
from app.services.retrieval.retriever import RetrievedChunk, Retriever

logger = get_logger(__name__)


class Source(BaseModel):
    text: str
    source: str
    page: int | None
    chunk_id: str
    scores: dict


class RAGResponse(BaseModel):
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


class RAGChain:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Retriever,
        reranker: BaseReranker,
        config: RetrievalConfig,
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.config = config

    def answer(self, query: str) -> RAGResponse:
        timings = {}

        # 1. Retrieve
        start_time = time.time()
        chunks = self.retriever.retrieve(query)
        timings["retrieval_ms"] = (time.time() - start_time) * 1000

        # 2. Refusal if empty
        if not chunks:
            refusal_text = (
                "I don't have enough information in the "
                "available documents to answer this question."
            )
            model_name = getattr(
                self.llm,
                "model_name",
                getattr(self.llm, "_model_name", "unknown"),
            )
            return RAGResponse(
                answer=refusal_text,
                sources=[],
                grounded=False,
                low_confidence=False,
                refusal=True,
                timings=timings,
                token_usage=TokenUsage(),
                model_name=model_name,
                retrieval_mode=self.config.mode,
                reranker_used=self.config.reranker,
            )

        # 3. Rerank
        start_time = time.time()
        ranked_chunks = self.reranker.rerank(query, chunks)
        timings["rerank_ms"] = (time.time() - start_time) * 1000

        return self.generate_from_chunks(query, ranked_chunks, timings=timings)

    def generate_from_chunks(
        self,
        query: str,
        ranked_chunks: list[RetrievedChunk],
        timings: dict | None = None,
    ) -> RAGResponse:
        """Generate an answer directly from an existing list of ranked chunks."""
        timings = timings or {}

        # 4. Build context
        context = ""
        # Assuming 8000 tokens context window
        budget_tokens = 8000 * self.config.context_budget_ratio
        current_tokens = 0.0
        used_chunks = []

        for i, chunk in enumerate(ranked_chunks):
            chunk_tokens = len(chunk.text) / 4
            if current_tokens + chunk_tokens > budget_tokens and current_tokens > 0:
                break

            used_chunks.append(chunk)
            context += f"[{i + 1}] Source: {chunk.source} (Page: {chunk.page})\n{chunk.text}\n\n"
            current_tokens += chunk_tokens

        # 5. Load and render prompt
        template = load_prompt("rag_answer")
        prompt = render_prompt(template, query=query, context=context)

        # 6. Generate
        messages = [Message(role="user", content=prompt)]
        start_time = time.time()
        llm_response = self.llm.generate(messages)
        timings["generation_ms"] = (time.time() - start_time) * 1000
        timings["total_ms"] = sum(timings.values())

        # 7. Parse citations
        cited_indices = set()
        citations = re.findall(r"\[(\d+)\]", llm_response.text)
        for c in citations:
            try:
                idx = int(c)
                if 1 <= idx <= len(used_chunks):
                    cited_indices.add(idx)
            except ValueError:
                pass

        # 8. Build Source list
        sources = []
        for i, chunk in enumerate(used_chunks):
            idx = i + 1
            if idx in cited_indices:
                sources.append(
                    Source(
                        text=chunk.text,
                        source=chunk.source,
                        page=chunk.page,
                        chunk_id=chunk.chunk_id,
                        scores={
                            "vector_score": chunk.vector_score,
                            "fts_score": chunk.fts_score,
                            "fused_score": chunk.fused_score,
                            "rerank_score": chunk.rerank_score,
                        },
                    )
                )

        grounded = len(sources) > 0

        # 9. Output guardrails
        guardrail_res = OutputGuardrails.check_response(llm_response.text, grounded)

        # Return
        return RAGResponse(
            answer=llm_response.text,
            sources=sources,
            grounded=grounded,
            low_confidence=guardrail_res["low_confidence"],
            refusal=False,
            timings=timings,
            token_usage=llm_response.token_usage,
            model_name=llm_response.model_name,
            retrieval_mode=self.config.mode,
            reranker_used=self.config.reranker,
        )
