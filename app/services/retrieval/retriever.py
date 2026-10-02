"""Retriever: vector search, FTS, Reciprocal Rank Fusion, similarity floor."""

from pydantic import BaseModel, Field
from app.core.config import RetrievalConfig
from app.services.providers.base import BaseEmbedder
from app.services.retrieval.vector_store import VectorStore

class RetrievedChunk(BaseModel):
    """A chunk returned from retrieval with associated scores."""
    text: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    source: str
    page: int | None = None
    vector_score: float = 0.0
    fts_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    final_rank: int = 0

class Retriever:
    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore, config: RetrievalConfig):
        self.embedder = embedder
        self.vector_store = vector_store
        self.config = config

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        query_vector = self.embedder.embed_query(query)
        
        vector_results = self.vector_store.vector_search(query_vector, top_k=self.config.top_k)
        
        # Normalize vector scores: LanceDB returns _distance. lower distance = higher similarity
        # Similarity = 1 / (1 + distance)
        for res in vector_results:
            distance = res.get("_distance", 0.0)
            res["vector_score"] = 1.0 / (1.0 + distance)
            
        fts_results = []
        if self.config.mode == "hybrid":
            fts_results = self.vector_store.fts_search(query, top_k=self.config.top_k)
            # FTS returns _score. Higher is better. 
            # Normalization to 0-1 within result set
            max_score = max((r.get("_score", 0.0) for r in fts_results), default=0.0)
            for res in fts_results:
                raw_score = res.get("_score", 0.0)
                res["fts_score"] = raw_score / max_score if max_score > 0 else 0.0
                
        chunk_dict: dict[str, RetrievedChunk] = {}
        
        def get_or_create(chunk_id, res):
            if chunk_id not in chunk_dict:
                chunk_dict[chunk_id] = RetrievedChunk(
                    text=res["text"],
                    chunk_id=res["chunk_id"],
                    doc_id=res["doc_id"],
                    chunk_index=res["chunk_index"],
                    source=res["source"],
                    page=res.get("page")
                )
            return chunk_dict[chunk_id]

        # RRF logic
        k = 60
        
        # Sort vector results by vector_score descending
        vector_results.sort(key=lambda x: x.get("vector_score", 0.0), reverse=True)
        for rank, res in enumerate(vector_results):
            chunk = get_or_create(res["chunk_id"], res)
            chunk.vector_score = res["vector_score"]
            chunk.fused_score += 1.0 / (k + rank + 1)
            
        if self.config.mode == "hybrid":
            # Sort FTS results by fts_score descending
            fts_results.sort(key=lambda x: x.get("fts_score", 0.0), reverse=True)
            for rank, res in enumerate(fts_results):
                chunk = get_or_create(res["chunk_id"], res)
                chunk.fts_score = res["fts_score"]
                chunk.fused_score += 1.0 / (k + rank + 1)
        else:
            # If vector only, fused_score = vector_score to allow sorting and similarity floor
            for chunk in chunk_dict.values():
                chunk.fused_score = chunk.vector_score
                
        # Filter by similarity floor
        results = []
        for chunk in chunk_dict.values():
            if chunk.fused_score >= self.config.similarity_threshold:
                results.append(chunk)
                
        # Sort by fused_score descending
        results.sort(key=lambda x: x.fused_score, reverse=True)
        
        # Take top_k
        return results[:self.config.top_k]
