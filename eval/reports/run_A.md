# Evaluation Report: run_A
**Timestamp:** 2026-10-06T14:10:25.362800Z  
**Dataset Version:** v1  
**Mode:** Retrieval-Only  
**Success Rate:** 30/30

## Configuration Snapshot
```json
{
  "provider": "gemini",
  "chunking": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "strategy": "recursive"
  },
  "retrieval": {
    "top_k": 12,
    "final_k": 4,
    "similarity_threshold": 0.01,
    "mode": "vector",
    "reranker": "none",
    "context_budget_ratio": 0.6,
    "max_query_length": 1000
  },
  "provider_config": {
    "model": "gemini-3.8-flash",
    "embedding_model": "gemini-embedding-001",
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "timeout": 30,
    "base_url": null,
    "region": null
  }
}
```

## Aggregate Metrics
### Retrieval Metrics
| Metric | Score |
|---|---|
| hit_rate@4 | 0.8333 |
| recall@4 | 1.0000 |
| mrr | 0.8333 |
| ndcg@4 | 1.8374 |
