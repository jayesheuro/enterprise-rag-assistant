"""
Retrieval Evaluation Metrics.

This module provides standard Information Retrieval (IR) metrics to evaluate the
quality of the retrieval phase in a RAG system.

Why evaluate retrieval separately from generation?
Isolating retrieval quality from generation quality is a core diagnostic principle.
If a RAG system fails to answer a question, we need to know if the retriever failed
to find the right information, or if the generator failed to synthesize it. Good
generation cannot fix bad retrieval (hallucinations occur), and good retrieval is
useless with bad generation.

Metrics provided:
- Hit Rate@k: Did we find at least one relevant document in the top k? (Binary measure)
- Recall@k: What fraction of all relevant documents did we find in the top k?
- MRR (Mean Reciprocal Rank): How high up in the ranking is the first relevant document?
- NDCG@k (Normalized Discounted Cumulative Gain): Measures ranking quality, giving higher
  scores when relevant documents appear higher in the ranking.
"""

import math

def hit_rate_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """
    Calculate Hit Rate at k.
    
    Binary: 1.0 if ANY expected doc appears in top-k retrieved, else 0.0
    Formula: Hit@k = 1 if |R_k ∩ E| > 0 else 0
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ordered by rank).
        expected_doc_ids: List of relevant/expected document IDs.
        k: The cutoff rank.
        
    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    top_k_retrieved = set(retrieved_doc_ids[:k])
    expected_set = set(expected_doc_ids)
    if top_k_retrieved.intersection(expected_set):
        return 1.0
    return 0.0

def recall_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """
    Calculate Recall at k.
    
    Fraction of expected docs found in top-k
    Formula: Recall@k = |R_k ∩ E| / |E|
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ordered by rank).
        expected_doc_ids: List of relevant/expected document IDs.
        k: The cutoff rank.
        
    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not expected_doc_ids:
        return 1.0
    
    top_k_retrieved = set(retrieved_doc_ids[:k])
    expected_set = set(expected_doc_ids)
    
    hits = len(top_k_retrieved.intersection(expected_set))
    return float(hits) / len(expected_set)

def mrr(retrieved_doc_ids: list[str], expected_doc_ids: list[str]) -> float:
    """
    Calculate Mean Reciprocal Rank for a single query.
    
    Reciprocal of the rank of the FIRST relevant document
    Formula: MRR = 1/rank_first_relevant, or 0.0 if no relevant doc found
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ordered by rank).
        expected_doc_ids: List of relevant/expected document IDs.
        
    Returns:
        Reciprocal rank score between 0.0 and 1.0.
    """
    expected_set = set(expected_doc_ids)
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in expected_set:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.
    
    Binary relevance: rel_i = 1 if doc is in expected, else 0
    DCG@k = Σ(i=1..k) rel_i / log2(i+1)
    IDCG@k = DCG of ideal ranking (all relevant docs first)
    NDCG@k = DCG@k / IDCG@k
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs (ordered by rank).
        expected_doc_ids: List of relevant/expected document IDs.
        k: The cutoff rank.
        
    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if not expected_doc_ids:
        return 0.0
    
    expected_set = set(expected_doc_ids)
    
    # Calculate DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in expected_set:
            dcg += 1.0 / math.log2(i + 2) # i is 0-indexed, so rank is i+1, formula needs rank+1 = i+2
            
    # Calculate IDCG@k
    idcg = 0.0
    # The ideal ranking would have all expected docs at the top
    ideal_hits = min(len(expected_set), k)
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg

def compute_retrieval_metrics(retrieved_doc_ids: list[str], expected_doc_ids: list[str], k: int = 5) -> dict[str, float]:
    """
    Compute all retrieval metrics for a single query.
    
    Args:
        retrieved_doc_ids: List of retrieved document IDs.
        expected_doc_ids: List of relevant/expected document IDs.
        k: The cutoff rank for @k metrics. Defaults to 5.
        
    Returns:
        Dictionary containing metric names and values.
    """
    return {
        f"hit_rate@{k}": hit_rate_at_k(retrieved_doc_ids, expected_doc_ids, k),
        f"recall@{k}": recall_at_k(retrieved_doc_ids, expected_doc_ids, k),
        "mrr": mrr(retrieved_doc_ids, expected_doc_ids),
        f"ndcg@{k}": ndcg_at_k(retrieved_doc_ids, expected_doc_ids, k),
    }

def aggregate_metrics(per_question_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Aggregate metrics across multiple queries by taking the mean.
    
    Args:
        per_question_metrics: List of metric dictionaries (one per query).
        
    Returns:
        Dictionary with averaged metrics.
    """
    if not per_question_metrics:
        return {}
        
    aggregated: dict[str, float] = {}
    num_queries = len(per_question_metrics)
    
    # Initialize sums
    for key in per_question_metrics[0].keys():
        aggregated[key] = 0.0
        
    # Sum up
    for metrics in per_question_metrics:
        for key, value in metrics.items():
            aggregated[key] += value
            
    # Average
    for key in aggregated.keys():
        aggregated[key] /= num_queries
        
    return aggregated
