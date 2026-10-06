"""Tests for evaluation metrics and reporting."""

import pytest
from app.services.evaluation.retrieval_metrics import (
    hit_rate_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    compute_retrieval_metrics
)
from app.services.evaluation.dataset import GoldenDataset, GoldenQuestion, QuestionType, CreatedBy
from app.services.evaluation.report import EvalReport, QuestionResult

def test_hit_rate_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc3", "doc4"]
    
    assert hit_rate_at_k(retrieved, expected, k=2) == 0.0
    assert hit_rate_at_k(retrieved, expected, k=3) == 1.0

def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc2", "doc3", "doc4"]
    
    assert recall_at_k(retrieved, expected, k=1) == 0.0
    assert recall_at_k(retrieved, expected, k=2) == 1/3
    assert recall_at_k(retrieved, expected, k=3) == 2/3

def test_mrr():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc3", "doc4"]
    
    # First relevant is doc3 at index 2 (rank 3) -> MRR = 1/3
    assert mrr(retrieved, expected) == 1/3
    
    assert mrr(retrieved, ["doc1"]) == 1.0
    assert mrr(retrieved, ["doc5"]) == 0.0

def test_ndcg_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc3"]
    
    # doc3 is at rank 3 (i=2). DCG = 1/log2(2+2) = 1/2 = 0.5
    # IDCG (ideal is rank 1) = 1/log2(1+1) = 1.0
    # NDCG = 0.5 / 1.0 = 0.5
    assert ndcg_at_k(retrieved, expected, k=3) == 0.5

def test_compute_metrics():
    retrieved = ["doc1", "doc2", "doc3"]
    expected = ["doc3"]
    
    metrics = compute_retrieval_metrics(retrieved, expected, k=3)
    assert metrics["hit_rate@3"] == 1.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr"] == 1/3
    assert metrics["ndcg@3"] == 0.5

def test_dataset_filtering():
    questions = [
        GoldenQuestion(
            question_id="1", question="Q1", ground_truth="A1",
            expected_source_docs=["doc1"], question_type=QuestionType.FACTUAL, created_by=CreatedBy.MANUAL
        ),
        GoldenQuestion(
            question_id="2", question="Q2", ground_truth="",
            expected_source_docs=[], question_type=QuestionType.OUT_OF_SCOPE, created_by=CreatedBy.MANUAL
        )
    ]
    dataset = GoldenDataset(version="test", questions=questions)
    
    assert len(dataset.filter_by_type(QuestionType.FACTUAL)) == 1
    assert len(dataset.filter_by_type(QuestionType.OUT_OF_SCOPE)) == 1

def test_report_serialization():
    result = QuestionResult(
        question_id="q1",
        question="What?",
        ground_truth="This.",
        question_type="factual",
        answer="This.",
        retrieved_contexts=["Context 1"],
        retrieved_doc_ids=["doc1"],
        expected_doc_ids=["doc1"],
        retrieval_metrics={"mrr": 1.0},
        ragas_metrics={"faithfulness": 0.9}
    )
    
    report = EvalReport(
        run_id="test_run",
        timestamp="2026-01-01T00:00:00Z",
        dataset_version="v1",
        retrieval_only=False,
        config_snapshot={"test": "config"},
        prompt_versions={"rag_answer": "1.0"},
        aggregate_retrieval_metrics={"mrr": 1.0},
        aggregate_ragas_metrics={"faithfulness": 0.9},
        aggregate_refusal_accuracy=1.0,
        total_questions=1,
        failed_questions=0,
        results=[result]
    )
    
    md = report.to_markdown()
    assert "test_run" in md
    assert "mrr" in md
    assert "faithfulness" in md
