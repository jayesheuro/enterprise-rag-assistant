"""CLI runner for the evaluation lab.

Executes the evaluation pipeline across a golden dataset.
Supports sequential execution with resumable checkpointing to handle rate limits.

Usage:
    python scripts/run_eval.py --dataset eval/golden/golden_v1.jsonl
    python scripts/run_eval.py --dataset eval/golden/golden_v1.jsonl --retrieval-only
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.evaluation.dataset import GoldenDataset, QuestionType
from app.services.evaluation.report import EvalReport, QuestionResult
from app.services.evaluation.retrieval_metrics import compute_retrieval_metrics, aggregate_metrics
from app.services.generation.rag_chain import RAGChain
from app.services.providers.factory import create_embedder, create_llm
from app.services.retrieval.reranker import get_reranker
from app.services.retrieval.retriever import Retriever
from app.services.retrieval.vector_store import VectorStore
from app.services.generation.prompt_builder import load_prompt

# Try importing RagasRunner, gracefully degrade if missing
try:
    from app.services.evaluation.ragas_runner import RagasRunner, compute_refusal_accuracy
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False

logger = get_logger(__name__)


class EvalRunner:
    def __init__(self, dataset_path: str, run_id: str, retrieval_only: bool = False):
        self.dataset = GoldenDataset.load(dataset_path)
        self.run_id = run_id
        self.retrieval_only = retrieval_only
        
        self.checkpoint_path = Path(f"eval/reports/{run_id}_checkpoint.json")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.config = get_settings()
        
        # Init components
        self.embedder = create_embedder(self.config)
        self.vector_store = VectorStore("data/lancedb")
        self.retriever = Retriever(self.embedder, self.vector_store, self.config.retrieval)
        
        if not self.retrieval_only or self.config.retrieval.reranker == "llm":
            self.llm = create_llm(self.config)
            self.reranker = get_reranker(self.config.retrieval, self.llm)
            
        if not self.retrieval_only:
            self.rag_chain = RAGChain(self.llm, self.retriever, self.reranker, self.config.retrieval)
            if HAS_RAGAS:
                self.ragas_runner = RagasRunner(self.config)
            else:
                logger.warning("RagasRunner not available. RAGAS metrics will be skipped.")

    def load_checkpoint(self) -> dict[str, dict]:
        """Load previously completed questions from checkpoint."""
        if not self.checkpoint_path.exists():
            return {}
            
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data

    def save_checkpoint(self, results: dict[str, dict]):
        """Save partial results to checkpoint."""
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    async def run(self):
        """Execute the evaluation pipeline."""
        logger.info(f"Starting evaluation run {self.run_id}")
        logger.info(f"Dataset: {self.dataset.path} ({self.dataset.size} questions)")
        logger.info(f"Mode: {'Retrieval-Only' if self.retrieval_only else 'End-to-End'}")

        completed = self.load_checkpoint()
        if completed:
            logger.info(f"Resuming from checkpoint with {len(completed)} completed questions.")

        results_list = []
        
        for q in self.dataset.questions:
            if q.question_id in completed:
                results_list.append(QuestionResult(**completed[q.question_id]))
                continue
                
            logger.info(f"Evaluating {q.question_id}: {q.question}")
            
            try:
                # 1. Retrieval
                retrieved_chunks = self.retriever.retrieve(q.question)
                
                # Rerank if configured
                if hasattr(self, 'reranker') and self.reranker:
                    retrieved_chunks = self.reranker.rerank(q.question, retrieved_chunks)
                    
                import os
                retrieved_doc_ids = [os.path.basename(c.source) for c in retrieved_chunks] # 'source' field holds the filename
                
                # Retrieval Metrics
                retrieval_metrics = compute_retrieval_metrics(
                    retrieved_doc_ids=retrieved_doc_ids,
                    expected_doc_ids=q.expected_source_docs,
                    k=self.config.retrieval.final_k
                )
                
                result = QuestionResult(
                    question_id=q.question_id,
                    question=q.question,
                    ground_truth=q.ground_truth,
                    question_type=q.question_type.value,
                    answer="",
                    retrieved_contexts=[c.text for c in retrieved_chunks],
                    retrieved_doc_ids=retrieved_doc_ids,
                    expected_doc_ids=q.expected_source_docs,
                    retrieval_metrics=retrieval_metrics
                )
                
                # 2. Generation & RAGAS
                if not self.retrieval_only:
                    rag_response = self.rag_chain.answer(q.question)
                    result.answer = rag_response.answer
                    result.timings = rag_response.timings
                    
                    # Refusal Accuracy
                    is_out_of_scope = (q.question_type == QuestionType.OUT_OF_SCOPE)
                    if HAS_RAGAS:
                        result.refusal_accuracy = compute_refusal_accuracy(result.answer, is_out_of_scope)
                    
                    # RAGAS metrics (skip for out of scope since ground truth is empty and behavior is refusal)
                    if HAS_RAGAS and not is_out_of_scope:
                        logger.info(f"Computing RAGAS metrics for {q.question_id}...")
                        ragas_metrics = self.ragas_runner.evaluate_question(
                            question=q.question,
                            answer=result.answer,
                            retrieved_contexts=result.retrieved_contexts,
                            ground_truth=q.ground_truth
                        )
                        result.ragas_metrics = ragas_metrics
                
                completed[q.question_id] = result.model_dump()
                self.save_checkpoint(completed)
                results_list.append(result)
                
                # Small sleep to prevent aggressive rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error evaluating {q.question_id}: {e}")
                import traceback
                traceback.print_exc()
                # Break to allow resuming later
                break

        # Generate Report
        if len(results_list) > 0:
            self._generate_report(results_list)
            
        if len(results_list) == self.dataset.size:
            logger.info("Evaluation complete. Deleting checkpoint.")
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()

    def _generate_report(self, results: list[QuestionResult]):
        successful_retrieval = [r.retrieval_metrics for r in results if r.retrieval_metrics and not r.error]
        successful_ragas = [r.ragas_metrics for r in results if r.ragas_metrics and not r.error]
        
        agg_retrieval = aggregate_metrics(successful_retrieval)
        agg_ragas = aggregate_metrics(successful_ragas)
        
        refusal_acc = 0.0
        if not self.retrieval_only:
            valid_refusals = [r.refusal_accuracy for r in results if not r.error]
            if valid_refusals:
                refusal_acc = sum(valid_refusals) / len(valid_refusals)
        
        # Capture config snapshot
        config_dict = self.config.model_dump()
        if "secrets" in config_dict:
            del config_dict["secrets"]  # Never snapshot secrets
            
        # Get prompt versions
        try:
            rag_prompt = load_prompt("rag_answer")
            rerank_prompt = load_prompt("rerank")
            prompts = {"rag_answer": rag_prompt.version, "rerank": rerank_prompt.version}
        except Exception:
            prompts = {}
            
        report = EvalReport(
            run_id=self.run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            dataset_version=self.dataset.version,
            retrieval_only=self.retrieval_only,
            config_snapshot=config_dict,
            prompt_versions=prompts,
            aggregate_retrieval_metrics=agg_retrieval,
            aggregate_ragas_metrics=agg_ragas,
            aggregate_refusal_accuracy=refusal_acc,
            total_questions=len(results),
            failed_questions=len([r for r in results if r.error]),
            results=results
        )
        
        report.save()


def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Enterprise RAG Assistant Evaluation Runner")
    parser.add_argument("--dataset", required=True, help="Path to golden dataset JSONL")
    parser.add_argument("--retrieval-only", action="store_true", help="Run only the retrieval phase")
    parser.add_argument("--run-id", default=None, help="Optional run ID (defaults to timestamp)")
    
    args = parser.parse_args()
    
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    runner = EvalRunner(
        dataset_path=args.dataset,
        run_id=run_id,
        retrieval_only=args.retrieval_only
    )
    
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
