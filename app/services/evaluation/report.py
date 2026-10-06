"""Evaluation report generator and persistence layer.

This module defines the EvalReport schema, which snapshots the full application
config, prompt versions, and metrics for a single evaluation run. It saves
reports as both JSON (for downstream processing/MLflow) and Markdown (for
human readability and A/B comparison).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import AppConfig, PROJECT_ROOT
from app.core.logging import get_logger

logger = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"


class QuestionResult(BaseModel):
    """Result of evaluating a single question."""

    question_id: str
    question: str
    ground_truth: str
    question_type: str
    answer: str
    retrieved_contexts: list[str]
    retrieved_doc_ids: list[str]
    expected_doc_ids: list[str]
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    ragas_metrics: dict[str, float] = Field(default_factory=dict)
    refusal_accuracy: float = 0.0
    error: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class EvalReport(BaseModel):
    """Full snapshot of an evaluation run."""

    run_id: str
    timestamp: str
    dataset_version: str
    retrieval_only: bool
    config_snapshot: dict[str, Any]
    prompt_versions: dict[str, str]
    aggregate_retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    aggregate_ragas_metrics: dict[str, float] = Field(default_factory=dict)
    aggregate_refusal_accuracy: float = 0.0
    total_questions: int
    failed_questions: int
    results: list[QuestionResult]

    def save(self) -> None:
        """Save the report to JSON and Markdown formats."""
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        json_path = REPORTS_DIR / f"{self.run_id}.json"
        md_path = REPORTS_DIR / f"{self.run_id}.md"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
            
        # Save Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

        logger.info(f"Saved evaluation report to {json_path} and {md_path}")

    def to_markdown(self) -> str:
        """Generate a human-readable Markdown report."""
        md = [
            f"# Evaluation Report: {self.run_id}",
            f"**Timestamp:** {self.timestamp}  ",
            f"**Dataset Version:** {self.dataset_version}  ",
            f"**Mode:** {'Retrieval-Only' if self.retrieval_only else 'End-to-End (Retrieval + RAGAS)'}  ",
            f"**Success Rate:** {self.total_questions - self.failed_questions}/{self.total_questions}",
            "",
            "## Configuration Snapshot",
            "```json",
            json.dumps(self.config_snapshot, indent=2),
            "```",
            "",
            "## Aggregate Metrics",
        ]

        if self.aggregate_retrieval_metrics:
            md.append("### Retrieval Metrics")
            md.append("| Metric | Score |")
            md.append("|---|---|")
            for k, v in self.aggregate_retrieval_metrics.items():
                md.append(f"| {k} | {v:.4f} |")
            md.append("")

        if not self.retrieval_only:
            md.append("### Generation Metrics (RAGAS)")
            md.append("| Metric | Score |")
            md.append("|---|---|")
            for k, v in self.aggregate_ragas_metrics.items():
                md.append(f"| {k} | {v:.4f} |")
            md.append(f"| refusal_accuracy | {self.aggregate_refusal_accuracy:.4f} |")
            md.append("")
            
            # Find worst questions by faithfulness
            results_with_faithfulness = [r for r in self.results if "faithfulness" in r.ragas_metrics]
            if results_with_faithfulness:
                results_with_faithfulness.sort(key=lambda x: x.ragas_metrics["faithfulness"])
                worst = results_with_faithfulness[:5]
                
                md.append("## Worst 5 Questions by Faithfulness (Hallucination Debugging)")
                for r in worst:
                    md.append(f"### Question: {r.question}")
                    md.append(f"**Score:** {r.ragas_metrics['faithfulness']:.4f}")
                    md.append(f"**Answer:** {r.answer}")
                    md.append("**Retrieved Contexts:**")
                    for i, ctx in enumerate(r.retrieved_contexts):
                        md.append(f"**[{i+1}]** {ctx[:200]}...")
                    md.append("")

        return "\n".join(md)

    @classmethod
    def load(cls, run_id: str) -> "EvalReport":
        """Load a report by run_id."""
        json_path = REPORTS_DIR / f"{run_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Report not found: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return cls(**data)

