#!/usr/bin/env python
"""Demo startup seeding script.

Run at container boot time. Idempotent:
  - If data/lancedb/ already contains chunks, skips ingestion (fast restart).
  - Copies pre-generated eval reports from eval/reports/ so the dashboard
    is populated on first launch without a live eval run.

Usage (called by start.sh):
    python scripts/seed_demo.py

Environment:
    GOOGLE_API_KEY - required for Gemini embedding during ingestion
    SEED_CORPUS_PATH - override corpus path (default: data/corpus)
    SEED_FORCE - set to "true" to force re-ingestion even if DB exists
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path so app imports resolve
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import get_logger, setup_logging  # noqa: E402
from app.services.ingestion.chunkers import get_chunker  # noqa: E402
from app.services.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.services.providers.factory import create_embedder  # noqa: E402
from app.services.retrieval.vector_store import VectorStore  # noqa: E402

setup_logging()
logger = get_logger(__name__)

CORPUS_PATH = Path(os.getenv("SEED_CORPUS_PATH", str(PROJECT_ROOT / "data" / "corpus")))
LANCEDB_PATH = PROJECT_ROOT / "data" / "lancedb"
FORCE_RESEED = os.getenv("SEED_FORCE", "false").lower() in ("true", "1")


def _lancedb_has_data() -> bool:
    """Return True if LanceDB already contains ingested chunks."""
    try:
        import lancedb

        db = lancedb.connect(str(LANCEDB_PATH))
        table = db.open_table("chunks")
        return table.count_rows() > 0
    except Exception as e:
        logger.info("LanceDB 'chunks' table not yet initialized (%s); starting fresh seed.", e)
        return False


def seed_corpus() -> None:
    """Ingest the bundled corpus into LanceDB if not already seeded."""
    if not FORCE_RESEED and _lancedb_has_data():
        logger.info("LanceDB already seeded — skipping corpus ingestion.")
        return

    if not CORPUS_PATH.exists() or not any(CORPUS_PATH.iterdir()):
        logger.error("Corpus path is empty or missing: %s", CORPUS_PATH)
        sys.exit(1)

    logger.info("Seeding corpus from %s ...", CORPUS_PATH)

    settings = get_settings()
    embedder = create_embedder(config=settings)
    chunker = get_chunker(config=settings.chunking)
    vector_store = VectorStore(db_path=LANCEDB_PATH)

    pipeline = IngestionPipeline(
        embedder=embedder,
        chunker=chunker,
        vector_store=vector_store,
    )

    model_name = (
        settings.provider_config.embedding_model if settings.provider_config else "default"
    )
    report = pipeline.run([CORPUS_PATH], embedding_model=model_name)

    logger.info(
        "Seed complete: %d docs, %d chunks, %.1fs",
        report.docs_processed,
        report.chunks_created,
        report.wall_time_seconds,
    )

    if report.failures:
        for f in report.failures:
            logger.warning("Ingestion failure: %s — %s", f["path"], f["reason"])


def ensure_eval_reports() -> None:
    """Eval reports are committed to eval/reports/; nothing to copy.

    LanceDB is ephemeral on HF Spaces — the reports directory is part of the
    container image (committed JSON/MD files) so they survive restarts.
    This function validates they are present and logs the count.
    """
    reports_dir = PROJECT_ROOT / "eval" / "reports"
    json_reports = list(reports_dir.glob("*.json"))
    md_reports = list(reports_dir.glob("*.md"))

    if not json_reports:
        logger.warning(
            "No JSON eval reports found in %s. "
            "The Eval Dashboard will be empty. "
            "Run `uv run python scripts/run_eval.py` locally and commit the reports.",
            reports_dir,
        )
    else:
        logger.info(
            "Eval reports ready: %d JSON, %d Markdown in %s",
            len(json_reports),
            len(md_reports),
            reports_dir,
        )


def main() -> None:
    logger.info("=== Enterprise RAG Assistant — Demo Seed ===")
    logger.info("PROJECT_ROOT: %s", PROJECT_ROOT)
    logger.info("CORPUS_PATH : %s", CORPUS_PATH)
    logger.info("LANCEDB_PATH: %s", LANCEDB_PATH)
    logger.info("FORCE_RESEED: %s", FORCE_RESEED)

    ensure_eval_reports()
    seed_corpus()

    logger.info("Seed complete. Ready to accept traffic.")


if __name__ == "__main__":
    main()

