"""Integration tests for the Ingestion Pipeline.

These tests verify how the components orchestrate together, specifically targeting
gaps in isolated unit tests (like default configurations and fail-fast behaviors).
"""

from pathlib import Path
import pytest

from app.core.config import get_settings
from app.core.exceptions import EmbeddingModelMismatchError, ProviderError
from app.services.ingestion.chunkers import get_chunker
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.providers.base import BaseEmbedder
from app.services.retrieval.vector_store import VectorStore
from tests.fakes import FakeEmbedder


class CrashEmbedder(BaseEmbedder):
    """An embedder that simulates a fatal SDK/Network crash (e.g., 404 Not Found)."""
    
    @property
    def dimension(self) -> int:
        return 768

    def _embed_impl(self, texts: list[str]) -> list[list[float]]:
        # If the pipeline reaches this point, the fail-fast check failed!
        raise ProviderError("API 404 NOT FOUND: Model does not exist", provider="fake")


def test_pipeline_default_config_chunking(tmp_path: Path):
    """Verify that default chunking config actually splits a realistic document."""
    # 1. Load real application defaults (not isolated unit test hardcodes)
    settings = get_settings()
    chunker = get_chunker(settings.chunking)
    embedder = FakeEmbedder()
    
    # 2. Set up a fresh local vector store
    db_path = tmp_path / "lancedb"
    vector_store = VectorStore(db_path)
    
    pipeline = IngestionPipeline(
        embedder=embedder,
        chunker=chunker,
        vector_store=vector_store
    )
    
    # 3. Create a synthetic document that is roughly 2500 characters
    # (Similar to the synthetic corpus generated in Phase 2)
    test_file = tmp_path / "realistic_doc.txt"
    content = "This is a sentence to add bulk. " * 80  # ~2560 chars
    test_file.write_text(content, encoding="utf-8")
    
    # 4. Run the pipeline
    report = pipeline.run([test_file], embedding_model="test-model")
    
    # 5. Assert the integration of chunker + defaults actually splits the doc
    assert report.docs_processed == 1
    assert report.chunks_created > 1, (
        f"Integration failure: Expected multiple chunks based on default config, "
        f"but got exactly {report.chunks_created}. Chunk size may be too large."
    )


def test_pipeline_fail_fast_mismatch(tmp_path: Path):
    """Verify the pipeline blocks mismatched models BEFORE executing expensive/failing API calls."""
    settings = get_settings()
    chunker = get_chunker(settings.chunking)
    db_path = tmp_path / "lancedb"
    vector_store = VectorStore(db_path)
    test_file = tmp_path / "doc.txt"
    test_file.write_text("Hello world", encoding="utf-8")
    
    # 1. Prime the database with "model_A"
    pipeline1 = IngestionPipeline(FakeEmbedder(), chunker, vector_store)
    pipeline1.run([test_file], embedding_model="model_A")
    
    # 2. Setup second pipeline with "model_B" and an embedder that crashes on invocation
    # If the pipeline doesn't fail fast, CrashEmbedder will throw a ProviderError.
    crash_embedder = CrashEmbedder()
    pipeline2 = IngestionPipeline(crash_embedder, chunker, vector_store)
    
    # 3. Assert we get the specific Mismatch Error, NOT the ProviderError from the embedder
    with pytest.raises(EmbeddingModelMismatchError):
        pipeline2.run([test_file], embedding_model="model_B")
