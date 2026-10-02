"""Pipeline for orchestrating document ingestion."""

import time
from pathlib import Path

from pydantic import BaseModel

from app.core.logging import get_logger
from app.services.ingestion.chunkers import BaseChunker
from app.services.ingestion.loaders import load_document, LOADER_REGISTRY
from app.services.providers.base import BaseEmbedder
from app.services.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class IngestionReport(BaseModel):
    """Summary report of an ingestion run."""
    docs_processed: int
    chunks_created: int
    tokens_embedded: int
    failures: list[dict[str, str]]
    wall_time_seconds: float
    embedding_model: str


class IngestionPipeline:
    """Orchestrates loading, chunking, embedding, and storing documents."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        chunker: BaseChunker,
        vector_store: VectorStore,
        batch_size: int = 50,
    ):
        self.embedder = embedder
        self.chunker = chunker
        self.vector_store = vector_store
        self.batch_size = batch_size

    def run(self, paths: list[Path], embedding_model: str) -> IngestionReport:
        """Run the ingestion pipeline on a list of paths."""
        start_time = time.time()
        
        # Fail fast if attempting to mix embedding models
        self.vector_store.check_embedding_model(embedding_model)
        
        docs_processed = 0
        chunks_created = 0
        tokens_embedded = 0
        failures = []
        
        all_chunks = []
        
        # 1. Load & Chunk
        for path in paths:
            try:
                # Load directory vs file
                if path.is_dir():
                    for file_path in path.rglob("*"):
                        if file_path.is_file():
                            if file_path.suffix.lower() not in LOADER_REGISTRY:
                                continue
                            try:
                                doc = load_document(file_path)
                                chunks = self.chunker.chunk(doc)
                                all_chunks.extend(chunks)
                                docs_processed += 1
                                chunks_created += len(chunks)
                                logger.info(f"Loaded and chunked {file_path}")
                            except Exception as e:
                                failures.append({"path": str(file_path), "reason": str(e)})
                else:
                    doc = load_document(path)
                    chunks = self.chunker.chunk(doc)
                    all_chunks.extend(chunks)
                    docs_processed += 1
                    chunks_created += len(chunks)
                    logger.info(f"Loaded and chunked {path}")
            except Exception as e:
                failures.append({"path": str(path), "reason": str(e)})
                logger.error(f"Failed to load {path}: {e}")

        # 2. Embed & Store in batches
        for i in range(0, len(all_chunks), self.batch_size):
            batch = all_chunks[i:i + self.batch_size]
            texts = [c.text for c in batch]
            
            try:
                vectors = self.embedder.embed_documents(texts)
                for c in batch:
                    tokens_embedded += c.token_count
                    
                self.vector_store.add_chunks(batch, vectors, embedding_model)
                logger.info(f"Embedded and stored batch {i // self.batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to embed/store batch: {e}")
                for c in batch:
                    failures.append({"path": c.metadata.get("source_path", "unknown"), "reason": f"Embedding failed: {e}"})

        # 3. Build FTS index
        try:
            self.vector_store.build_fts_index()
        except Exception as e:
            logger.warning(f"Failed to build FTS index: {e}")

        wall_time = time.time() - start_time
        
        return IngestionReport(
            docs_processed=docs_processed,
            chunks_created=chunks_created,
            tokens_embedded=tokens_embedded,
            failures=failures,
            wall_time_seconds=wall_time,
            embedding_model=embedding_model,
        )
