#!/usr/bin/env python
"""CLI for document ingestion into the vector store."""

import argparse
import sys
from pathlib import Path

from app.core.config import get_settings
from app.services.ingestion.chunkers import get_chunker
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.providers.factory import create_embedder
from app.services.retrieval.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store.")
    parser.add_argument("path", type=str, help="Path to file or directory to ingest")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store before ingesting")
    args = parser.parse_args()

    settings = get_settings()
    
    # Initialize components
    embedder = create_embedder(settings)
    chunker = get_chunker(settings.chunking)
    db_path = Path("data/lancedb")
    vector_store = VectorStore(db_path=db_path)
    
    if args.reset:
        vector_store.reset()
        print("Vector store reset.")

    pipeline = IngestionPipeline(
        embedder=embedder,
        chunker=chunker,
        vector_store=vector_store
    )

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path {path} does not exist.")
        sys.exit(1)

    print(f"Starting ingestion for {path}...")
    
    # Needs provider_config for embedding_model
    model_name = settings.provider_config.embedding_model if settings.provider_config else "default"
    
    report = pipeline.run([path], embedding_model=model_name)

    print("\n=== Ingestion Report ===")
    print(f"Documents processed: {report.docs_processed}")
    print(f"Chunks created: {report.chunks_created}")
    print(f"Tokens embedded: {report.tokens_embedded}")
    print(f"Time taken: {report.wall_time_seconds:.2f}s")
    print(f"Embedding model: {report.embedding_model}")
    
    if report.failures:
        print("\nFailures:")
        for failure in report.failures:
            print(f"  - {failure['path']}: {failure['reason']}")

if __name__ == "__main__":
    main()
