"""Vector store operations using LanceDB."""

import os
import shutil
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.index import FTS
from lancedb.pydantic import LanceModel, Vector

from app.core.exceptions import EmbeddingModelMismatchError
from app.core.logging import get_logger
from app.services.ingestion.chunkers import Chunk

logger = get_logger(__name__)


class VectorStore:
    """LanceDB wrapper for storing and retrieving document chunks."""

    def __init__(self, db_path: str | Path, table_name: str = "chunks"):
        self.db_path = Path(db_path)
        self.table_name = table_name
        
        # Ensure directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    def create_table(self, embedder_model: str, dimension: int) -> None:
        """Create the table with the proper schema if it does not exist."""
        # Using pyarrow schema to be precise
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), dimension)),
            pa.field("text", pa.string()),
            pa.field("chunk_id", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32(), nullable=True),
            pa.field("ingested_at", pa.string()),
            pa.field("embedding_model", pa.string()),
        ])
        
        if not self.table_exists():
            self.db.create_table(self.table_name, schema=schema)
            logger.info(f"Created new table '{self.table_name}' with dimension {dimension}")

    def table_exists(self) -> bool:
        """Check if the table exists."""
        try:
            self.db.open_table(self.table_name)
            return True
        except Exception:
            return False

    def check_embedding_model(self, model: str) -> None:
        """Guard against mixing different embedding models."""
        if not self.table_exists():
            return
            
        tbl = self.db.open_table(self.table_name)
        if tbl.count_rows() == 0:
            return
            
        try:
            df = tbl.search().limit(1).to_arrow()
            if df and len(df) > 0:
                current_model = df["embedding_model"][0].as_py()
                if current_model and current_model != model:
                    raise EmbeddingModelMismatchError(current_model, model)
        except Exception as e:
            if isinstance(e, EmbeddingModelMismatchError):
                raise
            logger.warning(f"Could not verify embedding model: {e}")

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a specific document."""
        if not self.table_exists():
            return
            
        tbl = self.db.open_table(self.table_name)
        try:
            tbl.delete(f"doc_id = '{doc_id}'")
            logger.debug(f"Deleted old chunks for doc_id {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to delete chunks for doc_id {doc_id}: {e}")

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]], embedding_model: str) -> None:
        """Add chunks and their embeddings to the store."""
        if not chunks:
            return
            
        self.check_embedding_model(embedding_model)
        
        data = []
        doc_ids = set()
        
        for chunk, vector in zip(chunks, vectors):
            doc_ids.add(chunk.doc_id)
            row = {
                "vector": vector,
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "source": chunk.metadata.get("source_path", ""),
                "page": chunk.metadata.get("page"),
                "ingested_at": chunk.metadata.get("ingested_at", ""),
                "embedding_model": embedding_model,
            }
            data.append(row)
            
        # Ensure table exists first with dimensions from first vector
        if not self.table_exists():
            self.create_table(embedding_model, len(vectors[0]))
            
        tbl = self.db.open_table(self.table_name)
        
        # Delete old chunks for these documents first
        for doc_id in doc_ids:
            self.delete_by_doc_id(doc_id)
            
        # Add new chunks
        tbl.add(data)
        logger.info(f"Added {len(data)} chunks for {len(doc_ids)} documents")

    def reset(self) -> None:
        """Drop and recreate the table."""
        if self.table_exists():
            self.db.drop_table(self.table_name)
            logger.info(f"Dropped table '{self.table_name}'")

    def build_fts_index(self) -> None:
        """Create Full-Text Search index on text column."""
        if not self.table_exists():
            return
            
        tbl = self.db.open_table(self.table_name)
        if tbl.count_rows() > 0:
            tbl.create_index("text", config=FTS())
            logger.info("Built FTS index on text column")

    def get_stats(self) -> dict[str, Any]:
        """Return basic statistics about the vector store."""
        if not self.table_exists():
            return {"row_count": 0, "doc_count": 0, "embedding_model": "unknown"}
            
        tbl = self.db.open_table(self.table_name)
        row_count = tbl.count_rows()
        
        if row_count == 0:
            return {"row_count": 0, "doc_count": 0, "embedding_model": "unknown"}
            
        df = tbl.search().limit(row_count).to_arrow()
        unique_docs = set(df["doc_id"].to_pylist())
        doc_count = len(unique_docs)
        
        embedding_model = df["embedding_model"][0].as_py()
        
        return {
            "row_count": row_count,
            "doc_count": doc_count,
            "embedding_model": embedding_model,
        }

    def vector_search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """Search by vector similarity. Returns list of dicts with chunk fields + _distance."""
        if not self.table_exists():
            return []
        tbl = self.db.open_table(self.table_name)
        results = tbl.search(query_vector).limit(top_k).to_arrow()
        return self._arrow_to_dicts(results)

    def fts_search(self, query_text: str, top_k: int = 10) -> list[dict]:
        """Full-text search. Returns list of dicts with chunk fields + _score."""
        if not self.table_exists():
            return []
        tbl = self.db.open_table(self.table_name)
        try:
            results = tbl.search(query_text, query_type='fts').limit(top_k).to_arrow()
            return self._arrow_to_dicts(results)
        except Exception as e:
            logger.warning(f"FTS search failed (index may not exist): {e}")
            return []

    def _arrow_to_dicts(self, arrow_table) -> list[dict]:
        """Convert a PyArrow table to a list of dicts."""
        if arrow_table is None or len(arrow_table) == 0:
            return []
        columns = arrow_table.column_names
        rows = []
        for i in range(len(arrow_table)):
            row = {}
            for col in columns:
                val = arrow_table[col][i].as_py()
                row[col] = val
            rows.append(row)
        return rows
