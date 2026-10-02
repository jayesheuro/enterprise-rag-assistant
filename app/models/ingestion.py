"""Request and response models for document ingestion API."""

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    paths: list[str] = Field(..., description="List of file or directory paths to ingest")


class IngestResponse(BaseModel):
    """Response model for document ingestion results."""
    docs_processed: int = Field(..., description="Number of documents successfully processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    tokens_embedded: int = Field(..., description="Estimated number of tokens embedded")
    failures: list[dict[str, str]] = Field(default_factory=list, description="List of failures (path and reason)")
    wall_time_seconds: float = Field(..., description="Total time taken for ingestion in seconds")
    embedding_model: str = Field(..., description="Embedding model used")


class IngestStatusResponse(BaseModel):
    """Response model for vector store status."""
    row_count: int = Field(..., description="Total number of chunks in the vector store")
    doc_count: int = Field(..., description="Number of unique documents in the vector store")
    embedding_model: str = Field(..., description="Embedding model used in the vector store")
