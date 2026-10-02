"""API routes for document ingestion."""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings, AppConfig
from app.models.ingestion import IngestRequest, IngestResponse, IngestStatusResponse
from app.services.ingestion.chunkers import get_chunker
from app.services.ingestion.pipeline import IngestionPipeline
from app.services.providers.factory import create_embedder
from app.services.retrieval.vector_store import VectorStore

router = APIRouter(prefix="/ingest", tags=["ingestion"])


def get_vector_store() -> VectorStore:
    return VectorStore(db_path=Path("data/lancedb"))


@router.post("", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    settings: AppConfig = Depends(get_settings),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """Ingest documents into the vector store."""
    try:
        embedder = create_embedder(settings)
        chunker = get_chunker(settings.chunking)
        
        pipeline = IngestionPipeline(
            embedder=embedder,
            chunker=chunker,
            vector_store=vector_store
        )
        
        paths = [Path(p) for p in request.paths]
        model_name = settings.provider_config.embedding_model if settings.provider_config else "default"
        
        report = pipeline.run(paths, embedding_model=model_name)
        
        return IngestResponse(**report.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=IngestStatusResponse)
async def get_ingest_status(vector_store: VectorStore = Depends(get_vector_store)):
    """Get the status of the vector store."""
    try:
        stats = vector_store.get_stats()
        return IngestStatusResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
