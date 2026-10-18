"""Typed API Client for communicating with the FastAPI backend.

Strictly communicates over HTTP (via BACKEND_URL) so that the Streamlit frontend
remains completely decoupled from backend service code.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# Backend URL defaults to localhost:8000
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "180.0"))


class APIClientError(Exception):
    """Base exception for API client errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class APIConnectionError(APIClientError):
    """Raised when the backend server is unreachable."""

    def __init__(self, base_url: str):
        message = (
            f"Backend server not reachable at {base_url}. "
            "Is the FastAPI service running? "
            "(Start it with: uv run uvicorn app.main:app)"
        )
        super().__init__(message)


@dataclass
class SourceItem:
    text: str
    source: str
    page: int | None
    chunk_id: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ChatResult:
    answer: str
    sources: list[SourceItem]
    grounded: bool
    low_confidence: bool
    refusal: bool
    timings: dict[str, float]
    token_usage: dict[str, int]
    model_name: str
    retrieval_mode: str
    reranker_used: str


@dataclass
class IngestStatus:
    row_count: int
    doc_count: int
    embedding_model: str


@dataclass
class IngestResult:
    docs_processed: int
    chunks_created: int
    tokens_embedded: int
    failures: list[dict[str, str]]
    wall_time_seconds: float
    embedding_model: str


@dataclass
class SystemConfig:
    provider: str
    model: str
    embedding_model: str
    retrieval: dict[str, Any]
    chunking: dict[str, Any]


class APIClient:
    """HTTP client wrapper for all backend endpoints."""

    def __init__(self, base_url: str = BACKEND_URL, timeout: float = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        json_data: Any = None,
        files: Any = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    json=json_data,
                    files=files,
                    params=params,
                )
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as e:
            raise APIConnectionError(self.base_url) from e
        except httpx.TimeoutException as e:
            raise APIClientError(
                f"Request to {path} timed out after {self.timeout}s. The model may be processing."
            ) from e
        except httpx.HTTPStatusError as e:
            detail = "Unknown error"
            try:
                err_data = e.response.json()
                detail = err_data.get("message") or err_data.get("detail") or str(err_data)
            except Exception:
                detail = e.response.text or f"HTTP {e.response.status_code}"
            raise APIClientError(
                f"API Error ({e.response.status_code}): {detail}", e.response.status_code
            ) from e
        except Exception as e:
            raise APIClientError(f"Unexpected error calling {path}: {str(e)}") from e

    def get_health(self) -> dict[str, Any]:
        """Check server health."""
        return self._request("GET", "/health")

    def get_config(self) -> SystemConfig:
        """Get system configuration (provider, retrieval, chunking settings)."""
        data = self._request("GET", "/config")
        return SystemConfig(
            provider=data.get("provider", "unknown"),
            model=data.get("model", "unknown"),
            embedding_model=data.get("embedding_model", "unknown"),
            retrieval=data.get("retrieval", {}),
            chunking=data.get("chunking", {}),
        )

    def send_chat(self, query: str) -> ChatResult:
        """Send a question to the RAG pipeline."""
        data = self._request("POST", "/chat", json_data={"query": query})
        sources = [
            SourceItem(
                text=s.get("text", ""),
                source=s.get("source", "unknown"),
                page=s.get("page"),
                chunk_id=s.get("chunk_id", ""),
                scores=s.get("scores", {}),
            )
            for s in data.get("sources", [])
        ]
        return ChatResult(
            answer=data.get("answer", ""),
            sources=sources,
            grounded=data.get("grounded", False),
            low_confidence=data.get("low_confidence", False),
            refusal=data.get("refusal", False),
            timings=data.get("timings", {}),
            token_usage=data.get("token_usage", {}),
            model_name=data.get("model_name", "unknown"),
            retrieval_mode=data.get("retrieval_mode", "unknown"),
            reranker_used=data.get("reranker_used", "unknown"),
        )

    def get_ingest_status(self) -> IngestStatus:
        """Get vector store row and document counts."""
        data = self._request("GET", "/ingest/status")
        return IngestStatus(
            row_count=data.get("row_count", 0),
            doc_count=data.get("doc_count", 0),
            embedding_model=data.get("embedding_model", "unknown"),
        )

    def ingest_paths(self, paths: list[str]) -> IngestResult:
        """Ingest existing file paths on disk."""
        data = self._request("POST", "/ingest", json_data={"paths": paths})
        return IngestResult(
            docs_processed=data.get("docs_processed", 0),
            chunks_created=data.get("chunks_created", 0),
            tokens_embedded=data.get("tokens_embedded", 0),
            failures=data.get("failures", []),
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
            embedding_model=data.get("embedding_model", "unknown"),
        )

    def upload_and_ingest(self, file_tuples: list[tuple[str, bytes, str]]) -> IngestResult:
        """Upload raw files and ingest them."""
        # httpx files parameter expects list of (field_name, (filename, content, content_type))
        upload_files = [
            ("files", (name, content, mime or "application/octet-stream"))
            for name, content, mime in file_tuples
        ]
        data = self._request("POST", "/ingest/upload", files=upload_files)
        return IngestResult(
            docs_processed=data.get("docs_processed", 0),
            chunks_created=data.get("chunks_created", 0),
            tokens_embedded=data.get("tokens_embedded", 0),
            failures=data.get("failures", []),
            wall_time_seconds=data.get("wall_time_seconds", 0.0),
            embedding_model=data.get("embedding_model", "unknown"),
        )

    def list_eval_reports(self) -> list[str]:
        """List all available evaluation report IDs."""
        data = self._request("GET", "/eval/reports")
        return data.get("reports", [])

    def get_eval_report(self, run_id: str) -> dict[str, Any]:
        """Get full evaluation report by run ID."""
        return self._request("GET", f"/eval/reports/{run_id}")

    def get_eval_narrative(self) -> str:
        """Get experiment narrative markdown text."""
        data = self._request("GET", "/eval/narrative")
        return data.get("content", "")
