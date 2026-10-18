"""Unit tests for ui/api_client.py using mocked HTTP responses."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ui.api_client import (
    APIClient,
    APIClientError,
    APIConnectionError,
    ChatResult,
    IngestResult,
    IngestStatus,
    SystemConfig,
)


@pytest.fixture
def client():
    return APIClient(base_url="http://testserver:8000")


def test_get_health_success(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "status": "healthy",
        "provider": "gemini",
        "model": "gemini-3.5-flash",
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.request", return_value=mock_resp):
        res = client.get_health()
        assert res["status"] == "healthy"
        assert res["provider"] == "gemini"


def test_get_config_success(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "embedding_model": "gemini-embedding-001",
        "retrieval": {"top_k": 12, "final_k": 4, "mode": "hybrid", "reranker": "llm"},
        "chunking": {"chunk_size": 1000, "chunk_overlap": 200},
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.request", return_value=mock_resp):
        cfg = client.get_config()
        assert isinstance(cfg, SystemConfig)
        assert cfg.provider == "gemini"
        assert cfg.model == "gemini-3.5-flash"
        assert cfg.retrieval["mode"] == "hybrid"


def test_send_chat_success(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "answer": "A PIR must be completed within 48 hours [1].",
        "sources": [
            {
                "text": "PIR must be drafted within 48 hours.",
                "source": "incident-response-plan.md",
                "page": 1,
                "chunk_id": "c1",
                "scores": {
                    "vector_score": 0.85,
                    "fts_score": 0.72,
                    "fused_score": 0.032,
                    "rerank_score": 9.5,
                },
            }
        ],
        "grounded": True,
        "low_confidence": False,
        "refusal": False,
        "timings": {
            "retrieval_ms": 120.0,
            "rerank_ms": 250.0,
            "generation_ms": 400.0,
            "total_ms": 770.0,
        },
        "token_usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 50,
            "total_tokens": 1250,
        },
        "model_name": "gemini-3.5-flash",
        "retrieval_mode": "hybrid",
        "reranker_used": "llm",
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.request", return_value=mock_resp):
        result = client.send_chat("What is the PIR deadline?")
        assert isinstance(result, ChatResult)
        assert result.grounded is True
        assert len(result.sources) == 1
        assert result.sources[0].source == "incident-response-plan.md"
        assert result.sources[0].scores["rerank_score"] == 9.5
        assert result.timings["total_ms"] == 770.0


def test_get_ingest_status_success(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "row_count": 142,
        "doc_count": 9,
        "embedding_model": "gemini-embedding-001",
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.request", return_value=mock_resp):
        status = client.get_ingest_status()
        assert isinstance(status, IngestStatus)
        assert status.row_count == 142
        assert status.doc_count == 9


def test_upload_and_ingest_success(client):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "docs_processed": 1,
        "chunks_created": 15,
        "tokens_embedded": 4500,
        "failures": [],
        "wall_time_seconds": 1.25,
        "embedding_model": "gemini-embedding-001",
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.request", return_value=mock_resp):
        report = client.upload_and_ingest([("sample.md", b"# Policy", "text/markdown")])
        assert isinstance(report, IngestResult)
        assert report.docs_processed == 1
        assert report.chunks_created == 15


def test_eval_reports_endpoints(client):
    mock_list = MagicMock()
    mock_list.json.return_value = {"reports": ["run_A", "run_B"]}
    mock_list.raise_for_status = MagicMock()

    mock_get = MagicMock()
    mock_get.json.return_value = {"run_id": "run_A", "aggregate_retrieval_metrics": {"mrr": 0.85}}
    mock_get.raise_for_status = MagicMock()

    mock_narrative = MagicMock()
    mock_narrative.json.return_value = {"content": "# Experiment Narrative"}
    mock_narrative.raise_for_status = MagicMock()

    with patch("httpx.Client.request", side_effect=[mock_list, mock_get, mock_narrative]):
        reports = client.list_eval_reports()
        assert reports == ["run_A", "run_B"]

        rep = client.get_eval_report("run_A")
        assert rep["run_id"] == "run_A"

        narrative = client.get_eval_narrative()
        assert "# Experiment Narrative" in narrative


def test_connection_error_raises_friendly_exception(client):
    with patch("httpx.Client.request", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(APIConnectionError) as exc_info:
            client.get_health()
        assert "not reachable at http://testserver:8000" in str(exc_info.value)
        assert "uv run uvicorn app.main:app" in str(exc_info.value)


def test_http_status_error_handling(client):
    req = httpx.Request("GET", "http://testserver:8000/eval/reports/missing")
    resp = httpx.Response(404, request=req, json={"message": "Report not found"})
    err = httpx.HTTPStatusError("Not Found", request=req, response=resp)

    with patch("httpx.Client.request", side_effect=err):
        with pytest.raises(APIClientError) as exc_info:
            client.get_eval_report("missing")
        assert "API Error (404)" in str(exc_info.value)
        assert "Report not found" in str(exc_info.value)
