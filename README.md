# Enterprise RAG Assistant

A document Q&A platform with a **built-in evaluation lab**. Every answer is grounded with citations, and the system measures its own retrieval quality, generation quality, and hallucination rate — with prompt versions and evaluation experiments tracked in MLflow.

> Most RAG demos stop at "it answers questions." This one **proves** its answers: retrieval metrics, RAGAS generation metrics, hallucination gating, and versioned prompts with tracked experiments.

## Features

- 📄 **Document Ingestion** — Upload PDFs, text files, and more; automatically chunked and embedded
- 🔍 **Hybrid Retrieval** — Vector similarity + keyword search via LanceDB
- 💬 **Grounded Q&A** — Answers with source citations and confidence scores
- 📊 **Evaluation Lab** — RAGAS metrics (faithfulness, relevance), retrieval metrics (Recall@K, MRR, NDCG)
- 🔬 **Experiment Tracking** — MLflow-tracked prompt versions and eval experiments
- 🔌 **Multi-Provider** — Swap between Gemini, Ollama, and Bedrock with a single config change

## Tech Stack

| Layer        | Technology                                              |
| ------------ | ------------------------------------------------------- |
| Backend      | FastAPI + Uvicorn                                       |
| UI           | Streamlit                                               |
| Vector Store | LanceDB (embedded, hybrid search)                       |
| LLM          | Gemini (default) / Ollama (local) / Bedrock (AWS)       |
| Embeddings   | Gemini `text-embedding-004` / Ollama `nomic-embed-text` |
| Evaluation   | RAGAS + custom IR metrics                               |
| Tracking     | MLflow                                                  |
| Language     | Python 3.11 with type hints + Pydantic v2               |

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- A [Google AI Studio API key](https://aistudio.google.com/apikey)

### Setup

```bash
# Clone and install
git clone <repo-url>
cd enterprise-rag-app
uv sync

# Configure secrets
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# Verify setup
uv run python scripts/smoke_test.py

# Start the API server
uv run uvicorn app.main:app --reload

# Visit http://127.0.0.1:8000/docs for the API documentation
```

### Switching Providers

Edit `configs/app.yaml` — change one line:

```yaml
provider: gemini # or: ollama, bedrock
```

No code changes required. See `configs/providers/` for provider-specific settings.

## Project Structure

```
├── app/                    # FastAPI backend
│   ├── api/routes/         # REST endpoints
│   ├── core/               # Config, logging, exceptions
│   ├── models/             # Pydantic schemas
│   └── services/
│       ├── providers/      # LLM/Embedding abstraction (Strategy + Factory)
│       ├── ingestion/      # Document loading & chunking
│       ├── retrieval/      # Vector search & reranking
│       ├── generation/     # Prompt building & RAG chain
│       └── evaluation/     # Metrics & reporting
├── ui/                     # Streamlit frontend
├── configs/                # YAML configuration files
├── prompts/                # Versioned prompt templates
├── data/corpus/            # Sample documents
├── eval/golden/            # Golden Q&A datasets
├── docs/architecture/      # Architecture Decision Records
├── tests/                  # pytest test suite
└── scripts/                # CLI utilities
```

## Architecture

The core design pattern is a **provider abstraction** using the Strategy + Factory pattern:

```
configs/app.yaml ──→ Factory ──→ BaseLLM / BaseEmbedder
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                      GeminiLLM   OllamaLLM   BedrockLLM
```

All configuration lives in YAML files; secrets live in `.env`. See `docs/architecture/decisions.md` for detailed ADRs.

## License

MIT
