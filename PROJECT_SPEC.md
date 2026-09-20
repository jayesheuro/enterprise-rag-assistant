# MASTER CONTEXT - Enterprise RAG Assistant

You are the lead engineer building **Enterprise RAG Assistant** with me. I am new to AI
Engineering but a capable software engineer with good AWS exposure. This project is for my
portfolio AND my learning - both are first-class requirements. Read this entire document
before doing anything; it defines the product, architecture, conventions, and the
documentation contract you must honor in every phase.

---

## 1. Product vision

A document Q&A platform ("chat with your enterprise docs") whose differentiator is a
**built-in evaluation lab**: every answer is grounded with citations, and the system can
measure its own retrieval quality, generation quality, and hallucination rate using RAGAS
and classic IR metrics - with prompt versions and eval experiments tracked in MLflow.

Elevator pitch for the portfolio: _"Most RAG demos stop at 'it answers questions.' This one
proves its answers: retrieval metrics, RAGAS generation metrics, hallucination gating, and
versioned prompts with tracked experiments."_

## 2. Tech stack (fixed - do not substitute without asking me)

| Layer         | Choice                                                                    | Notes                                                                                         |
| ------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Language      | Python 3.11                                                               | type hints everywhere, Pydantic v2 models                                                     |
| Backend       | FastAPI + uvicorn                                                         | REST API, OpenAPI docs auto-generated                                                         |
| UI            | Streamlit                                                                 | separate `ui/` app, talks to backend over HTTP                                                |
| Vector store  | LanceDB                                                                   | embedded, file-based, hybrid search capable                                                   |
| LLM providers | Gemini (cloud, default), Ollama (local dev), Bedrock (stub)               | swappable via config file ONLY - zero code changes to switch                                  |
| Embeddings    | Provider-matched: Gemini `text-embedding-004` / Ollama `nomic-embed-text` | dimension recorded in table metadata; never mix embeddings from different models in one table |
| Evaluation    | RAGAS + custom retrieval metrics (recall@k, MRR, NDCG)                    |                                                                                               |
| Tracking      | MLflow (local file/sqlite backend)                                        | prompt registry + eval experiment tracking                                                    |
| Packaging     | `uv` with `pyproject.toml` (fallback: pip + requirements.txt if I say so) |                                                                                               |
| Testing       | pytest                                                                    | unit tests per module; integration tests marked and skippable when no API key                 |
| Deployment    | Docker -> Hugging Face Spaces (free CPU)                                  | Phase 7                                                                                       |

## 3. Repository structure (canonical - create files exactly here)

```
enterprise-rag-app/
├── PROJECT_SPEC.md            # this document, saved into the repo in Phase 1
├── app/                       # FastAPI backend package
│   ├── main.py                # app factory, router registration
│   ├── core/
│   │   ├── config.py          # Pydantic Settings; loads configs/*.yaml + .env
│   │   └── logging.py
│   ├── api/routes/            # chat.py, ingest.py, eval.py, health.py
│   ├── models/                # Pydantic request/response schemas
│   └── services/
│       ├── providers/         # base.py (ABCs), gemini.py, ollama.py, bedrock.py, factory.py
│       ├── ingestion/         # loaders.py, chunkers.py, pipeline.py
│       ├── retrieval/         # vector_store.py (LanceDB), retriever.py, reranker.py
│       ├── generation/        # prompt_builder.py, rag_chain.py, guardrails.py
│       └── evaluation/        # dataset.py, retrieval_metrics.py, ragas_runner.py, report.py
├── ui/                        # Streamlit app (pages/, api_client.py, app.py)
├── configs/
│   ├── app.yaml               # active app config (chunking, retrieval, provider name)
│   └── providers/             # gemini.yaml, ollama.yaml, bedrock.yaml
├── prompts/                   # prompt templates as versioned .yaml files (name, version, template, changelog)
├── data/
│   ├── corpus/                # sample documents (committed)
│   └── lancedb/               # vector store files (GITIGNORED)
├── eval/
│   ├── golden/                # golden Q&A datasets (committed, versioned)
│   └── reports/               # generated eval reports (json + md)
├── docs/
│   ├── learning/              # phase1-*.md ... phase7-*.md <- LEARNING CONTRACT
│   ├── question-bank/         # phase1-questions.md ... phase7-questions.md
│   └── architecture/          # decisions.md (ADR log), diagrams (mermaid in md)
├── tests/
├── scripts/                   # ingest.py, run_eval.py, seed_demo.py (CLI entrypoints)
├── .env.example               # every env var documented; NEVER commit real .env
├── Dockerfile                 # Phase 7
└── README.md
```

## 4. The provider abstraction (core architectural pattern)

This is the project's most interview-relevant design. Implement it exactly:

- `app/services/providers/base.py` defines two ABCs:
    - `BaseLLM` with `generate(messages: list[Message], **kwargs) -> LLMResponse`
      (LLMResponse carries text, token usage, latency_ms, model_name)
    - `BaseEmbedder` with `embed_documents(texts: list[str]) -> list[list[float]]` and
      `embed_query(text: str) -> list[float]`, plus a `dimension` property
- Concrete implementations: `GeminiLLM`/`GeminiEmbedder` (google-genai SDK),
  `OllamaLLM`/`OllamaEmbedder` (local HTTP), `BedrockLLM`/`BedrockEmbedder` (boto3, fully
  written but clearly marked untested-stub; must import lazily so boto3 is optional)
- `factory.py` reads `configs/app.yaml` -> `provider: gemini|ollama|bedrock` -> loads the
  matching `configs/providers/<name>.yaml` -> returns instances. Switching providers =
  editing one line of YAML. This is a hard requirement; test it.
- Retries with exponential backoff and a per-request timeout live in the base layer, not
  duplicated per provider.

## 5. Configuration & secrets

- All tunables (chunk size/overlap, top_k, similarity threshold, reranking on/off, provider,
  model names, temperature) live in `configs/*.yaml` - never hardcoded.
- Secrets (API keys) live in `.env` only, loaded via Pydantic Settings. `.env` is gitignored;
  `.env.example` documents every variable. Never print or log a secret.

## 6. THE LEARNING CONTRACT (non-negotiable, every phase)

I am learning AI Engineering through this build. Every phase, alongside the code, you MUST
produce/update:

1. **`docs/learning/phaseN-<topic>.md`** containing:
    - **Concepts**: the theory behind what we built (e.g., why chunk overlap exists, what
      embedding similarity actually measures, how RAGAS faithfulness is computed) - explained
      for a strong SWE who is new to AI, with small concrete examples.
    - **Decisions & tradeoffs**: every significant choice, the alternatives we rejected, and
      why (e.g., LanceDB vs Chroma vs pgvector; fixed vs semantic chunking; ABC-based provider
      abstraction vs LiteLLM).
    - **Business relevance**: one short section - why an enterprise cares (cost, compliance,
      trust, latency).
    - **Mermaid diagram** of the phase's data/control flow.
2. **`docs/question-bank/phaseN-questions.md`**: 12-15 interview questions WITH model
   answers, mixing: conceptual ("what is MRR and why not just accuracy?"), decision-based
   ("why LanceDB over a managed vector DB?"), tradeoff ("semantic vs fixed chunking - when
   does each win?"), and scenario ("your faithfulness score dropped after a prompt change -
   debug it"). Answers should be the caliber of a strong candidate's spoken answer: 4-8
   sentences, specific, honest about tradeoffs.
3. **`docs/architecture/decisions.md`**: append an ADR entry (context -> decision ->
   consequences) for each significant choice.

Do NOT pause development for teaching sessions. Docs are written as a byproduct, at the end
of the phase, reflecting what was actually built.

## 7. Engineering conventions

- Every module: docstring stating its responsibility; every public function: type hints +
  concise docstring.
- Pydantic models for all API request/response bodies and inter-service data shapes.
- Errors: custom exception hierarchy in `app/core/`; FastAPI exception handlers return
  structured error JSON; never bare `except:`.
- Logging: structured (module, latency, token counts for LLM calls); never log document
  contents at INFO level or secrets at any level.
- Tests: every service module gets unit tests with fakes/mocks (a `FakeLLM` / `FakeEmbedder`
  test double lives in `tests/fakes.py`); network-dependent tests marked
  `@pytest.mark.integration` and skipped when the relevant key/host is absent.
- Commits: small, conventional (`feat:`, `docs:`, `test:`, ...); suggest the message, I run git.
- Windows-friendly: use `pathlib`, no POSIX-only assumptions; commands you tell me to run
  must work in Git Bash on Windows 11.

## 8. Working style with me

- At the START of each phase prompt there is a "Questions to ask me" section. Ask those (and
  anything genuinely blocking), wait for answers, then build. Don't re-ask what this spec
  already fixes.
- Deliver in the numbered order the phase prompt lists. After each deliverable, one-line
  status. If a library API differs from your training data (these libraries move fast -
  check PyPI for current versions of ragas/lancedb/mlflow/google-genai before pinning),
  verify against installed-version docs/help rather than guessing.
- Acceptance criteria at the end of each phase are MY checklist - make sure they pass
  before declaring the phase done, and show me the commands to verify myself.
- If something can't work as specced (version conflict, API change), STOP and present
  options with a recommendation - don't silently substitute.
