# Architecture Decision Records (ADRs)

## ADR-001: Provider Abstraction via Strategy + Factory Pattern

**Status**: Accepted  
**Date**: 2026-09-20  

### Context
The application needs to support multiple Large Language Model (LLM) and Embedding providers, specifically Gemini, Ollama (local), and AWS Bedrock. We need the ability to switch between these providers based on environment (local dev vs. production) or cost constraints without modifying the core business logic or RAG orchestration code. Hardcoding SDK calls directly into the application creates tight coupling and vendor lock-in.

### Decision
We will implement an ABC-based Strategy pattern to define a common interface (`BaseLLM`, `BaseEmbedder`) for all AI providers. We will pair this with a Factory pattern that reads from the application's YAML configuration to dynamically instantiate the correct concrete provider (e.g., `GeminiLLM`, `OllamaLLM`, `BedrockLLM`) at startup.

### Consequences
- **Positive:** We have zero dependency on heavy abstractions like LangChain or LiteLLM. We maintain absolute control over API boundaries, retry logic, and timeouts. Switching vendors requires a one-line configuration change. The architecture is highly interview-demonstrable.
- **Negative:** We incur the overhead of writing and maintaining the boilerplate mapping code for every new provider SDK we wish to support.

---

## ADR-002: LanceDB as Vector Store

**Status**: Accepted  
**Date**: 2026-09-20  

### Context
The Retrieval-Augmented Generation (RAG) system requires a vector database to store document chunks and their corresponding embeddings, and to perform low-latency cosine similarity searches (vector math) to retrieve context for user queries.

### Decision
We will use LanceDB as our primary vector store. It is an embedded, file-based database that runs in-process.

### Consequences
- **Positive:** Operations and deployment are heavily simplified. There is no separate database server or Docker container to manage. It drastically improves local development velocity and supports advanced features like hybrid search out-of-the-box.
- **Negative:** Because it is an embedded database, it will not scale horizontally across multiple application instances for concurrent writes or massive document corpora (millions of documents) without migrating to a managed service or cloud storage backend.

---

## ADR-003: YAML Config + .env Secrets System

**Status**: Accepted  
**Date**: 2026-09-20  

### Context
The application contains both structural tunables (e.g., chunk size, top-K retrieval limit, chosen LLM provider) and highly sensitive secrets (e.g., API keys). We need a configuration system that allows developers to tune the application without exposing secrets or violating 12-Factor App principles.

### Decision
We will use YAML files for all non-sensitive application and provider tunables, which will be committed to version control. We will use `.env` files (and environment variables) strictly for sensitive secrets. We will use Pydantic Settings to load, validate, and merge both sources into a single application config object.

### Consequences
- **Positive:** We strictly adhere to 12-Factor App principles. The risk of accidental secret exposure in git history is heavily mitigated. Application behavior can be heavily modified by non-engineers via simple YAML edits. Pydantic ensures the configuration is type-safe.
- **Negative:** Developers must maintain two separate configuration mechanisms and remember which variables belong where, increasing the cognitive load slightly during setup.
