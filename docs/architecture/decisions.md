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


## ADR-004: Chunking Strategy Design

**Status**: Accepted  
**Date**: 2026-09-27  

### Context
Documents must be split into smaller segments (chunks) before being embedded and stored in the vector database. Feeding entire documents into an LLM exceeds context limits and dilutes semantic meaning during embedding. We need to decide on a chunking strategy that balances implementation complexity with retrieval accuracy for our primarily text-based corpus (Markdown, plain text, and PDFs).

### Decision
We will default to **Recursive Character Chunking** (via LangChain's `RecursiveCharacterTextSplitter` algorithm or custom equivalent) as our primary strategy, rather than purely fixed-size or semantic chunking. We will fallback to Markdown-Aware structure chunking for highly structured internal documentation when explicitly configured.

### Consequences
- **Positive:** Recursive chunking attempts to respect linguistic boundaries (paragraphs, then sentences, then words), preventing the mid-sentence splits common in fixed-size chunking. This preserves context and semantic integrity.
- **Negative:** It is more computationally intensive than fixed-size chunking and requires tuning parameters (chunk size and overlap) based on the specific dataset. It may still occasionally break structure in complex tables or nested lists.

---

## ADR-005: Idempotent Upsert Design

**Status**: Accepted  
**Date**: 2026-09-27  

### Context
The ingestion pipeline will be run repeatedly as the document corpus evolves. If the pipeline is not idempotent, re-running it will result in duplicate chunks in the vector database, skewing retrieval results and wasting storage. We need a robust mechanism to handle document updates and deletions without requiring a full database wipe on every run.

### Decision
We will implement an idempotent ingestion design using a **Hash-Based Stable ID** and a **Delete-Then-Insert** pattern. 
1. We will generate a unique `doc_id` for each file by hashing its absolute path and content (e.g., SHA-256).
2. During ingestion, we will query the vector database for existing `doc_id`s.
3. If a document has changed (the hash differs) or is new, we will delete any existing chunks associated with the file path, and then insert the new chunks. 

### Consequences
- **Positive:** The vector database will perfectly mirror the state of the local file system corpus. The pipeline can be run safely on a cron job or webhook trigger without fear of data duplication. 
- **Negative:** Hashing large files adds overhead to the ingestion process. Deleting and re-inserting documents on minor changes is slightly less efficient than a diff-based patch, but far simpler to implement and debug.
