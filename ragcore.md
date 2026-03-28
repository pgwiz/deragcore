**RAGCORE**

Modular Multi-Provider RAG API Platform

━━━━━━━━━━━━━━━━━━━

System Architecture & Build Phases

v1.0 - 2026

| **Attribute** | **Value**                                        |
| ------------- | ------------------------------------------------ |
| Architecture  | Modular Microservices - FastAPI + Python         |
| Deployment    | VPS (Primary) + Azure AI Foundry (AI Layer)      |
| Database      | PostgreSQL + pgvector + Redis                    |
| AI Providers  | Anthropic, Azure AI Foundry, OpenAI, Ollama      |
| Real-time     | WebSockets + Outbound Webhooks                   |
| File Support  | PDF, DOCX - Parse → Chunk → Embed → Vector Store |

# **01 Vision & Design Philosophy**

RAGCORE is a self-hosted, modular RAG (Retrieval-Augmented Generation) API platform designed to be your personal AI intelligence gateway. It connects to multiple AI model providers, multiple web research APIs, and handles documents end-to-end - all through a single, unified API surface.

## **Core Design Principles**

- Provider-agnostic - swap Anthropic, Azure, OpenAI, Ollama behind one interface
- Module-first - every capability (research, files, chat) is independently pluggable
- Async-native - file processing, research jobs, AI calls are all non-blocking
- Observable - every request, token count, and job is tracked via Prometheus
- Real-time - WebSocket streaming + outbound webhooks for event-driven integrations

## **What RAGCORE Is Not**

- Not a chatbot UI - it is a backend API platform
- Not locked to one AI provider - the AI layer is fully abstracted
- Not a black box - every module has clear input/output contracts

# **02 System Architecture**

## **High-Level Overview**

The system is divided into five layers. Each layer communicates only through defined interfaces, making the system easy to extend, replace, or scale independently.

**LAYER 1 - Client / Consumer**

Any external client - your own app, webhook consumer, or browser - connects via REST endpoints or WebSocket connections.

**LAYER 2 - RAGCORE API Gateway (FastAPI on VPS)**

| **Module** | **Responsibility**                                               |
| ---------- | ---------------------------------------------------------------- |
| /research  | Fan out queries to SerpAPI, Tavily, DuckDuckGo, GPT Researcher   |
| /files     | Upload, parse, chunk, embed, store, retrieve documents           |
| /chat      | RAG completion - retrieve context, build prompt, stream response |
| /config    | Manage model configs, provider keys, session settings            |
| /webhooks  | Register outbound webhooks, replay events, view logs             |

**LAYER 3 - AI Controller & Provider Registry**

The AI Controller is the brain that routes every AI call. It reads the ModelConfig attached to a session, picks the correct provider, and normalizes the response. Provider parsers translate each platform's unique API into a single UnifiedResponse shape.

| **Provider**     | **Capabilities**                                            |
| ---------------- | ----------------------------------------------------------- |
| Anthropic        | Chat completion, streaming - Claude Sonnet / Haiku / Opus   |
| Azure AI Foundry | Chat, streaming, embeddings - Phi-4, Llama 3.3, Qwen 2.5    |
| OpenAI           | Chat, streaming, embeddings - GPT-4o, GPT-4o-mini           |
| Ollama           | Local models - Llama 3, Mistral, Phi-3 (fallback / offline) |

**LAYER 4 - Data Layer**

| **Store**   | **Purpose**                                                |
| ----------- | ---------------------------------------------------------- |
| PostgreSQL  | Sessions, file metadata, model configs, chat history, jobs |
| pgvector    | Document chunk embeddings - cosine similarity search       |
| Redis       | Job queue, WebSocket pub/sub, webhook retry buffer, cache  |
| Disk / Blob | Raw uploaded files (PDF, DOCX) before and after parsing    |

**LAYER 5 - External APIs**

| **Service**       | **Role**                                              |
| ----------------- | ----------------------------------------------------- |
| Tavily            | Best-in-class search API for RAG - structured results |
| SerpAPI           | Google search scraping - broad web coverage           |
| DuckDuckGo (DDGS) | Free search fallback - no API key required            |
| GPT Researcher    | Deep research agent - multi-source report generation  |
| Firecrawl / BS4   | Full-page web scraping and content extraction         |

# **03 Project Folder Structure**

Every concern has its own home. Modules are fully self-contained - router, logic, and external adapter all live together.

| **Path**              | **Purpose**                               |
| --------------------- | ----------------------------------------- |
| ragcore/              | Project root                              |
| main.py               | FastAPI app factory, mounts all routers   |
| config.py             | Env vars, API keys, feature flags         |
| docker-compose.yml    | PostgreSQL, Redis, pgvector, app          |
| Dockerfile            | Production container definition           |
| core/                 | Shared infrastructure - no business logic |
| ai_controller.py      | Routes requests to correct provider       |
| provider_registry.py  | Lazy-loads & caches provider instances    |
| embeddings.py         | Embedding facade - Azure or local         |
| vector_store.py       | pgvector CRUD operations                  |
| websocket_manager.py  | WS connection pool management             |
| webhook_dispatcher.py | Async outbound webhook sender             |
| core/providers/       | One file per AI provider                  |
| base.py               | Abstract BaseProvider - the contract      |
| anthropic_provider.py | Anthropic SDK adapter                     |
| azure_provider.py     | Azure AI Foundry adapter                  |
| openai_provider.py    | OpenAI SDK adapter                        |
| ollama_provider.py    | Ollama local model adapter                |
| modules/research/     | Research module                           |
| router.py             | POST /research/query, GET /research/jobs  |
| serpapi.py            | SerpAPI adapter                           |
| tavily.py             | Tavily adapter                            |
| duckduckgo.py         | DDGS adapter                              |
| gpt_researcher.py     | GPT Researcher subprocess runner          |
| aggregator.py         | Merges & deduplicates results             |
| modules/files/        | File pipeline module                      |
| router.py             | POST /files/upload, GET /files, DELETE    |
| parser.py             | PDF/DOCX → plain text extraction          |
| chunker.py            | Splits text into overlapping chunks       |
| pipeline.py           | Orchestrates parse→chunk→embed→store      |
| modules/chat/         | Chat & RAG module                         |
| router.py             | POST /chat, WS /chat/stream               |
| retriever.py          | Vector similarity search on pgvector      |
| context_builder.py    | Assembles prompt from retrieved chunks    |
| history.py            | Persists and retrieves chat turns         |
| modules/config/       | Model config management                   |
| router.py             | CRUD for ModelConfigs and API keys        |
| models/               | SQLAlchemy ORM models                     |
| file.py               | File upload record                        |
| chunk.py              | Text chunk + vector embedding             |
| session.py            | Chat session with model_config_id FK      |
| model_config.py       | Provider/model/params configuration       |
| job.py                | Background job tracking                   |
| db/                   | Database layer                            |
| database.py           | SQLAlchemy engine + async session         |
| migrations/           | Alembic migration files                   |
| workers/              | Background processing                     |
| file_worker.py        | ARQ worker - file pipeline jobs           |
| research_worker.py    | ARQ worker - async research jobs          |
| monitoring/           | Observability                             |
| prometheus.py         | Custom metrics - tokens, latency, jobs    |

# **04 Multi-Provider AI System**

Every AI call in RAGCORE flows through one unified interface. Modules never talk directly to any AI provider SDK - they call the AIController, which resolves the correct provider at runtime based on the session's ModelConfig.

## **The Contract - BaseProvider**

Every provider implements exactly four methods. This is the only interface modules ever call.

| **Method**                | **Description**                               |
| ------------------------- | --------------------------------------------- |
| complete(messages, model) | Single-shot completion → UnifiedResponse      |
| stream(messages, model)   | Async generator → yields UnifiedChunk tokens  |
| embed(text, model)        | Text → list\[float\] embedding vector         |
| list_models()             | Returns supported model IDs for this provider |

## **UnifiedResponse Shape**

All providers return this exact structure. Callers never parse provider-specific response formats.

| **Field**     | **Type & Meaning**                                   |
| ------------- | ---------------------------------------------------- |
| text          | str - the full completion text                       |
| model         | str - model ID actually used                         |
| provider      | str - "anthropic" \| "azure" \| "openai" \| "ollama" |
| input_tokens  | int - tokens consumed in prompt                      |
| output_tokens | int - tokens generated in response                   |
| raw           | dict - original provider response (always preserved) |

## **Provider Registry - Config Sync Solution**

The ModelConfig is stored in the database and attached to every chat session via a foreign key. Wherever a session travels - chat, research, file Q&A - it always carries its model_config_id, which the ProviderRegistry resolves at runtime.

| **ModelConfig Field** | **Purpose**                                     |
| --------------------- | ----------------------------------------------- |
| id (UUID)             | Primary key - referenced by sessions and jobs   |
| name                  | Human label e.g. "fast-claude", "research-phi4" |
| provider              | "anthropic" \| "azure" \| "openai" \| "ollama"  |
| model_id              | Exact model string sent to provider API         |
| temperature           | Float 0.0-2.0 - default 0.7                     |
| max_tokens            | Integer - default 2048                          |
| system_prompt         | Optional per-config system instruction          |
| is_default            | Boolean - fallback when session has no config   |
| extra (JSONB)         | Provider-specific extras e.g. api_version       |

## **Adding a New Provider**

Adding a new AI provider requires exactly three steps and zero changes to existing modules:

- Create core/providers/newprovider_provider.py - extend BaseProvider
- Register it in provider_registry.py: \_providers\["newname"\] = NewProvider
- Add the API key to config.py and the .env file

_💡 That's it. All modules immediately gain access to the new provider through the AIController._

# **05 Build Phases**

Each phase builds on the previous and delivers a fully testable milestone. Phases 1-3 form the MVP. Phases 4-5 complete the platform.

| **Phase 1** | **Core Skeleton - API + DB + AI Controller** | Week 1 | **Start Here** |
| ----------- | -------------------------------------------- | ------ | -------------- |

### **What gets built**

- FastAPI application factory with lifespan management
- PostgreSQL + pgvector + Redis via Docker Compose
- SQLAlchemy async models: ModelConfig, Session, Job
- Alembic migration for initial schema
- ProviderRegistry with Anthropic + Azure wired in
- BaseProvider contract + UnifiedResponse + UnifiedChunk
- AIController with complete() and stream() methods
- Config system - .env + Pydantic settings
- GET /health endpoint - DB ping, Redis ping, provider status

### **Deliverable**

A running API at localhost:8000. POST /chat/test returns a streamed response from Anthropic or Azure based on the default ModelConfig.

| **Phase 2** | **Files Module - Full Pipeline** | Week 1-2 | **Core Feature** |
| ----------- | -------------------------------- | -------- | ---------------- |

### **What gets built**

- POST /files/upload - accepts PDF and DOCX, stores raw file
- parser.py - PyMuPDF for PDF, python-docx for DOCX
- chunker.py - recursive text splitter (512 tokens, 50 overlap)
- embeddings.py - Azure text-embedding-3-small or local fallback
- pipeline.py - orchestrates the full parse → chunk → embed → store flow
- Async file worker via ARQ (processes pipeline in background)
- GET /files - list uploaded files with status
- GET /files/{id}/chunks - inspect chunks and embeddings
- DELETE /files/{id} - removes file, chunks, and vectors

### **The Pipeline Flow**

| **Step**   | **What Happens**                                            |
| ---------- | ----------------------------------------------------------- |
| 1\. Upload | File saved to disk, File record created with status=pending |
| 2\. Parse  | PDF/DOCX extracted to clean plain text                      |
| 3\. Chunk  | Text split into overlapping 512-token chunks                |
| 4\. Embed  | Each chunk sent to embedding model → float vector           |
| 5\. Store  | Vectors + text + metadata saved to pgvector table           |
| 6\. Ready  | File status updated to ready - available for chat           |

| **Phase 3** | **Chat Module - RAG Completion + WebSocket Streaming** | Week 2 | **Core Feature** |
| ----------- | ------------------------------------------------------ | ------ | ---------------- |

### **What gets built**

- POST /chat - single-shot RAG completion
- WS /chat/stream - streaming tokens over WebSocket
- retriever.py - cosine similarity search on pgvector (top-K chunks)
- context_builder.py - assembles system prompt + retrieved chunks + history
- history.py - stores and retrieves conversation turns per session
- Session scoping - file_ids filter limits retrieval to specific documents
- Source attribution - response includes which chunks were used

### **The RAG Flow**

| **Step**     | **What Happens**                                                      |
| ------------ | --------------------------------------------------------------------- |
| 1\. Receive  | POST /chat { message, session_id, file_ids }                          |
| 2\. Retrieve | Embed query → cosine search → top 5 chunks from pgvector              |
| 3\. Build    | context_builder assembles: system_prompt + chunks + history + message |
| 4\. Complete | AIController streams response from provider (via ModelConfig)         |
| 5\. Stream   | Tokens sent over WebSocket as they arrive                             |
| 6\. Persist  | Full response + sources saved to chat history                         |

| **Phase 4** | **Research Module - Web Intelligence Pipeline** | Week 2-3 | **Power Feature** |
| ----------- | ----------------------------------------------- | -------- | ----------------- |

### **What gets built**

- POST /research/query - synchronous quick research
- POST /research/query/async - background job + webhook on completion
- GET /research/jobs/{id} - poll job status and results
- Tavily adapter - structured search results optimised for RAG
- SerpAPI adapter - Google search with rich snippets
- DuckDuckGo adapter - free fallback via DDGS library
- GPT Researcher adapter - deep multi-source research reports
- aggregator.py - merges results, deduplicates URLs, scores relevance
- Optional: save research results as files → feed into RAG pipeline

### **Research Source Priority**

| **Priority** | **Provider → Use Case**                               |
| ------------ | ----------------------------------------------------- |
| 1 (Best)     | Tavily - RAG-optimised, structured, fast              |
| 2            | SerpAPI - broad Google coverage, rich snippets        |
| 3            | GPT Researcher - deep reports, multi-source synthesis |
| 4 (Fallback) | DuckDuckGo (DDGS) - free, no API key required         |

| **Phase 5** | **Webhooks, Monitoring & Production Hardening** | Week 3+ | **Production** |
| ----------- | ----------------------------------------------- | ------- | -------------- |

### **What gets built**

- Webhook registry - POST /webhooks/register with URL + events filter
- webhook_dispatcher.py - async send with retry (3x exponential backoff)
- Prometheus metrics - tokens used, latency per provider, job counts, errors
- API key auth middleware - bearer token on all endpoints
- Rate limiting per API key - via Redis token bucket
- Structured request logging - JSON logs with trace IDs
- Docker Compose production config - health checks, restart policies
- Grafana dashboard - pre-built for RAGCORE metrics

### **Webhook Event Types**

| **Event**         | **Fired When**                         |
| ----------------- | -------------------------------------- |
| file.ready        | File pipeline completes successfully   |
| file.error        | File pipeline fails at any step        |
| research.complete | Async research job finishes            |
| chat.complete     | Chat session turn finishes             |
| job.failed        | Any background job fails after retries |

# **06 Complete Tech Stack**

| **Layer**              | **Technology**               | **Notes**                                |
| ---------------------- | ---------------------------- | ---------------------------------------- |
| **API Framework**      | FastAPI + Uvicorn            | Async-native, WebSocket support built-in |
| **Language**           | Python 3.11+                 | Full async/await throughout              |
| **Database**           | PostgreSQL 16 + pgvector     | Relational + vector similarity in one DB |
| **Cache / Queue**      | Redis + ARQ                  | Job queue, pub/sub, rate limiting        |
| **File Parsing**       | PyMuPDF + python-docx        | PDF and DOCX to clean text               |
| **Chunking**           | LangChain TextSplitter       | Recursive with token awareness           |
| **Embeddings**         | Azure text-embedding-3-small | Or sentence-transformers fallback        |
| **AI - Anthropic**     | claude-sonnet-4 / haiku      | Via official anthropic Python SDK        |
| **AI - Azure**         | Phi-4, Llama 3.3, Qwen 2.5   | Via azure-ai-inference SDK               |
| **AI - OpenAI**        | GPT-4o, GPT-4o-mini          | Via openai Python SDK                    |
| **AI - Local**         | Ollama (Llama3, Mistral)     | Fallback / offline / low-cost            |
| **Search - Primary**   | Tavily                       | RAG-optimised structured results         |
| **Search - Secondary** | SerpAPI                      | Google coverage                          |
| **Search - Free**      | DuckDuckGo DDGS              | No API key required                      |
| **Deep Research**      | GPT Researcher               | Multi-source report generation           |
| **Monitoring**         | Prometheus + Grafana         | Metrics, dashboards, alerting            |
| **Containerisation**   | Docker + Docker Compose      | One command to run everything            |
| **Migrations**         | Alembic                      | Version-controlled schema changes        |
| **ORM**                | SQLAlchemy 2.0 async         | Async session + typed models             |

# **07 Real-Time Architecture**

## **WebSocket Streaming**

Clients connect to ws://api/chat/stream?session_id=xxx and receive tokens as they are generated by the AI provider. The WebSocket manager maintains a connection pool keyed by session ID. Multiple consumers can subscribe to the same session stream.

| **Event Type** | **Payload**                          |
| -------------- | ------------------------------------ |
| token          | { type: "token", delta: "Hello" }    |
| sources        | { type: "sources", chunks: \[...\] } |
| done           | { type: "done", total_tokens: 412 }  |
| error          | { type: "error", message: "..." }    |

## **Outbound Webhooks**

When async jobs complete (file processing, research), RAGCORE fires a POST request to all registered webhook URLs that subscribed to that event type. Redis buffers failed webhooks for retry.

_💡 Webhooks use exponential backoff: retry after 5s, 25s, 125s. After 3 failures the job is logged as dead-letter and an admin alert fires._

# **08 UI / Admin Console (Optional Phase 6)**

If a management UI is built, it follows the Ethereal Glassmorphism design philosophy. The backend API is fully functional without it - this is an optional overlay.

| **UI Screen**   | **What It Shows**                                  |
| --------------- | -------------------------------------------------- |
| Dashboard       | Live token usage, active sessions, job queue depth |
| Files           | Upload interface, pipeline status, chunk inspector |
| Chat Playground | Test RAG queries against uploaded files            |
| Research        | Run research queries, view source aggregation      |
| Model Configs   | Create/edit ModelConfigs, set defaults             |
| Webhooks        | Register endpoints, view delivery logs, replay     |
| Monitoring      | Embedded Grafana panels - latency, errors, costs   |

## **CSS Design Tokens**

:root {

\--glass-bg: rgba(255, 255, 255, 0.12);

\--glass-border: rgba(255, 255, 255, 0.25);

\--glass-blur: blur(18px);

\--glass-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);

\--primary: #6C63FF;

\--accent: #00D4FF;

\--font: "Poppins", sans-serif;

\--radius: 24px;

}

# **09 Agent Configuration**

See the companion AGENT.md document for the complete AI agent persona definition, system prompt templates, conversation guidelines, tool invocation rules, and tone specifications for the RAGCORE AI controller agent.

| **Document**               | **Purpose**                                 |
| -------------------------- | ------------------------------------------- |
| RAGCORE_PLAN.md (this doc) | Architecture, phases, tech stack            |
| AGENT.md                   | Agent persona, system prompts, tool configs |