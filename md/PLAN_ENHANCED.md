# RAGCORE - Enhanced Execution Plan

**Status:** Kickoff Ready
**Last Updated:** 2026-03-27
**Phase Target:** Phase 1 - Core Skeleton (Week 1)

---

## Quick Reference

| Item | Value |
|------|-------|
| **Project Root** | `ragcore/` |
| **Dev Entry Point** | `ragcore/main.py` |
| **Test Command** | `pytest tests/ -v` |
| **Dev Server** | `uvicorn ragcore.main:app --reload --port 8000` |
| **Docker Startup** | `docker-compose up -d` |
| **Python Version** | 3.11+ |

---

## Phase 1 - Detailed Breakdown

### **Week 1 Objectives**

Build a running FastAPI server that:
- ✅ Connects to PostgreSQL (with pgvector extension)
- ✅ Connects to Redis
- ✅ Implements AI Provider abstraction
- ✅ Routes requests to Anthropic & Azure (via ModelConfig)
- ✅ Streams responses over WebSocket
- ✅ Serves a `/health` endpoint

**Deliverable:** `POST /test/complete` and `WS /test/stream` return streaming chat responses.

---

## Task Breakdown & Dependencies

### **Sprint 1.1 - Project Foundation (Day 1-2)**

**Duration:** ~4 hours
**Dependencies:** None (Start here)

#### Task 1.1.1: Initialize Python Project
- [ ] Create `ragcore/` directory structure
- [ ] Initialize `pyproject.toml` with core dependencies
- [ ] Create `.env.example` with all required API keys
- [ ] Create `requirements.txt` for pip install fallback
- [ ] Initialize git repo + `.gitignore`

**Deliverable:**
```bash
ragcore/
├── main.py          # Blank, will add routes
├── config.py        # Env loading
├── pyproject.toml   # Dependencies
├── .env.example     # Template
└── docker-compose.yml
```

#### Task 1.1.2: Docker Compose Setup
- [ ] Create `docker-compose.yml` with:
  - PostgreSQL 16 + pgvector extension
  - Redis 7+
  - FastAPI dev server
- [ ] Volume mappings for code reloading
- [ ] Health check endpoints for services

**Validation:**
```bash
docker-compose up -d
docker-compose ps  # All running
docker-compose logs postgres  # No errors
```

#### Task 1.1.3: Database Models & Migrations
- [ ] Create `ragcore/models/` directory
- [ ] Define SQLAlchemy models:
  - `ModelConfig` (provider, model_id, temperature, max_tokens, etc.)
  - `Session` (foreign key to ModelConfig)
  - `Job` (for background tasks)
- [ ] Create Alembic migration directory
- [ ] First migration: init schema with all tables

**Validation:**
```bash
alembic upgrade head  # Schema created
psql -c "SELECT * FROM model_config"  # Table exists
```

---

### **Sprint 1.2 - AI Provider Abstraction (Day 2-3)**

**Duration:** ~6 hours
**Dependencies:** Sprint 1.1 complete

#### Task 1.2.1: BaseProvider Contract
- [ ] Create `ragcore/core/providers/base.py`
- [ ] Define abstract methods:
  - `complete(messages, model) → UnifiedResponse`
  - `stream(messages, model) → AsyncGenerator[UnifiedChunk]`
  - `embed(text, model) → list[float]`
  - `list_models() → list[str]`
- [ ] Define `UnifiedResponse` dataclass
- [ ] Define `UnifiedChunk` dataclass

**Code Reference:**
```python
@dataclass
class UnifiedResponse:
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    raw: dict  # Original provider response

@dataclass
class UnifiedChunk:
    delta: str
    input_tokens: int
    output_tokens: int
```

#### Task 1.2.2: Anthropic Provider
- [ ] Create `ragcore/core/providers/anthropic_provider.py`
- [ ] Implement `complete()` using anthropic SDK
- [ ] Implement `stream()` async generator
- [ ] Implement `embed()` - delegate to Azure or raise NotImplemented
- [ ] Handle API errors gracefully

#### Task 1.2.3: Azure Provider
- [ ] Create `ragcore/core/providers/azure_provider.py`
- [ ] Implement `complete()` using azure-ai-inference
- [ ] Implement `stream()` async generator
- [ ] Implement `embed()` using text-embedding-3-small
- [ ] Use env vars for key, endpoint, api_version

#### Task 1.2.4: Provider Registry
- [ ] Create `ragcore/core/provider_registry.py`
- [ ] Singleton registry with lazy loading
- [ ] `get_provider(name) → BaseProvider`
- [ ] Cache provider instances to reuse connections

**Validation:**
```python
from ragcore.core.provider_registry import registry
provider = registry.get_provider("anthropic")
assert provider is not None
```

---

### **Sprint 1.3 - AI Controller & Config (Day 3-4)**

**Duration:** ~4 hours
**Dependencies:** Sprint 1.2 complete

#### Task 1.3.1: AI Controller
- [ ] Create `ragcore/core/ai_controller.py`
- [ ] `complete(session, messages) → UnifiedResponse`
  - Fetch ModelConfig from DB via session.model_config_id
  - Get provider from registry
  - Call provider.complete()
- [ ] `stream(session, messages) → AsyncGenerator[UnifiedChunk]`
  - Same logic, yield tokens as they arrive

#### Task 1.3.2: Config System
- [ ] Create `ragcore/config.py`
- [ ] Use Pydantic BaseSettings
- [ ] Load from `.env`:
  - `ANTHROPIC_API_KEY`
  - `AZURE_API_KEY`
  - `AZURE_ENDPOINT`
  - `OPENAI_API_KEY`
  - `DATABASE_URL`
  - `REDIS_URL`
  - `LOG_LEVEL`

**Validation:**
```bash
source .env && python -c "from ragcore.config import settings; print(settings.anthropic_api_key)"
```

---

### **Sprint 1.4 - FastAPI App & Routes (Day 4-5)**

**Duration:** ~5 hours
**Dependencies:** Sprint 1.3 complete

#### Task 1.4.1: App Factory
- [ ] Create `ragcore/main.py`
- [ ] FastAPI app factory with lifespan context manager
- [ ] On startup:
  - Initialize DB connection pool
  - Initialize Redis connection
  - Test provider connectivity
  - Log all enabled providers
- [ ] On shutdown:
  - Gracefully close all connections

#### Task 1.4.2: Health Endpoint
- [ ] `GET /health → { status: "ok", timestamp, services: {} }`
- [ ] Ping PostgreSQL
- [ ] Ping Redis
- [ ] List available providers

**Response Example:**
```json
{
  "status": "ok",
  "timestamp": "2026-03-27T10:30:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "providers": ["anthropic", "azure"]
  }
}
```

#### Task 1.4.3: Test Endpoints (for manual testing)
- [ ] `POST /test/complete` - single completion
- [ ] `WS /test/stream` - streaming over WebSocket
- [ ] Both use default ModelConfig from DB
- [ ] Accept `message: str` in request body

**Request:**
```json
POST /test/complete
{
  "message": "Hello, explain quantum computing briefly"
}
```

**Response:**
```json
{
  "text": "Quantum computing...",
  "model": "claude-sonnet-4",
  "provider": "anthropic",
  "input_tokens": 15,
  "output_tokens": 234,
  "raw": { ... }
}
```

---

### **Sprint 1.5 - Database & Migrations (Day 5)**

**Duration:** ~3 hours
**Dependencies:** Sprint 1.4 complete

#### Task 1.5.1: Database Layer
- [ ] Create `ragcore/db/database.py`
- [ ] SQLAlchemy async engine
- [ ] Async session factory
- [ ] Connection pool settings

#### Task 1.5.2: Seed Default ModelConfigs
- [ ] Create migration or seed script
- [ ] Insert default ModelConfigs:
  - `anthropic-default` → claude-sonnet-4
  - `azure-default` → phi-4
  - `anthropic-fast` → claude-haiku-4
- [ ] Create default Session record

**Validation:**
```bash
psql -c "SELECT * FROM model_config" | head -5
```

---

### **Sprint 1.6 - Testing & Validation (Day 5+)**

**Duration:** ~4 hours
**Dependencies:** Sprint 1.5 complete

#### Task 1.6.1: Unit Tests
- [ ] `tests/test_provider_registry.py` - provider caching
- [ ] `tests/test_ai_controller.py` - routing logic
- [ ] `tests/test_config.py` - settings loading

#### Task 1.6.2: Integration Tests
- [ ] `tests/test_app_health.py` - /health endpoint
- [ ] `tests/test_endpoints_basic.py` - /test/complete endpoint
- [ ] Mocked providers (mock completion responses)

#### Task 1.6.3: Manual E2E Validation
- [ ] Start dev server: `uvicorn ragcore.main:app --reload`
- [ ] Curl `/health`
- [ ] POST `/test/complete` with Anthropic key
- [ ] Connect to WS `/test/stream`, receive tokens
- [ ] Verify tokens recorded in logs

**Validation Checklist:**
- [ ] Server starts without errors
- [ ] `/health` returns 200 OK
- [ ] `/test/complete` streams response
- [ ] WebSocket connection established
- [ ] Response timestamps realistic
- [ ] Error handling works (bad API key → 401)

---

## Critical Path (Must Happen First)

```
Sprint 1.1 (Foundation)
    ↓
Sprint 1.2 (Providers)
    ↓
Sprint 1.3 (Controller)
    ↓
Sprint 1.4 (App)
    ↓
Sprint 1.5 (DB)
    ↓
Sprint 1.6 (Testing)
```

**No parallelization** - each sprint depends on the previous.

---

## Key Dependencies (Python)

```toml
[build-system]
requires = ["setuptools", "wheel"]

[project]
dependencies = [
    "fastapi>=0.109",
    "uvicorn[standard]>=0.27",
    "sqlalchemy[asyncio]>=2.0",
    "psycopg[binary,asyncio]>=3.1",
    "pgvector>=0.3",
    "redis[asyncio]>=5.0",
    "pydantic-settings>=2.0",
    "anthropic>=0.25",
    "azure-ai-inference>=1.0",
    "openai>=1.20",
    "alembic>=1.13",
    "pytest>=7.4",
    "pytest-asyncio>=0.23",
]
```

---

## Environment Variables (.env)

```bash
# API Keys
ANTHROPIC_API_KEY=sk-...
AZURE_API_KEY=...
AZURE_ENDPOINT=https://....
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/ragcore

# Cache
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=DEBUG

# Server
WORKERS=1
PORT=8000
```

---

## Success Metrics (Phase 1 Complete)

- [x] API server starts without errors
- [x] All three providers (Anthropic, Azure, OpenAI ready) connect
- [x] `/health` returns provider status
- [x] `POST /test/complete` produces streaming response
- [x] `WS /test/stream` streams tokens in real-time
- [x] Database schema migrated and seeded
- [x] Unit tests pass (>80% coverage)
- [x] No API errors after 24 hours of running

---

## Next Phases (Preview)

| Phase | Focus | Duration |
|-------|-------|----------|
| **2** | Files Pipeline (upload → parse → chunk → embed) | Week 1-2 |
| **3** | Chat Module (RAG retrieval + conversation history) | Week 2 |
| **4** | Research Module (web search + aggregation) | Week 2-3 |
| **5** | Production Hardening (auth, monitoring, webhooks) | Week 3+ |

---

## Commands Reference

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Database
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head

# Development
uvicorn ragcore.main:app --reload --port 8000

# Docker
docker-compose up -d
docker-compose logs -f

# Testing
pytest tests/ -v --cov=ragcore

# Linting
black ragcore/
mypy ragcore/ --ignore-missing-imports
```

---

## Architecture Diagram (Phase 1)

```
┌─────────────────────────────────────────────────────────────┐
│ FastAPI App (main.py)                                       │
├─────────────────────────────────────────────────────────────┤
│ Routes: /health, /test/complete, /test/stream (WS)          │
└────────────┬────────────────────────────────────────────────┘
             │
        ┌────▼────────────────────────────────────────┐
        │ AIController (ai_controller.py)             │
        │ - Routes to correct provider                │
        │ - Normalizes responses                      │
        └────┬───────────────────────────┬─────┬──────┘
             │                           │     │
      ┌──────▼──────┐ ┌────────▼─┐  ┌──▼──────────┐
      │ Anthropic   │ │  Azure   │  │  OpenAI    │
      │ Provider    │ │ Provider │  │  Provider  │
      └──────┬──────┘ └────────┬─┘  └──┬─────────┘
             │                 │       │
      ┌──────▼──┐ ┌───────────▼──┐ ┌──▼──────────┐
      │anthropic│ │azure-ai-    │ │  openai    │
      │SDK      │ │inference    │ │  SDK       │
      └─────────┘ └──────────────┘ └────────────┘

        ┌────────────────────────────────────────┐
        │ Data Layer (config.py, models/)        │
        ├────────────────────────────────────────┤
        │ - PostgreSQL (ModelConfig, Session)    │
        │ - Redis (WebSocket pub/sub, cache)     │
        └────────────────────────────────────────┘
```

---

## Progress Tracker

Run this periodically to check status:

```bash
# Quick health check
curl http://localhost:8000/health

# Check migrations
alembic current

# Run test suite
pytest tests/ -v --tb=short

# View logs
docker-compose logs -f api
```

---

**Next Step:** Start Sprint 1.1 by initializing the project structure and dependencies.
