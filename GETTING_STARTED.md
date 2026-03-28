# RAGCORE - Getting Started

## Quick Start (5 minutes)

### Option 1: Using Docker Compose (Recommended)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Start all services
docker-compose up -d

# 3. Verify services
docker-compose ps

# 4. Check logs
docker-compose logs -f api

# 5. Test health endpoint
curl http://localhost:8000/health
```

### Option 2: Local Development (with Vendored Packages - Offline)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install from vendored packages (no internet needed)
# Linux/macOS:
./install-vendor.sh

# Windows:
install-vendor.bat

# 3. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start PostgreSQL and Redis
# (install locally or use docker-compose)
docker-compose up -d postgres redis

# 5. Run migrations
alembic upgrade head

# 6. Start dev server
uvicorn ragcore.main:app --reload --port 8000
```

### Option 2b: Local Development (Traditional - from PyPI)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies from PyPI
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your API keys

# ... rest same as above
```

---

## API Endpoints (Phase 1)

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2026-03-27T10:30:00",
  "services": {
    "database": "connected",
    "redis": "ready",
    "api": "online"
  },
  "providers": {
    "anthropic": true,
    "azure": true
  }
}
```

### Test Completion (Single Response)
```bash
curl -X POST http://localhost:8000/test/complete \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing in one sentence",
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

**Response:**
```json
{
  "text": "Quantum computing harnesses ...",
  "model": "claude-3-5-sonnet-20241022",
  "provider": "anthropic",
  "input_tokens": 15,
  "output_tokens": 42
}
```

### Test Streaming (WebSocket)
```bash
# Using websocat (install with: cargo install websocat)
websocat ws://localhost:8000/test/stream

# Then send:
{"message": "Hello", "temperature": 0.7}

# Receive streaming tokens:
{"type": "token", "delta": "Hello", "provider": "anthropic"}
{"type": "token", "delta": " there", "provider": "anthropic"}
{"type": "done"}
```

Or use a script:
```python
import asyncio
import websockets
import json

async def test_stream():
    async with websockets.connect("ws://localhost:8000/test/stream") as ws:
        await ws.send(json.dumps({"message": "Hi there"}))
        async for msg in ws:
            print(json.loads(msg))

asyncio.run(test_stream())
```

---

## Environment Variables

**Required:**
- `ANTHROPIC_API_KEY` - Get from https://console.anthropic.com/
- `AZURE_API_KEY` - Get from Azure Portal
- `AZURE_ENDPOINT` - Your Azure AI Foundry endpoint
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection URL

**Optional:**
- `OPENAI_API_KEY` - For OpenAI models
- `LOG_LEVEL` - DEBUG, INFO, WARNING, ERROR
- `PORT` - Server port (default 8000)

---

## Project Structure

```
ragcore/
├── main.py              # FastAPI app and routes
├── config.py            # Environment + settings
├── models.py            # SQLAlchemy ORM
├── core/
│   ├── schemas.py       # UnifiedResponse, UnifiedChunk
│   ├── ai_controller.py # Request routing
│   ├── provider_registry.py # Provider caching
│   └── providers/       # AI provider adapters
│       ├── base.py
│       ├── anthropic_provider.py
│       ├── azure_provider.py
│       ├── openai_provider.py
│       └── ollama_provider.py
├── db/
│   ├── database.py      # SQLAlchemy setup
│   └── migrations/      # Alembic (future)
└── tests/               # Test suite

docker-compose.yml      # Multi-container setup
pyproject.toml          # Dependencies
.env.example            # Environment template
```

---

## Troubleshooting

### Services won't start
```bash
# Check Docker logs
docker-compose logs postgres
docker-compose logs redis
docker-compose logs api

# Rebuild containers
docker-compose down -v
docker-compose up --build
```

### API returns "No providers available"
- Check `.env` - ensure at least one API key is set
- Check `/health` - see which providers are available
- Check logs: `docker-compose logs api`

### Database connection errors
```bash
# Verify postgres is running
docker-compose ps

# Connect directly
psql -h localhost -U ragcore -d ragcore

# Check logs
docker-compose logs postgres
```

### Port already in use
- Change port in `docker-compose.yml` or `.env`
- Kill existing process: `lsof -i :8000` then `kill -9 <PID>`

---

## Next Steps

1. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

2. **Explore API docs:**
   - Visit http://localhost:8000/docs (Swagger UI)
   - Visit http://localhost:8000/redoc (ReDoc)

3. **Proceed to Phase 2:** Files module (upload → parse → embed)

---

## Vendored Packages

All Python dependencies are pre-downloaded and included in the `vendor/` directory (26.6 MB, 47 packages). This enables:

- ✅ **Offline Installation** - No internet required
- ✅ **Supply Chain Security** - Verify package integrity
- ✅ **Air-Gapped Deployment** - Deploy to isolated networks
- ✅ **Reproducible Builds** - Same exact versions always

### Install from Vendor (Offline)

```bash
# Windows
install-vendor.bat

# Linux/macOS
./install-vendor.sh

# Or manually
pip install --no-index --find-links vendor/ vendor/*.whl
```

### Verify Vendored Packages

```bash
python vendor_install.py list
```

### Update Vendored Packages

```bash
# Edit requirements.txt with new versions
# Then download fresh packages
pip download -r requirements.txt -d vendor/

# Generate manifest with SHA256 hashes
python generate_vendor_manifest.py

# Commit to git
git add vendor/ requirements.txt
git commit -m "Update vendored dependencies"
```

For more details, see [VENDORING.md](VENDORING.md).

---

- **Architecture docs:** See `ragcore.md`
- **Enhanced plan:** See `PLAN_ENHANCED.md`
- **Issues:** Check project logs and `TROUBLESHOOTING.md`
