# Vendoring Summary
Generated: 2026-03-27

## Statistics

- **Total Packages:** 47
- **Total Size:** 26.6 MB
- **Python Version:** 3.11
- **Build System:** pip wheel format

## Dependency Tree

```
fastapi (0.135.2)
  ├── starlette (1.0.0)
  ├── pydantic (2.12.5)
  │   ├── pydantic-core
  │   ├── annotated-types (0.7.0)
  │   └── typing-extensions (4.15.0)
  └── typing-extensions (4.15.0)

uvicorn[standard] (0.42.0)
  ├── click (8.3.1)
  ├── h11 (0.16.0)
  ├── httptools (0.7.1)
  ├── python-dotenv (1.0.1)
  ├── pyyaml (6.0.3)
  ├── uvloop (0.21.0)
  └── watchfiles (1.1.1)

sqlalchemy[asyncio] (2.0.48)
  ├── greenlet (3.3.2)
  └── typing-extensions (4.15.0)

psycopg[binary,asyncio] (3.3.3)
  ├── psycopg-binary
  └── typing-extensions (4.15.0)

pgvector (0.4.2)
  └── sqlalchemy (2.0.48)

redis[asyncio] (7.4.0)
  ├── hiredis (3.0.3)
  └── pycparser (2.22)

pydantic-settings (2.13.1)
  ├── pydantic (2.12.5)
  └── python-dotenv (1.0.1)

anthropic (0.86.0)
  ├── httpx (0.28.1)
  │   ├── certifi (2026.2.25)
  │   ├── httpcore (1.0.9)
  │   · idna (3.11)
  │   └── sniffio (1.3.1)
  ├── pydantic (2.12.5)
  └── typing-extensions (4.15.0)

openai (1.20.0)
  ├── httpx (0.28.1)
  ├── pydantic (2.12.5)
  └── tqdm (4.67.3)

alembic (1.18.4)
  ├── sqlalchemy (2.0.48)
  ├── mako (1.3.10)
  │   └── markupsafe (3.0.3)
  └── typing-extensions (4.15.0)

pytest (7.4.4)
  ├── iniconfig (2.3.0)
  ├── packaging (26.0)
  └── pluggy (1.6.0)

pytest-asyncio (0.23.3)
  └── pytest (7.4.4)

httpx (0.28.1)
  ├── certifi (2026.2.25)
  ├── httpcore (1.0.9)
  ├── idna (3.11)
  └── sniffio (1.3.1)
```

## Installation Methods

### 1. Offline (Vendored - Recommended for Production)
```bash
./install-vendor.sh      # Linux/macOS
# OR
install-vendor.bat       # Windows
```

### 2. Online (PyPI - Fresh versions)
```bash
pip install -r requirements.txt
```

### 3. From Python Script
```bash
python vendor_install.py
```

## Package Integrity

All packages are verified with SHA256 hashes in `vendor/MANIFEST.json`:

```bash
# Verify all packages
for pkg in vendor/*.whl; do
    expected=$(grep "$(basename $pkg)" vendor/MANIFEST.json | grep -o '"sha256":"[^"]*"' | cut -d'"' -f4)
    actual=$(sha256sum "$pkg" | cut -d' ' -f1)
    [ "$expected" = "$actual" ] && echo "[OK] $(basename $pkg)" || echo "[FAIL] $(basename $pkg)"
done
```

## Storage Optimization

If storage is critical:

```bash
# Compress vendor directory
tar -czf vendor.tar.gz vendor/
# Size: ~8-10 MB compressed

# Extract when needed
tar -xzf vendor.tar.gz

# Remove source distributions (keep wheels only)
find vendor -name "*.tar.gz" -delete
find vendor -name "*.zip" -delete
# Size after: ~15-20 MB instead of 26.6 MB
```

## CI/CD Integration

### GitHub Actions
```yaml
- uses: actions/cache@v3
  with:
    path: vendor/
    key: vendor-${{ hashFiles('requirements.txt') }}

- run: python -m pip install --no-index --find-links vendor/ vendor/*.whl
```

### GitLab CI
```yaml
cache:
  paths:
    - vendor/
  key: "$CI_COMMIT_REF_SLICE"

install:
  script:
    - pip install --no-index --find-links vendor/ vendor/*.whl
```

## Docker with Vendored Packages

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY vendor/ vendor/
RUN pip install --no-index --find-links vendor/ vendor/*.whl --no-cache-dir
COPY ragcore/ ragcore/
CMD ["uvicorn", "ragcore.main:app", "--host", "0.0.0.0"]
```

```bash
# Build offline (no internet)
docker build -t ragcore:latest .
```

## Updating Vendors

Workflow for adding/updating dependencies:

1. **Edit requirements.txt**
   ```bash
   echo "new-package>=1.0" >> requirements.txt
   ```

2. **Download packages**
   ```bash
   pip download -r requirements.txt -d vendor/
   ```

3. **Generate manifest**
   ```bash
   python generate_vendor_manifest.py
   ```

4. **Verify size**
   ```bash
   du -sh vendor/
   ```

5. **Commit**
   ```bash
   git add vendor/ requirements.txt vendor/MANIFEST.json
   git commit -m "Update: Add new-package to dependencies"
   ```

## Troubleshooting

### "No distributions found"
- Check vendor directory exists: `ls vendor/`
- Run: `python vendor_install.py list`

### "pip subprocess failed"
- Upgrade pip: `python -m pip install --upgrade pip`
- Retry installation

### "Package wheel not compatible"
- Vendor is built for Python 3.11 on Windows
- For other platforms, rebuild: `pip download -r requirements.txt -d vendor/`

## FAQ

**Q: Why include vendor/ in git?**
A: For reproducible builds and offline deployment. Trade-off: adds 26.6 MB to repo.

**Q: Can I use older Python versions?**
A: Redownload packages for target Python version: `pip download -r requirements.txt -d vendor/ --python-version 310`

**Q: Is this secure?**
A: Yes - packages are verified with SHA256 hashes before installation. Supply chain is more secure than fetching from PyPI each time.

**Q: Do I need both vendor/ and requirements.txt?**
A: Yes. `requirements.txt` specifies versions, `vendor/` contains the actual files.

---

For detailed information, see [VENDORING.md](VENDORING.md).
