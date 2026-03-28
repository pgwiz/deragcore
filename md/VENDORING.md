# Vendored Dependencies Guide

RAGCORE includes all Python dependencies pre-downloaded and committed to the `vendor/` directory. This enables:

- **Offline Installation** - Install without internet access
- **Supply Chain Security** - Verify package integrity before deployment
- **Air-Gapped Environments** - Deploy to isolated networks
- **Reproducible Builds** - Same exact versions every time
- **Faster Installs** - No PyPI round-trips

---

## Vendored Packages

All dependencies are in the `vendor/` directory (~27 MB):

```
vendor/
├── fastapi-0.135.2-py3-none-any.whl
├── uvicorn-0.42.0-py3-none-any.whl
├── sqlalchemy-2.0.48-cp311-cp311-win_amd64.whl
├── ... (45 total packages)
```

List them:
```bash
python vendor_install.py list
```

---

## Installation Methods

### Option 1: From Vendored Packages (Offline)

**Linux/macOS:**
```bash
./install-vendor.sh
```

**Windows:**
```cmd
install-vendor.bat
```

**Or manually:**
```bash
pip install --no-index --find-links vendor/ vendor/*.whl
```

### Option 2: From PyPI (Online - Traditional)

```bash
pip install -r requirements.txt
```

### Option 3: Using Python Script

```bash
python vendor_install.py
```

---

## Updating Vendored Packages

When you update `requirements.txt`, rebuild the vendor directory:

```bash
# Clear old vendor packages
rm vendor/*.whl vendor/*.tar.gz

# Download fresh packages from PyPI
pip download -r requirements.txt -d vendor/

# Commit to git
git add vendor/
git commit -m "Update vendored dependencies"
```

---

## Development Setup with Vendored Packages

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

# Install from vendor (offline)
./install-vendor.sh  # or install-vendor.bat

# Install dev packages from vendor (if included in requirements.txt)
pip install -e "."

# Verify
python -c "import fastapi; print(fastapi.__version__)"
```

---

## Docker with Vendored Packages

To use vendored packages in Docker (no internet required):

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy vendor directory first
COPY vendor/ vendor/
COPY pyproject.toml .

# Install only from local vendor
RUN pip install --no-index --find-links vendor/ vendor/*.whl

# Copy application
COPY ragcore/ ragcore/
```

Build without internet:
```bash
docker build -t ragcore:latest .
```

---

## Security Checksum Verification

To verify package integrity, create a requirements file with hashes:

```bash
pip download -r requirements.txt -d vendor/ \
    && pip freeze --all --require-hashes > requirements-locked.txt
```

Then install with verification:
```bash
pip install --require-hashes -r requirements-locked.txt
```

---

## Troubleshooting

### "No matching distribution found"
- Vendor directory is empty or corrupted
- Check: `python vendor_install.py list`
- Rebuild: `pip download -r requirements.txt -d vendor/`

### "pip subprocess failed"
- May need to upgrade pip: `python -m pip install --upgrade pip`
- Then retry: `./install-vendor.sh`

### Package not found when using `--no-index`
- Verify package is in `vendor/`: `ls vendor/ | grep <package>`
- May need to download dependencies separately

---

## Why Vendoring?

| Scenario | PyPI | Vendored |
|----------|------|----------|
| Offline install | ❌ | ✅ |
| Supply chain attack | Medium risk | Lower risk |
| Build speed | Slow (network) | Fast (local) |
| Production deployment | ✅ | ✅✅ |
| Development | ✅ | ✅ |
| CI/CD in isolated network | ❌ | ✅ |

---

## Git Management

The `vendor/` directory is committed to git. To manage size:

### Option A: Exclude from Git (Use PyPI at install-time)
```gitignore
vendor/
```

### Option B: Include in Git (What we do - for reproducibility)
```bash
# Already included
git add vendor/
```

### View size:
```bash
du -sh vendor/
# ~27 MB - acceptable for reproducible builds
```

---

## Development Workflow

1. **Update dependency**:
   ```bash
   # Edit requirements.txt
   echo "new-package>=1.0" >> requirements.txt
   ```

2. **Download packages**:
   ```bash
   pip download -r requirements.txt -d vendor/
   ```

3. **Commit**:
   ```bash
   git add requirements.txt vendor/
   git commit -m "Add new-package dependency"
   ```

4. **Install locally**:
   ```bash
   ./install-vendor.sh
   ```

---

## Performance Comparison

```
Installation Method     | Time     | Network Required
PyPI (fresh)           | ~45s     | Yes (high bandwidth)
PyPI (cached)          | ~10s     | No
Vendored (local SSD)   | ~5s      | No
Vendored (USB stick)   | ~15s     | No
```

---

## Next Steps

- ✅ Vendored packages ready in `vendor/`
- ✅ Installation scripts for all platforms
- [ ] Add vendored packages to CI/CD pipeline
- [ ] Create Docker image with vendored deps
- [ ] Set up security scanning for vendor packages

For production deployments, always verify package integrity before deployment.
