# Quick Vendoring Reference

## One-Line Install

```bash
# Windows
install-vendor.bat

# Linux/macOS
./install-vendor.sh

# Manual
pip install --no-index --find-links vendor/ vendor/*.whl
```

## Check Vendored Packages

```bash
# List all vendored packages
python vendor_install.py list

# View manifest
cat vendor/MANIFEST.json | python -m json.tool

# Count packages
ls vendor/*.whl | wc -l

# Total size
du -sh vendor/
```

## Update Workflow

```bash
# 1. Edit requirements.txt
echo "new-package>=1.0" >> requirements.txt

# 2. Download
pip download -r requirements.txt -d vendor/

# 3. Generate manifest
python generate_vendor_manifest.py

# 4. Commit
git add vendor/ requirements.txt vendor/MANIFEST.json
git commit -m "Update vendored dependencies"
```

## Key Files

| File | Purpose |
|------|---------|
| `vendor/` | All 47 Python packages (26.6 MB) |
| `vendor/MANIFEST.json` | Package manifest with SHA256 hashes |
| `requirements.txt` | Python dependency specifications |
| `install-vendor.sh` | Bash installer (Linux/macOS) |
| `install-vendor.bat` | Batch installer (Windows) |
| `vendor_install.py` | Python installer (universal) |
| `generate_vendor_manifest.py` | Generate manifest with hashes |
| `pip.conf` | pip configuration for vendor mode |
| `VENDORING.md` | Detailed vendoring guide |
| `VENDOR_SUMMARY.md` | Dependencies and statistics |

## Deployment Options

### Developer Machine
```bash
./install-vendor.sh  # Fast, offline
```

### Docker/Container
```dockerfile
COPY vendor/ vendor/
RUN pip install --no-index --find-links vendor/ vendor/*.whl
```

### CI/CD Pipeline
```bash
pip install --no-index --find-links vendor/ vendor/*.whl
```

### Air-Gapped Network
```bash
# Copy vendor/ to USB/network drive
# Then on target: ./install-vendor.sh
```

## Status

✅ 47 packages vendorized
✅ 26.6 MB total
✅ SHA256 hashes generated
✅ All platforms supported
✅ Docker ready
✅ CI/CD ready

## Commands Cheat Sheet

```bash
# Install
./install-vendor.sh

# List packages
python vendor_install.py list

# Update all
pip download -r requirements.txt -d vendor/ && python generate_vendor_manifest.py

# Verify
python -c "import fastapi; print(f'FastAPI {fastapi.__version__}')"

# Docker build (offline)
docker build -t ragcore:latest .

# Docker run
docker run -p 8000:8000 ragcore:latest
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No matching distribution" | Run `pip download -r requirements.txt -d vendor/` |
| "pip subprocess failed" | Upgrade pip: `python -m pip install --upgrade pip` |
| Slow installation | Use vendored packages: `./install-vendor.sh` |
| Network timeout | Use vendored packages (no network needed) |
| Dependency missing | Check `vendor/MANIFEST.json` |

---

See [GETTING_STARTED.md](GETTING_STARTED.md) for full setup instructions.
