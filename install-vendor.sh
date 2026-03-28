#!/bin/bash
# Install dependencies from vendored packages (offline install)
# Usage: ./install-vendor.sh

set -e

VENDOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vendor"

if [ ! -d "$VENDOR_DIR" ]; then
    echo "ERROR: vendor/ directory not found"
    exit 1
fi

echo "Installing from vendor directory: $VENDOR_DIR"
echo "Found $(ls -1 "$VENDOR_DIR" | wc -l) packages"

pip install \
    --no-index \
    --find-links "$VENDOR_DIR" \
    "$VENDOR_DIR"/*.whl \
    "$VENDOR_DIR"/*.tar.gz \
    2>/dev/null || true

echo "✓ Installation complete"
