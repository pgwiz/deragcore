"""Generate manifest of vendored packages with hashes."""

import hashlib
import json
from pathlib import Path
from datetime import datetime


def generate_vendor_manifest():
    """Generate a manifest file for all vendored packages."""
    vendor_dir = Path("vendor")
    manifest = {
        "created": datetime.utcnow().isoformat(),
        "total_packages": 0,
        "total_size_bytes": 0,
        "python_version": "3.11",
        "packages": [],
    }

    packages = sorted(vendor_dir.glob("*.whl")) + sorted(vendor_dir.glob("*.tar.gz"))

    for pkg_path in packages:
        # Calculate size and hash
        size = pkg_path.stat().st_size
        sha256_hash = hashlib.sha256(pkg_path.read_bytes()).hexdigest()

        manifest["packages"].append({
            "filename": pkg_path.name,
            "size_bytes": size,
            "sha256": sha256_hash,
            "type": "wheel" if pkg_path.suffix == ".whl" else "source",
        })

        manifest["total_size_bytes"] += size
        manifest["total_packages"] += 1

    # Write manifest
    manifest_path = vendor_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"[OK] Generated vendor manifest: {manifest_path}")
    print(f"  - {manifest['total_packages']} packages")
    print(f"  - {manifest['total_size_bytes'] / 1024 / 1024:.1f} MB")

    # Also write requirements file format for verification
    requirements_path = Path("requirements-vendor.txt")
    lines = []
    for pkg in manifest["packages"]:
        # Extract package info from filename
        filename = pkg["filename"]
        lines.append(f"# {filename} sha256:{pkg['sha256']}")

    requirements_path.write_text("\n".join(lines))
    print(f"[OK] Generated requirements-vendor.txt with SHA256 hashes")


if __name__ == "__main__":
    generate_vendor_manifest()
