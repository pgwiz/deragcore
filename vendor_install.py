"""
Vendored dependencies installer for RAGCORE.

This module provides utilities to install dependencies from the vendored
directory instead of PyPI, enabling offline installs and supply chain security.
"""

import os
import sys
import subprocess
from pathlib import Path


def get_vendor_dir() -> Path:
    """Get the vendor directory path."""
    return Path(__file__).parent / "vendor"


def install_from_vendor() -> int:
    """
    Install all dependencies from the vendor directory.

    Returns:
        Exit code (0 for success)
    """
    vendor_dir = get_vendor_dir()

    if not vendor_dir.exists():
        print(f"ERROR: Vendor directory not found at {vendor_dir}")
        return 1

    print(f"Installing from vendor directory: {vendor_dir}")

    # Find all .whl and .tar.gz files
    packages = list(vendor_dir.glob("*.whl")) + list(vendor_dir.glob("*.tar.gz"))

    if not packages:
        print("ERROR: No packages found in vendor directory")
        return 1

    print(f"Found {len(packages)} packages")

    # Install using pip with --no-index to use only local packages
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(vendor_dir),
        *[str(p) for p in packages],
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    return result.returncode


def list_vendor_packages() -> None:
    """List all vendored packages."""
    vendor_dir = get_vendor_dir()
    packages = sorted(vendor_dir.glob("*.whl")) + sorted(vendor_dir.glob("*.tar.gz"))

    print(f"Vendored packages ({len(packages)}):\n")
    for pkg in packages:
        size_mb = pkg.stat().st_size / (1024 * 1024)
        print(f"  {pkg.name:60} {size_mb:6.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "list":
        list_vendor_packages()
    else:
        exit_code = install_from_vendor()
        sys.exit(exit_code)
