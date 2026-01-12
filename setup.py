#!/usr/bin/env python3
"""
Setup.py for Sampai

This setup.py wraps scikit-build-core and automatically installs
the samurai C++ dependency from conda-forge before building.

Usage:
    pip install .           # Build and install from source
    pip install -e .        # Editable install (development)
    python -m build         # Build wheel
"""

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup

# Import the samurai installer
from install_samurai import main as install_samurai_main


def install_samurai_if_needed() -> None:
    """Install samurai dependency before building."""
    print("=" * 70)
    print("Sampai Setup - Checking samurai dependency")
    print("=" * 70)

    # Run the samurai installer
    # It will skip if samurai is already found
    result = install_samurai_main()

    if result != 0:
        print()
        print("WARNING: Failed to automatically install samurai.")
        print("The build may fail if samurai is not available.")
        print()
        print("To manually install samurai, run:")
        print("  conda install -c conda-forge samurai=0.27.1")
        print()
        response = input("Continue with build anyway? [y/N] ")
        if response.lower() != 'y':
            sys.exit(1)


def get_version() -> str:
    """Get version from version.txt."""
    version_file = Path(__file__).parent / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.0"


# Install samurai dependency before build
if "install" in sys.argv or "develop" in sys.argv or "egg_info" in sys.argv:
    install_samurai_if_needed()


# Setup configuration using scikit-build-core
setup(
    name="sampai",
    version=get_version(),
    packages=["sampai"],
    package_dir={"": "src"},
    python_requires=">=3.9",
)
