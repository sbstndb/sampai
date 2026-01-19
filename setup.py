#!/usr/bin/env python3

# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Setup.py for Sampai

This setup.py is provided for backward compatibility only.
The actual build configuration is in pyproject.toml using meson-python.

Usage:
    pip install .           # Build and install from source
    pip install -e .        # Editable install (development)
    python -m build         # Build wheel
"""

from pathlib import Path
from setuptools import setup


def get_version() -> str:
    """Get version from pyproject.toml."""
    import tomli
    pyproject = Path(__file__).parent / "pyproject.toml"
    if pyproject.exists():
        data = tomli.loads(pyproject.read_text())
        return data["project"]["version"]
    return "0.1.0"


# Minimal setup configuration
# The actual build is handled by meson-python via pyproject.toml
setup(
    name="sampai",
    version=get_version(),
    packages=["sampai"],
    package_dir={"": "src"},
    python_requires=">=3.9",
)
