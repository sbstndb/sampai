# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Utility functions and helpers for Sampai.

This module provides various utilities for working with Sampai simulations,
including progress tracking, visualization, I/O operations, and timing.

The progress and timer modules are always available. The viz and io modules
require optional dependencies (matplotlib, h5py) and will raise ImportError
if those dependencies are not installed.

To install optional dependencies:
    pip install sampai[viz]    # For visualization
    pip install sampai[io]     # For I/O utilities
    pip install sampai[all]    # For all optional dependencies
"""

from . import progress

__all__ = ["progress", "timer"]

# Lazy imports for optional modules
import importlib

def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name == "viz":
        return importlib.import_module(".viz", package=__name__)
    if name == "io":
        return importlib.import_module(".io", package=__name__)
    if name == "timer":
        return importlib.import_module("sampai.utils.timer")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
