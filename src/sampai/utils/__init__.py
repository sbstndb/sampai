"""
Utility functions and helpers for Sampai.

This module provides various utilities for working with Sampai simulations,
including progress tracking, visualization, and I/O operations.

The progress module is always available. The viz and io modules require
optional dependencies (matplotlib, h5py) and will raise ImportError if
those dependencies are not installed.

To install optional dependencies:
    pip install sampai[viz]    # For visualization
    pip install sampai[io]     # For I/O utilities
    pip install sampai[all]    # For all optional dependencies
"""

from . import progress

__all__ = ["progress"]

# Lazy imports for optional modules
def __getattr__(name: str):
    """Lazy import for optional modules."""
    if name == "viz":
        from . import viz
        return viz
    if name == "io":
        from . import io
        return io
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
