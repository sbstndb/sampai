"""
Sampai: Python Interface for Samurai AMR Library

This package provides the Python bindings for Samurai, combining:
- Native C++ implementations (via pybind11 compiled module _sampai.so)
- Python utility modules (progress bars, visualization helpers)
"""

# Import everything from the compiled C++ module
# The _sampai.so file is installed alongside this __init__.py
# by scikit-build-core, so Python finds it automatically.
from ._sampai import *  # noqa: F401, F403

# Import the Python utility submodules
from . import utils  # noqa: E402

# Export utilities at package level for convenience
from .utils import progress  # noqa: E402

__all__ = [
    "progress",
    "utils",
]
