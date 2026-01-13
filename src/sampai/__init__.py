"""
Sampai: Python Interface for Samurai AMR Library

This package provides the Python bindings for Samurai, combining:
- Native C++ implementations (via pybind11 compiled module _sampai)
- Python utility modules (progress bars, visualization helpers)
"""

import sys

# Try to import the compiled extension module
# With meson-python, _sampai.so is installed alongside sampai/ package
try:
    import _sampai as _compiled_module

    # Copy all public symbols from compiled module to this package namespace
    for attr_name in dir(_compiled_module):
        if not attr_name.startswith('_'):
            globals()[attr_name] = getattr(_compiled_module, attr_name)

    # Copy __version__ even though it starts with __
    if hasattr(_compiled_module, "__version__"):
        __version__ = _compiled_module.__version__
except ImportError:
    # Compiled module not available (e.g., during documentation generation)
    _compiled_module = None

# Import the Python utility submodules
try:
    from . import utils  # noqa: E402

    # Export utilities at package level for convenience
    from .utils import progress  # noqa: E402

    __all__ = [
        "progress",
        "utils",
    ]
except ImportError:
    # Utils not available (e.g., during documentation generation)
    __all__ = []

# Add 'sam' as an alias to this package for backward compatibility
# Note: When imported as 'from sampai import sam', 'sam' refers to this package
sys.modules[__name__ + '.sam'] = sys.modules[__name__]
