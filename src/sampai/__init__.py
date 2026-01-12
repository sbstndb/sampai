"""
Sampai: Python Interface for Samurai AMR Library

This package provides the Python bindings for Samurai, combining:
- Native C++ implementations (via pybind11 compiled module)
- Python utility modules (progress bars, visualization helpers)
"""

import importlib.util
import os
import sys

# Get the build directory (project root/src -> project root -> build/lib)
_package_dir = os.path.dirname(__file__)
_project_root = os.path.abspath(os.path.join(_package_dir, '..', '..'))
_build_lib_dir = os.path.join(_project_root, 'build', 'lib')

# Find the compiled .so file from build directory
_so_files = []
if os.path.exists(_build_lib_dir):
    import glob
    _so_files = glob.glob(os.path.join(_build_lib_dir, 'sampai*.so'))

if _so_files:
    _so_path = _so_files[0]
else:
    raise ImportError(
        f"Cannot find sampai compiled module in {_build_lib_dir}. "
        "Please build the project first (run: cmake --build build)."
    )

# Load the compiled module using importlib
spec = importlib.util.spec_from_file_location("sampai", _so_path)
_compiled_module = importlib.util.module_from_spec(spec)
sys.modules["sampai_compiled"] = _compiled_module
spec.loader.exec_module(_compiled_module)

# Copy all public symbols from compiled module to this package namespace
for attr_name in dir(_compiled_module):
    if not attr_name.startswith('_'):
        globals()[attr_name] = getattr(_compiled_module, attr_name)

# Copy __version__ even though it starts with __
if hasattr(_compiled_module, "__version__"):
    __version__ = _compiled_module.__version__

# Import the Python utility submodules
from . import utils  # noqa: E402

# Export utilities at package level for convenience
from .utils import progress  # noqa: E402

__all__ = [
    "progress",
    "utils",
]
