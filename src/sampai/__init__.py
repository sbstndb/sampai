"""
Sampai: Python Interface for Samurai AMR Library

This package provides the Python bindings for Samurai, combining:
- Native C++ implementations (via pybind11 compiled module)
- Python utility modules (progress bars, visualization helpers)
"""

import importlib.util
import os
import sys

# Try to import _sampai using the normal import mechanism first
# This works with scikit-build-core installations (editable and regular)
_compiled_module = None
_import_error = None

try:
    _compiled_module = sys.modules.get("_sampai")
    if _compiled_module is None:
        import _sampai as _compiled_module
except ImportError as e:
    _import_error = e

# If normal import failed, try manual discovery for CMake builds
if _compiled_module is None:
    # Get the build directory (project root/src -> project root -> build/...)
    _package_dir = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_package_dir, '..', '..'))

    # Try multiple possible build directories for scikit-build-core
    _build_lib_dirs = [
        os.path.join(_project_root, 'build', 'lib'),  # Direct cmake build
        # scikit-build-core uses platform-specific directories
        os.path.join(_project_root, 'build', 'py3-none-linux_x86_64', 'lib'),
        os.path.join(_project_root, 'build', 'py3-none-linux_x86_64', 'sampai'),
        os.path.join(_project_root, 'build', 'lib', 'sampai'),
        # Also check platform-agnostic patterns
        os.path.join(_project_root, 'build', 'lib.*', 'sampai'),
        os.path.join(_project_root, 'build', '*-linux_x86_64', 'lib'),
        os.path.join(_project_root, 'build', '*-linux_x86_64', 'sampai'),
    ]

    # Find the compiled .so file from build directories
    _so_files = []
    for _build_lib_dir in _build_lib_dirs:
        if '*' in _build_lib_dir:
            import glob as glob_module
            _matching_dirs = glob_module.glob(_build_lib_dir)
            for _expanded_dir in _matching_dirs:
                if os.path.exists(_expanded_dir):
                    import glob
                    _found = glob.glob(os.path.join(_expanded_dir, '_sampai*.so'))
                    if _found:
                        _so_files = _found
                        break
            if _so_files:
                break
        elif os.path.exists(_build_lib_dir):
            import glob
            _so_files = glob.glob(os.path.join(_build_lib_dir, '_sampai*.so'))
            if _so_files:
                break

    if _so_files:
        _so_path = _so_files[0]
        # Load the compiled module using importlib
        spec = importlib.util.spec_from_file_location("_sampai", _so_path)
        _compiled_module = importlib.util.module_from_spec(spec)
        sys.modules["_sampai"] = _compiled_module
        spec.loader.exec_module(_compiled_module)
    else:
        raise ImportError(
            f"Cannot find _sampai compiled module. "
            f"Tried normal import (failed: {_import_error}) and manual search in {_build_lib_dirs}. "
            "Please build the project first (run: pip install . or cmake --build build)."
        )

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
