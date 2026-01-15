"""
Sampai: Python Interface for Samurai AMR Library

This package provides the Python bindings for Samurai, combining:
- Native C++ implementations (via pybind11 compiled module _sampai)
- Python utility modules (progress bars, visualization helpers)
"""

import sys
from importlib import util as importlib_util
from pathlib import Path


def _load_compiled_module():
    try:
        import _sampai as compiled
        return compiled
    except ImportError:
        pass

    try:
        from . import _sampai as compiled
        return compiled
    except ImportError:
        pass

    # Fallback for editable/dev builds: load from a local Meson build dir.
    repo_root = Path(__file__).resolve().parent.parent.parent
    if not (repo_root / "pyproject.toml").is_file():
        return None
    build_roots = (repo_root / "build", repo_root / "builddir")
    patterns = ("_sampai*.so", "_sampai*.pyd", "_sampai*.dylib")
    for build_root in build_roots:
        if not build_root.is_dir():
            continue
        for pattern in patterns:
            for ext_path in build_root.rglob(pattern):
                try:
                    spec = importlib_util.spec_from_file_location("_sampai", ext_path)
                    if spec and spec.loader:
                        compiled = importlib_util.module_from_spec(spec)
                        spec.loader.exec_module(compiled)
                        sys.modules["_sampai"] = compiled
                        return compiled
                except Exception:
                    continue
    return None

# Try to import the compiled extension module
# With meson-python, _sampai.so is installed alongside sampai/ package
try:
    _compiled_module = _load_compiled_module()
    if _compiled_module is None:
        raise ImportError("Compiled module _sampai not found")

    # Provide package-qualified alias for relative imports.
    sys.modules[__name__ + "._sampai"] = _compiled_module

    # Copy all public symbols from compiled module to this package namespace
    # Exclude 'mpi' to avoid conflicts with the sampai.mpi Python module
    for attr_name in dir(_compiled_module):
        if not attr_name.startswith('_') and attr_name != 'mpi':
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

# Try to import MPI module if available
try:
    from . import mpi  # noqa: E402
    __all__.append("mpi")
except (ImportError, RuntimeError):
    # MPI not enabled or not available
    pass

# Add 'sam' as an alias to this package for backward compatibility
# Note: When imported as 'from sampai import sam', 'sam' refers to this package
sys.modules[__name__ + '.sam'] = sys.modules[__name__]
