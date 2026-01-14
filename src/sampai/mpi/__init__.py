"""
MPI support for Sampai.

This module provides MPI communicator management functions for distributed
memory parallelization using the underlying Samurai MPI implementation.

Note: Samurai handles MPI mesh distribution and ghost cell exchange automatically.
When MPI is enabled (compiled with -Dmpi=true), the following operations automatically
use MPI:
    - sam.mesh.make(box, config) - creates mesh distributed across ranks
    - sam.adaptation.update_ghost_mr(field) - exchanges ghost cells across ranks

Example:
    >>> from sampai import mpi
    >>> comm = mpi.init()
    >>> print(f"Rank {mpi.rank()} of {mpi.size()}")
    >>>
    >>> # Mesh is automatically distributed in MPI
    >>> import sampai as sam
    >>> mesh = sam.mesh.make(box, config)
    >>>
    >>> # Ghost cells are automatically exchanged
    >>> u = sam.field.zeros(mesh, "u")
    >>> sam.adaptation.update_ghost_mr(u)

Note:
    MPI support must be enabled at build time with -Dmpi=true.
    If MPI is not enabled, this module will provide stub functions
    that raise RuntimeError when called.
"""

# Import compiled MPI functions
import _sampai

# Global communicator
_comm = None

# Import submodules
from .comm import (
    Communicator,
    init,
    finalize,
    rank,
    size,
    barrier,
)

# No parallel I/O functions needed - samurai.save() and samurai.load()
# already handle MPI automatically when SAMURAI_WITH_MPI is defined.

__all__ = [
    # Communicator
    'Communicator',
    'init',
    'finalize',
    'rank',
    'size',
    'barrier',
    # Query
    'is_initialized',
]

# Direct imports from _sampai for convenience
is_initialized = _sampai.mpi.is_initialized

__version__ = '0.1.0'
