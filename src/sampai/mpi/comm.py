# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
MPI communicator management.

Provides initialization, rank/size queries, and barrier synchronization.
"""

from importlib import import_module
from typing import Optional

try:
    import _sampai as _sampai
except ImportError:
    _sampai = import_module("sampai._sampai")

# Global communicator instance
_comm: Optional['Communicator'] = None


class Communicator:
    """
    MPI communicator wrapper.

    This class provides a Python interface to the MPI communicator
    managed by the underlying Samurai library.

    Attributes
    ----------
    rank : int
        Rank of this process (0-indexed)
    size : int
        Total number of processes in the communicator

    Examples
    --------
    >>> from sampai import mpi
    >>> comm = mpi.init()
    >>> print(f"I am rank {comm.rank} of {comm.size}")
    """

    def __init__(self):
        if not _sampai.mpi.is_initialized():
            raise RuntimeError(
                "MPI support is not enabled in this Sampai build. "
                "Rebuild with -Dmpi=true to enable MPI."
            )
        self._comm = _sampai.mpi.Communicator.world

    @property
    def rank(self) -> int:
        """int: Rank of this process (0-indexed)."""
        return self._comm.rank

    @property
    def size(self) -> int:
        """int: Total number of processes."""
        return self._comm.size

    def barrier(self):
        """
        Barrier synchronization.

        Blocks until all processes in the communicator call this function.

        Examples
        --------
        >>> comm.barrier()  # All processes synchronize here
        """
        self._comm.barrier()

    def __repr__(self) -> str:
        return f"Communicator(rank={self.rank}, size={self.size})"


def init() -> Communicator:
    """
    Initialize MPI and return the world communicator.

    This function initializes MPI (if not already initialized) and
    returns a Communicator object for MPI_COMM_WORLD.

    Returns
    -------
    Communicator
        The world communicator

    Examples
    --------
    >>> from sampai import mpi
    >>> comm = mpi.init()
    >>> print(f"Rank {mpi.rank()} of {mpi.size()}")
    """
    global _comm
    if _comm is None:
        _comm = Communicator()
    return _comm


def finalize():
    """
    Finalize MPI (no-op in this implementation).

    MPI finalization is handled automatically by the Samurai library.
    This function exists for API compatibility but does nothing.

    Examples
    --------
    >>> mpi.finalize()
    """
    pass


def rank() -> int:
    """
    Get the rank of the current process.

    Returns
    -------
    int
        Rank of this process (0-indexed). Returns 0 if MPI is not enabled.

    Examples
    --------
    >>> from sampai import mpi
    >>> r = mpi.rank()
    >>> if r == 0:
    ...     print("I am the root process")
    """
    # Directly call C++ binding for efficiency
    return _sampai.mpi.rank()


def size() -> int:
    """
    Get the total number of processes.

    Returns
    -------
    int
        Number of processes. Returns 1 if MPI is not enabled.

    Examples
    --------
    >>> from sampai import mpi
    >>> print(f"Running with {mpi.size()} processes")
    """
    # Directly call C++ binding for efficiency
    return _sampai.mpi.size()


def barrier():
    """
    Barrier synchronization across all processes.

    Blocks until all processes in MPI_COMM_WORLD call this function.

    Examples
    --------
    >>> mpi.barrier()  # All processes synchronize here
    """
    _sampai.mpi.barrier()
