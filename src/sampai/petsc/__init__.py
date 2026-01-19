# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
PETSc support for Sampai.

This module provides PETSc (Portable, Extensible Toolkit for Scientific Computation)
integration for linear solvers, nonlinear solvers, and time integrators.

PETSc is primarily used for:
    - Linear solvers (KSP): Krylov subspace methods (CG, GMRES, etc.)
    - Preconditioners (PC): Jacobi, ILU, Multigrid, Hypre, etc.
    - Nonlinear solvers (SNES): Newton-based methods
    - Time integrators (TS): ODE/DAE solvers
    - Matrix assembly: Discrete operator matrices from Samurai fields

Note: PETSc support must be enabled at build time with -Dmpi=true -Dpetsc=true.
    If PETSc is not enabled, this module will provide stub functions
    that raise RuntimeError when called.

Example:
    >>> from sampai import petsc
    >>> print(f"PETSc version: {petsc.version}")
    >>> print(f"PETSc enabled: {petsc.is_initialized()}")
    >>>
    >>> # Set PETSc options (like command-line arguments)
    >>> petsc.set_option("-ksp_type", "cg")
    >>> petsc.set_option("-pc_type", "hypre")

For advanced usage with petsc4py:
    >>> import petsc4py
    >>> from petsc4py import PETSc
    >>> # Use petsc4py directly for Vec, Mat, KSP objects
    >>> # Sampai provides field-to-vector conversion utilities (future)

Note:
    This module provides basic PETSc integration. For full PETSc functionality
    (Vec, Mat, KSP, SNES, TS objects), use petsc4py which is installed
    automatically with the PETSc environment.
"""

from importlib import import_module

# Import compiled PETSc functions
try:
    import _sampai as _sampai
except ImportError:
    _sampai = import_module("sampai._sampai")

# Direct imports from _sampai for convenience
is_initialized = _sampai.petsc.is_initialized
initialize = _sampai.petsc.initialize
finalize = _sampai.petsc.finalize
set_option = _sampai.petsc.set_option
set_options_prefix = _sampai.petsc.set_options_prefix
clear_options_prefix = _sampai.petsc.clear_options_prefix

# Read-only properties
version = _sampai.petsc.version
release_date = _sampai.petsc.release_date
world_size = _sampai.petsc.world_size
world_rank = _sampai.petsc.world_rank

# Enums (imported from _sampai)
KSPOption = _sampai.petsc.KSPOption
PCOption = _sampai.petsc.PCOption

__all__ = [
    # Query functions
    'is_initialized',
    'version',
    'release_date',
    # Initialization
    'initialize',
    'finalize',
    # Options
    'set_option',
    'set_options_prefix',
    'clear_options_prefix',
    # Communicator info
    'world_size',
    'world_rank',
    # Enums
    'KSPOption',
    'PCOption',
]

__version__ = '0.1.0'
