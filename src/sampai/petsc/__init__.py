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
    >>> print(f"PETSc version: {petsc.get_version()}")
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
get_version = _sampai.petsc.get_version
get_world_size = _sampai.petsc.get_world_size
get_world_rank = _sampai.petsc.get_world_rank
set_option = _sampai.petsc.set_option
set_options_prefix = _sampai.petsc.set_options_prefix
clear_options_prefix = _sampai.petsc.clear_options_prefix

# KSP solver type constants
KSP_RICHARDSON = _sampai.petsc.KSP_RICHARDSON
KSP_CHEBYSHEV = _sampai.petsc.KSP_CHEBYSHEV
KSP_CG = _sampai.petsc.KSP_CG
KSP_GMRES = _sampai.petsc.KSP_GMRES
KSP_TCQMR = _sampai.petsc.KSP_TCQMR
KSP_TFQMR = _sampai.petsc.KSP_TFQMR
KSP_BCGS = _sampai.petsc.KSP_BCGS
KSP_CGS = _sampai.petsc.KSP_CGS
KSP_BICG = _sampai.petsc.KSP_BICG
KSP_PREONLY = _sampai.petsc.KSP_PREONLY

# Preconditioner type constants
PC_NONE = _sampai.petsc.PC_NONE
PC_JACOBI = _sampai.petsc.PC_JACOBI
PC_SOR = _sampai.petsc.PC_SOR
PC_LU = _sampai.petsc.PC_LU
PC_ILU = _sampai.petsc.PC_ILU
PC_ICC = _sampai.petsc.PC_ICC
PC_ASM = _sampai.petsc.PC_ASM
PC_GASM = _sampai.petsc.PC_GASM
PC_BJACOBI = _sampai.petsc.PC_BJACOBI
PC_MG = _sampai.petsc.PC_MG
PC_HYPRE = _sampai.petsc.PC_HYPRE
PC_GAMG = _sampai.petsc.PC_GAMG

__all__ = [
    # Query functions
    'is_initialized',
    'get_version',
    # Initialization
    'initialize',
    'finalize',
    # Options
    'set_option',
    'set_options_prefix',
    'clear_options_prefix',
    # Communicator info
    'get_world_size',
    'get_world_rank',
    # KSP solver types
    'KSP_RICHARDSON',
    'KSP_CHEBYSHEV',
    'KSP_CG',
    'KSP_GMRES',
    'KSP_TCQMR',
    'KSP_TFQMR',
    'KSP_BCGS',
    'KSP_CGS',
    'KSP_BICG',
    'KSP_PREONLY',
    # Preconditioner types
    'PC_NONE',
    'PC_JACOBI',
    'PC_SOR',
    'PC_LU',
    'PC_ILU',
    'PC_ICC',
    'PC_ASM',
    'PC_GASM',
    'PC_BJACOBI',
    'PC_MG',
    'PC_HYPRE',
    'PC_GAMG',
]

__version__ = '0.1.0'
