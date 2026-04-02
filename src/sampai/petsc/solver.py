# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
PETSc Solver utilities for Sampai.

This module provides helper functions to create and configure PETSc KSP solvers.
These utilities use petsc4py for the actual PETSc solver objects.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from petsc4py import PETSc

# Optional petsc4py import
try:
    from petsc4py import PETSc
    HAS_PETSC4PY = True
except ImportError:
    HAS_PETSC4PY = False
    PETSc = None


def create_ksp_solver(
    ksp_type: str = "cg",
    pc_type: str = "jacobi",
    rtol: float = 1e-8,
    atol: float = 1e-50,
    max_it: int = 1000,
    comm=None,
) -> "PETSc.KSP":
    """
    Create a PETSc KSP solver with standard configuration.

    This function creates a Krylov subspace solver (KSP) with the specified
    solver type and preconditioner.

    Args:
        ksp_type: Type of KSP solver (e.g., "cg", "gmres", "bicgstab")
        pc_type: Type of preconditioner (e.g., "jacobi", "ilu", "hypre", "gamg")
        rtol: Relative convergence tolerance
        atol: Absolute convergence tolerance
        max_it: Maximum number of iterations
        comm: MPI communicator (defaults to PETSc.COMM_SELF)

    Returns:
        A configured petsc4py KSP object

    Raises:
        ImportError: If petsc4py is not available

    Example:
        >>> from sampai.petsc.solver import create_ksp_solver
        >>> from petsc4py import PETSc
        >>> # Create a CG solver with Jacobi preconditioning
        >>> ksp = create_ksp_solver("cg", "jacobi")
        >>> # Create matrix and RHS
        >>> A = PETSc.Mat().createAIJ(...)
        >>> b = PETSc.Vec().create(...)
        >>> # Set operators and solve
        >>> ksp.setOperators(A)
        >>> x = b.duplicate()
        >>> ksp.solve(b, x)
    """
    if not HAS_PETSC4PY:
        raise ImportError(
            "petsc4py is required to create KSP solvers. "
            "Install it with: conda install -c conda-forge petsc4py"
        )

    if comm is None:
        comm = PETSc.COMM_SELF

    # Create KSP solver
    ksp = PETSc.KSP().create(comm)

    # Set solver type
    ksp.setType(ksp_type)

    # Set tolerances
    ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)

    # Get and configure preconditioner
    pc = ksp.getPC()
    pc.setType(pc_type)

    # Set from options (allows command-line override)
    ksp.setFromOptions()

    return ksp


def create_gmres_solver(
    restart: int = 30,
    pc_type: str = "jacobi",
    rtol: float = 1e-8,
    atol: float = 1e-50,
    max_it: int = 1000,
    comm=None,
) -> "PETSc.KSP":
    """
    Create a GMRES solver with specified restart parameter.

    GMRES (Generalized Minimal Residual) is a popular Krylov method for
    non-symmetric systems.

    Args:
        restart: GMRES restart parameter (number of iterations before restart)
        pc_type: Type of preconditioner
        rtol: Relative convergence tolerance
        atol: Absolute convergence tolerance
        max_it: Maximum number of iterations
        comm: MPI communicator

    Returns:
        A configured GMRES solver

    Example:
        >>> from sampai.petsc.solver import create_gmres_solver
        >>> ksp = create_gmres_solver(restart=50, pc_type="hypre")
    """
    ksp = create_ksp_solver(
        ksp_type="gmres",
        pc_type=pc_type,
        rtol=rtol,
        atol=atol,
        max_it=max_it,
        comm=comm,
    )
    ksp.setGMRESRestart(restart)
    return ksp


def create_cg_solver(
    pc_type: str = "jacobi",
    rtol: float = 1e-8,
    atol: float = 1e-50,
    max_it: int = 1000,
    comm=None,
) -> "PETSc.KSP":
    """
    Create a Conjugate Gradient (CG) solver.

    CG is efficient for symmetric positive-definite matrices.

    Args:
        pc_type: Type of preconditioner
        rtol: Relative convergence tolerance
        atol: Absolute convergence tolerance
        max_it: Maximum number of iterations
        comm: MPI communicator

    Returns:
        A configured CG solver

    Example:
        >>> from sampai.petsc.solver import create_cg_solver
        >>> ksp = create_cg_solver(pc_type="hypre")
    """
    return create_ksp_solver(
        ksp_type="cg",
        pc_type=pc_type,
        rtol=rtol,
        atol=atol,
        max_it=max_it,
        comm=comm,
    )


def configure_monitor(ksp: "PETSc.KSP", verbose: bool = True) -> None:
    """
    Configure convergence monitoring for a KSP solver.

    This sets up the solver to print convergence information.

    Args:
        ksp: A petsc4py KSP object
        verbose: If True, print convergence information

    Example:
        >>> from sampai.petsc.solver import create_cg_solver, configure_monitor
        >>> ksp = create_cg_solver()
        >>> configure_monitor(ksp, verbose=True)
        >>> # Now ksp will print iteration information during solve
    """
    if not HAS_PETSC4PY:
        raise ImportError("petsc4py is required")

    if verbose:
        # Use PETSc's default monitor
        ksp.monitorSet(
            PETSc.Monitor(),
            None,
            None,
        )
    else:
        ksp.monitorCancel()


def solve_with_ksp(
    ksp: "PETSc.KSP",
    A: "PETSc.Mat",
    b: "PETSc.Vec",
    x: Optional["PETSc.Vec"] = None,
) -> "PETSc.Vec":
    """
    Solve a linear system Ax = b using the given KSP solver.

    This is a convenience wrapper around KSP.solve().

    Args:
        ksp: A configured petsc4py KSP object
        A: System matrix
        b: Right-hand side vector
        x: Solution vector (if None, created as duplicate of b)

    Returns:
        The solution vector x

    Example:
        >>> from sampai.petsc.solver import create_cg_solver, solve_with_ksp
        >>> from petsc4py import PETSc
        >>> ksp = create_cg_solver()
        >>> A = PETSc.Mat().createAIJ(...)
        >>> b = PETSc.Vec().create(...)
        >>> x = solve_with_ksp(ksp, A, b)
    """
    if not HAS_PETSC4PY:
        raise ImportError("petsc4py is required")

    # Set operators
    ksp.setOperators(A)

    # Create solution vector if not provided
    if x is None:
        x = b.duplicate()

    # Solve
    ksp.solve(b, x)

    return x


class KSPSolver:
    """
    High-level wrapper for PETSc KSP solvers.

    This class provides a convenient interface for solving linear systems
    with PETSc Krylov solvers.

    Attributes:
        ksp: The underlying petsc4py KSP object
        comm: MPI communicator

    Example:
        >>> from sampai.petsc.solver import KSPSolver
        >>> from petsc4py import PETSc
        >>> # Create solver
        >>> solver = KSPSolver(ksp_type="cg", pc_type="hypre")
        >>> # Set up problem
        >>> A = PETSc.Mat().createAIJ(...)
        >>> b = PETSc.Vec().create(...)
        >>> # Solve
        >>> x = solver.solve(A, b)
        >>> print(f"Converged in {solver.iters} iterations")
    """

    def __init__(
        self,
        ksp_type: str = "cg",
        pc_type: str = "jacobi",
        rtol: float = 1e-8,
        atol: float = 1e-50,
        max_it: int = 1000,
        comm=None,
    ):
        """
        Initialize a KSP solver.

        Args:
            ksp_type: Type of KSP solver
            pc_type: Type of preconditioner
            rtol: Relative tolerance
            atol: Absolute tolerance
            max_it: Maximum iterations
            comm: MPI communicator
        """
        if not HAS_PETSC4PY:
            raise ImportError("petsc4py is required")

        self.comm = comm if comm is not None else PETSc.COMM_SELF
        self.ksp = create_ksp_solver(
            ksp_type=ksp_type,
            pc_type=pc_type,
            rtol=rtol,
            atol=atol,
            max_it=max_it,
            comm=self.comm,
        )

    def solve(
        self,
        A: "PETSc.Mat",
        b: "PETSc.Vec",
        x: Optional["PETSc.Vec"] = None,
    ) -> "PETSc.Vec":
        """
        Solve the linear system Ax = b.

        Args:
            A: System matrix
            b: Right-hand side vector
            x: Solution vector (if None, created as duplicate of b)

        Returns:
            The solution vector x
        """
        return solve_with_ksp(self.ksp, A, b, x)

    @property
    def iters(self) -> int:
        """Get the number of iterations from the last solve."""
        return self.ksp.getIterationNumber()

    @property
    def converged_reason(self) -> int:
        """Get the convergence reason from the last solve."""
        return self.ksp.getConvergedReason()

    @property
    def is_converged(self) -> bool:
        """Check if the last solve converged."""
        reason = self.converged_reason
        return reason > 0

    def set_monitor(self, enabled: bool = True) -> None:
        """Enable or disable convergence monitoring."""
        configure_monitor(self.ksp, verbose=enabled)


__all__ = [
    'create_ksp_solver',
    'create_gmres_solver',
    'create_cg_solver',
    'configure_monitor',
    'solve_with_ksp',
    'KSPSolver',
    'HAS_PETSC4PY',
]
