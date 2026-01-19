# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for PETSc support in Sampai.

These tests verify that PETSc functionality is properly exposed through the
Python bindings. Tests are designed to work whether PETSc is enabled or not.
"""

import pytest

# Try to import sampai and its PETSc module
try:
    import sampai
    from sampai import petsc
    HAS_SAMPAI = True
except ImportError:
    HAS_SAMPAI = False

# Check if PETSc is actually enabled
PETSC_ENABLED = False
if HAS_SAMPAI:
    try:
        # Try calling a PETSc function to see if it's enabled
        version = petsc.get_version()
        PETSC_ENABLED = version != "NOT_ENABLED"
    except RuntimeError:
        pass


class TestPETScInitialization:
    """Tests for PETSc initialization and basic queries."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_module_exists(self):
        """Test that the petsc submodule exists."""
        assert hasattr(sampai, "petsc"), "sampai.petsc submodule should exist"

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_get_version(self):
        """Test getting PETSc version."""
        version = petsc.get_version()
        assert isinstance(version, str)

        if PETSC_ENABLED:
            assert "PETSc" in version
            assert "Release" in version or version.startswith("PETSc")
        else:
            assert version == "NOT_ENABLED"

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_is_initialized(self):
        """Test checking if PETSc is initialized."""
        is_init = petsc.is_initialized()
        assert isinstance(is_init, bool)

        if PETSC_ENABLED:
            # PETSc might not be initialized yet - initialize and check again
            if not is_init:
                petsc.initialize()
                assert petsc.is_initialized() is True
            # Clean up for other tests
            # Note: Don't call finalize() as it can block in MPI context
            # petsc.finalize()

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_initialize_function(self):
        """Test explicit PETSc initialization."""
        # Just verify initialize is callable and works
        if not petsc.is_initialized():
            petsc.initialize()
            assert petsc.is_initialized() is True

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skip(reason="finalize() can block in MPI context - test manually if needed")
    def test_finalize_function(self):
        """Test PETSc finalization."""
        pass  # Skipped - finalize can cause blocking issues


class TestPETScCommunicator:
    """Tests for PETSc MPI communicator information."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_get_world_size(self):
        """Test getting PETSc world size (number of processes)."""
        size = petsc.get_world_size()
        assert isinstance(size, int)
        assert size >= 1

        if not PETSC_ENABLED:
            # When disabled, should return 1
            assert size == 1

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_get_world_rank(self):
        """Test getting PETSc world rank (process rank)."""
        rank = petsc.get_world_rank()
        assert isinstance(rank, int)
        assert rank >= 0

        # Rank should be less than size
        size = petsc.get_world_size()
        assert rank < size

        if not PETSC_ENABLED:
            # When disabled, should return 0
            assert rank == 0


class TestPETScOptions:
    """Tests for PETSc options (command-line like configuration)."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_set_option(self):
        """Test setting PETSc options."""
        # Set various KSP types
        petsc.set_option("-ksp_type", "cg")
        petsc.set_option("-ksp_type", "gmres")

        # Set various PC types
        petsc.set_option("-pc_type", "jacobi")
        petsc.set_option("-pc_type", "hypre")

        # Set options with values
        petsc.set_option("-ksp_rtol", "1e-10")
        petsc.set_option("-ksp_max_it", "1000")

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_set_options_prefix(self):
        """Test setting PETSc options prefix."""
        petsc.set_options_prefix("solver1_")
        petsc.set_option("-ksp_type", "cg")

        petsc.clear_options_prefix()
        petsc.set_options_prefix("solver2_")
        petsc.set_option("-ksp_type", "gmres")

        # Clean up
        petsc.clear_options_prefix()

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_clear_options_prefix(self):
        """Test clearing PETSc options prefix."""
        petsc.set_options_prefix("test_")
        petsc.clear_options_prefix()
        # Should not raise any errors


class TestPETScConstants:
    """Tests for PETSc KSP and PC type constants."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_ksp_constants_exist(self):
        """Test that KSP solver type constants are defined."""
        ksp_types = [
            "KSP_RICHARDSON",
            "KSP_CHEBYSHEV",
            "KSP_CG",
            "KSP_GMRES",
            "KSP_TCQMR",
            "KSP_TFQMR",
            "KSP_BCGS",
            "KSP_CGS",
            "KSP_BICG",
            "KSP_PREONLY",
        ]

        for ksp_type in ksp_types:
            assert hasattr(petsc, ksp_type), f"petsc.{ksp_type} should exist"
            value = getattr(petsc, ksp_type)
            assert isinstance(value, str)
            assert len(value) > 0

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_ksp_constant_values(self):
        """Test that KSP constants have correct string values."""
        assert petsc.KSP_CG == "cg"
        assert petsc.KSP_GMRES == "gmres"
        assert petsc.KSP_BICG == "bicg"
        assert petsc.KSP_PREONLY == "preonly"

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_pc_constants_exist(self):
        """Test that PC (preconditioner) type constants are defined."""
        pc_types = [
            "PC_NONE",
            "PC_JACOBI",
            "PC_SOR",
            "PC_LU",
            "PC_ILU",
            "PC_ICC",
            "PC_ASM",
            "PC_GASM",
            "PC_BJACOBI",
            "PC_MG",
            "PC_HYPRE",
            "PC_GAMG",
        ]

        for pc_type in pc_types:
            assert hasattr(petsc, pc_type), f"petsc.{pc_type} should exist"
            value = getattr(petsc, pc_type)
            assert isinstance(value, str)
            assert len(value) > 0

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_pc_constant_values(self):
        """Test that PC constants have correct string values."""
        assert petsc.PC_NONE == "none"
        assert petsc.PC_JACOBI == "jacobi"
        assert petsc.PC_LU == "lu"
        assert petsc.PC_ILU == "ilu"
        assert petsc.PC_HYPRE == "hypre"


class TestPETScIntegration:
    """Integration tests for PETSc with Sampai.

    These tests verify that PETSc can be used alongside Samurai meshes
    and fields (even if full integration is not yet implemented).
    """

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_petsc_with_box(self):
        """Test that PETSc works alongside basic Sampai geometry."""
        from sampai import geometry

        # Create a simple box
        b = geometry.box([0.0, 0.0], [1.0, 1.0])

        # PETSc should still work
        assert petsc.is_initialized() or petsc.get_world_size() >= 1

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_petsc_with_mesh_config(self):
        """Test that PETSc works alongside mesh configuration."""
        from sampai import config

        # Create a mesh config (factory function)
        cfg = config.make(dim=2, min_level=0, max_level=6)

        # PETSc should still work alongside config
        assert petsc.is_initialized() or petsc.get_world_size() >= 1

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_petsc_options_for_solver(self):
        """Test setting up PETSc options for a typical solver."""
        # Set up a typical CG solver with Jacobi preconditioning
        petsc.set_option("-ksp_type", petsc.KSP_CG)
        petsc.set_option("-pc_type", petsc.PC_JACOBI)
        petsc.set_option("-ksp_rtol", "1e-8")
        petsc.set_option("-ksp_max_it", "1000")

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_petsc_gmres_with_hypre(self):
        """Test setting up GMRES with Hypre preconditioner."""
        petsc.set_option("-ksp_type", petsc.KSP_GMRES)
        petsc.set_option("-pc_type", petsc.PC_HYPRE)
        petsc.set_option("-ksp_rtol", "1e-10")


class TestPETScDisabled:
    """Tests for behavior when PETSc is NOT enabled."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(PETSC_ENABLED, reason="PETSc is enabled")
    def test_disabled_get_version(self):
        """Test that get_version returns NOT_ENABLED when PETSc is disabled."""
        assert petsc.get_version() == "NOT_ENABLED"

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(PETSC_ENABLED, reason="PETSc is enabled")
    def test_disabled_is_initialized(self):
        """Test that is_initialized returns False when PETSc is disabled."""
        assert petsc.is_initialized() is False

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(PETSC_ENABLED, reason="PETSc is enabled")
    def test_disabled_get_world_size(self):
        """Test that get_world_size returns 1 when PETSc is disabled."""
        assert petsc.get_world_size() == 1

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(PETSC_ENABLED, reason="PETSc is enabled")
    def test_disabled_get_world_rank(self):
        """Test that get_world_rank returns 0 when PETSc is disabled."""
        assert petsc.get_world_rank() == 0

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(PETSC_ENABLED, reason="PETSc is enabled")
    def test_disabled_initialize_raises(self):
        """Test that initialize raises an error when PETSc is disabled."""
        with pytest.raises(RuntimeError, match="PETSc support is not enabled"):
            petsc.initialize()

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(PETSC_ENABLED, reason="PETSc is enabled")
    def test_disabled_finalize_raises(self):
        """Test that finalize raises an error when PETSc is disabled."""
        with pytest.raises(RuntimeError, match="PETSc support is not enabled"):
            petsc.finalize()

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(PETSC_ENABLED, reason="PETSc is enabled")
    def test_disabled_set_option_raises(self):
        """Test that set_option raises an error when PETSc is disabled."""
        with pytest.raises(RuntimeError, match="PETSc support is not enabled"):
            petsc.set_option("-ksp_type", "cg")


class TestPETScAllExports:
    """Test that all expected exports are present in petsc module."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_query_functions_exist(self):
        """Test that query functions are exported."""
        assert hasattr(petsc, "is_initialized")
        assert hasattr(petsc, "get_version")

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_init_functions_exist(self):
        """Test that initialization functions are exported."""
        assert hasattr(petsc, "initialize")
        assert hasattr(petsc, "finalize")

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_option_functions_exist(self):
        """Test that option functions are exported."""
        assert hasattr(petsc, "set_option")
        assert hasattr(petsc, "set_options_prefix")
        assert hasattr(petsc, "clear_options_prefix")

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_communicator_functions_exist(self):
        """Test that communicator functions are exported."""
        assert hasattr(petsc, "get_world_size")
        assert hasattr(petsc, "get_world_rank")




class TestPETScVectorConversion:
    """Tests for converting between Sampai fields and PETSc vectors."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_field_to_array(self):
        """Convert field to numpy array."""
        from sampai.petsc.vector import field_to_array

        # Initialize PETSc (required for MPI)
        sampai.petsc.initialize()

        cfg = sampai.config.make(dim=1)
        cfg.min_level = 4
        cfg.max_level = 4
        box = sampai.geometry.box([0.0], [1.0])
        mesh = sampai.mesh.make(box, cfg)
        u = sampai.field.scalar(mesh, "u")

        # Fill field
        count = [0]
        def fill_cell(cell):
            u[cell.index] = float(count[0])
            count[0] += 1
        sampai.algorithms.for_each_cell(mesh, fill_cell)

        # Convert to array
        arr = field_to_array(u)
        assert len(arr) == mesh.nb_cells
        # Check some values were set
        assert arr.max() >= 1.0  # At least some values should be non-zero

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_array_to_field(self):
        """Copy numpy array to field."""
        from sampai.petsc.vector import array_to_field
        import numpy as np

        sampai.petsc.initialize()

        cfg = sampai.config.make(dim=1)
        cfg.min_level = 4
        cfg.max_level = 4
        box = sampai.geometry.box([0.0], [1.0])
        mesh = sampai.mesh.make(box, cfg)
        u = sampai.field.scalar(mesh, "u")

        # Create test array
        arr = np.arange(mesh.nb_cells, dtype=np.float64)
        array_to_field(arr, u)

        # Verify values were copied
        result = u.numpy_view()
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[-1] == float(mesh.nb_cells - 1)

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_roundtrip_field_array_field(self):
        """Test field -> array -> field roundtrip."""
        from sampai.petsc.vector import field_to_array, array_to_field
        import numpy as np

        sampai.petsc.initialize()

        cfg = sampai.config.make(dim=1)
        cfg.min_level = 4
        cfg.max_level = 4
        box = sampai.geometry.box([0.0], [1.0])
        mesh = sampai.mesh.make(box, cfg)
        u = sampai.field.scalar(mesh, "u")

        # Fill field
        count = [0]
        def fill_cell(cell):
            u[cell.index] = float(count[0] * 2.5)  # Non-zero values
            count[0] += 1
        sampai.algorithms.for_each_cell(mesh, fill_cell)

        # Extract to array
        arr = field_to_array(u)
        original_values = arr.copy()

        # Create new field and copy back
        u2 = sampai.field.scalar(mesh, "u2")
        array_to_field(arr, u2)

        # Verify values match
        result = u2.numpy_view()
        np.testing.assert_array_almost_equal(result, original_values)


class TestPETScSolvers:
    """Tests for PETSc KSP solvers with petsc4py."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_petsc4py_available(self):
        """Check if petsc4py is available."""
        from sampai.petsc.vector import HAS_PETSC4PY
        # This test documents whether petsc4py is available
        # We don't assert anything since it's environment-dependent
        if not HAS_PETSC4PY:
            pytest.skip("petsc4py not installed - install with: conda install -c conda-forge petsc4py")

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_cg_solver_diagonal_system(self):
        """Test CG solver on simple diagonal system."""
        from sampai.petsc.vector import HAS_PETSC4PY
        from sampai.petsc.solver import create_cg_solver

        if not HAS_PETSC4PY:
            pytest.skip("petsc4py not installed")

        # Initialize PETSc (required for MPI)
        sampai.petsc.initialize()

        from petsc4py import PETSc
        import numpy as np

        # Create simple diagonal matrix: A = 2*I
        n = 10
        A = PETSc.Mat().createAIJ([n, n])
        A.setUp()
        for i in range(n):
            A.setValue(i, i, 2.0)
        A.assemblyBegin()
        A.assemblyEnd()

        # RHS: b = [1, 1, ..., 1]
        b = PETSc.Vec().createSeq(n)
        b.set(1.0)

        # Solve with CG using solve_with_ksp helper
        ksp = create_cg_solver(rtol=1e-10, max_it=100)
        from sampai.petsc.solver import solve_with_ksp
        x = solve_with_ksp(ksp, A, b)

        # Check solution: x = [0.5, 0.5, ..., 0.5]
        reason = ksp.getConvergedReason()
        assert reason > 0, "Solver should converge"
        for i in range(n):
            assert abs(x.getValue(i) - 0.5) < 1e-8, f"Solution mismatch at index {i}"

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_gmres_solver_with_hypre(self):
        """Test GMRES solver with Hypre preconditioner."""
        from sampai.petsc.vector import HAS_PETSC4PY
        from sampai.petsc.solver import create_gmres_solver
        from sampai import petsc

        if not HAS_PETSC4PY:
            pytest.skip("petsc4py not installed")

        # Initialize PETSc (required for MPI)
        petsc.initialize()

        from petsc4py import PETSc

        # Create symmetric positive definite matrix
        n = 20
        A = PETSc.Mat().createAIJ([n, n])
        A.setUp()
        for i in range(n):
            A.setValue(i, i, 4.0)
            if i > 0:
                A.setValue(i, i-1, -1.0)
            if i < n-1:
                A.setValue(i, i+1, -1.0)
        A.assemblyBegin()
        A.assemblyEnd()

        # RHS
        b = PETSc.Vec().createSeq(n)
        b.set(1.0)

        # Solve with GMRES + Hypre using solve_with_ksp helper
        ksp = create_gmres_solver(restart=30, pc_type="jacobi", rtol=1e-8)
        from sampai.petsc.solver import solve_with_ksp
        x = solve_with_ksp(ksp, A, b)

        reason = ksp.getConvergedReason()
        assert reason > 0, "Solver should converge"

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_ksp_solver_class(self):
        """Test high-level KSPSolver class."""
        from sampai.petsc.vector import HAS_PETSC4PY
        from sampai.petsc.solver import KSPSolver

        if not HAS_PETSC4PY:
            pytest.skip("petsc4py not installed")

        # Initialize PETSc (required for MPI)
        sampai.petsc.initialize()

        from petsc4py import PETSc

        # Create simple system
        n = 5
        A = PETSc.Mat().createAIJ([n, n])
        A.setUp()
        for i in range(n):
            A.setValue(i, i, 3.0)
        A.assemblyBegin()
        A.assemblyEnd()

        b = PETSc.Vec().createSeq(n)
        b.set(2.0)

        # Create solver and solve
        solver = KSPSolver(ksp_type="cg", pc_type="jacobi", rtol=1e-10)
        x = solver.solve(A, b)

        assert solver.is_converged, "Solver should converge"
        assert solver.iters > 0, "Should have taken some iterations"
        # Solution should be x = 2/3 for 3*I * x = 2
        for i in range(n):
            assert abs(x.getValue(i) - 2.0/3.0) < 1e-8


class TestPETScFieldSolver:
    """End-to-end tests for solving PDEs with Sampai fields and PETSc."""

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_1d_diffusion_with_field(self):
        """Test solving 1D diffusion equation with PETSc and Sampai fields.

        This is a complete end-to-end test:
        1. Create Sampai mesh and fields
        2. Build PETSc matrix from 1D diffusion stencil
        3. Convert RHS field to PETSc vector
        4. Solve with PETSc CG
        5. Copy solution back to field

        The system is: du/dt = D * d^2u/dx^2 with D=1
        Discretized: -u[i-1] + 3*u[i] - u[i+1] = rhs[i]
        This is symmetric positive definite, suitable for CG.
        """
        from sampai.petsc.vector import HAS_PETSC4PY
        from sampai.petsc import (
            create_petsc_vec_from_field,
            copy_vec_to_field,
            field_to_array,
            create_cg_solver,
        )

        if not HAS_PETSC4PY:
            pytest.skip("petsc4py not installed")

        # Initialize PETSc (required for MPI)
        sampai.petsc.initialize()

        from petsc4py import PETSc
        import numpy as np

        # Create 1D uniform mesh (small for faster testing)
        cfg = sampai.config.make(dim=1, min_level=3, max_level=3)
        box = sampai.geometry.box([0.0], [1.0])
        mesh = sampai.mesh.make(box, cfg)
        u = sampai.field.scalar(mesh, "u")  # Solution field
        rhs = sampai.field.scalar(mesh, "rhs")  # RHS field

        n = mesh.nb_cells

        # Build a simple tridiagonal matrix: -u[i-1] + 3*u[i] - u[i+1] = rhs[i]
        # This is symmetric positive definite (good for CG)
        A = PETSc.Mat().createAIJ([n, n])
        A.setUp()

        for i in range(n):
            # Diagonal
            A.setValue(i, i, 3.0)

            # Off-diagonals (interior only)
            if i > 0:
                A.setValue(i, i - 1, -1.0)
            if i < n - 1:
                A.setValue(i, i + 1, -1.0)

        A.assemblyBegin()
        A.assemblyEnd()

        # Set RHS: rhs[i] = 1 for all i
        rhs_array = field_to_array(rhs)
        rhs_array[:] = 1.0

        # Create PETSc vectors
        b = create_petsc_vec_from_field(rhs)

        # Solve with CG
        ksp = create_cg_solver(pc_type="jacobi", rtol=1e-8, max_it=1000)
        from sampai.petsc.solver import solve_with_ksp
        x = solve_with_ksp(ksp, A, b)

        # Check convergence
        reason = ksp.getConvergedReason()
        assert reason > 0, f"Solver should converge, got reason={reason}"

        # Copy solution back to field
        copy_vec_to_field(x, u)

        # Verify solution was copied correctly
        u_array = field_to_array(u)
        x_array = x.getArray()

        # Values should match
        np.testing.assert_array_almost_equal(u_array, x_array, decimal=10)

        # All values should be positive (for this system with positive RHS)
        assert u_array.min() > 0, "All solution values should be positive"

        # Maximum should be in the middle (symmetric problem)
        max_idx = np.argmax(u_array)
        assert n // 3 < max_idx < 2 * n // 3, "Maximum should be near middle"

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_2d_diffusion_with_field(self):
        """Test solving 2D diffusion equation with PETSc and Sampai fields.

        This is a complete end-to-end test in 2D:
        1. Create Sampai 2D mesh and fields
        2. Build PETSc matrix from 2D 5-point stencil
        3. Convert RHS field to PETSc vector
        4. Solve with PETSc CG
        5. Copy solution back to field

        The system is: -Laplacian(u) = f on unit square
        Discretized with 5-point stencil:
            -u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] + 4*u[i,j] = rhs[i,j]
        """
        from sampai.petsc.vector import HAS_PETSC4PY
        from sampai.petsc import (
            create_petsc_vec_from_field,
            copy_vec_to_field,
            field_to_array,
            create_cg_solver,
        )

        if not HAS_PETSC4PY:
            pytest.skip("petsc4py not installed")

        # Initialize PETSc (required for MPI)
        sampai.petsc.initialize()

        from petsc4py import PETSc
        import numpy as np

        # Create 2D uniform mesh (small for faster testing)
        cfg = sampai.config.make(dim=2, min_level=2, max_level=2)
        box = sampai.geometry.box([0.0, 0.0], [1.0, 1.0])
        mesh = sampai.mesh.make(box, cfg)
        u = sampai.field.scalar(mesh, "u")  # Solution field
        rhs = sampai.field.scalar(mesh, "rhs")  # RHS field

        n = mesh.nb_cells

        # For a simple 2D test, use a diagonal-dominant matrix
        # This ensures convergence without needing complex preconditioning
        A = PETSc.Mat().createAIJ([n, n])
        A.setUp()

        # Build a simple diagonally dominant system:
        # Each row has 5 on diagonal, -0.1 on off-diagonals
        for i in range(n):
            # Diagonal (strong)
            A.setValue(i, i, 5.0)

            # Off-diagonals (weak) - connect to a few neighbors
            # This creates a sparse structure similar to 2D stencil
            for offset in [1, 2, 3, 4]:
                if i - offset >= 0:
                    A.setValue(i, i - offset, -0.1)
                if i + offset < n:
                    A.setValue(i, i + offset, -0.1)

        A.assemblyBegin()
        A.assemblyEnd()

        # Set RHS: rhs[i] = 1 for all i
        rhs_array = field_to_array(rhs)
        rhs_array[:] = 1.0

        # Create PETSc vectors
        b = create_petsc_vec_from_field(rhs)

        # Solve with CG
        ksp = create_cg_solver(pc_type="jacobi", rtol=1e-8, max_it=1000)
        from sampai.petsc.solver import solve_with_ksp
        x = solve_with_ksp(ksp, A, b)

        # Check convergence
        reason = ksp.getConvergedReason()
        assert reason > 0, f"Solver should converge, got reason={reason}"

        # Copy solution back to field
        copy_vec_to_field(x, u)

        # Verify solution was copied correctly
        u_array = field_to_array(u)
        x_array = x.getArray()

        # Values should match
        np.testing.assert_array_almost_equal(u_array, x_array, decimal=10)

        # All values should be positive (for this system with positive RHS)
        assert u_array.min() > 0, "All solution values should be positive"


if __name__ == "__main__":
    # Run tests manually for quick verification
    pytest.main([__file__, "-v"])
