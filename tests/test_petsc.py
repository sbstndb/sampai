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
            # PETSc should be initialized after calling get_version
            # (which auto-initializes)
            assert is_init is True

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_initialize_function(self):
        """Test explicit PETSc initialization."""
        if PETSC_ENABLED:
            # Finalize first if already initialized
            try:
                petsc.finalize()
            except RuntimeError:
                pass

            # Now initialize
            petsc.initialize()
            assert petsc.is_initialized() is True
        else:
            with pytest.raises(RuntimeError, match="PETSc support is not enabled"):
                petsc.initialize()

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    def test_finalize_function(self):
        """Test PETSc finalization."""
        if PETSC_ENABLED:
            # Make sure we're initialized
            if not petsc.is_initialized():
                petsc.initialize()

            # Finalize
            petsc.finalize()
            assert petsc.is_initialized() is False

            # Re-initialize for other tests
            petsc.initialize()
        else:
            with pytest.raises(RuntimeError, match="PETSc support is not enabled"):
                petsc.finalize()


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
        from sampai.geometry import box

        # Create a simple box
        b = box([0.0, 0.0], [1.0, 1.0])

        # PETSc should still work
        assert petsc.is_initialized() or petsc.get_world_size() >= 1

    @pytest.mark.skipif(not HAS_SAMPAI, reason="sampai not installed")
    @pytest.mark.skipif(not PETSC_ENABLED, reason="PETSc not enabled")
    def test_petsc_with_mesh_config(self):
        """Test that PETSc works alongside mesh configuration."""
        from sampai.config import MeshConfig

        # Create a mesh config
        cfg = MeshConfig()
        cfg.level = 4

        # PETSc should still work
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


if __name__ == "__main__":
    # Run tests manually for quick verification
    pytest.main([__file__, "-v"])
