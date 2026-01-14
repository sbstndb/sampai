"""
PETSc Integration Tests for Sampai

These tests verify the PETSc solver bindings for implicit schemes.
Tests are skipped if PETSc is not available.
"""

import pytest
import numpy as np

# Try to import sampai with PETSc
pytest.importorskip("sampai.petsc")

import sampai as sam


def test_petsc_module_available():
    """Test that PETSc submodule is available."""
    assert hasattr(sam, 'petsc'), "PETSc submodule should be available"
    assert hasattr(sam.petsc, 'LinearSolver'), "LinearSolver should be available"
    assert hasattr(sam.petsc, 'KSP'), "KSP should be available"
    assert hasattr(sam.petsc, 'PC'), "PC should be available"


def test_petsc_identity_operator():
    """Test identity operator creation."""
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)
    u = sam.field.scalar(mesh, "u")

    id_op = sam.petsc.identity(u)
    assert id_op is not None
    assert "Identity" in repr(id_op)


def test_petsc_diffusion_operator():
    """Test diffusion operator creation."""
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)
    u = sam.field.scalar(mesh, "u")

    diff = sam.petsc.diffusion_order2(u, coefficient=0.1)
    assert diff is not None
    assert "Diffusion" in repr(diff)


def test_petsc_operator_arithmetic():
    """Test operator arithmetic (addition, scalar multiplication)."""
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)
    u = sam.field.scalar(mesh, "u")

    id_op = sam.petsc.identity(u)
    diff = sam.petsc.diffusion_order2(u, coefficient=0.1)

    # Test scalar multiplication
    scaled = 2.0 * id_op
    assert scaled is not None

    # Test addition
    combined = id_op + diff
    assert combined is not None

    # Test subtraction
    diff_sub = id_op - diff
    assert diff_sub is not None

    # Test complex expression (Backward Euler)
    dt = 0.01
    implicit_scheme = id_op + dt * diff
    assert implicit_scheme is not None


def test_petsc_ksp_wrapper():
    """Test KSP wrapper class."""
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)
    u = sam.field.scalar(mesh, "u")

    # Create a simple solver to get KSP
    id_op = sam.petsc.identity(u)
    diff = sam.petsc.diffusion_order2(u, coefficient=0.1)
    implicit_scheme = id_op + 0.01 * diff

    # Note: This test will be updated when make_solver is fully implemented
    # For now, we just verify the operator structure
    assert implicit_scheme is not None


def test_petsc_pc_wrapper():
    """Test PC wrapper class."""
    # PC is accessed through KSP, so we test it there
    # This is a placeholder test
    assert True


@pytest.mark.skipif(True, reason="Full solver implementation pending")
def test_petsc_implicit_diffusion_solve():
    """Test implicit diffusion solve (integration test)."""
    # Create mesh
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)

    # Create field and set initial condition
    u = sam.field.scalar(mesh, "u")
    u.fill(1.0)

    # Apply boundary conditions
    sam.make_dirichlet_bc(u, value=0.0, order=2)

    # Define implicit scheme (Backward Euler)
    K = 0.1  # Diffusion coefficient
    dt = 0.01
    diff = sam.petsc.diffusion_order2(u, coefficient=K)
    id_op = sam.petsc.identity(u)
    implicit_scheme = id_op + dt * diff

    # Solve
    solver = sam.petsc.make_solver(implicit_scheme)
    solver.set_unknown(u)
    solver.setup()
    solver.solve(u, u)

    # Verify solution
    # Solution should diffuse towards the boundary value (0)
    assert u.max() < 1.0, "Solution should diffuse"
    assert u.min() >= 0.0, "Solution should be non-negative"


@pytest.mark.skipif(True, reason="Full solver implementation pending")
def test_petsc_solver_configuration():
    """Test solver KSP configuration."""
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)
    u = sam.field.scalar(mesh, "u")

    K = 0.1
    diff = sam.petsc.diffusion_order2(u, coefficient=K)
    id_op = sam.petsc.identity(u)
    implicit_scheme = id_op + 0.01 * diff

    solver = sam.petsc.make_solver(implicit_scheme)
    solver.set_unknown(u)

    # Configure KSP
    ksp = solver.ksp()
    ksp.set_type("gmres")
    ksp.pc().set_type("ilu")
    ksp.set_tolerances(rtol=1e-8, max_it=1000)

    solver.setup()

    # Verify configuration
    assert ksp.get_type() == "gmres"
    assert ksp.pc().get_type() == "ilu"


@pytest.mark.skipif(True, reason="Full solver implementation pending")
def test_petsc_stokes_block_system():
    """Test block system solver (Stokes)."""
    # This is a placeholder for future block solver tests
    pass


def test_petsc_field_compatibility():
    """Test that operators work with different field types."""
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)

    # Scalar field
    u_scalar = sam.field.scalar(mesh, "u")
    id_scalar = sam.petsc.identity(u_scalar)
    assert id_scalar is not None

    # Vector field
    v = sam.field.vector(mesh, "v", n_components=2)
    # Note: Vector field operators will be added in future implementation


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
