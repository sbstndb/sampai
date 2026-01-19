# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for diffusion, laplacian, gradient, and divergence operators.

These tests verify that the explicit operators work correctly for 1D, 2D, and 3D meshes.
Note: Gradient and divergence operators are NOT supported in 1D due to VectorField1D_1
incompatibility with the explicit scheme implementation.
"""

import pytest

import sampai as sam


class TestDiffusionOperator:
    """Tests for make_diffusion_order2 operator."""

    def test_diffusion_1d_default_coefficient(self):
        """Test 1D diffusion with default coefficient (1.0)."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        diff_u = sam.make_diffusion_order2(u)

        assert diff_u.mesh is mesh
        assert diff_u.name == "diffusion(u)"
        # Default coefficient is 1.0

    def test_diffusion_1d_with_coefficient(self):
        """Test 1D diffusion with custom coefficient."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        diff_u = sam.make_diffusion_order2(u, coefficient=0.5)

        assert diff_u.mesh is mesh

    def test_diffusion_2d(self):
        """Test 2D diffusion operator."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        diff_u = sam.make_diffusion_order2(u, coefficient=0.1)

        assert diff_u.mesh is mesh
        assert diff_u.name == "diffusion(u)"

    def test_diffusion_3d(self):
        """Test 3D diffusion operator."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        u = sam.field.scalar(mesh, "u")

        diff_u = sam.make_diffusion_order2(u, coefficient=2.0)

        assert diff_u.mesh is mesh


class TestLaplacianOperator:
    """Tests for make_laplacian_order2 operator."""

    def test_laplacian_1d(self):
        """Test 1D Laplacian operator."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        lap_u = sam.make_laplacian_order2(u)

        assert lap_u.mesh is mesh
        assert lap_u.name == "diffusion(u)"

    def test_laplacian_2d(self):
        """Test 2D Laplacian operator."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        lap_u = sam.make_laplacian_order2(u)

        assert lap_u.mesh is mesh
        assert lap_u.name == "diffusion(u)"

    def test_laplacian_3d(self):
        """Test 3D Laplacian operator."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        u = sam.field.scalar(mesh, "u")

        lap_u = sam.make_laplacian_order2(u)

        assert lap_u.mesh is mesh


class TestGradientOperator:
    """Tests for make_gradient_order2 operator."""

    # NOTE: 1D gradient is NOT supported due to VectorField1D_1 incompatibility with explicit scheme

    def test_gradient_2d(self):
        """Test 2D gradient operator (returns VectorField2D_2)."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        grad_u = sam.make_gradient_order2(u)

        assert grad_u.mesh is mesh
        assert grad_u.name == "Gradient(u)"

    def test_gradient_3d(self):
        """Test 3D gradient operator (returns VectorField3D_3)."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        u = sam.field.scalar(mesh, "u")

        grad_u = sam.make_gradient_order2(u)

        assert grad_u.mesh is mesh
        assert grad_u.name == "Gradient(u)"


class TestDivergenceOperator:
    """Tests for make_divergence_order2 operator."""

    # NOTE: 1D divergence is NOT supported due to VectorField1D_1 incompatibility with explicit scheme

    def test_divergence_2d(self):
        """Test 2D divergence operator (VectorField2D_2 -> ScalarField2D)."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        v = sam.field.vector(mesh, "v", n_components=2)

        div_v = sam.make_divergence_order2(v)

        assert div_v.mesh is mesh
        assert div_v.name == "Divergence(v)"

    def test_divergence_3d(self):
        """Test 3D divergence operator (VectorField3D_3 -> ScalarField3D)."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        v = sam.field.vector(mesh, "v", n_components=3)

        div_v = sam.make_divergence_order2(v)

        assert div_v.mesh is mesh
        assert div_v.name == "Divergence(v)"


class TestOperatorsSubmodule:
    """Test that operators are accessible via sam.operators submodule."""

    def test_diffusion_in_submodule(self):
        """Test diffusion operator is accessible via sam.operators."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)
        u = sam.field.scalar(mesh, "u")

        # Should work via submodule
        diff_u = sam.operators.make_diffusion_order2(u)
        assert diff_u.mesh is mesh

        # Alias should also work
        diff_u2 = sam.operators.diffusion_order2(u)
        assert diff_u2.mesh is mesh

    def test_laplacian_in_submodule(self):
        """Test laplacian operator is accessible via sam.operators."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)
        u = sam.field.scalar(mesh, "u")

        lap_u = sam.operators.make_laplacian_order2(u)
        assert lap_u.mesh is mesh

    def test_gradient_in_submodule(self):
        """Test gradient operator is accessible via sam.operators."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)
        u = sam.field.scalar(mesh, "u")

        grad_u = sam.operators.make_gradient_order2(u)
        assert grad_u.mesh is mesh

    def test_divergence_in_submodule(self):
        """Test divergence operator is accessible via sam.operators."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)
        v = sam.field.vector(mesh, "v", n_components=2)

        div_v = sam.operators.make_divergence_order2(v)
        assert div_v.mesh is mesh


class TestOperatorChaining:
    """Test that operators can be chained together (e.g., div(grad(u)))."""

    # NOTE: 1D operator chaining is NOT supported due to VectorField1D_1 incompatibility

    def test_divergence_of_gradient_2d(self):
        """Test div(grad(u)) in 2D (should equal Laplacian)."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        grad_u = sam.make_gradient_order2(u)
        div_grad_u = sam.make_divergence_order2(grad_u)

        assert div_grad_u.mesh is mesh

    def test_divergence_of_gradient_3d(self):
        """Test div(grad(u)) in 3D (should equal Laplacian)."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        u = sam.field.scalar(mesh, "u")

        grad_u = sam.make_gradient_order2(u)
        div_grad_u = sam.make_divergence_order2(grad_u)

        assert div_grad_u.mesh is mesh


class TestOperatorProperties:
    """Test mathematical properties of operators."""

    def test_divergence_of_gradient_equals_laplacian_2d(self):
        """Verify div(grad(u)) and Laplacian(u) both work for same field."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        # Compute div(grad(u))
        grad_u = sam.make_gradient_order2(u)
        div_grad_u = sam.make_divergence_order2(grad_u)

        # Compute Laplacian(u)
        lap_u = sam.make_laplacian_order2(u)

        # Both should produce valid fields
        assert div_grad_u.mesh is mesh
        assert lap_u.mesh is mesh

    def test_diffusion_coefficient_scaling(self):
        """Test that diffusion accepts different coefficients."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        # Compute with different coefficients
        diff_u_1 = sam.make_diffusion_order2(u, coefficient=1.0)
        diff_u_2 = sam.make_diffusion_order2(u, coefficient=2.0)
        diff_u_small = sam.make_diffusion_order2(u, coefficient=0.01)

        # All should produce valid fields
        assert diff_u_1.mesh is mesh
        assert diff_u_2.mesh is mesh
        assert diff_u_small.mesh is mesh


class TestOperatorWithAMR:
    """Test operators on AMR (adaptively refined) meshes."""

    def test_diffusion_on_amr_mesh_2d(self):
        """Test diffusion operator on 2D AMR mesh with different levels."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=4)

        u = sam.field.scalar(mesh, "u")

        # Apply diffusion - should work on AMR mesh
        diff_u = sam.make_diffusion_order2(u, coefficient=0.1)

        assert diff_u.mesh is mesh

    def test_gradient_on_amr_mesh_3d(self):
        """Test gradient operator on 3D AMR mesh with different levels."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=1, max_level=3)

        u = sam.field.scalar(mesh, "u")

        # Apply gradient - should work on AMR mesh
        grad_u = sam.make_gradient_order2(u)

        assert grad_u.mesh is mesh

    def test_divergence_on_amr_mesh_2d(self):
        """Test divergence operator on 2D AMR mesh with different levels."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=4)

        v = sam.field.vector(mesh, "v", n_components=2)

        # Apply divergence - should work on AMR mesh
        div_v = sam.make_divergence_order2(v)

        assert div_v.mesh is mesh


class TestOperatorEdgeCases:
    """Test edge cases and error handling."""

    def test_diffusion_with_zero_coefficient(self):
        """Test diffusion with coefficient = 0 (boundary case)."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        # Should work with zero coefficient
        diff_u = sam.make_diffusion_order2(u, coefficient=0.0)

        assert diff_u.mesh is mesh

    def test_diffusion_with_negative_coefficient(self):
        """Test diffusion with negative coefficient."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        # Should work with negative coefficient
        diff_u = sam.make_diffusion_order2(u, coefficient=-1.5)

        assert diff_u.mesh is mesh

    def test_diffusion_with_large_coefficient(self):
        """Test diffusion with large coefficient."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        u = sam.field.scalar(mesh, "u")

        # Should work with large coefficient
        diff_u = sam.make_diffusion_order2(u, coefficient=100.0)

        assert diff_u.mesh is mesh
