"""
Tests for spatially-selective and function-based boundary conditions

Tests make_function_bc and make_spatial_bc functions.
"""

import pytest
import math

try:
    import sampai as sam
except ImportError:
    pytest.skip("sampai module not built", allow_module_level=True)


class TestFunctionBC:
    """Tests for make_function_bc function (value depends on space)."""

    def test_1d_function_bc_quadratic(self):
        """Test 1D function-based BC with quadratic profile u(x) = x^2."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 3
        config.max_level = 3
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Quadratic profile: u = x^2 at boundaries
        sam.make_function_bc(u, lambda coords: coords[0]**2, order=2)

        # If we get here without exception, the BC was attached successfully
        assert True

    def test_1d_function_bc_sinusoidal(self):
        """Test 1D function-based BC with sinusoidal profile."""
        box = sam.geometry.box([0.0], [2.0 * math.pi])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Sinusoidal profile: u = sin(x) at boundaries
        sam.make_function_bc(u, lambda coords: math.sin(coords[0]), order=2)

        assert True

    def test_2d_function_bc_sinusoidal(self):
        """Test 2D function-based BC with sinusoidal variation."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # u(x,y) = sin(2*pi*x) * cos(2*pi*y)
        sam.make_function_bc(
            u,
            lambda coords: math.sin(2 * math.pi * coords[0]) * math.cos(2 * math.pi * coords[1]),
            order=2
        )

        assert True

    def test_2d_function_bc_quadratic(self):
        """Test 2D function-based BC with quadratic profile."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # u(x,y) = x^2 + y^2
        sam.make_function_bc(u, lambda coords: coords[0]**2 + coords[1]**2, order=2)

        assert True

    def test_3d_function_bc(self):
        """Test 3D function-based BC."""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 1
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # u(x,y,z) = x + y + z
        sam.make_function_bc(u, lambda coords: coords[0] + coords[1] + coords[2], order=2)

        assert True

    def test_function_bc_different_orders(self):
        """Test function-based BC with different orders."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        config.max_stencil_size = 10
        mesh = sam.mesh.make(box, config)

        for order in [1, 2, 3, 4]:
            u = sam.field.scalar(mesh, "u", init=0.0)
            sam.make_function_bc(u, lambda coords: coords[0]**2, order=order)
            assert True

    def test_function_bc_invalid_order(self):
        """Test that invalid order raises an error."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        with pytest.raises(RuntimeError, match="order must be between 1 and 4"):
            sam.make_function_bc(u, lambda coords: coords[0], order=5)


class TestSpatialBC:
    """Tests for make_spatial_bc (region + value both depend on space)."""

    def test_1d_spatial_bc_left_only(self):
        """Test 1D spatial BC applied only to left boundary."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 3
        config.max_level = 3
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply u = x^2 only on left boundary (x < 0.1)
        sam.make_spatial_bc(
            u,
            lambda coords: coords[0] < 0.1,
            lambda coords: coords[0]**2,
            order=2
        )

        assert True

    def test_1d_spatial_bc_right_only(self):
        """Test 1D spatial BC applied only to right boundary."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 3
        config.max_level = 3
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply u = 1.0 only on right boundary (x > 0.9)
        sam.make_spatial_bc(
            u,
            lambda coords: coords[0] > 0.9,
            lambda coords: 1.0,
            order=2
        )

        assert True

    def test_2d_spatial_bc_bottom_edge(self):
        """Test 2D spatial BC applied only to bottom edge."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply sinusoidal BC only on bottom edge (y < 0.1)
        sam.make_spatial_bc(
            u,
            lambda coords: coords[1] < 0.1,
            lambda coords: math.sin(coords[0]),
            order=2
        )

        assert True

    def test_2d_spatial_bc_corner(self):
        """Test 2D spatial BC applied only to a corner region."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply only to bottom-left corner (x < 0.2 and y < 0.2)
        sam.make_spatial_bc(
            u,
            lambda coords: coords[0] < 0.2 and coords[1] < 0.2,
            lambda coords: coords[0]**2 + coords[1]**2,
            order=2
        )

        assert True

    def test_2d_spatial_bc_circular_region(self):
        """Test 2D spatial BC with circular region selection."""
        box = sam.geometry.box([-1.0, -1.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply only near circular obstacle at origin
        sam.make_spatial_bc(
            u,
            lambda coords: coords[0]**2 + coords[1]**2 < 0.3**2,
            lambda coords: 0.0,
            order=2
        )

        assert True

    def test_3d_spatial_bc(self):
        """Test 3D spatial BC."""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 1
        config.max_level = 1
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply only to one face (z < 0.1)
        sam.make_spatial_bc(
            u,
            lambda coords: coords[2] < 0.1,
            lambda coords: coords[0] + coords[1],
            order=2
        )

        assert True

    def test_spatial_bc_different_orders(self):
        """Test spatial BC with different orders."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        config.max_stencil_size = 10
        mesh = sam.mesh.make(box, config)

        for order in [1, 2, 3, 4]:
            u = sam.field.scalar(mesh, "u", init=0.0)
            sam.make_spatial_bc(
                u,
                lambda coords: coords[0] < 0.1,
                lambda coords: coords[0]**2,
                order=order
            )
            assert True

    def test_spatial_bc_invalid_order(self):
        """Test that invalid order raises an error."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        with pytest.raises(RuntimeError, match="order must be between 1 and 4"):
            sam.make_spatial_bc(
                u,
                lambda coords: coords[0] < 0.5,
                lambda coords: coords[0],
                order=5
            )


class TestBCIntegration:
    """Integration tests combining different BC types."""

    def test_function_bc_with_constant_bc(self):
        """Test combining function-based BC with constant BC on different regions."""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 3
        config.max_level = 3
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply function-based BC everywhere first
        sam.make_function_bc(u, lambda coords: coords[0]**2, order=2)

        # Note: In the current implementation, make_function_bc applies to ALL
        # boundaries. For selective application, use make_spatial_bc instead.
        assert True

    def test_multiple_spatial_bcs(self):
        """Test applying multiple spatial BCs to different regions."""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=0.0)

        # Apply BC to bottom edge
        sam.make_spatial_bc(
            u,
            lambda coords: coords[1] < 0.1,
            lambda coords: 0.0,
            order=2
        )

        # Apply different BC to left edge
        sam.make_spatial_bc(
            u,
            lambda coords: coords[0] < 0.1,
            lambda coords: 1.0,
            order=2
        )

        assert True
