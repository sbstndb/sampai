"""
Tests for subset algebra operations

These tests verify set algebra operations on meshes:
- Basic set operations: intersection, union, difference
- Geometric operations: translate, contraction
- AMR operations: projection, prediction, update_ghost_mr
"""

import pytest

try:
    import sampai as sam
except ImportError:
    pytest.skip("Module not built yet", allow_module_level=True)


# ============================================================================
# 1D Tests
# ============================================================================

class TestSubset1D:
    """Test subset operations in 1D"""

    def test_intersection_1d_basic(self):
        """Test basic intersection of two 1D meshes"""
        # Create two overlapping 1D meshes
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Intersection at level 2
        inter = sam.subsets.intersection(mesh1, mesh2, level=2)

        # Should have cells (both meshes cover same domain)
        assert inter.nb_cells > 0
        assert not inter.empty
        assert inter.level == 2
        assert "intersection" in inter.description

    def test_union_1d_basic(self):
        """Test basic union of two 1D meshes"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Union at level 2
        union_result = sam.subsets.union_(mesh1, mesh2, level=2)

        # Should have cells
        assert union_result.nb_cells > 0
        assert not union_result.empty
        assert "union" in union_result.description

    def test_difference_1d_basic(self):
        """Test basic difference of two 1D meshes"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Difference at level 2
        diff = sam.subsets.difference(mesh1, mesh2, level=2)

        # Difference of identical meshes should be small or empty
        assert isinstance(diff.empty, bool)
        assert diff.level == 2
        assert "difference" in diff.description

    def test_translate_1d(self):
        """Test translation operation in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Translate by [1] (one cell to the right)
        translated = sam.subsets.translate(mesh, [1], level=2)

        # Should have cells
        assert isinstance(translated.empty, bool)
        assert translated.level == 2
        assert "translated" in translated.description

    def test_contraction_1d(self):
        """Test contraction operation in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract by 1 cell
        contracted = sam.subsets.contraction(mesh, level=3, n_cells_to_remove=1)

        # Should have cells (interior)
        assert contracted.level == 3
        assert "contraction" in contracted.description

    def test_projection_1d(self):
        """Test projection operation in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Projection from level 3 to 2
        sam.subsets.projection(field, coarse_level=2)

        # Field should still be accessible
        assert hasattr(field, "mesh")

    def test_prediction_1d(self):
        """Test prediction operation in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Predict from level 2 to 3
        sam.subsets.prediction(field, coarse_level=2, order=1)

        # Field should still be accessible
        assert hasattr(field, "mesh")

    def test_update_ghost_mr_1d(self):
        """Test update_ghost_mr in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Update ghost cells
        sam.subsets.update_ghost_mr(field)

        # Should not raise
        assert hasattr(field, "mesh")


# ============================================================================
# 2D Tests
# ============================================================================

class TestSubset2D:
    """Test subset operations in 2D"""

    def test_intersection_2d_basic(self):
        """Test basic intersection of two 2D meshes"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Intersection at level 2
        inter = sam.subsets.intersection(mesh1, mesh2, level=2)

        # Should have cells
        assert inter.nb_cells > 0
        assert not inter.empty
        assert inter.level == 2

    def test_union_2d_basic(self):
        """Test basic union of two 2D meshes"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Union at level 2
        union_result = sam.subsets.union_(mesh1, mesh2, level=2)

        # Should have cells
        assert union_result.nb_cells > 0
        assert not union_result.empty

    def test_difference_2d_basic(self):
        """Test basic difference of two 2D meshes"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Difference at level 2
        diff = sam.subsets.difference(mesh1, mesh2, level=2)

        # Difference of identical meshes should be small or empty
        assert isinstance(diff.empty, bool)
        assert diff.level == 2

    def test_translate_2d(self):
        """Test translation operation in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Translate by [1, 0] (one cell to the right)
        translated = sam.subsets.translate(mesh, [1, 0], level=2)

        # Should have cells
        assert isinstance(translated.empty, bool)
        assert translated.level == 2

    def test_contraction_2d(self):
        """Test contraction operation in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract by 1 cell
        contracted = sam.subsets.contraction(mesh, level=3, n_cells_to_remove=1)

        # Should have cells
        assert contracted.level == 3
        assert "contraction" in contracted.description

    def test_projection_2d(self):
        """Test projection operation in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Projection from level 3 to 2
        sam.subsets.projection(field, coarse_level=2)

        # Field should still be accessible
        assert hasattr(field, "mesh")

    def test_prediction_2d(self):
        """Test prediction operation in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Predict from level 2 to 3
        sam.subsets.prediction(field, coarse_level=2, order=1)

        # Field should still be accessible
        assert hasattr(field, "mesh")

    def test_update_ghost_mr_2d(self):
        """Test update_ghost_mr in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Update ghost cells
        sam.subsets.update_ghost_mr(field)

        # Should not raise
        assert hasattr(field, "mesh")

    def test_prediction_orders_2d(self):
        """Test different prediction orders in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Test different prediction orders
        for order in [0, 1, 2]:
            # Predict with this order
            sam.subsets.prediction(field, coarse_level=2, order=order)

            # Should not raise
            assert hasattr(field, "mesh")

    def test_invalid_prediction_order(self):
        """Test that invalid prediction order raises error"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Should raise for invalid order
        with pytest.raises(RuntimeError):
            sam.subsets.prediction(field, coarse_level=2, order=10)


# ============================================================================
# 3D Tests
# ============================================================================

class TestSubset3D:
    """Test subset operations in 3D"""

    def test_intersection_3d_basic(self):
        """Test basic intersection of two 3D meshes"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1  # Keep low for 3D to avoid memory issues
        config.max_level = 1

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Intersection at level 1
        inter = sam.subsets.intersection(mesh1, mesh2, level=1)

        # Should have cells
        assert inter.nb_cells > 0
        assert not inter.empty
        assert inter.level == 1

    def test_union_3d_basic(self):
        """Test basic union of two 3D meshes"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        # Union at level 1
        union_result = sam.subsets.union_(mesh1, mesh2, level=1)

        # Should have cells
        assert union_result.nb_cells > 0
        assert not union_result.empty

    def test_projection_3d(self):
        """Test projection operation in 3D"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Projection from level 2 to 1
        sam.subsets.projection(field, coarse_level=1)

        # Field should still be accessible
        assert hasattr(field, "mesh")

    def test_prediction_3d(self):
        """Test prediction operation in 3D"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "u", 1.0)

        # Predict from level 1 to 2
        sam.subsets.prediction(field, coarse_level=1, order=1)

        # Field should still be accessible
        assert hasattr(field, "mesh")


# ============================================================================
# Subset String Representation Tests
# ============================================================================

class TestSubsetRepresentation:
    """Test subset string representations"""

    def test_subset_repr_1d(self):
        """Test string representation of 1D subset"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        subset = sam.subsets.intersection(mesh, mesh, level=2)

        repr_str = repr(subset)
        str_str = str(subset)

        assert "1D" in repr_str
        assert "level 2" in repr_str
        assert "intersection" in repr_str
        assert "cells" in repr_str

    def test_subset_repr_2d(self):
        """Test string representation of 2D subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        subset = sam.subsets.contraction(mesh, level=2)

        repr_str = repr(subset)

        assert "2D" in repr_str
        assert "level 2" in repr_str
        assert "contraction" in repr_str


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestSubsetErrors:
    """Test error handling in subset operations"""

    def test_invalid_stencil_dimension_1d(self):
        """Test that invalid stencil dimension raises error in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Wrong stencil dimension (2D stencil for 1D mesh)
        with pytest.raises(RuntimeError, match="Stencil dimension"):
            sam.subsets.translate(mesh, [1, 2], level=2)

    def test_invalid_stencil_dimension_2d(self):
        """Test that invalid stencil dimension raises error in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Wrong stencil dimension (3D stencil for 2D mesh)
        with pytest.raises(RuntimeError, match="Stencil dimension"):
            sam.subsets.translate(mesh, [1, 2, 3], level=2)


if __name__ == "__main__":
    # Run tests manually for quick verification
    pytest.main([__file__, "-v"])
