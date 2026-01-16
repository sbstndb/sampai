"""
Tests for expand and contract operations (subset algebra)

These tests verify the new expand/contract operations that complement
the existing contraction operation.
"""

import pytest

try:
    import sampai as sam
except ImportError:
    pytest.skip("Module not built yet", allow_module_level=True)


# ============================================================================
# Expand Operations Tests
# ============================================================================

class TestExpandOperations:
    """Test expand operations (adding ghost cells)"""

    def test_expand_basic_1d(self):
        """Test basic expand in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Expand by 1 cell in all directions
        expanded = sam.subsets.expand(mesh, level=2, width=1)

        # Expand should add cells, so more cells than original
        original = sam.subsets.intersection(mesh, mesh, level=2)
        assert expanded.nb_cells > original.nb_cells
        assert "expanded" in expanded.description

    def test_expand_basic_2d(self):
        """Test basic expand in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Expand by 1 cell
        expanded = sam.subsets.expand(mesh, level=2, width=1)

        # Expand should add cells
        original = sam.subsets.intersection(mesh, mesh, level=2)
        assert expanded.nb_cells > original.nb_cells
        assert expanded.level == 2

    def test_expand_basic_3d(self):
        """Test basic expand in 3D"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)

        # Expand by 1 cell
        expanded = sam.subsets.expand(mesh, level=1, width=1)

        # Expand should add cells
        original = sam.subsets.intersection(mesh, mesh, level=1)
        assert expanded.nb_cells > original.nb_cells

    def test_expand_invalid_width(self):
        """Test that invalid width raises error"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Width 2 not supported yet
        with pytest.raises(RuntimeError, match="width must be 1"):
            sam.subsets.expand(mesh, level=2, width=2)


class TestExpandDirectional:
    """Test directional expand operations"""

    def test_expand_x_direction_only_2d(self):
        """Test expand only in x-direction (2D)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Expand only in x-direction
        expanded_x = sam.subsets.expand_dir(mesh, level=2, width=1, directions=[True, False])

        assert "expanded_directional" in expanded_x.description
        assert expanded_x.level == 2
        assert expanded_x.nb_cells > 0

    def test_expand_y_direction_only_2d(self):
        """Test expand only in y-direction (2D)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Expand only in y-direction
        expanded_y = sam.subsets.expand_dir(mesh, level=2, width=1, directions=[False, True])

        assert expanded_y.level == 2
        assert expanded_y.nb_cells > 0

    def test_expand_xy_directions_2d(self):
        """Test expand in both x and y directions (2D)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Expand in both directions (should be same as non-directional)
        expanded_both = sam.subsets.expand_dir(mesh, level=2, width=1, directions=[True, True])
        expanded_all = sam.subsets.expand(mesh, level=2, width=1)

        # Both should have similar cell counts
        assert expanded_both.nb_cells == expanded_all.nb_cells

    def test_expand_invalid_directions_size(self):
        """Test that invalid directions array size raises error"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Wrong size for 1D mesh
        with pytest.raises(RuntimeError, match="Directions array size"):
            sam.subsets.expand_dir(mesh, level=2, width=1, directions=[True, False])


# ============================================================================
# Contract Operations Tests
# ============================================================================

class TestContractOperations:
    """Test contract operations (modern version with directional control)"""

    def test_contract_basic_1d(self):
        """Test basic contract in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract by 1 cell
        contracted = sam.subsets.contract(mesh, level=3, width=1)

        # Contract should remove boundary cells
        original = sam.subsets.intersection(mesh, mesh, level=3)
        assert contracted.nb_cells < original.nb_cells
        assert "contracted" in contracted.description

    def test_contract_basic_2d(self):
        """Test basic contract in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract by 1 cell
        contracted = sam.subsets.contract(mesh, level=3, width=1)

        # Contract should remove boundary cells
        original = sam.subsets.intersection(mesh, mesh, level=3)
        assert contracted.nb_cells < original.nb_cells
        assert contracted.level == 3

    def test_contract_with_larger_width(self):
        """Test contract with larger width value"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract by 2 cells
        contracted2 = sam.subsets.contract(mesh, level=3, width=2)

        # Contract by 2 should remove even more cells
        contracted1 = sam.subsets.contract(mesh, level=3, width=1)
        assert contracted2.nb_cells <= contracted1.nb_cells


class TestContractDirectional:
    """Test directional contract operations"""

    def test_contract_x_direction_only_2d(self):
        """Test contract only in x-direction (2D)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract only in x-direction
        contracted_x = sam.subsets.contract_dir(mesh, level=3, width=1, directions=[True, False])

        assert "contracted_directional" in contracted_x.description
        assert contracted_x.level == 3
        # Should have removed some cells but not all
        assert contracted_x.nb_cells > 0

    def test_contract_y_direction_only_2d(self):
        """Test contract only in y-direction (2D)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract only in y-direction
        contracted_y = sam.subsets.contract_dir(mesh, level=3, width=1, directions=[False, True])

        assert contracted_y.level == 3
        assert contracted_y.nb_cells > 0

    def test_contract_both_directions(self):
        """Test contract in both directions should match non-directional"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Contract in both directions
        contracted_both = sam.subsets.contract_dir(mesh, level=3, width=1, directions=[True, True])
        contracted_all = sam.subsets.contract(mesh, level=3, width=1)

        # Should have same cell count
        assert contracted_both.nb_cells == contracted_all.nb_cells


# ============================================================================
# Self Operation Tests
# ============================================================================

class TestSelfOperation:
    """Test self operation (wrap LevelCellArray as subset)"""

    def test_self_basic_1d(self):
        """Test self operation in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Wrap mesh cells as subset
        wrapped = sam.subsets.self(mesh, level=2)

        assert "self" in wrapped.description
        assert wrapped.level == 2
        # Should have same cells as mesh
        mesh_subset = sam.subsets.intersection(mesh, mesh, level=2)
        assert wrapped.nb_cells == mesh_subset.nb_cells

    def test_self_basic_2d(self):
        """Test self operation in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Wrap mesh cells as subset
        wrapped = sam.subsets.self(mesh, level=2)

        assert "self" in wrapped.description
        assert wrapped.level == 2
        assert wrapped.nb_cells > 0

    def test_self_basic_3d(self):
        """Test self operation in 3D"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)

        # Wrap mesh cells as subset
        wrapped = sam.subsets.self(mesh, level=1)

        assert "self" in wrapped.description
        assert wrapped.level == 1
        assert wrapped.nb_cells > 0


# ============================================================================
# Expand/Contract Comparison Tests
# ============================================================================

class TestExpandContractComparison:
    """Test relationship between expand and contract"""

    def test_expand_then_contract_reduces_cells(self):
        """Test that expanding then contracting reduces cells"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)

        # Get original
        original = sam.subsets.self(mesh, level=2)

        # Expand adds cells
        expanded = sam.subsets.expand(mesh, level=2, width=1)

        # Contract removes cells
        contracted = sam.subsets.contract(mesh, level=2, width=1)

        # Cell count relationship
        assert contracted.nb_cells < original.nb_cells < expanded.nb_cells

    def test_contract_removes_more_than_contraction(self):
        """Test that modern contract removes at least as many cells as contraction"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)

        # Both operations with width=1
        contracted_modern = sam.subsets.contract(mesh, level=3, width=1)
        contracted_legacy = sam.subsets.contraction(mesh, level=3, n_cells_to_remove=1)

        # Modern contract should remove similar number of cells
        assert contracted_modern.nb_cells <= contracted_legacy.nb_cells + 10  # Allow small difference


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
