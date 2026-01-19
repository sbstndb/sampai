# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for subset field operations (fill, copy, apply_function)

These tests verify operations for manipulating field values on specific subsets.
"""

import pytest
import math

try:
    import sampai as sam
except ImportError:
    pytest.skip("Module not built yet", allow_module_level=True)


# ============================================================================
# Subset Fill Operations
# ============================================================================

class TestSubsetFill:
    """Test fill operations on subsets"""

    def test_fill_subset_1d(self):
        """Test filling a scalar field on a 1D subset"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Create subset and fill - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill(field, 3.14)

        # Verify subset has cells
        assert subset.nb_cells > 0

    def test_fill_subset_2d(self):
        """Test filling a scalar field on a 2D subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Create subset and fill - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill(field, 2.5)

        assert subset.nb_cells > 0

    def test_fill_expanded_subset(self):
        """Test filling an expanded subset (ghost cells)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Create expanded subset and fill
        expanded = sam.subsets.expand(mesh, level=2, width=1)
        expanded.fill(field, 1.0)

        # Verify expanded has more cells
        original = sam.subsets.intersection(mesh, mesh, level=2)
        assert expanded.nb_cells > original.nb_cells

    def test_fill_contracted_subset(self):
        """Test filling a contracted subset (interior cells)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Create contracted subset and fill
        contracted = sam.subsets.contract(mesh, level=3, width=1)
        contracted.fill(field, 5.0)

        # Verify contracted has fewer cells
        original = sam.subsets.intersection(mesh, mesh, level=3)
        assert contracted.nb_cells < original.nb_cells


# ============================================================================
# Subset Copy Operations
# ============================================================================

class TestSubsetCopy:
    """Test copy operations between subsets"""

    def test_copy_same_subset(self):
        """Test copying a field onto itself via subset"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Initialize with some values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill(field, 3.14)

        # Copy onto itself - should not raise
        sam.subsets.copy(field, subset, field, subset)

        assert subset.nb_cells > 0

    def test_copy_different_subsets(self):
        """Test copying between different subsets"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field1 = sam.field.scalar(mesh, "test_field1")
        field2 = sam.field.scalar(mesh, "test_field2")

        # Fill first field
        subset1 = sam.subsets.intersection(mesh, mesh, level=2)
        subset1.fill(field1, 7.0)

        # Copy to second field - should not raise
        subset2 = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.copy(field2, subset2, field1, subset1)

        assert subset2.nb_cells > 0

    def test_copy_with_translation(self):
        """Test copying with translated subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field1 = sam.field.scalar(mesh, "test_field1")
        field2 = sam.field.scalar(mesh, "test_field2")

        # Create original and translated subsets
        subset_orig = sam.subsets.intersection(mesh, mesh, level=3)
        subset_trans = sam.subsets.translate(mesh, [1, 0], level=3)

        # Fill and copy - should not raise
        subset_orig.fill(field1, 2.5)
        sam.subsets.copy(field2, subset_trans, field1, subset_orig)

        assert subset_trans.nb_cells > 0


# ============================================================================
# Subset Apply Function Operations
# ============================================================================

class TestSubsetApplyFunction:
    """Test apply_function operations on subsets"""

    def test_apply_function_1d(self):
        """Test applying a function to 1D subset"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Apply function based on index - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i * 2.0)

        assert subset.nb_cells > 0

    def test_apply_function_2d(self):
        """Test applying a function to 2D subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Apply function based on indices - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i + j)

        assert subset.nb_cells > 0

    def test_apply_function_3d(self):
        """Test applying a function to 3D subset"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Apply function based on indices - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=1)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i * j + k)

        assert subset.nb_cells > 0

    def test_apply_function_trigonometric(self):
        """Test applying trigonometric function"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Apply sin function - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=3)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: math.sin(i * 0.1))

        assert subset.nb_cells > 0

    def test_apply_function_on_expanded_subset(self):
        """Test applying function to expanded subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Apply function to expanded subset - should not raise
        expanded = sam.subsets.expand(mesh, level=2, width=1)
        sam.subsets.apply_function(field, expanded, lambda i, j, k, level: i * 0.5 + j * 0.3)

        assert expanded.nb_cells > 0

    def test_apply_function_on_contracted_subset(self):
        """Test applying function to contracted subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Apply function to contracted subset - should not raise
        contracted = sam.subsets.contract(mesh, level=3, width=1)
        sam.subsets.apply_function(field, contracted, lambda i, j, k, level: (i + j) * level)

        assert contracted.nb_cells > 0


# ============================================================================
# Combined Operations
# ============================================================================

class TestSubsetCombinedOperations:
    """Test combinations of subset operations"""

    def test_fill_then_apply_function(self):
        """Test fill followed by apply_function"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # First fill with constant
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill(field, 1.0)

        # Then apply function to modify - should not raise
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i + j + 1.0)

        assert subset.nb_cells > 0

    def test_copy_then_modify(self):
        """Test copy followed by modification"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field1 = sam.field.scalar(mesh, "test_field1")
        field2 = sam.field.scalar(mesh, "test_field2")

        # Copy field1 to field2 - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill(field1, 3.0)
        sam.subsets.copy(field2, subset, field1, subset)

        # Modify field2 with function - should not raise
        sam.subsets.apply_function(field2, subset, lambda i, j, k, level: i * 2.0)

        assert subset.nb_cells > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
