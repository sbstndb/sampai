"""
Tests for subset field operations on vector fields (fill, copy, apply_function)

These tests verify operations for manipulating vector field values on specific subsets.
"""

import pytest
import math

try:
    import sampai as sam
except ImportError:
    pytest.skip("Module not built yet", allow_module_level=True)


# ============================================================================
# Subset Fill Operations for Vector Fields
# ============================================================================

class TestSubsetFillVector:
    """Test fill operations on vector fields"""

    def test_fill_vector_2d_2comp(self):
        """Test filling a 2-component vector field on a 2D subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Create subset and fill - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill_vector(field, [1.5, 2.5])

        assert subset.nb_cells > 0

    def test_fill_vector_3d_3comp(self):
        """Test filling a 3-component vector field on a 3D subset"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=3)

        # Create subset and fill - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=1)
        subset.fill_vector(field, [1.0, 2.0, 3.0])

        assert subset.nb_cells > 0

    def test_fill_vector_expanded_subset(self):
        """Test filling an expanded subset with vector field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Create expanded subset and fill
        expanded = sam.subsets.expand(mesh, level=2, width=1)
        expanded.fill_vector(field, [0.5, 1.5])

        assert expanded.nb_cells > 0

    def test_fill_vector_wrong_size_raises_error(self):
        """Test that wrong value size raises error"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        subset = sam.subsets.intersection(mesh, mesh, level=2)

        # Wrong size (3 instead of 2)
        with pytest.raises(RuntimeError, match="Value size must match"):
            subset.fill_vector(field, [1.0, 2.0, 3.0])


# ============================================================================
# Subset Copy Operations for Vector Fields
# ============================================================================

class TestSubsetCopyVector:
    """Test copy operations between vector fields"""

    def test_copy_vector_2comp_same_subset(self):
        """Test copying a 2-component vector field onto itself via subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Initialize with some values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill_vector(field, [1.0, 2.0])

        # Copy onto itself - should not raise
        sam.subsets.copy_vector(field, subset, field, subset)

        assert subset.nb_cells > 0

    def test_copy_vector_3comp_different_subsets(self):
        """Test copying 3-component vector fields between different subsets"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field1 = sam.field.vector(mesh, "test_field1", n_components=3)
        field2 = sam.field.vector(mesh, "test_field2", n_components=3)

        # Fill first field
        subset1 = sam.subsets.intersection(mesh, mesh, level=2)
        subset1.fill_vector(field1, [1.0, 2.0, 3.0])

        # Copy to second field - should not raise
        subset2 = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.copy_vector(field2, subset2, field1, subset1)

        assert subset2.nb_cells > 0

    def test_copy_vector_with_translation(self):
        """Test copying vector fields with translated subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field1 = sam.field.vector(mesh, "test_field1", n_components=2)
        field2 = sam.field.vector(mesh, "test_field2", n_components=2)

        # Create original and translated subsets
        subset_orig = sam.subsets.intersection(mesh, mesh, level=3)
        subset_trans = sam.subsets.translate(mesh, [1, 0], level=3)

        # Fill and copy - should not raise
        subset_orig.fill_vector(field1, [2.0, 3.0])
        sam.subsets.copy_vector(field2, subset_trans, field1, subset_orig)

        assert subset_trans.nb_cells > 0


# ============================================================================
# Subset Apply Function Operations for Vector Fields
# ============================================================================

class TestSubsetApplyFunctionVector:
    """Test apply_function_vector operations on subsets"""

    def test_apply_function_vector_2d_2comp(self):
        """Test applying a function to 2D 2-component vector field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Apply function based on indices - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: comp * 1.0 + i * 0.1)

        assert subset.nb_cells > 0

    def test_apply_function_vector_3d_3comp(self):
        """Test applying a function to 3D 3-component vector field"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=3)

        # Apply function based on indices - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=1)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: comp + i + j + k)

        assert subset.nb_cells > 0

    def test_apply_function_vector_trigonometric(self):
        """Test applying trigonometric function to vector field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Apply sin function - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=3)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: comp * math.sin(i * 0.1) + math.cos(j * 0.1))

        assert subset.nb_cells > 0

    def test_apply_function_vector_on_expanded_subset(self):
        """Test applying function to vector field on expanded subset"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Apply function to expanded subset - should not raise
        expanded = sam.subsets.expand(mesh, level=2, width=1)
        sam.subsets.apply_function_vector(field, expanded,
                                          lambda comp, i, j, k, level: (comp + 1) * (i + j) * 0.5)

        assert expanded.nb_cells > 0


# ============================================================================
# Combined Vector Field Operations
# ============================================================================

class TestSubsetCombinedVectorOperations:
    """Test combinations of subset operations on vector fields"""

    def test_fill_vector_then_apply_function(self):
        """Test fill followed by apply_function on vector field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=3)

        # First fill with constant
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill_vector(field, [1.0, 2.0, 3.0])

        # Then apply function to modify - should not raise
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: comp + i + j)

        assert subset.nb_cells > 0

    def test_copy_vector_then_modify(self):
        """Test copy followed by modification on vector fields"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field1 = sam.field.vector(mesh, "test_field1", n_components=2)
        field2 = sam.field.vector(mesh, "test_field2", n_components=2)

        # Copy field1 to field2 - should not raise
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill_vector(field1, [1.0, 2.0])
        sam.subsets.copy_vector(field2, subset, field1, subset)

        # Modify field2 with function - should not raise
        sam.subsets.apply_function_vector(field2, subset,
                                          lambda comp, i, j, k, level: comp * i + j)

        assert subset.nb_cells > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
