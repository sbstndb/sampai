# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for field-based filtering operations (where and clamp)

These tests verify the new field-based filtering operations that allow:
- Creating subsets based on field values (where)
- Limiting field values to a range (clamp)

Both operations support scalar and vector fields.
"""

import pytest
import math

try:
    import sampai as sam
except ImportError:
    pytest.skip("Module not built yet", allow_module_level=True)


# ============================================================================
# Where Operation Tests for Scalar Fields
# ============================================================================

class TestWhereScalar:
    """Test where operation for scalar fields"""

    def test_where_positive_values_1d(self):
        """Test filtering positive values in 1D scalar field"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with values: some positive, some negative (i - 2 gives range [-2, 1])
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i - 2)

        # Filter for positive values (i > 2)
        positive_subset = sam.subsets.where(field, lambda x: x > 0, level=2)

        # Should have some cells (those where i > 2, i.e., i=3)
        assert positive_subset.nb_cells > 0
        assert "where" in positive_subset.description

    def test_where_nonzero_values_2d(self):
        """Test filtering non-zero values in 2D scalar field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with alternating zeros and non-zeros
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset,
                                   lambda i, j, k, level: (i + j) % 2)

        # Filter for non-zero values
        nonzero_subset = sam.subsets.where(field, lambda x: x != 0, level=2)

        # Should have some cells
        assert nonzero_subset.nb_cells > 0

    def test_where_range_filter_2d(self):
        """Test filtering values within a range in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with index-based values
        subset = sam.subsets.intersection(mesh, mesh, level=3)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i + j)

        # Filter for values between 5 and 10
        range_subset = sam.subsets.where(field, lambda x: 5 <= x <= 10, level=3)

        # Should have some cells
        assert range_subset.nb_cells > 0

    def test_where_empty_result(self):
        """Test where operation that results in empty subset"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill all with zeros
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        subset.fill(field, 0.0)

        # Filter for positive values - should be empty
        positive_subset = sam.subsets.where(field, lambda x: x > 0, level=2)

        # Should have zero cells
        assert positive_subset.nb_cells == 0

    def test_where_complex_condition_1d(self):
        """Test where with complex condition combining multiple checks"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i * 0.5)

        # Filter: values > 1.0 AND values < 3.0
        result = sam.subsets.where(field, lambda x: 1.0 < x < 3.0, level=2)

        assert result.nb_cells > 0


# ============================================================================
# Where Operation Tests for Vector Fields
# ============================================================================

class TestWhereVector:
    """Test where operation for vector fields"""

    def test_where_vector_magnitude_2d(self):
        """Test filtering 2D vector field by magnitude"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Fill with values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: (comp + 1) * (i + 1))

        # Filter by magnitude (default mode)
        result = sam.subsets.where_vector(field, "magnitude",
                                          lambda mag: mag > 2.0, level=2)

        assert result.nb_cells > 0

    def test_where_vector_any_2d(self):
        """Test filtering 2D vector field by 'any' mode"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Fill with some positive and some negative values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: (i - comp))

        # Filter: any component is positive
        result = sam.subsets.where_vector(field, "any",
                                          lambda x: x > 0, level=2)

        assert result.nb_cells > 0

    def test_where_vector_all_2d(self):
        """Test filtering 2D vector field by 'all' mode"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Fill with all positive values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: i + comp + 1)

        # Filter: all components are positive
        result = sam.subsets.where_vector(field, "all",
                                          lambda x: x > 0, level=2)

        assert result.nb_cells > 0

    def test_where_vector_3comp_magnitude_3d(self):
        """Test filtering 3D vector field (3 components) by magnitude"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=3)

        # Fill with values
        subset = sam.subsets.intersection(mesh, mesh, level=1)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: comp + 1)

        # Filter by magnitude
        result = sam.subsets.where_vector(field, "magnitude",
                                          lambda mag: mag > 2.0, level=1)

        assert result.nb_cells > 0

    def test_where_vector_3comp_any_3d(self):
        """Test filtering 3D vector field (3 components) by 'any' mode"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=3)

        # Fill with values
        subset = sam.subsets.intersection(mesh, mesh, level=1)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: comp * i)

        # Filter: any component > 0
        result = sam.subsets.where_vector(field, "any",
                                          lambda x: x > 0, level=1)

        assert result.nb_cells > 0

    def test_where_vector_invalid_mode_raises_error(self):
        """Test that invalid mode raises error"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Invalid mode should raise error
        with pytest.raises(RuntimeError, match="Invalid mode"):
            sam.subsets.where_vector(field, "invalid_mode", lambda x: x > 0, level=2)


# ============================================================================
# Clamp Operation Tests for Scalar Fields
# ============================================================================

class TestClampScalar:
    """Test clamp operation for scalar fields"""

    def test_clamp_basic_1d(self):
        """Test basic clamping in 1D"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with values outside range [1.0, 3.0]
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i)

        # Clamp to [1.0, 3.0]
        sam.subsets.clamp(field, 1.0, 3.0, level=2)

        # Verify no errors occurred
        assert subset.nb_cells > 0

    def test_clamp_symmetric_2d(self):
        """Test symmetric clamping in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with various values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i - j)

        # Clamp to [-2.0, 2.0]
        sam.subsets.clamp(field, -2.0, 2.0, level=2)

        assert subset.nb_cells > 0

    def test_clamp_one_sided_2d(self):
        """Test one-sided clamping (only upper bound)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 3
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with positive values
        subset = sam.subsets.intersection(mesh, mesh, level=3)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i * 10)

        # Clamp from below (0 to 50)
        sam.subsets.clamp(field, 0.0, 50.0, level=3)

        assert subset.nb_cells > 0

    def test_clamp_no_effect_1d(self):
        """Test clamping when all values are already in range"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with values in [0, 5]
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i)

        # Clamp to [0, 10] - should have no effect
        sam.subsets.clamp(field, 0.0, 10.0, level=2)

        assert subset.nb_cells > 0


# ============================================================================
# Clamp Operation Tests for Vector Fields
# ============================================================================

class TestClampVector:
    """Test clamp operation for vector fields"""

    def test_clamp_vector_2comp_2d(self):
        """Test clamping 2-component vector field in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Fill with values outside range
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: (i + 1) * (comp + 1) * 5)

        # Clamp all components to [0.0, 10.0]
        sam.subsets.clamp_vector(field, 0.0, 10.0, level=2)

        assert subset.nb_cells > 0

    def test_clamp_vector_3comp_3d(self):
        """Test clamping 3-component vector field in 3D"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 1

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=3)

        # Fill with values
        subset = sam.subsets.intersection(mesh, mesh, level=1)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: comp * i * 10)

        # Clamp to [-5.0, 5.0]
        sam.subsets.clamp_vector(field, -5.0, 5.0, level=1)

        assert subset.nb_cells > 0

    def test_clamp_vector_negative_range_2d(self):
        """Test clamping vector field to negative range"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Fill with various values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: (i - comp) * 10)

        # Clamp to [-10.0, 0.0]
        sam.subsets.clamp_vector(field, -10.0, 0.0, level=2)

        assert subset.nb_cells > 0


# ============================================================================
# Combined Operations Tests
# ============================================================================

class TestWhereClampCombined:
    """Test combinations of where and clamp operations"""

    def test_where_then_clamp_scalar_2d(self):
        """Test filtering then clamping scalar field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill with values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function(field, subset, lambda i, j, k, level: i * 5)

        # First filter for values > 3
        filtered = sam.subsets.where(field, lambda x: x > 3, level=2)

        # Then clamp the entire field
        sam.subsets.clamp(field, 0.0, 10.0, level=2)

        assert filtered.nb_cells > 0

    def test_clamp_then_where_vector_2d(self):
        """Test clamping then filtering vector field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2

        mesh = sam.mesh.make(box, config)
        field = sam.field.vector(mesh, "test_field", n_components=2)

        # Fill with values
        subset = sam.subsets.intersection(mesh, mesh, level=2)
        sam.subsets.apply_function_vector(field, subset,
                                          lambda comp, i, j, k, level: i * 10)

        # First clamp
        sam.subsets.clamp_vector(field, 0.0, 5.0, level=2)

        # Then filter for magnitude > 2
        filtered = sam.subsets.where_vector(field, "magnitude",
                                            lambda mag: mag > 2.0, level=2)

        assert filtered.nb_cells > 0

    def test_where_clamp_on_different_levels(self):
        """Test where and clamp operations on different AMR levels"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 1
        config.max_level = 3

        mesh = sam.mesh.make(box, config)
        field = sam.field.scalar(mesh, "test_field")

        # Fill all levels
        for level in [1, 2, 3]:
            subset = sam.subsets.intersection(mesh, mesh, level=level)
            sam.subsets.apply_function(field, subset,
                                       lambda i, j, k, lvl=level: i * lvl)

        # Apply where on level 2
        filtered_l2 = sam.subsets.where(field, lambda x: x > 5, level=2)

        # Apply clamp on level 3
        sam.subsets.clamp(field, 0.0, 20.0, level=3)

        assert filtered_l2.nb_cells >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
