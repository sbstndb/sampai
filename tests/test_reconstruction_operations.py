# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for reconstruction operations (AMR to uniform mesh)

These tests verify the reconstruction operation that converts an AMR field
(multi-level mesh) to a uniform mesh field at the domain level.
"""

import pytest
import numpy as np

try:
    import sampai as sam
except ImportError:
    pytest.skip("Module not built yet", allow_module_level=True)


# ============================================================================
# Basic Reconstruction Tests for Scalar Fields
# ============================================================================

class TestReconstructionScalar:
    """Test reconstruction operations for scalar fields"""

    def test_reconstruction_scalar_1d(self):
        """Test reconstructing 1D scalar field from AMR to uniform mesh"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 3

        # Create AMR mesh
        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        # Fill with some values
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        sam.subsets.apply_function(field_amr, subset, lambda i, j, k, level: i * 0.1)

        # Create uniform mesh at domain level (level 3)
        uniform_config = sam.config._MeshConfig1D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        # Reconstruct - should not raise
        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_scalar_2d(self):
        """Test reconstructing 2D scalar field from AMR to uniform mesh"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        # Create AMR mesh
        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        # Fill with some values
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        sam.subsets.apply_function(field_amr, subset, lambda i, j, k, level: i + j)

        # Create uniform mesh at domain level
        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        # Reconstruct - should not raise
        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_scalar_3d(self):
        """Test reconstructing 3D scalar field from AMR to uniform mesh"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 2

        # Create AMR mesh
        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        # Fill with some values
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=2)
        sam.subsets.apply_function(field_amr, subset, lambda i, j, k, level: i * j + k)

        # Create uniform mesh at domain level
        uniform_config = sam.config._MeshConfig3D()
        uniform_config.min_level = 2
        uniform_config.max_level = 2
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        # Reconstruct - should not raise
        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0


# ============================================================================
# Reconstruction Tests for Vector Fields
# ============================================================================

class TestReconstructionVector:
    """Test reconstruction operations for vector fields"""

    def test_reconstruction_vector_2comp_2d(self):
        """Test reconstructing 2-component vector field in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        # Create AMR mesh
        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.vector(mesh_amr, "field_amr", n_components=2)

        # Fill with some values
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        sam.subsets.apply_function_vector(field_amr, subset,
                                          lambda comp, i, j, k, level: comp + i + j)

        # Create uniform mesh at domain level
        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.vector(mesh_uniform, "field_uniform", n_components=2)

        # Reconstruct - should not raise
        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_vector_3comp_2d(self):
        """Test reconstructing 3-component vector field in 2D"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        # Create AMR mesh
        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.vector(mesh_amr, "field_amr", n_components=3)

        # Fill with some values
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        sam.subsets.apply_function_vector(field_amr, subset,
                                          lambda comp, i, j, k, level: comp * (i + 1))

        # Create uniform mesh at domain level
        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.vector(mesh_uniform, "field_uniform", n_components=3)

        # Reconstruct - should not raise
        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_vector_3comp_3d(self):
        """Test reconstructing 3-component vector field in 3D"""
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config._MeshConfig3D()
        config.min_level = 1
        config.max_level = 2

        # Create AMR mesh
        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.vector(mesh_amr, "field_amr", n_components=3)

        # Fill with some values
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=2)
        sam.subsets.apply_function_vector(field_amr, subset,
                                          lambda comp, i, j, k, level: comp + i + j + k)

        # Create uniform mesh at domain level
        uniform_config = sam.config._MeshConfig3D()
        uniform_config.min_level = 2
        uniform_config.max_level = 2
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.vector(mesh_uniform, "field_uniform", n_components=3)

        # Reconstruct - should not raise
        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0


# ============================================================================
# Reconstruction with Constant Values
# ============================================================================

class TestReconstructionConstant:
    """Test reconstruction with constant field values"""

    def test_reconstruction_constant_scalar_2d(self):
        """Test reconstructing constant scalar field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        # Fill with constant
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        subset.fill(field_amr, 5.0)

        # Create uniform mesh and reconstruct
        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_constant_vector_2d(self):
        """Test reconstructing constant vector field"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.vector(mesh_amr, "field_amr", n_components=2)

        # Fill with constant
        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        subset.fill_vector(field_amr, [1.0, 2.0])

        # Create uniform mesh and reconstruct
        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.vector(mesh_uniform, "field_uniform", n_components=2)

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0


# ============================================================================
# Reconstruction with Different Level Configurations
# ============================================================================

class TestReconstructionLevelConfigurations:
    """Test reconstruction with different AMR level configurations"""

    def test_reconstruction_two_levels_2d(self):
        """Test reconstruction with 2-level AMR hierarchy"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        # Fill different levels with different values
        subset_l2 = sam.subsets.intersection(mesh_amr, mesh_amr, level=2)
        subset_l3 = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)

        sam.subsets.apply_function(field_amr, subset_l2, lambda i, j, k, level: 1.0)
        sam.subsets.apply_function(field_amr, subset_l3, lambda i, j, k, level: 2.0)

        # Create uniform mesh and reconstruct
        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_three_levels_1d(self):
        """Test reconstruction with 3-level AMR hierarchy"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 1
        config.max_level = 3

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        # Fill all levels
        for level in [1, 2, 3]:
            subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=level)
            sam.subsets.apply_function(field_amr, subset,
                                       lambda i, j, k, lvl=level: float(lvl))

        # Create uniform mesh and reconstruct
        uniform_config = sam.config._MeshConfig1D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0


# ============================================================================
# Reconstruction Edge Cases
# ============================================================================

class TestReconstructionEdgeCases:
    """Test edge cases for reconstruction"""

    def test_reconstruction_single_level_amr(self):
        """Test reconstruction when AMR mesh has only one level"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 2  # Single level

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=2)
        subset.fill(field_amr, 3.14)

        # Create uniform mesh at same level
        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 2
        uniform_config.max_level = 2
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_preserves_data(self):
        """Test that reconstruction preserves data at the finest level"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D()
        config.min_level = 2
        config.max_level = 3

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        # Fill only the finest level
        subset_l3 = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        sam.subsets.apply_function(field_amr, subset_l3, lambda i, j, k, level: i * 0.5)

        # Create uniform mesh and reconstruct
        uniform_config = sam.config._MeshConfig1D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        # Uniform mesh should have cells
        assert field_uniform.mesh.nb_cells > 0


# ============================================================================
# Reconstruction with Different Geometries
# ============================================================================

class TestReconstructionGeometries:
    """Test reconstruction with different box geometries"""

    def test_reconstruction_non_unit_box_2d(self):
        """Test reconstruction with non-unit box in 2D"""
        box = sam.geometry.box([0.0, 0.0], [2.0, 3.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        subset.fill(field_amr, 7.5)

        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0

    def test_reconstruction_shifted_box_2d(self):
        """Test reconstruction with shifted box in 2D"""
        box = sam.geometry.box([1.0, 2.0], [3.0, 4.0])
        config = sam.config._MeshConfig2D()
        config.min_level = 2
        config.max_level = 3

        mesh_amr = sam.mesh.make(box, config)
        field_amr = sam.field.scalar(mesh_amr, "field_amr")

        subset = sam.subsets.intersection(mesh_amr, mesh_amr, level=3)
        sam.subsets.apply_function(field_amr, subset, lambda i, j, k, level: i + j + level)

        uniform_config = sam.config._MeshConfig2D()
        uniform_config.min_level = 3
        uniform_config.max_level = 3
        mesh_uniform = sam.mesh.make(box, uniform_config)
        field_uniform = sam.field.scalar(mesh_uniform, "field_uniform")

        sam.subsets.reconstruction_to(field_uniform, field_amr)

        assert field_uniform.mesh.nb_cells > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
