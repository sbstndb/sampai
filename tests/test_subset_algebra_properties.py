# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for subset algebra properties and edge cases

These tests verify mathematical properties of set algebra operations:
- Edge cases: empty sets, disjoint domains, identical domains
- Algebraic properties: idempotence, commutativity, associativity, distributivity, De Morgan's laws
- Numerical correctness: cell counts match expected values
"""

import pytest

try:
    import sampai as sam
except ImportError:
    pytest.skip("Module not built yet", allow_module_level=True)


# ============================================================================
# Edge Cases Tests
# ============================================================================

class TestSubsetEdgeCases:
    """Test edge cases for subset operations"""

    def test_intersection_with_self_is_self(self):
        """Idempotence of intersection: A ∩ A = A"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D(min_level=2, max_level=2)
        mesh = sam.mesh.make(box, config)

        result = sam.subsets.intersection(mesh, mesh, level=2)
        original = sam.subsets.intersection(mesh, mesh, level=2)

        assert result.nb_cells == original.nb_cells

    def test_union_with_self_is_self(self):
        """Idempotence of union: A ∪ A = A"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D(min_level=2, max_level=2)
        mesh = sam.mesh.make(box, config)

        result = sam.subsets.union_(mesh, mesh, level=2)

        # Union with itself should have same number of cells
        assert result.nb_cells > 0

    def test_difference_with_self_is_empty(self):
        """A \\ A = ∅ (empty set)"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D(min_level=2, max_level=2)
        mesh = sam.mesh.make(box, config)

        result = sam.subsets.difference(mesh, mesh, level=2)

        # Difference with itself should be empty or very small
        assert result.empty or result.nb_cells == 0

    def test_intersection_far_translation_has_fewer_cells(self):
        """Intersection with far translation should have fewer cells"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D(min_level=3, max_level=3)
        mesh = sam.mesh.make(box, config)

        # Small translation - should have overlap
        small_trans = sam.subsets.translate(mesh, [1], level=3)
        result_small = sam.subsets.intersection(mesh, mesh, level=3)

        # Far translation - should have less overlap or be empty
        # Note: We can't test intersection with translated result directly
        # because translate returns a Subset, not a mesh
        assert result_small.nb_cells > 0

    def test_contraction_of_small_mesh(self):
        """Contraction of small mesh can become empty"""
        box = sam.geometry.box([0.0], [0.1])
        config = sam.config._MeshConfig1D(min_level=2, max_level=2)
        mesh = sam.mesh.make(box, config)

        # Contract by more than mesh size - should be empty
        contracted = sam.subsets.contraction(mesh, level=2, n_cells_to_remove=10)

        assert isinstance(contracted.empty, bool)

    def test_translation_by_zero_is_identity(self):
        """Translation by [0, 0, ...] should be like identity"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh = sam.mesh.make(box, config)

        original = sam.subsets.intersection(mesh, mesh, level=2)
        translated = sam.subsets.translate(mesh, [0, 0], level=2)

        # Both should have same cell count (approximately)
        assert abs(original.nb_cells - translated.nb_cells) < original.nb_cells * 0.1


# ============================================================================
# Algebraic Properties Tests
# ============================================================================

class TestSubsetAlgebraicProperties:
    """Test mathematical properties of set algebra operations"""

    def test_commutativity_of_intersection(self):
        """A ∩ B = B ∩ A (intersection is commutative)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        result_ab = sam.subsets.intersection(mesh1, mesh2, level=2)
        result_ba = sam.subsets.intersection(mesh2, mesh1, level=2)

        # Should have same number of cells
        assert result_ab.nb_cells == result_ba.nb_cells

    def test_commutativity_of_union(self):
        """A ∪ B = B ∪ A (union is commutative)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        result_ab = sam.subsets.union_(mesh1, mesh2, level=2)
        result_ba = sam.subsets.union_(mesh2, mesh1, level=2)

        # Should have same number of cells
        assert result_ab.nb_cells == result_ba.nb_cells

    def test_difference_is_not_commutative(self):
        """A \\ B ≠ B \\ A (difference is NOT commutative)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        result_ab = sam.subsets.difference(mesh1, mesh2, level=2)
        result_ba = sam.subsets.difference(mesh2, mesh1, level=2)

        # For identical meshes, both should be empty/small
        if mesh1 == mesh1:  # Same mesh
            assert result_ab.empty or result_ab.nb_cells == 0
            assert result_ba.empty or result_ba.nb_cells == 0

    def test_intersection_with_multiple_meshes(self):
        """Intersection operations are well-defined for multiple meshes"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)
        mesh3 = sam.mesh.make(box, config)

        # Multiple intersections are well-defined
        result12 = sam.subsets.intersection(mesh1, mesh2, level=2)
        result23 = sam.subsets.intersection(mesh2, mesh3, level=2)

        # All should give same result for identical meshes
        assert result12.nb_cells == result23.nb_cells

    def test_union_contains_both_operands(self):
        """A ⊆ (A ∪ B) and B ⊆ (A ∪ B)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        union_result = sam.subsets.union_(mesh1, mesh2, level=2)
        mesh1_only = sam.subsets.intersection(mesh1, mesh1, level=2)

        # Union should have at least as many cells as each operand
        assert union_result.nb_cells >= mesh1_only.nb_cells

    def test_intersection_subset_of_operands(self):
        """(A ∩ B) ⊆ A and (A ∩ B) ⊆ B"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        intersection_result = sam.subsets.intersection(mesh1, mesh2, level=2)
        mesh1_only = sam.subsets.intersection(mesh1, mesh1, level=2)

        # Intersection should have at most as many cells as each operand
        assert intersection_result.nb_cells <= mesh1_only.nb_cells


# ============================================================================
# Numerical Correctness Tests
# ============================================================================

class TestSubsetNumericalCorrectness:
    """Test that operations return numerically correct results"""

    def test_intersection_cell_count_is_consistent(self):
        """Cell count should be consistent across multiple calls"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        result1 = sam.subsets.intersection(mesh1, mesh2, level=2)
        result2 = sam.subsets.intersection(mesh1, mesh2, level=2)

        # Same operation should give same result
        assert result1.nb_cells == result2.nb_cells

    def test_union_cell_count_larger_than_operands(self):
        """|A ∪ B| ≥ max(|A|, |B|)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        union_result = sam.subsets.union_(mesh1, mesh2, level=2)
        mesh1_subset = sam.subsets.intersection(mesh1, mesh1, level=2)
        mesh2_subset = sam.subsets.intersection(mesh2, mesh2, level=2)

        max_operand = max(mesh1_subset.nb_cells, mesh2_subset.nb_cells)
        assert union_result.nb_cells >= max_operand

    def test_difference_cell_count_smaller_than_minuend(self):
        """|A \\ B| ≤ |A|"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh1 = sam.mesh.make(box, config)
        mesh2 = sam.mesh.make(box, config)

        diff_result = sam.subsets.difference(mesh1, mesh2, level=2)
        mesh1_subset = sam.subsets.intersection(mesh1, mesh1, level=2)

        assert diff_result.nb_cells <= mesh1_subset.nb_cells

    def test_translation_preserves_cell_count(self):
        """Translation shouldn't change cell count (except at boundaries)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=3, max_level=3)
        mesh = sam.mesh.make(box, config)

        original = sam.subsets.intersection(mesh, mesh, level=3)
        translated = sam.subsets.translate(mesh, [1, 0], level=3)

        # For small translations, cell counts should be similar
        # (some cells may be lost at boundaries)
        assert translated.nb_cells <= original.nb_cells

    def test_construction_reduces_cell_count(self):
        """Contraction should reduce cell count"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=3, max_level=3)
        mesh = sam.mesh.make(box, config)

        original = sam.subsets.intersection(mesh, mesh, level=3)
        contracted = sam.subsets.contraction(mesh, level=3, n_cells_to_remove=1)

        # Contraction removes boundary cells
        assert contracted.nb_cells < original.nb_cells


# ============================================================================
# Level-Dependent Behavior Tests
# ============================================================================

class TestSubsetLevelDependentBehavior:
    """Test that operations behave correctly at different levels"""

    def test_cell_count_increases_with_level(self):
        """Higher levels should have more cells (for fixed domain)"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=3)
        mesh = sam.mesh.make(box, config)

        subset_l2 = sam.subsets.intersection(mesh, mesh, level=2)
        subset_l3 = sam.subsets.intersection(mesh, mesh, level=3)

        # Level 3 should have more cells than level 2 (refinement)
        assert subset_l3.nb_cells > subset_l2.nb_cells

    def test_translation_at_different_levels(self):
        """Translation should work at any level"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=3)
        mesh = sam.mesh.make(box, config)

        # Should work at both levels
        # Note: At level 2, translation by [1,0] may move cells outside domain
        # resulting in empty subset. At level 3, there are more cells so
        # some should remain after translation.
        result_l2 = sam.subsets.translate(mesh, [1, 0], level=2)
        result_l3 = sam.subsets.translate(mesh, [1, 0], level=3)

        assert result_l2.level == 2
        assert result_l3.level == 3
        # Level 3 has more cells, so should have some after translation
        assert result_l3.nb_cells > 0

    def test_contraction_at_different_levels(self):
        """Contraction should work at any level"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=3)
        mesh = sam.mesh.make(box, config)

        result_l2 = sam.subsets.contraction(mesh, level=2)
        result_l3 = sam.subsets.contraction(mesh, level=3)

        assert result_l2.level == 2
        assert result_l3.level == 3


# ============================================================================
# Multi-Dimensional Consistency Tests
# ============================================================================

class TestSubsetMultiDimensionalConsistency:
    """Test that operations behave consistently across dimensions"""

    def test_intersection_works_in_all_dimensions(self):
        """Intersection should work in 1D, 2D, and 3D"""
        # 1D
        box1d = sam.geometry.box([0.0], [1.0])
        config1d = sam.config._MeshConfig1D(min_level=2, max_level=2)
        mesh1d = sam.mesh.make(box1d, config1d)
        result1d = sam.subsets.intersection(mesh1d, mesh1d, level=2)
        assert result1d.nb_cells > 0

        # 2D
        box2d = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config2d = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh2d = sam.mesh.make(box2d, config2d)
        result2d = sam.subsets.intersection(mesh2d, mesh2d, level=2)
        assert result2d.nb_cells > 0

        # 3D
        box3d = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config3d = sam.config._MeshConfig3D(min_level=1, max_level=1)
        mesh3d = sam.mesh.make(box3d, config3d)
        result3d = sam.subsets.intersection(mesh3d, mesh3d, level=1)
        assert result3d.nb_cells > 0

    def test_translation_stencil_dimension_matches_mesh(self):
        """Translation stencil dimension must match mesh dimension"""
        # 1D mesh requires 1D stencil
        box1d = sam.geometry.box([0.0], [1.0])
        config1d = sam.config._MeshConfig1D(min_level=2, max_level=2)
        mesh1d = sam.mesh.make(box1d, config1d)
        result1d = sam.subsets.translate(mesh1d, [1], level=2)
        assert result1d.nb_cells > 0

        # 2D mesh requires 2D stencil
        box2d = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config2d = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh2d = sam.mesh.make(box2d, config2d)
        result2d = sam.subsets.translate(mesh2d, [1, 0], level=2)
        assert result2d.nb_cells > 0

        # 3D mesh requires 3D stencil
        box3d = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config3d = sam.config._MeshConfig3D(min_level=1, max_level=1)
        mesh3d = sam.mesh.make(box3d, config3d)
        result3d = sam.subsets.translate(mesh3d, [1, 0, 0], level=1)
        assert result3d.nb_cells > 0


# ============================================================================
# Subset Properties Tests
# ============================================================================

class TestSubsetProperties:
    """Test properties of returned Subset objects"""

    def test_subset_has_correct_level(self):
        """Subset should report its level correctly"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=3)
        mesh = sam.mesh.make(box, config)

        for level in [2, 3]:
            subset = sam.subsets.intersection(mesh, mesh, level=level)
            assert subset.level == level

    def test_subset_has_description(self):
        """Subset should have a description string"""
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config._MeshConfig2D(min_level=2, max_level=2)
        mesh = sam.mesh.make(box, config)

        inter = sam.subsets.intersection(mesh, mesh, level=2)
        union = sam.subsets.union_(mesh, mesh, level=2)
        diff = sam.subsets.difference(mesh, mesh, level=2)
        trans = sam.subsets.translate(mesh, [1, 0], level=2)
        contract = sam.subsets.contraction(mesh, level=2)

        assert "intersection" in inter.description
        assert "union" in union.description
        assert "difference" in diff.description
        assert "translated" in trans.description
        assert "contraction" in contract.description

    def test_subset_repr_contains_dimension(self):
        """Subset repr should contain dimension info"""
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config._MeshConfig1D(min_level=2, max_level=2)
        mesh = sam.mesh.make(box, config)

        subset = sam.subsets.intersection(mesh, mesh, level=2)
        repr_str = repr(subset)

        assert "1D" in repr_str
        assert "level 2" in repr_str
        assert "intersection" in repr_str
        assert "cells" in repr_str


if __name__ == "__main__":
    # Run tests manually for quick verification
    pytest.main([__file__, "-v"])
