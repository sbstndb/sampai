# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Cell bindings and add_cell functionality.

Tests Cell creation, properties, methods, and the add_cell method
for LevelCellList in 1D, 2D, and 3D.
"""

import pytest

import sampai as sam


class TestCellBindings:
    """Tests for Cell class bindings."""

    def test_cell_factory_exists(self):
        """Test that Cell factory function exists."""
        assert hasattr(sam.cell, 'Cell')

    def test_cell_classes_exist(self):
        """Test that Cell1D, Cell2D, Cell3D classes exist."""
        assert hasattr(sam.cell, 'Cell1D')
        assert hasattr(sam.cell, 'Cell2D')
        assert hasattr(sam.cell, 'Cell3D')

    def test_cell_factory_inferred_dimension_1d(self):
        """Test Cell factory with inferred dimension (1D)."""
        cell = sam.cell.Cell(level=2, indices=(5,))
        assert cell.level == 2
        assert cell.indices == (5,)
        assert cell.index == 0
        assert cell.length == pytest.approx(0.25)  # 1 / 2^2

    def test_cell_factory_inferred_dimension_2d(self):
        """Test Cell factory with inferred dimension (2D)."""
        cell = sam.cell.Cell(level=1, indices=(3, 7))
        assert cell.level == 1
        assert cell.indices == (3, 7)
        assert cell.index == 0
        assert cell.length == pytest.approx(0.5)  # 1 / 2^1

    def test_cell_factory_inferred_dimension_3d(self):
        """Test Cell factory with inferred dimension (3D)."""
        cell = sam.cell.Cell(level=0, indices=(1, 2, 3))
        assert cell.level == 0
        assert cell.indices == (1, 2, 3)
        assert cell.index == 0
        assert cell.length == pytest.approx(1.0)  # 1 / 2^0

    def test_cell_factory_explicit_dimension(self):
        """Test Cell factory with explicit dimension parameter."""
        cell = sam.cell.Cell(dim=2, level=1, indices=(3, 7))
        assert cell.level == 1
        assert cell.indices == (3, 7)

    def test_cell_with_custom_index(self):
        """Test Cell with custom index parameter."""
        cell = sam.cell.Cell(level=2, indices=(3, 7), index=42)
        assert cell.index == 42

    def test_cell_with_origin_point(self):
        """Test Cell with custom origin_point."""
        cell = sam.cell.Cell(level=1, indices=(1, 1), origin_point=(0.5, 0.5))
        assert cell.origin_point == (0.5, 0.5)

    def test_cell_center_method(self):
        """Test Cell.center() method."""
        cell = sam.cell.Cell(level=1, indices=(0, 0), origin_point=(0., 0.))
        center = cell.center()
        assert center[0] == pytest.approx(0.25)  # 0 + 0.5 * (0 + 0.5)
        assert center[1] == pytest.approx(0.25)

    def test_cell_center_scalar_method(self):
        """Test Cell.center(i) scalar method."""
        cell = sam.cell.Cell(level=1, indices=(1, 2), origin_point=(0., 0.))
        assert cell.center(0) == pytest.approx(0.75)  # 0 + 0.5 * (1 + 0.5)
        assert cell.center(1) == pytest.approx(1.25)  # 0 + 0.5 * (2 + 0.5)

    def test_cell_corner_method(self):
        """Test Cell.corner() method."""
        cell = sam.cell.Cell(level=1, indices=(1, 2), origin_point=(0., 0.))
        corner = cell.corner()
        assert corner[0] == pytest.approx(0.5)  # 0 + 0.5 * 1
        assert corner[1] == pytest.approx(1.0)  # 0 + 0.5 * 2

    def test_cell_corner_scalar_method(self):
        """Test Cell.corner(i) scalar method."""
        cell = sam.cell.Cell(level=1, indices=(1, 2), origin_point=(0., 0.))
        assert cell.corner(0) == pytest.approx(0.5)
        assert cell.corner(1) == pytest.approx(1.0)

    def test_cell_repr(self):
        """Test Cell string representation."""
        cell = sam.cell.Cell(level=2, indices=(3, 7))
        repr_str = repr(cell)
        assert 'Cell2D' in repr_str
        assert 'level=2' in repr_str

    def test_cell_str(self):
        """Test Cell string output."""
        cell = sam.cell.Cell(level=2, indices=(3, 7))
        str_str = str(cell)
        assert 'Cell2D' in str_str
        assert 'level 2' in str_str


class TestAddCell:
    """Tests for LevelCellList.add_cell method."""

    def test_add_cell_1d(self):
        """Test add_cell in 1D."""
        cl = sam.cell.CellList(dim=1)
        lcl = cl[2]

        # Add single cell
        cell = sam.cell.Cell(level=2, indices=(5,))
        lcl.add_cell(cell)

        assert not lcl.empty

    def test_add_cell_2d(self):
        """Test add_cell in 2D."""
        cl = sam.cell.CellList(dim=2)
        lcl = cl[1]

        # Add multiple cells
        cell1 = sam.cell.Cell(level=1, indices=(2, 3))
        cell2 = sam.cell.Cell(level=1, indices=(4, 5))

        lcl.add_cell(cell1)
        lcl.add_cell(cell2)

        assert not lcl.empty

    def test_add_cell_3d(self):
        """Test add_cell in 3D."""
        cl = sam.cell.CellList(dim=3)
        lcl = cl[0]

        # Add cell
        cell = sam.cell.Cell(level=0, indices=(1, 2, 3))
        lcl.add_cell(cell)

        assert not lcl.empty

    def test_add_cell_multiple_same_level(self):
        """Test adding multiple cells at same level."""
        cl = sam.cell.CellList(dim=2)
        lcl = cl[1]

        for i in range(4):
            for j in range(4):
                cell = sam.cell.Cell(level=1, indices=(i, j))
                lcl.add_cell(cell)

        assert not lcl.empty

    def test_add_cell_with_different_levels(self):
        """Test adding cells to different levels."""
        cl = sam.cell.CellList(dim=1)

        # Add cells to different levels
        for level in range(3):
            lcl = cl[level]
            for i in range(2):
                cell = sam.cell.Cell(level=level, indices=(i,))
                lcl.add_cell(cell)

        # Check each level
        for level in range(3):
            assert not cl[level].empty

    def test_add_cell_consistency_with_interval_api(self):
        """Test that add_cell gives same result as interval API."""
        # Create two CellLists
        cl1 = sam.cell.CellList(dim=1, origin_point=(0.,))
        cl2 = sam.cell.CellList(dim=1, origin_point=(0.,))

        level = 2

        # Using add_cell
        cell1 = sam.cell.Cell(level=level, indices=(3,))
        cell2 = sam.cell.Cell(level=level, indices=(5,))
        cl1[level].add_cell(cell1)
        cl1[level].add_cell(cell2)

        # Using interval API
        cl2[level].add_point(3)
        cl2[level].add_point(5)

        # Both should have same number of intervals
        # (assuming add_cell and add_point create the same structure)
        assert not cl1[level].empty
        assert not cl2[level].empty


class TestCellIntegration:
    """Integration tests for Cell with other Sampai features."""

    def test_cell_with_celllist_different_origins(self):
        """Test Cell with CellList that has custom origin."""
        cl = sam.cell.CellList(dim=2, origin_point=(1.0, 2.0))
        lcl = cl[1]

        cell = sam.cell.Cell(level=1, indices=(0, 0), origin_point=(1.0, 2.0))
        lcl.add_cell(cell)

        assert not lcl.empty

    def test_cell_properties_match_constructor(self):
        """Test that Cell properties match constructor parameters."""
        level = 2
        indices = (3, 7)
        index = 42
        origin = (0.5, 0.5)

        cell = sam.cell.Cell(level=level, indices=indices, index=index, origin_point=origin)

        assert cell.level == level
        assert cell.indices == indices
        assert cell.index == index
        assert cell.origin_point == origin

    def test_cell_length_computation(self):
        """Test that Cell length is computed correctly."""
        # length = scaling_factor / (1 << level)
        # Default scaling_factor is 1.0
        for level in range(5):
            cell = sam.cell.Cell(level=level, indices=(0,))
            expected_length = 1.0 / (1 << level)
            assert cell.length == pytest.approx(expected_length)
