# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for find_cell functionality in algorithms submodule.

Tests coordinate-to-cell lookup for 1D, 2D, and 3D meshes.
"""

import pytest

import sampai as sam
import numpy as np


class TestFindCellBasic:
    """Basic tests for find_cell existence and API."""

    def test_find_cell_exists(self):
        """Test that find_cell function exists."""
        assert hasattr(sam.algorithms, 'find_cell')

    def test_find_cell_is_callable(self):
        """Test that find_cell is callable."""
        assert callable(sam.algorithms.find_cell)


class TestFindCell1D:
    """Tests for find_cell in 1D meshes."""

    def test_find_cell_1d_center(self):
        """Test finding cell at center of 1D mesh."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        # Find cell at center
        cell = sam.algorithms.find_cell(mesh, [0.5])

        assert cell.length > 0
        assert cell.level == 3
        center = cell.center()
        assert abs(center[0] - 0.5) < 0.125  # Should be near 0.5

    def test_find_cell_1d_multiple_coords(self):
        """Test finding cells at multiple coordinates."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Test at 0.25
        cell1 = sam.algorithms.find_cell(mesh, [0.25])
        assert cell1.length > 0

        # Test at 0.75
        cell2 = sam.algorithms.find_cell(mesh, [0.75])
        assert cell2.length > 0

        # Cells should be at same level
        assert cell1.level == cell2.level

    def test_find_cell_1d_tuple(self):
        """Test find_cell with tuple coordinates."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, (0.5,))
        assert cell.length > 0

    def test_find_cell_1d_numpy(self):
        """Test find_cell with numpy array coordinates."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, np.array([0.5]))
        assert cell.length > 0

    def test_find_cell_1d_not_found(self):
        """Test find_cell with coordinates outside mesh."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Coordinates outside mesh
        cell = sam.algorithms.find_cell(mesh, [2.0])
        assert cell.length == 0

        cell = sam.algorithms.find_cell(mesh, [-1.0])
        assert cell.length == 0


class TestFindCell2D:
    """Tests for find_cell in 2D meshes."""

    def test_find_cell_2d_center(self):
        """Test finding cell at center of 2D mesh."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=3, max_level=3)

        # Find cell at center
        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])

        assert cell.length > 0
        assert cell.level == 3
        center = cell.center()
        assert abs(center[0] - 0.5) < 0.125
        assert abs(center[1] - 0.5) < 0.125

    def test_find_cell_2d_corners(self):
        """Test finding cells at corners of 2D mesh."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Test at origin corner
        cell1 = sam.algorithms.find_cell(mesh, [0.1, 0.1])
        assert cell1.length > 0

        # Test at opposite corner
        cell2 = sam.algorithms.find_cell(mesh, [0.9, 0.9])
        assert cell2.length > 0

        # Cells should be at same level
        assert cell1.level == cell2.level

    def test_find_cell_2d_tuple(self):
        """Test find_cell with tuple coordinates."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, (0.5, 0.5))
        assert cell.length > 0

    def test_find_cell_2d_numpy(self):
        """Test find_cell with numpy array coordinates."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, np.array([0.5, 0.5]))
        assert cell.length > 0

    def test_find_cell_2d_not_found(self):
        """Test find_cell with coordinates outside 2D mesh."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Coordinates outside mesh
        cell = sam.algorithms.find_cell(mesh, [2.0, 0.5])
        assert cell.length == 0

        cell = sam.algorithms.find_cell(mesh, [0.5, -1.0])
        assert cell.length == 0


class TestFindCell3D:
    """Tests for find_cell in 3D meshes."""

    def test_find_cell_3d_center(self):
        """Test finding cell at center of 3D mesh."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Find cell at center
        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5, 0.5])

        assert cell.length > 0
        assert cell.level == 2
        center = cell.center()
        assert abs(center[0] - 0.5) < 0.25
        assert abs(center[1] - 0.5) < 0.25
        assert abs(center[2] - 0.5) < 0.25

    def test_find_cell_3d_tuple(self):
        """Test find_cell with tuple coordinates in 3D."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, (0.5, 0.5, 0.5))
        assert cell.length > 0

    def test_find_cell_3d_numpy(self):
        """Test find_cell with numpy array coordinates in 3D."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, np.array([0.5, 0.5, 0.5]))
        assert cell.length > 0

    def test_find_cell_3d_not_found(self):
        """Test find_cell with coordinates outside 3D mesh."""
        box = sam.geometry.box([0., 0., 0.], [1., 1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Coordinates outside mesh
        cell = sam.algorithms.find_cell(mesh, [2.0, 0.5, 0.5])
        assert cell.length == 0


class TestFindCellProperties:
    """Tests for CellWrapper properties returned by find_cell."""

    def test_cell_has_level(self):
        """Test that returned cell has level property."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        assert hasattr(cell, 'level')
        assert isinstance(cell.level, int)

    def test_cell_has_index(self):
        """Test that returned cell has index property."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        assert hasattr(cell, 'index')
        assert isinstance(cell.index, int)

    def test_cell_has_length(self):
        """Test that returned cell has length property."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        assert hasattr(cell, 'length')
        assert isinstance(cell.length, float)

    def test_cell_has_center(self):
        """Test that returned cell has center method."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        assert hasattr(cell, 'center')
        assert callable(cell.center)

        center = cell.center()
        assert isinstance(center, tuple)
        assert len(center) == 2

    def test_cell_has_corner(self):
        """Test that returned cell has corner method."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        assert hasattr(cell, 'corner')
        assert callable(cell.corner)

        corner = cell.corner()
        assert isinstance(corner, tuple)
        assert len(corner) == 2

    def test_cell_center_and_corner_match(self):
        """Test that center and corner are consistent."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        center = cell.center()
        corner = cell.corner()

        # Center should be corner + length/2
        for i in range(2):
            assert abs(center[i] - (corner[i] + cell.length / 2)) < 1e-10


class TestFindCellFieldAccess:
    """Tests for using find_cell to access field data."""

    def test_find_cell_field_access_2d(self):
        """Test accessing field data via find_cell in 2D."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)
        u = sam.field.scalar(mesh, "u")

        # Find cell and verify index is valid
        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        assert cell.length > 0
        assert isinstance(cell.index, int)


class TestFindCellAMR:
    """Tests for find_cell with AMR (adaptive mesh refinement)."""

    def test_find_cell_with_amr(self):
        """Test find_cell on mesh with AMR."""
        box = sam.geometry.box([0., 0.], [1., 1.])

        mesh = sam.mesh.make(box, min_level=2, max_level=4)
        u = sam.field.scalar(mesh, "u")

        # Perform some AMR
        MRadapt = sam.adaptation.make_MRAdapt(u)
        MRadapt(sam.config.MRAConfig(epsilon=1e-4))

        # Find cell at various locations
        cell1 = sam.algorithms.find_cell(mesh, [0.25, 0.25])
        cell2 = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        cell3 = sam.algorithms.find_cell(mesh, [0.75, 0.75])

        # All should be found
        assert cell1.length > 0
        assert cell2.length > 0
        assert cell3.length > 0


class TestFindCellCoordinateTypes:
    """Tests for different coordinate input types."""

    def test_coordinate_list_2d(self):
        """Test find_cell with list coordinates in 2D."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, [0.5, 0.5])
        assert cell.length > 0

    def test_coordinate_tuple_2d(self):
        """Test find_cell with tuple coordinates in 2D."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, (0.5, 0.5))
        assert cell.length > 0

    def test_coordinate_numpy_2d(self):
        """Test find_cell with numpy array coordinates in 2D."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        cell = sam.algorithms.find_cell(mesh, np.array([0.5, 0.5]))
        assert cell.length > 0


class TestFindCellErrors:
    """Tests for error handling in find_cell."""

    def test_wrong_dimension_1d_mesh_2d_coords(self):
        """Test that 2D coords on 1D mesh raises error."""
        box = sam.geometry.box([0.], [1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Should raise error for wrong dimension
        with pytest.raises(RuntimeError):
            sam.algorithms.find_cell(mesh, [0.5, 0.5])  # 2D coords on 1D mesh

    def test_wrong_dimension_2d_mesh_3d_coords(self):
        """Test that 3D coords on 2D mesh raises error."""
        box = sam.geometry.box([0., 0.], [1., 1.])
        mesh = sam.mesh.make(box, min_level=2, max_level=2)

        # Should raise error for wrong dimension
        with pytest.raises(RuntimeError):
            sam.algorithms.find_cell(mesh, [0.5, 0.5, 0.5])  # 3D coords on 2D mesh
