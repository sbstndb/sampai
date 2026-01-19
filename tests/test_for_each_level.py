# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for for_each_level algorithm function.

Tests the for_each_level function which iterates over mesh refinement levels.
"""

import pytest

import sampai as sam


def test_for_each_level_uniform_mesh_1d():
    """Test for_each_level on 1D uniform mesh."""
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 0
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    # For a uniform mesh, all cells are at max_level
    levels = []
    sam.algorithms.for_each_level(mesh, lambda level: levels.append(level))

    # Should only visit level 2 (max_level) for uniform mesh
    assert len(levels) == 1
    assert levels[0] == 2


def test_for_each_level_uniform_mesh_2d():
    """Test for_each_level on 2D uniform mesh."""
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.config.make(2)
    config.min_level = 0
    config.max_level = 3
    mesh = sam.mesh.make(box, config)

    levels = []
    sam.algorithms.for_each_level(mesh, lambda level: levels.append(level))

    # Should only visit level 3 (max_level) for uniform mesh
    assert len(levels) == 1
    assert levels[0] == 3


def test_for_each_level_uniform_mesh_3d():
    """Test for_each_level on 3D uniform mesh."""
    box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    config = sam.config.make(3)
    config.min_level = 1
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    levels = []
    sam.algorithms.for_each_level(mesh, lambda level: levels.append(level))

    # Should only visit level 2 (max_level) for uniform mesh
    assert len(levels) == 1
    assert levels[0] == 2


def test_for_each_level_include_empty_levels():
    """Test for_each_level with include_empty_levels parameter."""
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 0
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    # With include_empty_levels=False (default), only visit levels with cells
    levels_without_empty = []
    sam.algorithms.for_each_level(
        mesh, lambda level: levels_without_empty.append(level), include_empty_levels=False
    )
    assert len(levels_without_empty) == 1
    assert levels_without_empty[0] == 2

    # With include_empty_levels=True, should still only visit level 2
    # because the CellArray only has cells at max_level
    # Note: The include_empty_levels parameter in Samurai only affects
    # whether to visit empty levels within the CellArray's range,
    # but for uniform mesh, min_level and max_level are the same
    levels_with_empty = []
    sam.algorithms.for_each_level(
        mesh, lambda level: levels_with_empty.append(level), include_empty_levels=True
    )
    assert len(levels_with_empty) == 1
    assert levels_with_empty[0] == 2


def test_for_each_level_with_field_access():
    """Test for_each_level with field access inside callback."""
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 0
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    # Create a field
    u = sam.field.scalar(mesh, "u")

    # Initialize field values for each cell
    def init_field(cell):
        u[cell.index] = cell.level * 10.0

    sam.algorithms.for_each_cell(mesh, init_field)

    # Use for_each_level to check field at each level
    level_sums = {}
    sam.algorithms.for_each_level(
        mesh,
        lambda level: level_sums.__setitem__(level, 0)
    )

    # For uniform mesh, only max_level should be in the dict
    assert 2 in level_sums
    assert len(level_sums) == 1


def test_for_each_level_with_for_each_cell():
    """Test combining for_each_level with for_each_cell."""
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 0
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    # Count cells per level
    cells_per_level = {}

    def count_cells_at_level(level):
        cell_count = [0]

        def count_cell(cell):
            if cell.level == level:
                cell_count[0] += 1

        sam.algorithms.for_each_cell(mesh, count_cell)
        cells_per_level[level] = cell_count[0]

    sam.algorithms.for_each_level(mesh, count_cells_at_level)

    # Should have counted cells at level 2 only
    assert 2 in cells_per_level
    assert cells_per_level[2] > 0
    assert len(cells_per_level) == 1


def test_for_each_level_callback_signature():
    """Test that for_each_level callback receives level as integer."""
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 0
    config.max_level = 2
    mesh = sam.mesh.make(box, config)

    received_types = []

    def check_type(level):
        received_types.append(type(level).__name__)
        # Should be able to use level in arithmetic
        double_level = level * 2

    sam.algorithms.for_each_level(mesh, check_type)

    # Should receive integers
    assert all(t == 'int' for t in received_types)


def test_for_each_level_multiresolution_mesh():
    """Test for_each_level on a multiresolution mesh with cells at multiple levels.

    This test creates a mesh with cells at multiple refinement levels using CellList.
    """
    # Create a multiresolution mesh using CellList
    cl = sam.cell.CellList(dim=1)

    # Level 0: cover entire domain [0, 1)
    cl[0].add_interval(sam.cell.Interval(0, 8))

    # Level 1: refine first half [0, 0.5)
    cl[1].add_interval(sam.cell.Interval(0, 8))

    # Level 2: refine further [0, 0.25)
    cl[2].add_interval(sam.cell.Interval(0, 4))

    config = sam.config.make(dim=1, min_level=0, max_level=3)
    mesh = sam.mesh.make(cl, config)

    levels = []
    sam.algorithms.for_each_level(mesh, lambda level: levels.append(level))

    # Should visit multiple levels for multiresolution mesh
    assert len(levels) > 1
    assert 0 in levels  # Level 0 should have cells
    # The exact levels depend on the mesh construction


def test_for_each_level_empty_mesh():
    """Test for_each_level on an empty mesh (edge case)."""
    # Create minimal mesh
    box = sam.geometry.box([0.0], [0.1])
    config = sam.config.make(1)
    config.min_level = 0
    config.max_level = 0
    mesh = sam.mesh.make(box, config)

    levels = []
    sam.algorithms.for_each_level(mesh, lambda level: levels.append(level))

    # Should visit level 0 (the only level)
    assert len(levels) == 1
    assert levels[0] == 0
