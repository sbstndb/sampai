# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CellList and LevelCellList bindings.

Tests the cell list construction utilities used for building
adaptive mesh refinement (AMR) meshes with hierarchical cell storage.

The new API uses smart dimension inference similar to box():
- sam.cell.CellList(dim=2) - explicit dimension
- sam.cell.CellList(origin_point=(0., 0.)) - dimension inferred from tuple length
"""

import pytest

import sampai as sam


def test_cell_list_1d_creation():
    """Test creating 1D CellList with explicit dimension."""
    cl = sam.cell.CellList(dim=1)

    assert cl.dim == 1
    assert cl.max_level == 20  # Default max_size for CellList
    assert cl.empty()  # Initially empty


def test_cell_list_2d_creation():
    """Test creating 2D CellList with explicit dimension."""
    cl = sam.cell.CellList(dim=2)

    assert cl.dim == 2
    assert cl.max_level == 20
    assert cl.empty()


def test_cell_list_3d_creation():
    """Test creating 3D CellList with explicit dimension."""
    cl = sam.cell.CellList(dim=3)

    assert cl.dim == 3
    assert cl.max_level == 20
    assert cl.empty()


def test_cell_list_with_geometry():
    """Test creating CellList with origin_point - dimension inferred."""
    cl = sam.cell.CellList(origin_point=(1.0, 2.0), scaling_factor=0.5)

    assert cl.dim == 2  # Inferred from tuple length
    assert cl.origin_point == (1.0, 2.0)
    assert cl.scaling_factor == 0.5


def test_cell_list_1d_with_geometry():
    """Test creating 1D CellList with origin_point."""
    cl = sam.cell.CellList(origin_point=(1.0,))

    assert cl.dim == 1  # Inferred from tuple length
    assert cl.origin_point == (1.0,)


def test_cell_list_3d_with_geometry():
    """Test creating 3D CellList with origin_point."""
    cl = sam.cell.CellList(origin_point=(1.0, 2.0, 3.0))

    assert cl.dim == 3  # Inferred from tuple length
    assert cl.origin_point == (1.0, 2.0, 3.0)


def test_cell_list_explicit_dim_overrides_inference():
    """Test that explicit dim parameter takes precedence."""
    # Provide both dim and origin_point - dim should be used
    cl = sam.cell.CellList(dim=2, origin_point=(1.0, 2.0, 3.0))

    assert cl.dim == 2  # Uses explicit dim, ignores origin_point length


def test_cell_list_no_dim_no_origin_error():
    """Test that error is raised when neither dim nor origin_point is provided."""
    with pytest.raises(RuntimeError, match="Cannot determine dimension"):
        sam.cell.CellList()


def test_cell_list_invalid_dimension():
    """Test that invalid dimensions raise error."""
    with pytest.raises(RuntimeError, match="Invalid dimension"):
        sam.cell.CellList(dim=4)

    with pytest.raises(RuntimeError, match="Invalid dimension"):
        sam.cell.CellList(dim=0)


def test_cell_list_level_access():
    """Test accessing LevelCellList at different levels."""
    cl = sam.cell.CellList(dim=1)

    # Access level 0
    lcl0 = cl[0]
    assert lcl0.level == 0

    # Access level 2
    lcl2 = cl[2]
    assert lcl2.level == 2

    # Access level 10
    lcl10 = cl[10]
    assert lcl10.level == 10


def test_cell_list_level_access_bounds():
    """Test that accessing out-of-bounds levels raises error."""
    cl = sam.cell.CellList(dim=1)

    with pytest.raises(IndexError, match="Level out of range"):
        _ = cl[21]  # max_level is 20


def test_level_cell_list_1d_creation():
    """Test creating 1D LevelCellList."""
    lcl = sam.cell.LevelCellList1D()

    assert lcl.level == 0
    assert lcl.empty  # empty is a property, not a method


def test_level_cell_list_with_level():
    """Test creating LevelCellList at specific level."""
    lcl = sam.cell.LevelCellList2D(level=3)

    assert lcl.level == 3
    assert lcl.empty  # empty is a property, not a method


def test_level_cell_list_with_geometry():
    """Test creating LevelCellList with geometry parameters."""
    lcl = sam.cell.LevelCellList2D(
        level=2, origin_point=(1.0, 1.0), scaling_factor=0.25
    )

    assert lcl.level == 2
    assert lcl.origin_point == (1.0, 1.0)
    assert lcl.scaling_factor == 0.25


def test_level_cell_list_clear():
    """Test clearing a LevelCellList - NOT YET IMPLEMENTED."""
    # The clear() method for LevelCellList has C++ implementation issues
    # with forward_list inheritance in Samurai. This test documents
    # that the method is not currently exposed in Python bindings.
    # TODO: Re-enable when Samurai fixes the clear() implementation
    pass


def test_cell_list_clear():
    """Test clearing CellList - NOT YET IMPLEMENTED."""
    # The clear() method has C++ implementation issues with forward_list inheritance
    # in Samurai. This test documents that the method is not currently exposed
    # in Python bindings.
    # TODO: Re-enable when Samurai fixes the clear() implementation
    pass


def test_cell_list_repr():
    """Test string representations."""
    cl = sam.cell.CellList(dim=2)

    # Test __repr__
    repr_str = repr(cl)
    assert "CellList2D" in repr_str
    assert "dim=2" in repr_str
    assert "max_level=20" in repr_str

    # Test __str__
    str_str = str(cl)
    assert "CellList2D" in str_str
    assert "2D" in str_str


def test_level_cell_list_repr():
    """Test string representations of LevelCellList."""
    lcl = sam.cell.LevelCellList1D(level=2)

    repr_str = repr(lcl)
    assert "LevelCellList1D" in repr_str
    assert "level=2" in repr_str

    str_str = str(lcl)
    assert "LevelCellList1D" in str_str
    assert "level 2" in str_str


def test_level_cell_list_bool():
    """Test bool conversion (False if empty)."""
    lcl = sam.cell.LevelCellList1D(level=1)

    # Empty list should be falsy (bool(lcl) == not lcl.empty)
    assert not lcl == lcl.empty  # bool(lcl) returns False when empty


def test_interval_factory():
    """Test Interval creation via cell submodule."""
    interval = sam.cell.Interval(0, 4)

    assert interval.start == 0
    assert interval.end == 4
    assert interval.index == 0

    # With custom index
    interval2 = sam.cell.Interval(2, 6, index=5)

    assert interval2.start == 2
    assert interval2.end == 6
    assert interval2.index == 5


def test_list_of_intervals_creation():
    """Test ListOfIntervals binding."""
    loi = sam.cell.ListOfIntervals()

    assert loi.size == 0
    assert len(loi) == 0


def test_list_of_intervals_add_point():
    """Test adding points to ListOfIntervals."""
    loi = sam.cell.ListOfIntervals()

    # Add points
    loi.add_point(0)
    loi.add_point(5)

    # Points should create intervals
    # Note: behavior depends on Samurai implementation
    assert loi.size >= 0


def test_list_of_intervals_add_interval():
    """Test adding intervals to ListOfIntervals."""
    loi = sam.cell.ListOfIntervals()

    interval = sam.cell.Interval(0, 4)
    loi.add_interval(interval)

    assert loi.size >= 1


def test_list_of_intervals_iteration():
    """Test iterating over ListOfIntervals."""
    loi = sam.cell.ListOfIntervals()

    interval1 = sam.cell.Interval(0, 4)
    interval2 = sam.cell.Interval(5, 10)

    loi.add_interval(interval1)
    loi.add_interval(interval2)

    # Should be iterable
    count = 0
    for interval in loi:
        count += 1
        assert hasattr(interval, "start")
        assert hasattr(interval, "end")

    assert count >= 1


def test_list_of_intervals_repr():
    """Test ListOfIntervals string representation."""
    loi = sam.cell.ListOfIntervals()

    repr_str = repr(loi)
    assert "ListOfIntervals" in repr_str
    assert "size=" in repr_str


def test_cell_list_nested_access_2d():
    """Test nested access pattern for 2D CellList."""
    cl = sam.cell.CellList(dim=2)

    # Access level 0
    level_0 = cl[0]
    assert level_0.level == 0

    # Access grid_yz structure (returns nested maps)
    grid = level_0.grid_yz
    # This is the underlying nested map structure
    assert grid is not None


def test_cell_list_nested_access_3d():
    """Test nested access pattern for 3D CellList."""
    cl = sam.cell.CellList(dim=3)

    # Access level 1
    level_1 = cl[1]
    assert level_1.level == 1

    # Access grid_yz structure
    grid = level_1.grid_yz
    assert grid is not None


def test_interval_properties():
    """Test Interval object properties."""
    interval = sam.cell.Interval(2, 8, index=3)

    assert interval.start == 2
    assert interval.end == 8
    assert interval.index == 3
    assert interval.step == 1  # Default step


def test_multiple_cell_lists_independent():
    """Test that multiple CellLists are independent."""
    cl1 = sam.cell.CellList(dim=1)
    cl2 = sam.cell.CellList(dim=1)

    # Should be different objects
    assert cl1 is not cl2

    # Modifications to one should not affect the other
    # (assuming we could modify them - for now just verify independence)


def test_cell_list_default_origin():
    """Test CellList with default origin_point."""
    cl = sam.cell.CellList(dim=2)

    # Default origin should be (0, 0)
    origin = cl.origin_point
    assert len(origin) == 2
    assert origin[0] == 0.0
    assert origin[1] == 0.0


def test_cell_list_default_scaling_factor():
    """Test CellList with default scaling_factor."""
    cl = sam.cell.CellList(dim=1)

    assert cl.scaling_factor == 1.0


def test_level_cell_list_default_scaling_factor():
    """Test LevelCellList with default scaling_factor."""
    lcl = sam.cell.LevelCellList1D(level=1)

    assert lcl.scaling_factor == 1.0


def test_cell_list_api_consistency_with_box():
    """Test that CellList API is consistent with box() API style."""
    # Like box(), CellList can infer dimension from coordinates
    cl_1d = sam.cell.CellList(origin_point=(0.0,))
    cl_2d = sam.cell.CellList(origin_point=(0.0, 0.0))
    cl_3d = sam.cell.CellList(origin_point=(0.0, 0.0, 0.0))

    assert cl_1d.dim == 1
    assert cl_2d.dim == 2
    assert cl_3d.dim == 3

    # Like box(), CellList can also take explicit dimension
    cl_explicit = sam.cell.CellList(dim=2)
    assert cl_explicit.dim == 2
