# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
I/O utilities for Sampai.

This module provides tools for HDF5 file operations, comparison,
visualization, and mesh statistics management.

Submodules:
    compare: HDF5 file comparison utilities
    read_hdf5_1d: 1D mesh and field visualization from HDF5
    mesh_stats: Mesh statistics export and visualization

Main exports from compare:
    compare_hdf5_files: Compare two HDF5 files
    compare_hdf5_time_series: Compare a time series of files
    assert_hdf5_files_equal: Assert files are equal (for testing)

Main exports from read_hdf5_1d:
    read_mesh_1d: Read 1D mesh data from HDF5
    plot_field_1d: Plot a 1D field from HDF5
    animate_field_1d: Create animation from time series
    list_fields: List available fields in HDF5 file

Main exports from mesh_stats:
    export_mesh_stats: Export stats to JSON
    load_mesh_stats: Load stats from JSON
    plot_mesh_stats: Create visualization from stats
    MeshStatsRecorder: Record stats during simulation

Example:
    >>> from sampai.utils import io
    >>>
    >>> # Compare files
    >>> io.compare_hdf5_files("sim1.h5", "sim2.h5")
    >>>
    >>> # Visualize 1D data
    >>> io.plot_field_1d("solution", "u")
    >>>
    >>> # Record and plot stats
    >>> recorder = io.MeshStatsRecorder("test")
    >>> recorder.record(0, stats_dict)
    >>> recorder.save("stats.json")
    >>> io.plot_mesh_stats("stats.json")

Note:
    This module requires h5py and matplotlib as optional dependencies.
    Install with: pip install sampai[io]
"""

# Import from compare submodule
from .compare import (
    assert_hdf5_files_equal,
    compare_hdf5_files,
    compare_hdf5_time_series,
)

# Import from read_hdf5_1d submodule
from .read_hdf5_1d import (
    FieldAnimator1D,
    animate_field_1d,
    get_field_data_1d,
    list_fields,
    plot_field_1d,
    plot_mesh_1d,
    read_mesh_1d,
)

# Import from mesh_stats submodule
from .mesh_stats import (
    MeshStatsRecorder,
    export_mesh_stats,
    load_mesh_stats,
    plot_mesh_stats,
)

__all__ = [
    # compare
    "compare_hdf5_files",
    "compare_hdf5_time_series",
    "assert_hdf5_files_equal",
    # read_hdf5_1d
    "read_mesh_1d",
    "get_field_data_1d",
    "plot_field_1d",
    "plot_mesh_1d",
    "animate_field_1d",
    "list_fields",
    "FieldAnimator1D",
    # mesh_stats
    "export_mesh_stats",
    "load_mesh_stats",
    "plot_mesh_stats",
    "MeshStatsRecorder",
]
