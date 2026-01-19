# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
1D HDF5 mesh and field visualization utilities for Sampai.

This module provides tools for reading and visualizing 1D mesh and field
data from HDF5 files. Supports animation of time series and MPI
distributed data.

Functions:
    read_mesh_1d: Read 1D mesh data from HDF5 file
    plot_field_1d: Plot a 1D field from HDF5
    animate_field_1d: Create animation from time series

Example:
    >>> from sampai.utils.io import read_hdf5_1d
    >>> # Static plot
    >>> fig, ax = read_hdf5_1d.plot_field_1d("solution", field="u")
    >>> plt.show()
    >>>
    >>> # Animation
    >>> ani = read_hdf5_1d.animate_field_1d("sol", "u", start=0, end=100)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def read_mesh_1d(filename: str, mpi_rank: Optional[int] = None) -> h5py.Group:
    """Read 1D mesh data from HDF5 file.

    Args:
        filename: HDF5 file path (with or without .h5 extension)
        mpi_rank: MPI rank to read (for distributed data). If None, reads all ranks.

    Returns:
        HDF5 mesh group containing points, connectivity, and fields

    Example:
        >>> from sampai.utils.io import read_hdf5_1d
        >>> mesh = read_hdf5_1d.read_mesh_1d("solution.h5")
        >>> points = mesh["points"][:]
        >>> data = mesh["fields"]["u"][:]
    """
    if not filename.endswith('.h5'):
        filename = filename + '.h5'

    f = h5py.File(filename, 'r')

    if mpi_rank is not None:
        # Read specific MPI rank
        mesh = f[f"mesh/{mpi_rank}"]
    elif "points" in f["mesh"].keys():
        # Single rank file
        mesh = f["mesh"]
    else:
        # Multi-rank file, return the mesh group (user can iterate)
        mesh = f["mesh"]

    return mesh


def get_field_data_1d(mesh: h5py.Group, field_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract field data and coordinates from 1D mesh.

    Args:
        mesh: HDF5 mesh group
        field_name: Name of the field to extract

    Returns:
        Tuple of (coordinates, field_values) sorted by coordinate

    Raises:
        ValueError: If field not found in mesh
    """
    # Check if field exists
    if "fields" not in mesh.keys():
        raise ValueError(f"No fields found in mesh")
    if field_name not in mesh["fields"].keys():
        available = ' '.join(mesh["fields"].keys())
        raise ValueError(f"Field '{field_name}' not found. Available: {available}")

    points = mesh['points'][:]
    connectivity = mesh['connectivity'][:]
    data = mesh['fields'][field_name][:]

    # Compute cell centers
    segments = points[connectivity[:]]
    centers = 0.5 * (segments[:, 0] + segments[:, 1])

    # Sort by coordinate
    index = np.argsort(centers)
    return centers[index], data[index]


def plot_field_1d(
    filename: str,
    field: str,
    ax: Optional[plt.Axes] = None,
    mpi_rank: Optional[int] = None,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 1D field from HDF5 file.

    Args:
        filename: HDF5 file path (with or without .h5 extension)
        field: Name of the field to plot
        ax: Matplotlib axes (creates new if None)
        mpi_rank: MPI rank to read (for distributed data)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label (defaults to field name)
        **kwargs: Additional arguments passed to plot()

    Returns:
        Tuple of (figure, axes)

    Example:
        >>> from sampai.utils.io import read_hdf5_1d
        >>> fig, ax = read_hdf5_1d.plot_field_1d("solution", "u")
        >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    mesh = read_mesh_1d(filename, mpi_rank=mpi_rank)
    x, y = get_field_data_1d(mesh, field)

    line = ax.plot(x, y, 'o-', linewidth=1, markersize=3, alpha=0.5, **kwargs)[0]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if ylabel else field)
    ax.set_title(title if title else field)
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_mesh_1d(
    filename: str,
    ax: Optional[plt.Axes] = None,
    mpi_rank: Optional[int] = None,
    title: str = "Mesh",
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot 1D mesh structure (cells and connectivity).

    Args:
        filename: HDF5 file path (with or without .h5 extension)
        ax: Matplotlib axes (creates new if None)
        mpi_rank: MPI rank to read (for distributed data)
        title: Plot title
        **kwargs: Additional arguments passed to scatter()

    Returns:
        Tuple of (figure, axes)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    mesh = read_mesh_1d(filename, mpi_rank=mpi_rank)
    points = mesh['points'][:]
    connectivity = mesh['connectivity'][:]

    # Plot points
    ax.scatter(points[:, 0], np.zeros_like(points[:, 0]), marker='+', **kwargs)

    # Plot cells as line segments
    segments = np.zeros((connectivity.shape[0], 2, 2))
    segments[:, :, 0] = points[connectivity[:]][:, :, 0]
    segments[:, :, 1] = 0

    lc = LineCollection(segments, colors='b', linewidths=2)
    ax.add_collection(lc)

    ax.set_xlabel("x")
    ax.set_yticks([])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax


class FieldAnimator1D:
    """Animator for 1D field time series.

    Useful for creating animations from sequential HDF5 files.

    Args:
        filename_prefix: Prefix for file series (e.g., "solution" for "solution0.h5")
        field: Name of the field to animate
        start: Starting iteration number
        end: Ending iteration number
        mpi_size: Number of MPI ranks (for distributed data)
        interval: Delay between frames in milliseconds

    Example:
        >>> from sampai.utils.io import read_hdf5_1d
        >>> animator = read_hdf5_1d.FieldAnimator1D(
        ...     "solution", "u", start=0, end=100
        ... )
        >>> ani = animator.create_animation()
        >>> plt.show()
    """

    def __init__(
        self,
        filename_prefix: str,
        field: str,
        start: int = 0,
        end: Optional[int] = None,
        mpi_size: int = 1,
        interval: int = 200
    ):
        self.prefix = filename_prefix
        self.field = field
        self.start = start
        self.end = end if end is not None else start
        self.mpi_size = mpi_size
        self.interval = interval
        self.fig = None
        self.ax = None
        self.line = None

    def _init_plot(self):
        """Initialize the plot."""
        self.fig, self.ax = plt.subplots()

        # Read first iteration to set up axes
        if self.mpi_size == 1:
            mesh = read_mesh_1d(f"{self.prefix}{self.start}")
            x, y = get_field_data_1d(mesh, self.field)
        else:
            # Concatenate all ranks
            x_all, y_all = [], []
            for rank in range(self.mpi_size):
                mesh = read_mesh_1d(f"{self.prefix}{self.start}_rank_{rank}")
                x, y = get_field_data_1d(mesh, self.field)
                x_all.append(x)
                y_all.append(y)
            x = np.concatenate(x_all)
            y = np.concatenate(y_all)

        self.line, = self.ax.plot(x, y, 'o-', linewidth=1, markersize=3, alpha=0.5)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel(self.field)
        self.ax.grid(True, alpha=0.3)

    def _update(self, frame: int):
        """Update plot for a given frame."""
        iter_num = self.start + frame

        if self.mpi_size == 1:
            mesh = read_mesh_1d(f"{self.prefix}{iter_num}")
            x, y = get_field_data_1d(mesh, self.field)
        else:
            x_all, y_all = [], []
            for rank in range(self.mpi_size):
                mesh = read_mesh_1d(f"{self.prefix}{iter_num}_rank_{rank}")
                x, y = get_field_data_1d(mesh, self.field)
                x_all.append(x)
                y_all.append(y)
            x = np.concatenate(x_all)
            y = np.concatenate(y_all)

        index = np.argsort(x)
        self.line.set_data(x[index], y[index])
        self.ax.set_title(f"{self.field} - iteration {iter_num}")
        return self.line,

    def create_animation(self) -> animation.FuncAnimation:
        """Create the animation object.

        Returns:
            matplotlib.animation.FuncAnimation object
        """
        self._init_plot()
        n_frames = self.end - self.start + 1
        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            frames=n_frames,
            interval=self.interval,
            repeat=True,
            blit=True
        )
        return ani


def animate_field_1d(
    filename_prefix: str,
    field: str,
    start: int = 0,
    end: Optional[int] = None,
    mpi_size: int = 1,
    interval: int = 200,
    save_path: Optional[str] = None
) -> animation.FuncAnimation:
    """Create animation from 1D field time series.

    Creates an animation from sequential HDF5 files matching the pattern
    {filename_prefix}{i}.h5 for i in [start, end].

    Args:
        filename_prefix: Prefix for file series (e.g., "solution")
        field: Name of the field to animate
        start: Starting iteration number
        end: Ending iteration number (inclusive). If None, only plots start.
        mpi_size: Number of MPI ranks (for distributed data)
        interval: Delay between frames in milliseconds
        save_path: If provided, save animation to this path (adds .mp4 or .gif)

    Returns:
        matplotlib.animation.FuncAnimation object

    Example:
        >>> from sampai.utils.io import read_hdf5_1d
        >>> ani = read_hdf5_1d.animate_field_1d(
        ...     "solution", "u", start=0, end=100, save_path="animation"
        ... )
        >>> # Saves to animation.mp4
    """
    animator = FieldAnimator1D(filename_prefix, field, start, end, mpi_size, interval)
    ani = animator.create_animation()

    if save_path:
        path = Path(save_path)
        if end is None:
            # Static plot
            animator.fig.savefig(str(path.with_suffix('.png')), dpi=300)
        else:
            # Animation
            if path.suffix not in ['.mp4', '.gif']:
                path = path.with_suffix('.mp4')
            writer = animation.FFMpegWriter(fps=1) if path.suffix == '.mp4' else 'pillow'
            ani.save(str(path), dpi=300, writer=writer)

    return ani


def list_fields(filename: str) -> List[str]:
    """List all available fields in an HDF5 file.

    Args:
        filename: HDF5 file path (with or without .h5 extension)

    Returns:
        List of field names

    Example:
        >>> from sampai.utils.io import read_hdf5_1d
        >>> fields = read_hdf5_1d.list_fields("solution.h5")
        >>> print(fields)
        ['u', 'rho', 'v']
    """
    mesh = read_mesh_1d(filename)
    if "fields" not in mesh.keys():
        return []
    return list(mesh["fields"].keys())
