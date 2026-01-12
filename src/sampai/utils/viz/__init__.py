"""
Visualization utilities for Sampai AMR fields.

This module provides matplotlib-based visualization tools for adaptive
mesh refinement (AMR) data. It handles multiresolution meshes correctly,
rendering cells at different refinement levels with proper sizing.

Main exports:
    plot_field: One-shot plotting of scalar fields
    plot_vector: One-shot plotting of vector fields (quiver)
    plot_mesh: Plot mesh structure (cell outlines by level)
    set_axes_equal: Helper to set equal aspect ratio
    FieldPlotter: Real-time scalar field visualization
    VectorPlotter: Real-time vector field visualization

Example:
    >>> from sampai.utils import viz
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Static plot
    >>> fig, ax = plt.subplots()
    >>> viz.plot_field(u, ax=ax, cmap='viridis')
    >>> plt.show()
    >>>
    >>> # Real-time monitoring
    >>> plotter = viz.FieldPlotter(u)
    >>> for i in range(100):
    >>>     # ... update simulation ...
    >>>     plotter.update(u)
    >>>     plt.pause(0.01)

Note:
    This module requires matplotlib as an optional dependency.
    Install with: pip install sampai[viz]
"""

from .samplotlib import (
    FieldPlotter,
    VectorPlotter,
    plot_field,
    plot_mesh,
    plot_vector,
    set_axes_equal,
)

__all__ = [
    "plot_field",
    "plot_vector",
    "plot_mesh",
    "set_axes_equal",
    "FieldPlotter",
    "VectorPlotter",
]
