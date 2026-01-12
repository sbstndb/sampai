"""
Mesh statistics export and visualization utilities for Sampai.

This module provides tools for exporting, analyzing, and visualizing
mesh adaptation statistics over time. Useful for monitoring long-running
simulations and analyzing AMR behavior.

Functions:
    export_mesh_stats: Export mesh statistics to JSON
    load_mesh_stats: Load mesh statistics from JSON
    plot_mesh_stats: Create visualization from stats data

Example:
    >>> from sampai.utils.io import mesh_stats
    >>> # Export stats during simulation
    >>> mesh_stats.export_mesh_stats(stats_dict, "stats.json")
    >>>
    >>> # Load and plot
    >>> fig = mesh_stats.plot_mesh_stats("stats.json")
    >>> plt.show()
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def export_mesh_stats(
    stats: Dict[str, Any],
    filepath: str,
    test_case: Optional[str] = None
):
    """Export mesh statistics to JSON file.

    Args:
        stats: Statistics dictionary (e.g., from MeshStatistics.to_dict())
        filepath: Output JSON file path
        test_case: Optional test case name for grouping

    Example:
        >>> from sampai.utils import progress
        >>> from sampai.utils.io import mesh_stats
        >>> stats = progress.MeshStatistics()
        >>> # ... populate stats ...
        >>> mesh_stats.export_mesh_stats(stats.to_dict(), "stats.json")
    """
    data = {test_case: stats} if test_case else stats

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_mesh_stats(filepath: str, test_case: Optional[str] = None) -> Dict[str, Any]:
    """Load mesh statistics from JSON file.

    Args:
        filepath: Input JSON file path
        test_case: Test case name to extract (if file contains multiple cases)

    Returns:
        Statistics dictionary

    Example:
        >>> from sampai.utils.io import mesh_stats
        >>> stats = mesh_stats.load_mesh_stats("stats.json")
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    if test_case is not None:
        return data.get(test_case, {})
    return data


def plot_mesh_stats(
    filepath: str,
    test_case: Optional[str] = None,
    figsize: tuple = (20, 15),
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create comprehensive mesh statistics visualization.

    Creates a multi-panel figure showing:
    - Number of cells per level over time
    - Intervals per axis
    - Cells per interval statistics

    Args:
        filepath: Path to JSON statistics file
        test_case: Test case name (if file contains multiple)
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object

    Example:
        >>> from sampai.utils.io import mesh_stats
        >>> fig = mesh_stats.plot_mesh_stats("stats.json")
        >>> plt.show()
    """
    import pandas as pd

    data = pd.read_json(filepath)

    # Handle test case extraction
    if test_case is not None:
        if test_case in data.columns:
            data = pd.json_normalize(data[test_case])
        else:
            # Assume data is already normalized
            pass
    else:
        # If multiple test cases, use the first one
        if len(data.columns) == 1:
            test_case = data.columns[0]
            data = pd.json_normalize(data[test_case])

    # Extract level range
    if 'min_level' in data.columns and 'max_level' in data.columns:
        min_level = int(data.min_level.min())
        max_level = int(data.max_level.max())
    else:
        # Auto-detect from 'by_level' keys
        level_keys = [c for c in data.columns if c.startswith('by_level.')]
        if level_keys:
            levels = []
            for k in level_keys:
                parts = k.split('.')
                if len(parts) >= 2:
                    try:
                        levels.append(int(parts[1]))
                    except ValueError:
                        pass
            if levels:
                min_level = min(levels)
                max_level = max(levels)
            else:
                min_level, max_level = 0, 1
        else:
            min_level, max_level = 0, 1

    levels = range(min_level, max_level + 1)

    # Create figure
    nrow, ncol = 3, 4
    fig = plt.figure(figsize=figsize)

    def plot(suffix, title, xlabel, ylabel, ax, kind='box', legend=None, stacked=True):
        """Helper to plot a statistic."""
        fields = [f'by_level.{l:02d}.{suffix}' for l in levels]
        # Filter to only existing columns
        fields = [f for f in fields if f in data.columns]

        if not fields:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            return

        new_name = {f: l for f, l in zip(fields, levels)}
        to_plot = data[fields].rename(columns=new_name)
        to_plot.plot(kind=kind, ax=ax, stacked=stacked)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        if legend:
            ax.legend(title=legend)

    # Plot various statistics
    plot("cells", "Number of cells per level", "Time iteration", "number of cells",
         fig.add_subplot(nrow, 2, 1), kind='area', legend="Level")

    plot("axis-0.number of intervals", "Intervals per level in x-axis", "Time iteration",
         "number of intervals", fig.add_subplot(nrow, 2, 3), kind='area', legend="Level", stacked=False)
    plot("axis-1.number of intervals", "Intervals per level in y-axis", "Time iteration",
         "number of intervals", fig.add_subplot(nrow, 2, 5), kind='area', legend="Level", stacked=False)

    plot("axis-0.cells per interval.min", "Min cells per interval in x-axis", "level",
         "number of cells", fig.add_subplot(nrow, ncol, 3))
    plot("axis-0.cells per interval.max", "Max cells per interval in x-axis", "level",
         "number of cells", fig.add_subplot(nrow, ncol, 4))

    plot("axis-1.cells per interval.min", "Min cells per interval in y-axis", "level",
         "number of cells", fig.add_subplot(nrow, ncol, 7))
    plot("axis-1.cells per interval.max", "Max cells per interval in y-axis", "level",
         "number of cells", fig.add_subplot(nrow, ncol, 8))

    plot("axis-1.number of intervals per component.min", "Min intervals in x-axis per y-component",
         "level", "number of intervals", fig.add_subplot(nrow, ncol, 11))
    plot("axis-1.number of intervals per component.max", "Max intervals in x-axis per y-component",
         "level", "number of intervals", fig.add_subplot(nrow, ncol, 12))

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


class MeshStatsRecorder:
    """Recorder for accumulating mesh statistics over time.

    Useful for tracking mesh evolution during long simulations.

    Args:
        test_case: Name of the test case
        auto_save_interval: Auto-save to file every N iterations (0 to disable)

    Example:
        >>> from sampai.utils.io import mesh_stats
        >>> recorder = mesh_stats.MeshStatsRecorder("my_sim", auto_save_interval=100)
        >>> for i in range(1000):
        ...     # ... run simulation ...
        ...     stats = get_mesh_stats()
        ...     recorder.record(i, stats)
        >>> recorder.save("final_stats.json")
    """

    def __init__(self, test_case: str = "simulation", auto_save_interval: int = 0):
        self.test_case = test_case
        self.auto_save_interval = auto_save_interval
        self._data: List[Dict[str, Any]] = []
        self._iteration_count = 0

    def record(self, iteration: int, stats: Dict[str, Any], auto_save: bool = True):
        """Record statistics for an iteration.

        Args:
            iteration: Iteration number
            stats: Statistics dictionary
            auto_save: Whether to auto-save if interval reached
        """
        stats_with_iter = stats.copy()
        stats_with_iter['iteration'] = iteration
        self._data.append(stats_with_iter)
        self._iteration_count += 1

        if (auto_save and self.auto_save_interval > 0 and
                self._iteration_count % self.auto_save_interval == 0):
            self.save(f"{self.test_case}_checkpoint.json")

    def save(self, filepath: str):
        """Save recorded statistics to JSON file.

        Args:
            filepath: Output file path
        """
        output = {self.test_case: self._data}
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

    def get_data(self) -> List[Dict[str, Any]]:
        """Get all recorded data.

        Returns:
            List of statistics dictionaries
        """
        return self._data.copy()

    def clear(self):
        """Clear all recorded data."""
        self._data.clear()
        self._iteration_count = 0
