"""
HDF5 file comparison utilities for Sampai.

This module provides tools for comparing two HDF5 files containing
mesh and field data, cell-by-cell. Useful for validation, regression
testing, and debugging.

Functions:
    compare_hdf5_files: Compare two HDF5 files
    compare_hdf5_time_series: Compare a time series of HDF5 files

Example:
    >>> from sampai.utils.io import compare
    >>> # Compare single files
    >>> result = compare.compare_hdf5_files("sim1.h5", "sim2.h5")
    >>> # Compare time series
    >>> results = compare.compare_hdf5_time_series("sol", "ref", start=0, end=100)
"""

import sys
from typing import Dict, Optional, Tuple

import h5py
import numpy as np


def _construct_cells(mesh: h5py.Group) -> np.ndarray:
    """Extract cell coordinates from mesh group.

    Args:
        mesh: HDF5 mesh group

    Returns:
        Array of cell coordinates
    """
    if "points" in mesh.keys():
        points = mesh["points"]
        conn = mesh["connectivity"]
        return points[:][conn[:]]
    else:
        # MPI distributed case - concatenate all ranks
        output = None
        for k in mesh.keys():
            points = mesh[k]["points"]
            conn = mesh[k]["connectivity"]
            cells = points[:][conn[:]]
            if output is None:
                output = cells
            else:
                output = np.concatenate((output, cells))
        return output


def _construct_fields(mesh: h5py.Group) -> Dict[str, np.ndarray]:
    """Extract field data from mesh group.

    Args:
        mesh: HDF5 mesh group

    Returns:
        Dictionary mapping field names to data arrays
    """
    if "points" in mesh.keys():
        if "fields" not in mesh.keys():
            return {}
        return dict(mesh["fields"])
    else:
        # MPI distributed case - concatenate fields from all ranks
        output = {}
        for k in mesh.keys():
            if "fields" in mesh[k]:
                for f in mesh[k]["fields"].keys():
                    data = mesh[k]["fields"][f][:]
                    if f not in output:
                        output[f] = data
                    else:
                        output[f] = np.concatenate((output[f], data))
        return output


def compare_meshes(
    file1: str,
    file2: str,
    tol: float = 1e-14,
    verbose: bool = False
) -> Tuple[bool, str]:
    """Compare two HDF5 mesh files.

    Args:
        file1: Path to first HDF5 file
        file2: Path to second HDF5 file
        tol: Tolerance for floating point comparison
        verbose: If True, print details of comparison

    Returns:
        Tuple of (are_identical, message)
    """
    with h5py.File(file1, "r") as f1, h5py.File(file2, "r") as f2:
        mesh1 = f1["mesh"]
        mesh2 = f2["mesh"]

        cells1 = _construct_cells(mesh1)
        cells2 = _construct_cells(mesh2)

        # Sort cells by their byte representation for consistent ordering
        index1 = np.argsort(np.asarray([c.tobytes() for c in cells1]))
        index2 = np.argsort(np.asarray([c.tobytes() for c in cells2]))

        # Check shape compatibility
        if cells1.shape != cells2.shape:
            msg = f"Shape mismatch: {cells1.shape} vs {cells2.shape}"
            if verbose:
                print(msg)
            return False, msg

        # Check cell coordinates
        if np.any(cells1[index1] != cells2[index2]):
            msg = "Cell coordinates differ"
            if verbose:
                print(msg)
            return False, msg

        # Compare fields
        field1 = _construct_fields(mesh1)
        field2 = _construct_fields(mesh2)

        for field_name in field1.keys():
            if field_name not in field2.keys():
                msg = f"Field '{field_name}' not in second file"
                if verbose:
                    print(msg)
                return False, msg

            data1 = field1[field_name][:][index1]
            data2 = field2[field_name][:][index2]

            diff = np.abs(data1 - data2)
            max_diff = np.max(diff)
            max_idx = np.argmax(diff)

            if max_diff > tol:
                msg = (f"Field '{field_name}' differs: max_diff={max_diff:.2e} "
                       f"(data1={data1[max_idx]:.6e}, data2={data2[max_idx]:.6e})")
                if verbose:
                    print(msg)
                    print(f"  At cell: {cells1[index1[max_idx]]}")
                return False, msg

        msg = f"Files {file1} and {file2} are identical"
        if verbose:
            print(msg)
        return True, msg


def compare_hdf5_files(
    file1: str,
    file2: str,
    tol: float = 1e-14,
    verbose: bool = True
) -> bool:
    """Compare two HDF5 files containing mesh and field data.

    This function compares the mesh structure and all fields between
    two HDF5 files. Cells are sorted by coordinates to handle different
    ordering (e.g., MPI distributed data).

    Args:
        file1: Path to first HDF5 file
        file2: Path to second HDF5 file
        tol: Tolerance for floating point comparison (default: 1e-14)
        verbose: If True, print comparison results (default: True)

    Returns:
        True if files are identical within tolerance, False otherwise

    Raises:
        FileNotFoundError: If either file doesn't exist
        KeyError: If files don't contain mesh data

    Example:
        >>> from sampai.utils.io import compare
        >>> identical = compare.compare_hdf5_files("simulation.h5", "reference.h5")
        >>> if identical:
        ...     print("Test passed!")
    """
    identical, _ = compare_meshes(file1, file2, tol=tol, verbose=verbose)
    return identical


def compare_hdf5_time_series(
    prefix1: str,
    prefix2: str,
    start: int,
    end: int,
    tol: float = 1e-14,
    verbose: bool = True
) -> Dict[int, Tuple[bool, str]]:
    """Compare a time series of HDF5 files.

    Compares files matching the pattern:
    - {prefix1}{i}.h5 vs {prefix2}{i}.h5 for i in [start, end]

    Args:
        prefix1: Prefix for first file series (without .h5 extension)
        prefix2: Prefix for second file series (without .h5 extension)
        start: Starting iteration number
        end: Ending iteration number (inclusive)
        tol: Tolerance for floating point comparison
        verbose: If True, print comparison results

    Returns:
        Dictionary mapping iteration number to (identical, message) tuple

    Example:
        >>> from sampai.utils.io import compare
        >>> results = compare.compare_hdf5_time_series(
        ...     "simulation", "reference", start=0, end=100
        ... )
        >>> all_passed = all(r[0] for r in results.values())
    """
    results = {}
    for i in range(start, end + 1):
        file1 = f"{prefix1}{i}.h5"
        file2 = f"{prefix2}{i}.h5"
        identical, msg = compare_meshes(file1, file2, tol=tol, verbose=verbose)
        results[i] = (identical, msg)
        if not identical and verbose:
            print(f"Iteration {i}: FAILED")
    return results


def assert_hdf5_files_equal(
    file1: str,
    file2: str,
    tol: float = 1e-14,
    msg: Optional[str] = None
):
    """Assert that two HDF5 files are equal.

    Useful for testing and validation. Raises AssertionError if files differ.

    Args:
        file1: Path to first HDF5 file
        file2: Path to second HDF5 file
        tol: Tolerance for floating point comparison
        msg: Optional custom error message

    Raises:
        AssertionError: If files differ beyond tolerance

    Example:
        >>> from sampai.utils.io import compare
        >>> compare.assert_hdf5_files_equal("test.h5", "reference.h5")
    """
    identical, detail = compare_meshes(file1, file2, tol=tol, verbose=False)
    if not identical:
        if msg is None:
            msg = f"Files differ: {detail}"
        raise AssertionError(msg)
