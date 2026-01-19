# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
PETSc Vector utilities for Sampai.

This module provides utilities to convert between Sampai fields and PETSc vectors.
These utilities use petsc4py for the actual PETSc Vec objects.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

# Optional petsc4py import
try:
    from petsc4py import PETSc
    HAS_PETSC4PY = True
except ImportError:
    HAS_PETSC4PY = False
    PETSc = None


def field_to_array(field) -> "npt.NDArray[np.float64]":
    """
    Convert a Sampai ScalarField or VectorField to a NumPy array.

    This extracts the underlying data from the field as a contiguous array,
    which can then be used to create a PETSc Vec.

    Args:
        field: A Sampai ScalarField or VectorField

    Returns:
        A NumPy array containing the field data

    Example:
        >>> from sampai import field, mesh
        >>> from sampai.petsc.vector import field_to_array
        >>> m = mesh.make(MeshConfig(), box([0, 0], [1, 1]), level=4)
        >>> f = field.scalar(m, "u")
        >>> arr = field_to_array(f)
        >>> # Now arr can be used to create a PETSc Vec
    """
    # The field should have an array() method that returns the underlying data
    # This is exposed via the C++ bindings
    if hasattr(field, 'array'):
        # Get the numpy array from the field
        return np.asarray(field.array())
    else:
        raise TypeError(
            f"Expected a Sampai field with array() method, got {type(field)}"
        )


def array_to_field(array: "npt.NDArray[np.float64]", field) -> None:
    """
    Copy data from a NumPy array into a Sampai field.

    This is the inverse of field_to_array.

    Args:
        array: A NumPy array containing the data
        field: A Sampai ScalarField or VectorField to copy data into

    Example:
        >>> from sampai import field, mesh
        >>> from sampai.petsc.vector import array_to_field
        >>> from petsc4py import PETSc
        >>> m = mesh.make(MeshConfig(), box([0, 0], [1, 1]), level=4)
        >>> f = field.scalar(m, "u")
        >>> # After solving with PETSc, copy result back to field
        >>> # x is a PETSc Vec from the solver
        >>> array_to_field(x.getArray(), f)
    """
    if hasattr(field, 'array'):
        field_array = field.array()
        if hasattr(field_array, '__setitem__'):
            # Field array supports item assignment
            field_array[:] = array
        else:
            raise TypeError(
                f"Field array does not support item assignment"
            )
    else:
        raise TypeError(
            f"Expected a Sampai field with array() method, got {type(field)}"
        )


def create_petsc_vec_from_field(field, comm=None) -> "PETSc.Vec":
    """
    Create a PETSc Vec from a Sampai field.

    This function creates a PETSc vector that shares its data with the field.
    The vector is created using petsc4py.

    Args:
        field: A Sampai ScalarField or VectorField
        comm: MPI communicator (defaults to PETSc.COMM_SELF)

    Returns:
        A petsc4py Vec object

    Raises:
        ImportError: If petsc4py is not available

    Example:
        >>> from sampai import field, mesh
        >>> from sampai.petsc.vector import create_petsc_vec_from_field
        >>> from sampai.config import MeshConfig
        >>> from sampai.geometry import box
        >>> m = mesh.make(MeshConfig(), box([0, 0], [1, 1]), level=4)
        >>> f = field.scalar(m, "u")
        >>> vec = create_petsc_vec_from_field(f)
    """
    if not HAS_PETSC4PY:
        raise ImportError(
            "petsc4py is required to create PETSc Vec objects. "
            "Install it with: conda install -c conda-forge petsc4py"
        )

    if comm is None:
        comm = PETSc.COMM_SELF

    # Convert field to numpy array
    arr = field_to_array(field)

    # Create PETSc Vec with the array data
    vec = PETSc.Vec().createWithArray(arr, comm=comm)

    # Set the vector name if the field has one
    if hasattr(field, 'name') and field.name:
        vec.setName(field.name)

    return vec


def create_petsc_vec(array: "npt.NDArray[np.float64]", comm=None) -> "PETSc.Vec":
    """
    Create a PETSc Vec from a NumPy array.

    Args:
        array: A NumPy array
        comm: MPI communicator (defaults to PETSc.COMM_SELF)

    Returns:
        A petsc4py Vec object

    Raises:
        ImportError: If petsc4py is not available

    Example:
        >>> import numpy as np
        >>> from sampai.petsc.vector import create_petsc_vec
        >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
        >>> vec = create_petsc_vec(arr)
    """
    if not HAS_PETSC4PY:
        raise ImportError(
            "petsc4py is required to create PETSc Vec objects. "
            "Install it with: conda install -c conda-forge petsc4py"
        )

    if comm is None:
        comm = PETSc.COMM_SELF

    return PETSc.Vec().createWithArray(array, comm=comm)


def copy_vec_to_field(vec: "PETSc.Vec", field) -> None:
    """
    Copy data from a PETSc Vec to a Sampai field.

    Args:
        vec: A petsc4py Vec object
        field: A Sampai ScalarField or VectorField

    Example:
        >>> from sampai import field, mesh
        >>> from sampai.petsc.vector import copy_vec_to_field, create_petsc_vec_from_field
        >>> # After solving with PETSc...
        >>> copy_vec_to_field(solution_vec, f)
    """
    # Get the array from the PETSc Vec
    arr = vec.getArray()
    # Copy it to the field
    array_to_field(arr, field)


def copy_field_to_vec(field, vec: "PETSc.Vec") -> None:
    """
    Copy data from a Sampai field to a PETSc Vec.

    Args:
        field: A Sampai ScalarField or VectorField
        vec: A petsc4py Vec object

    Example:
        >>> from sampai import field, mesh
        >>> from sampai.petsc.vector import copy_field_to_vec, create_petsc_vec_from_field
        >>> # Set up RHS from field
        >>> copy_field_to_vec(rhs_field, rhs_vec)
    """
    # Get the array from the field
    arr = field_to_array(field)
    # Copy it to the PETSc Vec
    vec_array = vec.getArray()
    vec_array[:] = arr


__all__ = [
    'field_to_array',
    'array_to_field',
    'create_petsc_vec_from_field',
    'create_petsc_vec',
    'copy_vec_to_field',
    'copy_field_to_vec',
    'HAS_PETSC4PY',
]
