"""
Exception hierarchy for Sampai.

This module defines custom exception types for different categories of errors
that can occur when using the Sampai library.
"""

from typing import Any


class SampaiError(RuntimeError):
    """Base exception class for all Sampai errors.

    Inherits from RuntimeError for backward compatibility with existing code
    that catches RuntimeError.
    """

    pass


class MeshError(SampaiError):
    """Exception raised for errors related to mesh operations.

    This includes:
    - Mesh creation failures
    - Dimension mismatches
    - Invalid mesh configurations
    - Domain/box configuration errors
    """

    pass


class FieldError(SampaiError):
    """Exception raised for errors related to field operations.

    This includes:
    - Invalid field dimensions
    - Component count mismatches
    - Field operation failures
    - Invalid field types for specific operations
    """

    pass


class ConfigError(SampaiError):
    """Exception raised for errors related to configuration.

    This includes:
    - Invalid mesh configurations
    - Invalid MRA configurations
    - Parameter validation failures
    """

    pass


class OperatorError(SampaiError):
    """Exception raised for errors related to differential operators.

    This includes:
    - Invalid velocity field dimensions
    - Operator application failures
    - Stencil size errors
    """

    pass


class BoundaryConditionError(SampaiError):
    """Exception raised for errors related to boundary conditions.

    This includes:
    - Invalid BC order
    - Invalid number of values for BC
    - Polynomial extrapolation errors
    """

    pass


class AdaptationError(SampaiError):
    """Exception raised for errors related to mesh adaptation.

    This includes:
    - Dimension mismatches between mesh and velocity field
    - Adaptation algorithm failures
    """

    pass


class IOError(SampaiError):
    """Exception raised for errors related to I/O operations.

    This includes:
    - File not found errors
    - HDF5 read/write errors
    - Invalid file formats
    """

    pass
