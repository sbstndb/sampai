# Tests conftest.py
# The sampai package must be installed via `pip install -e .` before running tests

"""
pytest configuration for validation tests.

This module provides fixtures and markers for comparing Python outputs
with C++ reference files.
"""

import tempfile
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add custom command-line options for validation tests."""
    group = parser.getgroup("validation", "Validation tests vs C++")
    group.addoption(
        "--val-tol",
        type=float,
        default=1e-10,
        help="Tolerance for HDF5 comparison (default: 1e-10)"
    )
    group.addoption(
        "--val-generate-ref",
        action="store_true",
        help="Generate reference files from Python outputs (for debugging)"
    )


@pytest.fixture
def val_tol(request):
    """Get the validation tolerance from command-line option."""
    return request.config.getoption("--val-tol")


@pytest.fixture(scope="session")
def val_generate_ref(request):
    """Check if we're generating reference files."""
    return request.config.getoption("--val-generate-ref")


@pytest.fixture(scope="class")
def val_output_dir(val_generate_ref):
    """
    Get output directory for validation tests.

    If --val-generate-ref is set, returns a persistent directory
    for reference files. Otherwise, returns a temporary directory.
    """
    if val_generate_ref:
        # Use a persistent directory for reference files
        ref_dir = Path(__file__).parent / "reference" / "validation"
        ref_dir.mkdir(parents=True, exist_ok=True)
        return ref_dir
    else:
        # Use a temporary directory for test outputs
        return Path(tempfile.mkdtemp())


@pytest.fixture
def cpp_reference_dir():
    """
    Get path to C++ reference files.

    Points to subprojects/samurai/tests/reference/finite_volume/
    """
    samurai_tests = Path(__file__).parent.parent / "subprojects" / "samurai" / "tests"
    return samurai_tests / "reference" / "finite_volume"
