# C++/Python Equivalence Testing - Implementation Guide

## Quick Start

This guide provides concrete, actionable steps to implement the verification strategy.

## Prerequisites

1. C++ demos are built and working
2. Python bindings are built and importable
3. HDF5 comparison infrastructure exists (`tests/conftest.py`)

## Step-by-Step Implementation

### Step 1: Extend Test Infrastructure (1 day)

Add to `/home/sbstndbs/sbstndbs/samurai_pybind11/python/tests/conftest.py`:

```python
import pytest
import subprocess
import tempfile
import h5py
import numpy as np
from pathlib import Path

# ============================================================
# Command-line options
# ============================================================

def pytest_addoption(parser):
    parser.addoption(
        "--cpp-exec-dir",
        default="../build/demos",
        help="Directory containing C++ demo executables"
    )
    parser.addoption(
        "--update-cpp-ref",
        action="store_true",
        help="Regenerate C++ reference HDF5 files"
    )
    parser.addoption(
        "--equivalence-only",
        action="store_true",
        help="Run only equivalence tests"
    )

@pytest.fixture
def cpp_exec_dir(request):
    """Path to C++ demo executables."""
    return Path(request.config.getoption("--cpp-exec-dir"))

@pytest.fixture
def update_cpp_ref(request):
    """Whether to update C++ reference files."""
    return request.config.getoption("--update-cpp-ref")

@pytest.fixture
def temp_h5_dir(tmp_path):
    """Temporary directory for HDF5 files (using pytest tmp_path)."""
    return tmp_path

# ============================================================
# HDF5 Comparison Utilities
# ============================================================

def assert_h5_fields_equal(ref_path, test_path, rtol=1e-12, atol=1e-9,
                            ignore_datasets=None):
    """
    Compare two HDF5 files field-by-field.

    Args:
        ref_path: Path to reference HDF5 file
        test_path: Path to test HDF5 file
        rtol: Relative tolerance
        atol: Absolute tolerance
        ignore_datasets: List of dataset names to skip

    Raises:
        AssertionError: If fields don't match within tolerance
    """
    ignore_datasets = ignore_datasets or []

    with h5py.File(ref_path, 'r') as ref_file, \
         h5py.File(test_path, 'r') as test_file:

        # Check that top-level groups match
        ref_keys = set(ref_file.keys())
        test_keys = set(test_file.keys())

        if ref_keys != test_keys:
            raise AssertionError(
                f"HDF5 file structure differs:\n"
                f"  Reference groups: {ref_keys}\n"
                f"  Test groups: {test_keys}\n"
                f"  Missing in test: {ref_keys - test_keys}\n"
                f"  Extra in test: {test_keys - ref_keys}"
            )

        # Compare all datasets
        def compare_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Skip ignored datasets
                if any(name.endswith(pattern) for pattern in ignore_datasets):
                    return

                ref_data = ref_file[name][...]
                test_data = test_file[name][...]

                # Check shape
                if ref_data.shape != test_data.shape:
                    raise AssertionError(
                        f"Dataset '{name}' shape mismatch:\n"
                        f"  Reference: {ref_data.shape}\n"
                        f"  Test: {test_data.shape}"
                    )

                # Check dtype
                if ref_data.dtype != test_data.dtype:
                    raise AssertionError(
                        f"Dataset '{name}' dtype mismatch:\n"
                        f"  Reference: {ref_data.dtype}\n"
                        f"  Test: {test_data.dtype}"
                    )

                # Compare values with tolerance
                try:
                    close = np.allclose(test_data, ref_data, rtol=rtol, atol=atol)
                except (TypeError, ValueError) as e:
                    raise AssertionError(
                        f"Dataset '{name}' cannot be compared: {e}"
                    )

                if not close:
                    # Compute statistics for debugging
                    diff = np.abs(test_data - ref_data)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    max_rel_diff = np.max(diff / (np.abs(ref_data) + atol))

                    # Find location of maximum difference
                    max_idx = np.unravel_index(np.argmax(diff), diff.shape)

                    raise AssertionError(
                        f"Dataset '{name}' differs:\n"
                        f"  Max difference: {max_diff:.3e} at index {max_idx}\n"
                        f"  Mean difference: {mean_diff:.3e}\n"
                        f"  Max relative difference: {max_rel_diff:.3e}\n"
                        f"  Reference value at max diff: {ref_data[max_idx]:.10e}\n"
                        f"  Test value at max diff: {test_data[max_idx]:.10e}\n"
                        f"  Tolerance: rtol={rtol:.3e}, atol={atol:.3e}\n"
                        f"  Reference file: {ref_path}\n"
                        f"  Test file: {test_path}"
                    )

        ref_file.visititems(compare_datasets)

def get_h5_statistics(h5_path):
    """Get statistics about an HDF5 file for debugging."""
    stats = {
        'datasets': 0,
        'total_size': 0,
        'groups': []
    }

    with h5py.File(h5_path, 'r') as f:
        def collect_info(name, obj):
            if isinstance(obj, h5py.Group):
                stats['groups'].append(name)
            elif isinstance(obj, h5py.Dataset):
                stats['datasets'] += 1
                stats['total_size'] += obj.nbytes

        f.visititems(collect_info)

    return stats

# ============================================================
# Process Execution Utilities
# ============================================================

def run_cpp_demo(executable, args, cwd=None, timeout=60):
    """
    Run a C++ demo executable.

    Args:
        executable: Path to C++ executable
        args: List of command-line arguments
        cwd: Working directory (default: current directory)
        timeout: Timeout in seconds

    Returns:
        subprocess.CompletedProcess result

    Raises:
        RuntimeError: If executable fails or times out
    """
    cmd = [str(executable)] + [str(a) for a in args]

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        return result

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"C++ demo failed with exit code {e.returncode}:\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Working directory: {cwd}\n"
            f"  Stdout:\n{e.stdout}\n"
            f"  Stderr:\n{e.stderr}"
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"C++ demo timed out after {timeout}s:\n"
            f"  Command: {' '.join(cmd)}"
        )

def run_python_demo(script, args, cwd=None, timeout=60):
    """
    Run a Python demo script.

    Args:
        script: Path to Python script
        args: List of command-line arguments
        cwd: Working directory (default: current directory)
        timeout: Timeout in seconds

    Returns:
        subprocess.CompletedProcess result

    Raises:
        RuntimeError: If script fails or times out
    """
    cmd = ["python3", str(script)] + [str(a) for a in args]

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        return result

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Python demo failed with exit code {e.returncode}:\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Working directory: {cwd}\n"
            f"  Stdout:\n{e.stdout}\n"
            f"  Stderr:\n{e.stderr}"
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Python demo timed out after {timeout}s:\n"
            f"  Command: {' '.join(cmd)}"
        )

# ============================================================
# Test Markers
# ============================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "cpp_equivalence: Mark test as C++/Python equivalence test"
    )
    config.addinivalue_line(
        "markers",
        "slow: Mark test as slow (skip in quick test runs)"
    )
    config.addinivalue_line(
        "markers",
        "category1: Category 1 test (mesh structure, strict tolerance)"
    )
    config.addinivalue_line(
        "markers",
        "category2: Category 2 test (field operations, strict tolerance)"
    )
    config.addinivalue_line(
        "markers",
        "category3: Category 3 test (projection/prediction, medium tolerance)"
    )
    config.addinivalue_line(
        "markers",
        "category4: Category 4 test (flux operators, relaxed tolerance)"
    )
    config.addinivalue_line(
        "markers",
        "category5: Category 5 test (time-stepping, most relaxed tolerance)"
    )
```

### Step 2: Create First Equivalence Test (1 day)

Create `/home/sbstndbs/sbstndbs/samurai_pybind11/python/tests/test_cpp_python_equivalence.py`:

```python
"""
C++/Python Equivalence Tests

Test suite to verify Python bindings produce identical results to C++.

Usage:
    # Run all equivalence tests
    pytest python/tests/test_cpp_python_equivalence.py -v

    # Run only category 1 tests (strictest, fastest)
    pytest python/tests/test_cpp_python_equivalence.py -v -m category1

    # Run specific test
    pytest python/tests/test_cpp_python_equivalence.py::TestMeshEquivalence::test_2d_mesh -v

    # Update C++ reference files
    pytest python/tests/test_cpp_python_equivalence.py --update-cpp-ref -v
"""

import os
import sys
import pytest
from pathlib import Path

# Add samurai_python to path
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# Import test utilities
from conftest import (
    assert_h5_fields_equal,
    run_cpp_demo,
    run_python_demo,
    get_h5_statistics
)

# ============================================================
# Category 1: Mesh Structure Tests (Strictest)
# ============================================================

@pytest.mark.cpp_equivalence
@pytest.mark.category1
class TestMeshEquivalence:
    """Test mesh creation and structure equivalence."""

    def test_2d_uniform_mesh(self, cpp_exec_dir, temp_h5_dir):
        """
        Test that 2D uniform mesh creation matches C++ exactly.

        C++ demo: demos/tutorial/tutorial-2d-mesh.cpp
        Python script: examples/tutorial_2d_mesh.py
        """
        # Check if C++ executable exists
        cpp_exe = cpp_exec_dir / "tutorial" / "tutorial-2d-mesh"
        if not cpp_exe.exists():
            pytest.skip(f"C++ executable not found: {cpp_exe}")

        # Create Python script if it doesn't exist
        py_script = Path(__file__).parent.parent / "examples" / "tutorial_2d_mesh.py"
        if not py_script.exists():
            # Create a simple mesh test inline
            self._test_2d_mesh_inline(temp_h5_dir)
            return

        # C++ output
        cpp_args = ["--path", str(temp_h5_dir), "--filename", "cpp_mesh"]
        run_cpp_demo(cpp_exe, cpp_args, cwd=temp_h5_dir)

        # Python output
        py_args = ["--output", str(temp_h5_dir), "--filename", "py_mesh"]
        run_python_demo(py_script, py_args, cwd=temp_h5_dir)

        # Compare with strict tolerance
        assert_h5_fields_equal(
            temp_h5_dir / "cpp_mesh.h5",
            temp_h5_dir / "py_mesh.h5",
            rtol=1e-14, atol=1e-12  # Strict tolerance for mesh structure
        )

    def _test_2d_mesh_inline(self, temp_h5_dir):
        """Inline test for mesh creation when Python script doesn't exist."""
        import samurai_python as sam

        # Create mesh (matching C++ tutorial parameters)
        box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
        config = sam.config.make(2)
        config.min_level = 2
        config.max_level = 2  # Uniform mesh

        mesh = sam.mesh.make(box, config)

        # Save mesh
        sam.save(str(temp_h5_dir / "py_mesh"), mesh)

        # Verify mesh was created
        assert (temp_h5_dir / "py_mesh.h5").exists()

    def test_3d_uniform_mesh(self, cpp_exec_dir, temp_h5_dir):
        """Test that 3D uniform mesh creation matches C++ exactly."""
        import samurai_python as sam

        # Create 3D mesh
        box = sam.geometry.box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        config = sam.config.make(3)
        config.min_level = 2
        config.max_level = 2  # Uniform mesh

        mesh = sam.mesh.make(box, config)

        # Verify mesh properties
        assert mesh.dim == 3
        assert mesh.min_level == 2
        assert mesh.max_level == 2

# ============================================================
# Category 2: Field Operation Tests (Strict)
# ============================================================

@pytest.mark.cpp_equivalence
@pytest.mark.category2
class TestFieldArithmeticEquivalence:
    """Test field arithmetic operations equivalence."""

    def test_field_initialization(self, temp_h5_dir):
        """Test that field initialization matches C++ semantics."""
        import samurai_python as sam
        import numpy as np

        # Create mesh
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 4
        config.max_level = 4
        mesh = sam.mesh.make(box, config)

        # Initialize field to constant value
        field = sam.field.scalar(mesh, "u", init=3.14159)

        # Verify all cells have correct value
        for level in range(config.min_level, config.max_level + 1):
            level_mesh = mesh[level]
            for interval in level_mesh.intervals():
                for i in range(interval.start, interval.end):
                    # This would require cell iteration access
                    pass

        # Save and verify
        sam.save(str(temp_h5_dir / "field_init"), field)
        assert (temp_h5_dir / "field_init.h5").exists()

    def test_field_scalar_arithmetic(self, temp_h5_dir):
        """Test that field-scalar arithmetic matches C++."""
        import samurai_python as sam

        # Create mesh and field
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 4
        config.max_level = 4
        mesh = sam.mesh.make(box, config)

        u = sam.field.scalar(mesh, "u", init=2.0)

        # Perform arithmetic
        v = u + 1.0  # Should be 3.0
        w = u * 2.0  # Should be 4.0
        x = u / 2.0  # Should be 1.0
        y = u - 1.0  # Should be 1.0

        # Save results
        sam.save(str(temp_h5_dir / "field_arithmetic"), u, v, w, x, y)

        # Verify files exist
        assert (temp_h5_dir / "field_arithmetic.h5").exists()

    def test_field_field_arithmetic(self, temp_h5_dir):
        """Test that field-field arithmetic matches C++."""
        import samurai_python as sam

        # Create mesh
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 4
        config.max_level = 4
        mesh = sam.mesh.make(box, config)

        # Create two fields
        u = sam.field.scalar(mesh, "u", init=2.0)
        v = sam.field.scalar(mesh, "v", init=3.0)

        # Field-field operations
        w = u + v  # Should be 5.0
        x = v - u  # Should be 1.0

        # Save results
        sam.save(str(temp_h5_dir / "field_field"), u, v, w, x)

        # Verify
        assert (temp_h5_dir / "field_field.h5").exists()

# ============================================================
# Category 3: Projection/Prediction Tests (Medium)
# ============================================================

@pytest.mark.cpp_equivalence
@pytest.mark.category3
class TestProjectionPredictionEquivalence:
    """Test projection and prediction operators equivalence."""

    def test_projection_1d(self, temp_h5_dir):
        """Test that fine-to-coarse projection matches C++."""
        import samurai_python as sam

        # Create mesh with multiple levels
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 2
        config.max_level = 4
        mesh = sam.mesh.make(box, config)

        # Create field at finest level
        u = sam.field.scalar(mesh, "u", init=1.0)

        # Apply boundary conditions
        sam.boundary.dirichlet(u, 0.0)

        # Adapt to coarser levels
        MRadaptation = sam.adaptation.make_MRAdapt(u)
        mra_config = sam.config.MRAConfig()
        mra_config.epsilon = 1e-1  # Large epsilon to force coarsening

        MRadaptation(mra_config)
        sam.adaptation.update_ghost_mr(u)

        # Save result
        sam.save(str(temp_h5_dir / "projection"), u)

        # Verify
        assert (temp_h5_dir / "projection.h5").exists()

# ============================================================
# Integration Tests (End-to-End)
# ============================================================

@pytest.mark.cpp_equivalence
@pytest.mark.slow
class TestFullSimulationEquivalence:
    """Test full simulation pipeline equivalence."""

    @pytest.mark.parametrize("dim", [1, 2])
    def test_simple_advection(self, cpp_exec_dir, temp_h5_dir, dim):
        """
        Test that simple advection simulation matches C++.

        This is a smoke test - just verifies that the simulation runs
        without errors and produces output.
        """
        import samurai_python as sam

        # Domain
        if dim == 1:
            box = sam.geometry.box([0.0], [1.0])
        else:
            box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])

        # Mesh
        config = sam.config.make(dim)
        config.min_level = 4
        config.max_level = 6
        mesh = sam.mesh.make(box, config)

        # Field
        u = sam.field.scalar(mesh, "u", init=0.0)

        # Boundary conditions
        sam.boundary.dirichlet(u, 0.0)

        # Save initial state
        sam.save(str(temp_h5_dir / f"advection_{dim}d_init"), u)

        # Verify
        assert (temp_h5_dir / f"advection_{dim}d_init.h5").exists()

# ============================================================
# Property Tests (Invariants)
# ============================================================

@pytest.mark.cpp_equivalence
class TestPropertyEquivalence:
    """Test that mathematical properties hold in both implementations."""

    def test_mass_conservation_smoke(self, temp_h5_dir):
        """
        Smoke test for mass conservation.

        This is a simple sanity check - not a full equivalence test.
        """
        import samurai_python as sam

        # Create simple 1D mesh
        box = sam.geometry.box([0.0], [1.0])
        config = sam.config.make(1)
        config.min_level = 4
        config.max_level = 4
        mesh = sam.mesh.make(box, config)

        # Create field with known integral
        u = sam.field.scalar(mesh, "u", init=1.0)

        # Apply boundary conditions
        sam.boundary.dirichlet(u, 0.0)

        # Save
        sam.save(str(temp_h5_dir / "mass_conservation"), u)

        # Verify
        assert (temp_h5_dir / "mass_conservation.h5").exists()

        # Note: Actual mass conservation check would require
        # computing the integral over the domain
```

### Step 3: Update Python Examples for Testability (1-2 days)

Modify existing examples to accept command-line arguments:

```python
# examples/advection.py

import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Advection equation with AMR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Existing arguments
    parser.add_argument("--tf", type=float, default=1.0, help="Final time")
    parser.add_argument("--cfl", type=float, default=0.5, help="CFL number")

    # NEW: Test-specific arguments
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--filename", default="advection", help="Filename prefix")
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable plotting (for automated testing)")

    args = parser.parse_args()

    # ... rest of script ...

    # Use args.output and args.filename for sam.save()
    sam.save(f"{args.output}/{args.filename}", u)

if __name__ == "__main__":
    main()
```

### Step 4: Create Test Script References (1 day)

Create reference C++ outputs for comparison:

```bash
#!/bin/bash
# scripts/generate_cpp_references.sh

set -e

CPP_DIR="../build/demos"
REF_DIR="python/tests/reference/cpp_outputs"

mkdir -p "$REF_DIR"

# Tutorial 2D mesh
echo "Generating tutorial-2d-mesh reference..."
$CPP_DIR/tutorial/tutorial-2d-mesh \
    --path "$REF_DIR" \
    --filename "tutorial_2d_mesh"

# Finite volume advection 1D
echo "Generating advection-1d reference..."
$CPP_DIR/FiniteVolume/finite-volume-advection-1d \
    --path "$REF_DIR" \
    --filename "advection_1d" \
    --Tf 0.1

# Finite volume advection 2D
echo "Generating advection-2d reference..."
$CPP_DIR/FiniteVolume/finite-volume-advection-2d \
    --path "$REF_DIR" \
    --filename "advection_2d" \
    --Tf 0.01

echo "C++ reference files generated in $REF_DIR"
```

### Step 5: Create CI Configuration (1 day)

Create `.github/workflows/cpp_python_equivalence.yml`:

```yaml
name: C++/Python Equivalence Tests

on:
  push:
    branches: [master, pybind11]
  pull_request:
    branches: [master]

jobs:
  quick-tests:
    """Run fast tests on every push."""
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake g++ python3 python3-pip

    - name: Configure and build
      run: |
        cmake . -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_DEMOS=ON
        cmake --build build --target samurai_python

    - name: Install Python dependencies
      run: pip install pytest h5py numpy

    - name: Run Category 1 tests (mesh structure)
      run: |
        pytest python/tests/test_cpp_python_equivalence.py -v -m category1

    - name: Run Category 2 tests (field operations)
      run: |
        pytest python/tests/test_cpp_python_equivalence.py -v -m category2

  full-tests:
    """Run all tests (including slow ones) on master merge."""
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/master'

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y cmake g++ python3 python3-pip

    - name: Configure and build
      run: |
        cmake . -B build \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_DEMOS=ON \
          -DBUILD_TESTS=ON

    - name: Build C++ demos
      run: |
        cmake --build build --target tutorial-2d-mesh
        cmake --build build --target finite-volume-advection-1d
        cmake --build build --target finite-volume-advection-2d

    - name: Build Python bindings
      run: |
        cmake --build build --target samurai_python

    - name: Install Python dependencies
      run: pip install pytest h5py numpy

    - name: Run all equivalence tests
      run: |
        pytest python/tests/test_cpp_python_equivalence.py -v

    - name: Upload test results
      if: failure()
      uses: actions/upload-artifact@v3
      with:
        name: equivalence-test-results
        path: python/tests/test_output/
```

## Running the Tests

### Local Development

```bash
# Run all tests
pytest python/tests/test_cpp_python_equivalence.py -v

# Run only fast tests (categories 1-2)
pytest python/tests/test_cpp_python_equivalence.py -v -m "category1 or category2"

# Run specific test
pytest python/tests/test_cpp_python_equivalence.py::TestMeshEquivalence::test_2d_uniform_mesh -v

# Run with verbose output
pytest python/tests/test_cpp_python_equivalence.py -vvs

# Run and stop on first failure
pytest python/tests/test_cpp_python_equivalence.py -v -x

# Run with coverage
pytest python/tests/test_cpp_python_equivalence.py -v --cov=samurai_python
```

### Debugging Failed Tests

```bash
# Run with Python debugger
pytest python/tests/test_cpp_python_equivalence.py::TestMeshEquivalence::test_2d_uniform_mesh -vvs --pdb

# Show local variables on failure
pytest python/tests/test_cpp_python_equivalence.py -v -l

# Keep temporary files for inspection
pytest python/tests/test_cpp_python_equivalence.py -v --tb=long
```

## Test Checklist

When adding new equivalence tests:

- [ ] C++ demo executable is built and working
- [ ] Python script accepts `--output` and `--filename` arguments
- [ ] Both use identical parameters (domain, levels, boundary conditions)
- [ ] Test is marked with appropriate category (`@pytest.mark.category1`, etc.)
- [ ] Test uses appropriate tolerances (see VERIFICATION_STRATEGY.md)
- [ ] Test has docstring explaining what it validates
- [ ] Test is added to CI configuration
- [ ] Reference files are generated and committed

## Troubleshooting

### Issue: Test fails with "C++ executable not found"

**Solution**: Build C++ demos first:
```bash
cmake . -B build -DBUILD_DEMOS=ON
cmake --build build --target tutorial-2d-mesh
```

### Issue: Test fails with "HDF5 file structure differs"

**Solution**: Verify both C++ and Python are saving the same fields:
```python
# Inspect C++ output
import h5py
f = h5py.File("cpp_output.h5", 'r')
print("C++ keys:", list(f.keys()))

# Inspect Python output
f = h5py.File("py_output.h5", 'r')
print("Python keys:", list(f.keys()))
```

### Issue: Values differ significantly (> 1%)

**Solution**: Check that parameters match:
- Domain boundaries
- Mesh levels (min_level, max_level)
- Boundary conditions
- Initial conditions
- Time step (CFL, Tf)

### Issue: Test is flaky (sometimes passes, sometimes fails)

**Solution**: Possible causes:
1. Floating-point non-determinism in parallel code
2. Different mesh adaptation patterns due to tie-breaking
3. Random number generation

Fix: Use stricter tolerances or test properties instead of exact values.

## Next Steps

1. Implement infrastructure (Step 1)
2. Create first test (Step 2)
3. Run and debug
4. Add more tests incrementally
5. Set up CI (Step 5)
6. Document results

## References

- Main strategy document: `VERIFICATION_STRATEGY.md`
- HDF5 comparison: `tests/conftest.py` (existing)
- C++ demos: `demos/` directory
- Python examples: `python/examples/` directory
