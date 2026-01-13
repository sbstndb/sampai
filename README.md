# Sampai - Python Interface for Samurai AMR Library

[![CI](https://github.com/sbstndb/sampai/actions/workflows/ci.yml/badge.svg)](https://github.com/sbstndb/sampai/actions/workflows/ci.yml)

Python bindings for the **Samurai** library - Adaptive Mesh Refinement (AMR) and Multiresolution Analysis for numerical PDE solvers.

## Status: Standalone Package with Meson Build

**Sampai** (formerly samurai-python) is now a standalone package that uses the **Samurai C++ library headers from source** (`../samurai/include`) while using conda-forge for transitive dependencies. The build system has been migrated from CMake to **Meson** for faster builds.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development](#development)
- [Documentation](#documentation)
- [Examples](#examples)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Installation

### Option 1: Install via Conda Environment (Recommended)

Create a new conda environment with all dependencies:

```bash
# Create the environment from the environment file
conda env create -f environment.yml

# Activate the environment
conda activate sampai
```

Or create manually:

```bash
# Create environment with dependencies (samurai headers must be from source)
conda create -n sampai -c conda-forge python=3.11 meson ninja xtensor fmt cli11 hdf5
conda activate sampai

# Install sampai from source (requires ../samurai for headers)
pip install .
```

### Option 2: Build from Source with Meson

```bash
# Configure with Meson
meson setup builddir

# Build
meson compile -C builddir

# Install
pip install .
```

For development/editable mode:

```bash
pip install -e .
```

### Option 3: Build with pip

```bash
# pip will use Meson backend
pip install .
```

### Option 4: Build Conda Package

```bash
# Build the conda package
conda build conda-recipe/

# Install locally
conda install --use-local sampai
```

### Verify Installation

```python
import sampai as sam
print(sam.__version__)  # Should print version number
```

> **Note:** The samurai C++ headers must be available at `../samurai/include` relative to this repository. Clone the samurai repository next to sampai:
> ```bash
> git clone https://github.com/hpc-maths/samurai.git ../samurai
> ```

---

## Quick Start

Here's a minimal example to get you started:

```python
import sampai as sam

# 1. Create a computational domain
box = sam.geometry.box([0., 0.], [1., 1.])

# 2. Create a mesh with AMR capability
mesh = sam.mesh.make(box, min_level=2, max_level=6)

# 3. Create a field
u = sam.field.scalar(mesh, "u")

# 4. Initialize the field
for cell in mesh.cells():
    u[cell] = initial_condition(cell.center)

# 5. Apply boundary conditions
bc = sam.boundary.dirichlet(u, value=0.0)

# 6. Perform mesh adaptation
MRadapt = sam.adaptation.make_MRAdapt(u)
MRadapt(epsilon=1e-4, regularity=1)
```

For more detailed tutorials, see the [examples/](examples/) directory.

---

## Development

### Setting Up Development Environment

```bash
# Create a conda environment from the environment file
conda env create -f environment.yml
conda activate sampai

# Or create manually
conda create -n sampai-dev -c conda-forge python=3.11 meson ninja xtensor fmt cli11 hdf5
conda activate sampai-dev
pip install numpy pytest black isort ruff mypy

# Install in editable mode
pip install -e .
```

### Building from Source

#### Using Meson directly

```bash
# Configure
meson setup builddir

# Build
meson compile -C builddir

# Test
python -c "import sampai; print(sampai.__version__)"
```

### Code Quality

```bash
# Format code
make fmt

# Lint
make lint

# Type check
make check

# Run all checks
make check-all
```

---

## Documentation

### API Documentation

The Python bindings are organized into submodules:

- **`sam.geometry`** - Geometric primitives (`Box`, `DomainBuilder`)
- **`sam.config`** - Mesh configuration (`MeshConfig`, `MRAConfig`)
- **`sam.mesh`** - Mesh types (`UniformMesh`, `MRMesh`)
- **`sam.field`** - Fields (`ScalarField`, `VectorField`)
- **`sam.algorithms`** - Iteration algorithms (`for_each_cell`, `for_each_interval`)
- **`sam.operators`** - Differential operators (`upwind`, `diffusion`)
- **`sam.boundary`** - Boundary conditions (`Dirichlet`, `Neumann`)
- **`sam.adaptation`** - Mesh adaptation (`MRAdapt`, `update_ghost_mr`)
- **`sam.io`** - HDF5 I/O (`save`, `load`)

### Full Documentation

For complete documentation, visit: [https://hpc-math-samurai.readthedocs.io](https://hpc-math-samurai.readthedocs.io)

---

## Examples

The `examples/` directory contains complete, runnable examples:

| Example | Description |
|---------|-------------|
| `advection.py` | 2D advection equation with AMR |
| `burgers.py` | 2D Burgers equation with WENO5 |
| `convection.py` | Linear convection with obstacles |
| `demo_progress.py` | Progress bar demonstration |
| `demo_visualization.py` | Real-time visualization |

Run an example:

```bash
cd examples/
python advection.py
```

---

## Testing

### Run All Tests

```bash
cd python/
pytest tests/ -v
```

### Run Specific Tests

```bash
# Basic tests
pytest tests/test_basic.py -v

# Field tests
pytest tests/test_field.py -v

# Adaptation tests
pytest tests/test_adapt.py -v
```

### Standalone Tests

For testing the standalone build specifically:

```bash
pytest tests/test_standalone.py -v
```

---

## Troubleshooting

### "Samurai source not found"

**Error:** `Samurai source not found at ../samurai/include`

**Solution:** Clone the samurai repository next to sampai:
```bash
git clone https://github.com/hpc-maths/samurai.git ../samurai
```

### Import Errors

**Error:** `ImportError: cannot import name 'sampai'`

**Solution:** Make sure you've installed the package:
```bash
pip install -e .
```

### Build Errors

**Error:** Missing dependencies (xtensor, fmt, etc.)

**Solution:** Install the dependencies via conda:
```bash
conda install -c conda-forge meson ninja xtensor fmt cli11 hdf5
```

### "meson not found"

**Error:** `meson: command not found`

**Solution:** Install meson via conda or pip:
```bash
conda install -c conda-forge meson
# or
pip install meson
```

---

## License

This project is licensed under the **BSD-3-Clause License**. See the [LICENSE](../LICENSE) file for details.
