# Sampai - Python Interface for Samurai AMR Library

Python bindings for the **Samurai** library - Adaptive Mesh Refinement (AMR) and Multiresolution Analysis for numerical PDE solvers.

## Status: Standalone Package

**Sampai** (formerly samurai-python) is now a standalone package that uses the **Samurai C++ library as a conda dependency**. The Samurai C++ library (version 0.27.1) is automatically installed from conda-forge during the build process.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Development](#development)
- [Documentation](#documentation)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)
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
# Create environment and install samurai C++ library
conda create -n sampai -c conda-forge python=3.11 samurai=0.27.1
conda activate sampai

# Install sampai from source
pip install .
```

### Option 2: Build from Source with pip (Auto-install samurai)

The setup.py will automatically install the samurai C++ library if it's not found:

```bash
# pip will automatically install samurai from conda-forge
pip install .
```

For development/editable mode:

```bash
pip install -e .
```

### Option 3: Build with CMake Directly

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSAMURAI_PYTHON_STANDALONE=ON
cmake --build . -j4
cmake --install .
```

**Samurai Headers Source (`SAMURAI_SOURCE` option):**
- `AUTO` (default): conda first, fallback to `../samurai` if needed
- `CONDA`: conda-only (fails if not found) - for reproducible builds
- `LOCAL`: `../samurai/include` only - for development with latest headers

> **Note:** conda-forge samurai 0.27.1 is missing `mesh_config.hpp`. Use `LOCAL` if you need newer headers.

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
conda create -n sampai-dev -c conda-forge python=3.11 samurai=0.27.1
conda activate sampai-dev
pip install numpy pytest black isort ruff mypy

# Install in editable mode
pip install -e .
```

### Building from Source

#### Using CMake directly

```bash
mkdir build && cd build

# Configure (samurai must be installed via conda first)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSAMURAI_PYTHON_STANDALONE=ON

# Build
cmake --build . -j4

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

## Contributing

We welcome contributions! Please see the main [Samurai repository](https://github.com/hpc-maths/samurai) for guidelines.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests
5. Run code quality checks: `make check-all`
6. Submit a pull request

---

## Migration from Old Build System

If you were using the old `BUILD_PYTHON_BINDINGS` CMake option, you need to update your workflow:

### Old Way (Deprecated)

```bash
# From samurai_pybind11 root
cmake -B build -DBUILD_PYTHON_BINDINGS=ON
cmake --build build
```

### New Way

```bash
# 1. Build and install the C++ library
cd /path/to/samurai_pybind11
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
sudo cmake --install build --prefix /usr/local

# 2. Build and install Python bindings separately
cd python/
pip install .
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed migration instructions.

---

## Troubleshooting

### "samurai C++ library not found"

**Error:** `Could not find samurai`

**Solution:** Install the samurai C++ library via conda:
```bash
conda install -c conda-forge samurai=0.27.1
```

### Import Errors

**Error:** `ImportError: cannot import name 'sampai'`

**Solution:** Make sure you've installed the package:
```bash
pip install -e .
```

### Version Mismatch

**Error:** Compilation errors or template instantiation failures

**Solution:** Make sure you have the correct version of samurai installed:
```bash
conda install -c conda-forge samurai=0.27.1
```

---

## License

This project is licensed under the **BSD-3-Clause License**. See the [LICENSE](../LICENSE) file for details.

---

## Citation

If you use Samurai in your research, please cite:

```bibtex
@software{samurai,
  author = {Samurai Development Team},
  title = {Samurai: Adaptive Mesh Refinement Library},
  url = {https://github.com/hpc-maths/samurai},
  year = {2024}
}
```

---

## Contact

- **Issues:** [GitHub Issues](https://github.com/hpc-maths/samurai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/hpc-maths/samurai/discussions)
- **Email:** samurai@lists.sciencesconf.org

---

## Acknowledgments

- Built with [pybind11](https://github.com/pybind/pybind11)
- Uses [scikit-build-core](https://github.com/scikit-build/scikit-build-core) for modern Python packaging
- Depends on [xtensor](https://github.com/xtensor-stack/xtensor) for array operations
