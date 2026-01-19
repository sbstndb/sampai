# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Sampai** (formerly samurai-python) is a Python interface for the Samurai C++ library, providing Adaptive Mesh Refinement (AMR) and Multiresolution Analysis for numerical PDE solvers. The project automatically downloads Samurai headers from GitHub during build.

- **Languages:** C++20 (bindings), Python 3.9+ (utilities/tests)
- **Build System:** Meson with meson-python backend
- **License:** Apache 2.0

## Build and Development Commands

### Environment Setup
```bash
conda env create -f environment.yml
conda activate sampai
pip install -e .
```

### Building
```bash
# Preferred methods
make build          # Using Makefile
python dev.py build # Using dev.py helper

# Direct Meson
meson setup builddir
meson compile -C builddir
```

### Testing
```bash
pytest tests/ -v                              # All tests
pytest tests/test_basic.py -v                # Single file
pytest tests/test_basic.py::test_box -v      # Single test
pytest tests/ -m "not slow" -v               # Exclude slow tests
pytest tests/test_validation.py -v --val-tol 1e-10  # Validation vs C++
```

### Code Quality
```bash
make fmt       # Format (black + isort, line-length: 100)
make lint      # Lint (ruff)
make check     # Type check (mypy)
make check-all # Run all checks
```

## Architecture

### C++ Bindings (`src/bindings/`)

Each major component has a `.cpp`/`.hpp` pair. All bindings are initialized in `main.cpp` via `init_*_bindings()` functions.

**Core Components:**
- `box_bindings.cpp` - `Box<dim>` geometry
- `mesh_bindings.cpp` - `MRMesh<dim>`, `UniformMesh<dim>`
- `mesh_config_bindings.cpp` - `MeshConfig` configuration
- `field_bindings.cpp` - `ScalarField<dim>`, `VectorField<dim, n_comp>`
- `algorithm_bindings.cpp` - `for_each_cell`, `for_each_interval` iteration
- `operator_bindings.cpp` - `upwind`, `convection_weno5`, `diffusion`
- `bc_bindings.cpp` - `dirichlet`, `neumann` boundary conditions
- `adapt_bindings.cpp` - `MRAdapt`, `update_ghost_mr`
- `mra_config_bindings.cpp` - `MRAConfig` for multiresolution analysis
- `io_bindings.cpp` - HDF5 save/load (Paraview-compatible)
- `domain_builder_bindings.cpp` - `DomainBuilder` for complex domains
- `common_types.hpp` - Centralized type definitions

### Python Utilities (`src/sampai/utils/`)

- `io/` - HDF5 comparison, mesh stats export, 1D plotting/animation
- `viz/samplotlib.py` - Real-time matplotlib visualization (`FieldPlotter`, `VectorPlotter`)
- `progress/` - Progress bars for time loops (`TimeLoop`, `MeshStatistics`)

### Module Organization (v0.30.0+)

The API is organized into submodules:
```python
import sampai as sam

sam.geometry.box()           # Box creation
sam.geometry.DomainBuilder() # Complex domains
sam.config.MeshConfig()      # Mesh configuration
sam.config.MRAConfig()       # Multiresolution config
sam.mesh.make()              # Mesh factory
sam.field.scalar()           # Scalar field factory
sam.field.vector()           # Vector field factory
sam.algorithms.for_each_cell()      # Iteration
sam.algorithms.for_each_interval()  # Iteration
sam.operators.upwind()       # Operators (also available at module level)
sam.boundary.dirichlet()     # Boundary conditions
sam.adaptation.make_MRAdapt() # Adaptation
sam.adaptation.update_ghost_mr()
sam.io.save()                # HDF5 I/O
sam.io.load()
```

Legacy module-level functions (e.g., `make_dirichlet_bc()`, `upwind()`) are retained for backward compatibility.

## Key Architectural Patterns

1. **Factory Pattern** - `sam.mesh.make()`, `sam.field.scalar()`, `sam.field.vector()` hide C++ template complexity
2. **Type Erasure** - `PyAdaptBase` provides polymorphic Adapt objects
3. **Set Algebra** - Interval-based mesh representation with union/intersection/difference
4. **Dimension-Independent Design** - C++ templates for 1D/2D/3D, Python factories handle dispatch
5. **Context Managers** - Progress bars use `with` statements for resource management

## Dependencies

**Build (from conda-forge):**
- C++: xtensor, xtensor-blas, hdf5, highfive, pugixml, fmt, cli11
- Tools: meson >=0.64.0, ninja >=1.10, pybind11 >=2.13.0

**Runtime:**
- Python >=3.9, numpy >=1.20

**Optional:**
- h5py >=3.0 (HDF5 I/O), matplotlib >=3.0 (viz), tqdm >=4.60 (progress)

## Testing

- **Framework:** pytest
- **Markers:** `slow`, `integration`, `unit`, `viz`
- **Validation:** `tests/test_validation.py` compares Python outputs against C++ reference files in `subprojects/samurai/tests/reference/finite_volume/`
- **Fixtures:** `tests/conftest.py` provides `val_tol`, `val_generate_ref`, `cpp_reference_dir`

## CI/CD

`.github/workflows/ci.yml` defines two jobs:
1. `build-and-test` - Standard build and test
2. `validation-vs-cpp` - Validation against C++ references with `--val-tol 1e-10`

## Common Issues

- **Import errors:** Run `pip install -e .` after building
- **Build errors:** Install C++ dependencies via conda-forge
- **meson not found:** Install via `conda install meson ninja` or `pip install meson ninja`
