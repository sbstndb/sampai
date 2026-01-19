# Sampai Dependencies

This document provides a central reference for all Sampai dependencies.

## Overview

Sampai has dependencies in three categories:
1. **Build dependencies** - Required to build the C++ bindings
2. **Runtime dependencies** - Required to use Sampai
3. **Optional dependencies** - For additional features (testing, visualization, etc.)

## Build Dependencies

### Core Build Tools

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| Meson | >= 0.64.0 | conda-forge | Build system |
| Ninja | >= 1.10 | conda-forge | Build backend |
| C++ Compiler | C++20 | system | Compile bindings |

### Python Build Dependencies

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| Python | >= 3.9 | conda-forge | Python runtime |
| pybind11 | >= 2.13.0 | conda-forge | C++/Python bindings |
| meson-python | >= 0.15.0 | PyPI | Meson backend for Python |
| numpy | >= 1.20 | conda-forge / PyPI | Array operations |

### C++ Dependencies (via conda-forge)

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| xtensor | >= 0.26 | conda-forge | Multi-dimensional arrays |
| xtensor-blas | latest | conda-forge | BLAS integration for xtensor |
| hdf5 | latest | conda-forge | HDF5 I/O |
| highfive | >= 2.10 | conda-forge | HDF5 C++ wrapper |
| pugixml | latest | conda-forge | XML parsing |
| fmt | latest | conda-forge | String formatting |
| CLI11 | latest | conda-forge | Command-line interface |

### Samurai C++ Library

Samurai is fetched automatically during build using Meson wrap files.

**Default:** Latest `main` branch from GitHub

**Custom versions:**
```bash
# Use a specific tag
meson setup builddir -Dsamurai_commit=v0.27.1

# Use a specific commit
meson setup builddir -Dsamurai_commit=b39857a2780426f96562cfd8d0a4e31c929fc681

# Use a local copy
meson setup builddir -Dsamurai_local_path=/path/to/samurai

# Full clone (no shallow)
meson setup builddir -Dsamurai_depth=0
```

**File locations:**
- Wrap file: `subprojects/samurai.wrap`
- Downloaded headers: `subprojects/samurai/include/samurai/`

**Repository:** https://github.com/hpc-maths/samurai.git

---

## Runtime Dependencies

### Python Runtime

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| Python | >= 3.9 | conda-forge / system | Python runtime |
| numpy | >= 1.20 | conda-forge / PyPI | Array operations |

### C++ Runtime (via conda)

The C++ dependencies listed above must also be available at runtime.

---

## Optional Dependencies

### Testing

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| pytest | >= 7.0 | conda-forge / PyPI | Test runner |
| pytest-cov | >= 3.0 | conda-forge / PyPI | Coverage plugin |

### Visualization

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| matplotlib | >= 3.0 | conda-forge / PyPI | Plotting |
| ipywidgets | >= 7.0 | conda-forge / PyPI | Jupyter widgets |

### HDF5 I/O

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| h5py | >= 3.0 | conda-forge / PyPI | HDF5 Python interface |

### Progress Bars

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| tqdm | >= 4.60 | conda-forge / PyPI | Progress bars |

### Code Quality

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| black | >= 22.0 | conda-forge / PyPI | Code formatting |
| isort | >= 5.0 | conda-forge / PyPI | Import sorting |
| ruff | >= 0.1.0 | conda-forge / PyPI | Linting |
| mypy | >= 1.0 | conda-forge / PyPI | Type checking |

### Documentation

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| sphinx | >= 5.0 | conda-forge / PyPI | Documentation |
| sphinx-rtd-theme | >= 1.0 | conda-forge / PyPI | Docs theme |
| myst-parser | >= 0.18 | conda-forge / PyPI | Markdown in docs |
| breathe | >= 4.0 | conda-forge / PyPI | C++ docs integration |

---

## MPI Support (Optional)

When building with MPI support (`-Dmpi=true`), additional dependencies are required:

| Package | Version | Source | Purpose |
|---------|---------|--------|---------|
| mpich OR openmpi | latest | conda-forge | MPI implementation |
| mpi4py | latest | conda-forge / PyPI | MPI Python bindings |
| boost-mpi | latest | conda-forge | Boost MPI |
| libboost-mpi | latest | conda-forge | Boost MPI library |
| libboost-serialization | latest | conda-forge | Boost serialization |
| HDF5 (with MPI) | latest | conda-forge | Parallel HDF5 |

---

## Dependency Files Summary

Dependencies are specified in multiple files. When adding a new dependency, update:

| File | Purpose |
|------|---------|
| `environment.yml` | Conda environment for development |
| `pyproject.toml` | Python package metadata and optional deps |
| `meson.build` | C++ build dependencies |
| `conda-recipe/meta.yaml` | Conda package recipe |

---

## Installing Dependencies

### Quick Start (Recommended)

```bash
conda env create -f environment.yml
conda activate sampai
pip install -e .
```

### From Scratch

```bash
# Install conda dependencies
conda install -c conda-forge \
    xtensor xtensor-blas hdf5 highfive pugixml fmt cli11 \
    python numpy pybind11 meson ninja

# Install Python dependencies
pip install -e .

# Optional: install development tools
pip install pytest h5py matplotlib black isort ruff mypy
```

### Without Conda

You can use system packages, but you'll need to ensure all C++ dependencies are available and Meson can find them.

---

## Troubleshooting

### Samurai Download Issues

If Samurai fails to download:

```bash
# Manually remove cached version
rm -rf subprojects/samurai

# Rebuild
meson setup builddir --reconfigure
```

### Missing Dependencies

If Meson can't find a dependency:

```bash
# Check what Meson finds
meson setup builddir --wipe

# Provide explicit paths
export CMAKE_PREFIX_PATH=/path/to/dependency
```

### Conda vs System Conflicts

If you have both conda and system versions:

```bash
# Ensure conda environment is active
conda activate sampai

# Check which versions are being used
which python
meson setup builddir
```

---

## Version Pinning

For reproducible builds, you may want to pin specific versions:

**In `environment.yml`:**
```yaml
dependencies:
  - xtensor=0.25.0
  - hdf5=1.14.0
  - ...
```

**For Samurai:**
```bash
meson setup builddir -Dsamurai_commit=v0.27.1
```

---

## Updating Dependencies

### Update conda packages:
```bash
conda env update -f environment.yml --prune
```

### Update Python packages:
```bash
pip install --upgrade -e .
```

### Update Samurai to latest main:
```bash
rm -rf subprojects/samurai
meson setup builddir --reconfigure
```
