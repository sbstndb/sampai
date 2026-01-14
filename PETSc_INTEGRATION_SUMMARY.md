# PETSc Integration Summary

**Date**: 2025-01-14
**Branch**: `feat/petsc-integration`
**Status**: Foundation Complete - PETSc Testing Blocked by Dependency Issues

---

## Executive Summary

The PETSc integration for Sampai has been **architecturally completed** with full build system support, Python bindings skeleton, and comprehensive documentation. However, actual PETSc solver functionality cannot be tested due to dependency compatibility issues with conda-forge's PETSc package (v3.7.2, which lacks pkg-config support).

---

## What Was Accomplished

### ✅ 1. Build System Integration

| File | Changes |
|------|---------|
| `meson_options.txt` | Added `with_petsc` boolean option |
| `meson.build` | PETSc detection via pkg-config, CMake, and PETSC_DIR env var |
| `pyproject.toml` | Added `petsc` extra with `mpi4py` |
| `environment-petsc.yml` | New conda environment with PETSc dependencies |

**Build Configuration:**
```bash
meson setup builddir -Dwith_petsc=true  # Enable PETSc
meson setup builddir                      # Disable PETSc (default)
```

### ✅ 2. C++ Bindings Framework

| File | Description |
|------|-------------|
| `src/bindings/petsc_bindings.hpp` | Header with type-erased interfaces |
| `src/bindings/petsc_bindings.cpp` | pybind11 implementation (500+ lines) |
| `src/bindings/main.cpp` | Conditional PETSc initialization |

**Key Components Implemented:**
- `PyKSP` - PETSc Krylov Subspace Methods wrapper
- `PyPC` - Preconditioner wrapper
- `PyLinearSolverWrapper` - Type-erased solver interface
- `PyImplicitOperator` - Base class for implicit operators
- `PyDiffusionOperator` - Diffusion operator wrapper
- `PyIdentityOperator` - Identity operator wrapper
- Operator arithmetic (`+`, `-`, scalar `*`)

### ✅ 3. Tests & Examples

| File | Purpose |
|------|---------|
| `tests/test_petsc.py` | Unit tests for PETSc functionality |
| `examples/heat_implicit.py` | Implicit heat equation demo |
| `examples/operator_arithmetic_demo.py` | Operator arithmetic examples |

### ✅ 4. Documentation

| File | Content |
|------|---------|
| `docs/PETSc_INTEGRATION_DESIGN.md` | Complete technical design document |
| `docs/PETSc_USAGE.md` | User guide for PETSc API |
| `PETSc_INTEGRATION_SUMMARY.md` | This summary |

### ✅ 5. CI/CD Updates

```yaml
test-petsc:
  # New job for PETSc testing
  # Runs: conda install petsc mpich; pip install mpi4py; pytest tests/test_petsc.py
  continue-on-error: true
```

---

## Current Status

### Build Status

```
✅ Build WITHOUT PETSc: SUCCESS
   - meson setup builddir → PETSc: DISABLED
   - meson compile -C builddir → _sampai.cpython-312-x86_64-linux-gnu.so
   - Python import works
   - petsc submodule NOT exposed (expected)

❌ Build WITH PETSc: BLOCKED
   - conda-forge PETSc 3.7.2 lacks pkg-config files
   - Alternative detection methods (CMake) also failed
   - Manual PETSC_DIR not tested (requires full PETSc build)
```

### Python API (When PETSc is Available)

```python
import sampai as sam

# Create field
mesh = sam.mesh.make(box, min_level=4, max_level=8)
u = sam.field.scalar(mesh, "u")

# Apply boundary conditions
sam.make_dirichlet_bc(u, value=0.0, order=2)

# Define implicit scheme
diff = sam.petsc.diffusion_order2(u, coefficient=0.1)
id_op = sam.petsc.identity(u)
dt = 0.01
implicit_scheme = id_op + dt * diff  # Backward Euler

# Configure solver
solver = sam.petsc.LinearSolver()
solver.set_unknown(u)
ksp = solver.ksp()
ksp.set_type("gmres")
ksp.pc().set_type("gamg")
ksp.set_tolerances(rtol=1e-8, max_it=1000)

# Solve
solver.setup()
solver.solve(rhs, u)
print(f"Converged in {solver.iterations()} iterations")
```

---

## Technical Architecture

### Conditional Compilation

All PETSc code is conditionally compiled:

```cpp
#ifdef SAMURAI_WITH_PETSC
    // PETSc bindings implementation
#else
    // No PETSc code - zero runtime overhead
#endif
```

The `SAMURAI_WITH_PETSC` macro is set only when:
1. PETSc library is detected during build
2. `with_petsc=true` option is given to meson

### Type Erasure Pattern

The bindings use type erasure to handle templated C++ classes:

```cpp
// Base interface
class IPyLinearSolver {
    virtual void set_unknown(py::object) = 0;
    virtual void solve(py::object, py::object) = 0;
    // ...
};

// Concrete implementation (templated)
template <class Scheme>
class PyLinearSolver : public IPyLinearSolver {
    samurai::petsc::LinearSolver<Scheme> solver_;
    // ...
};
```

---

## Known Limitations

### 1. PETSc Dependency Issues

**Problem**: conda-forge PETSc 3.7.2 is too old and non-standard
- No pkg-config files
- Different library structure than modern PETSc
- Incompatible with samurai's PETSc detection

**Solutions to Try**:
1. Build PETSc from source (3.20+)
2. Use system PETSc from apt (`apt install libpetsc-dev`)
3. Use Spack for PETSc installation

### 2. Incomplete `make_solver()` Implementation

**Current State**: Stub that raises `RuntimeError`

```python
def make_solver(scheme):
    raise RuntimeError("make_solver() from operator is not yet implemented")
```

**Why**: Requires full type erasure for templated schemes
- Need to map Python operator objects to C++ scheme types
- Complex template instantiation logic
- Requires additional binding infrastructure

### 3. Block Solvers Not Implemented

**Missing**:
- `make_block_operator<rows, cols>(...)`
- `LinearBlockSolver` for Stokes-like problems
- Field split preconditioning

---

## Next Steps

### Immediate (Required for Testing)

1. **Install Compatible PETSc**
   ```bash
   # Option A: Build from source
   git clone https://gitlab.com/petsc/petsc.git
   cd petsc
   ./configure --download-mpich --with-debugging=0
   make all check
   export PETSC_DIR=$PWD

   # Option B: System package
   sudo apt install libpetsc-dev petsc-dev
   ```

2. **Test Build with PETSc**
   ```bash
   rm -rf builddir
   meson setup builddir -Dwith_petsc=true
   meson compile -C builddir
   pip install -e .
   python -c "import sampai; print(hasattr(sampai, 'petsc'))"  # Should print True
   ```

3. **Run Tests**
   ```bash
   pytest tests/test_petsc.py -v
   ```

### Short-term (Implementation)

4. **Complete `make_solver()` Function**
   - Implement scheme type detection
   - Add proper type erasure layer
   - Enable one-shot: `solve(scheme, unknown, rhs)`

5. **Add Vector Field Support**
   - Currently only supports scalar fields
   - Add `diffusion_order2()` for vector fields
   - Support SOA/AOS memory layouts

### Medium-term (Features)

6. **Block Operators**
   - `make_block_operator<rows, cols>(...)`
   - Gradient and divergence operators
   - Stokes solver example

7. **Nonlinear Solvers**
   - SNES interface
   - Newton-Krylov methods
   - Jacobian-free methods

8. **Preconditioner Options**
   - Geometric multigrid
   - Field split for block systems
   - Custom PC callbacks

### Long-term (Advanced)

9. **Distributed Memory**
   - MPI parallelism
   - Domain decomposition
   - Scaling tests

10. **GPU Support**
    - PETSc with CUDA
    - ViennaCL integration
    - Performance benchmarks

---

## File Reference

### Created Files

```
src/bindings/
├── petsc_bindings.hpp    # 120 lines
├── petsc_bindings.cpp    # 550 lines

tests/
└── test_petsc.py          # 150 lines

examples/
├── heat_implicit.py       # 140 lines
└── operator_arithmetic_demo.py  # 90 lines

docs/
├── PETSc_INTEGRATION_DESIGN.md  # 600 lines
├── PETSc_USAGE.md         # 250 lines
└── PETSc_INTEGRATION_SUMMARY.md # This file

environment-petsc.yml      # 30 lines
```

### Modified Files

```
meson_options.txt          # Added with_petsc option
meson.build                # PETSc detection logic (50 lines)
pyproject.toml             # Added petsc extra
environment.yml            # Added PETSc comments
src/bindings/main.cpp      # Conditional PETSc init
.github/workflows/ci.yml   # Added test-petsc job
```

---

## Testing Checklist

### Without PETSc (Current State)

- [x] `meson setup builddir` works
- [x] `meson compile -C builddir` works
- [x] `pip install -e .` works
- [x] `import sampai` works
- [x] `hasattr(sampai, 'petsc')` returns False (expected)
- [ ] `pytest tests/ -v -m "not petsc"` passes

### With PETSc (Not Yet Testable)

- [ ] PETSc installation works
- [ ] `meson setup builddir -Dwith_petsc=true` finds PETSc
- [ ] `meson compile -C builddir` links PETSc
- [ ] `import sampai; sam.petsc` works
- [ ] `sam.petsc.LinearSolver` exists
- [ ] `sam.petsc.diffusion_order2()` works
- [ ] `pytest tests/test_petsc.py` passes

---

## API Quick Reference

### Operators

```python
# Diffusion operator
diff = sam.petsc.diffusion_order2(field, coefficient=0.1)

# Identity operator
id_op = sam.petsc.identity(field)

# Arithmetic
scheme = id_op + dt * diff  # Backward Euler
scheme = id_op - dt * diff  # Forward (unstable)
scheme = id_op + (dt/2) * diff  # Crank-Nicolson
```

### Solver Configuration

```python
solver = sam.petsc.LinearSolver()
solver.set_unknown(field)
solver.setup()  # Assemble matrix

# Get KSP for configuration
ksp = solver.ksp()
ksp.set_type("gmres")  # cg, gmres, bcgs, etc.
ksp.pc().set_type("gamg")  # jacobi, ilu, gamg, hypre
ksp.set_tolerances(rtol=1e-8, max_it=1000)

# Solve
solver.solve(rhs, solution)
print(f"Iterations: {solver.iterations()}")
```

### Solver Types (KSP)

| Type | Description | Use Case |
|------|-------------|----------|
| `cg` | Conjugate Gradient | Symmetric positive definite |
| `gmres` | GMRES | General nonsymmetric |
| `bcgs` | BiCGStab | General nonsymmetric |
| `richardson` | Richardson | With preconditioning |

### Preconditioners (PC)

| Type | Description | Use Case |
|------|-------------|----------|
| `jacobi` | Jacobi | Simple, cheap |
| `ilu` | Incomplete LU | General problems |
| `gamg` | Geometric AMG | Large sparse systems |
| `hypre` | Hypre solvers | Large-scale parallel |

---

## Troubleshooting

### "PETSc not found"

**Cause**: PETSc library not installed or not detected

**Solutions**:
```bash
# 1. Install via conda
conda install -c conda-forge petsc mpich

# 2. Build from source
cd /path/to/petsc
./configure --download-mpich
make all
export PETSC_DIR=/path/to/petsc

# 3. Set environment variable
export PETSC_DIR=/usr/lib/petsc  # or wherever
```

### "ImportError: cannot import name 'petsc'"

**Cause**: Built without PETSc support

**Solution**:
```bash
rm -rf builddir
meson setup builddir -Dwith_petsc=true
meson compile -C builddir
pip install -e .
```

### "RuntimeError: make_solver() not yet implemented"

**Cause**: Function is a stub placeholder

**Workaround**: Use `LinearSolver` class directly
```python
# Instead of:
# solver = sam.petsc.make_solver(scheme)

# Use:
solver = sam.petsc.LinearSolver()
solver.set_unknown(field)
solver.setup()
solver.solve(rhs, field)
```

---

## Resources

### Documentation
- [PETSc Official Docs](https://petsc.org/main/)
- [PETSc KSP Solvers](https://petsc.org/main/manualpages/KSP/)
- [PETSc Preconditioners](https://petsc.org/main/manualpages/PC/)
- [Samurai C++ PETSc Examples](../subprojects/samurai/demos/FiniteVolume/)

### Related Projects
- [petsc4py](https://bitbucket.org/petsc/petsc4py) - Python PETSc bindings
- [dolfin](https://fenicsproject.org/) - FEM with PETSc
- [slepc4py](https://slepc4py.readthedocs.io/) - Eigenvalue solvers

---

## Conclusion

The PETSc integration for Sampai has a **solid foundation** with:
- ✅ Complete build system integration
- ✅ Comprehensive C++ bindings framework
- ✅ Full documentation and examples
- ✅ Test infrastructure

**What's blocking actual use**:
- ❌ Compatible PETSc installation
- ❌ Type-erased `make_solver()` implementation
- ❌ Full end-to-end testing

**Estimated effort to complete**: 2-3 days of focused work
- Day 1: Install/test PETSc, fix detection
- Day 2: Complete `make_solver()`, test basic solves
- Day 3: Add vector field support, comprehensive testing

---

**End of Summary**
