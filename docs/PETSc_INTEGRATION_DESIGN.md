# PETSc Integration Design Document

**Version**: 1.0
**Date**: 2025-01-14
**Status**: Implementation
**Author**: Claude Code

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background and Motivation](#background-and-motivation)
3. [Technical Requirements](#technical-requirements)
4. [Architecture Design](#architecture-design)
5. [API Specification](#api-specification)
6. [Implementation Details](#implementation-details)
7. [Build System Integration](#build-system-integration)
8. [Testing Strategy](#testing-strategy)
9. [Documentation](#documentation)
10. [Future Enhancements](#future-enhancements)

---

## Executive Summary

This document describes the integration of PETSc (Portable, Extensible Toolkit for Scientific Computation) into Sampai, the Python bindings for the Samurai AMR library. The integration enables:

- **Implicit solvers** for diffusion equations
- **Block solvers** for coupled systems (e.g., Stokes flow)
- **Nonlinear solvers** for reaction-diffusion systems
- **Flexible solver configuration** via PETSc options database

The integration follows Sampai's existing pybind11 architecture and makes PETSc support optional to maintain backward compatibility.

---

## Background and Motivation

### Current Limitations

Sampai currently supports only **explicit operators**:
- `upwind()` - First-order upwind convection
- `convection_weno5()` - Fifth-order WENO convection

These explicit schemes require:
```python
u_next = u + dt * convection_op(u)  # CFL limited
```

### Why PETSc?

Many problems require **implicit solvers**:
1. **Stiff diffusion equations** - Unconditional stability
2. **Elliptic problems** - Poisson, Helmholtz
3. **Coupled systems** - Stokes, Navier-Stokes
4. **Nonlinear problems** - Newton-Krylov methods

### Samurai's PETSc Integration

The Samurai C++ library already has sophisticated PETSc support:
- Linear solvers (KSP)
- Nonlinear solvers (SNES)
- Block operators with field split preconditioning
- AMR-aware matrix assembly
- Ghost cell elimination

**Goal**: Expose this functionality to Python users.

---

## Technical Requirements

### Functional Requirements

| Requirement | Description | Priority |
|-------------|-------------|----------|
| FR-1 | Implicit diffusion solver | High |
| FR-2 | Linear solver with configuration options | High |
| FR-3 | Block solver for coupled systems | Medium |
| FR-4 | Nonlinear solver | Medium |
| FR-5 | Operator arithmetic (+, -, *) | High |

### Non-Functional Requirements

| Requirement | Description |
|-------------|-------------|
| NFR-1 | PETSc must be optional (not required for basic usage) |
| NFR-2 | API must follow existing Sampai patterns |
| NFR-3 | Performance comparable to C++ |
| NFR-4 | Type-safe bindings |
| NFR-5 | Clear error messages when PETSc is not available |

---

## Architecture Design

### Overall Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python User Code                         │
│                                                                   │
│  import sampai as sam                                             │
│  solver = sam.petsc.make_solver(scheme)                          │
│  solver.solve(unknown, rhs)                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Python sampai Package                       │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  samurai.*  │  │ samurai.*   │  │   samurai.petsc         │  │
│  │  (existing) │  │  (existing) │  │   ┌───────────────────┐ │  │
│  └─────────────┘  └─────────────┘  │   │ make_solver()     │ │  │
│                                     │   │ solve()           │ │  │
│                                     │   │ LinearSolver      │ │  │
│                                     │   │ BlockOperator     │ │  │
│                                     │   │ diffusion()       │ │  │
│                                     │   └───────────────────┘ │  │
│                                     └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    pybind11 Bindings (_sampai.so)                │
│                                                                   │
│  ┌───────────────────┐  ┌─────────────────────────────────────┐ │
│  │  existing bindings│  │      petsc_bindings.cpp (NEW)       │ │
│  │  - mesh           │  │                                     │ │
│  │  - field          │  │  PYBIND11_MODULE(petsc) {           │ │
│  │  - operators      │  │      class_<LinearSolver<>>         │ │
│  │  - bc             │  │      def("make_solver", ...);       │ │
│  │  - adapt          │  │      def("solve", ...);             │ │
│  └───────────────────┘  │      def("diffusion", ...);         │ │
│                         │  }                                   │ │
│                         └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Samurai C++ Library (Header-Only)            │
│                                                                   │
│  samurai/petsc/                                                   │
│  ├── linear_solver.hpp      ────────┐                            │
│  ├── nonlinear_solver.hpp    │       │                            │
│  ├── linear_block_solver.hpp │       │                            │
│  ├── solver_helpers.hpp      │       │ Existing C++              │
│  ├── matrix_assembly.hpp     ├───────┘ Implementation            │
│  ├── block_assembly.hpp      │                                    │
│  └── utils.hpp              │                                    │
│                             │                                    │
│  samurai/schemes/fv/         │                                    │
│  ├── ..._lin_hom.hpp ───────┘  Implicit schemes                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PETSc Library (Optional)                    │
│                                                                   │
│  KSP (Linear Solvers)  │  SNES (Nonlinear Solvers)              │
│  Mat (Matrix)          │  Vec (Vector)                          │
│  PC (Preconditioners)  │  DM (Data Management)                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Organization

```
samurai (Python package)
├── geometry          # Existing: Box, DomainBuilder
├── mesh              # Existing: UniformMesh, MRMesh
├── field             # Existing: ScalarField, VectorField
├── operators         # Existing: upwind, convection_weno5
├── boundary          # Existing: Dirichlet, Neumann
├── adaptation        # Existing: MRAdapt
├── petsc             # NEW: PETSc solvers and operators
│   ├── make_solver()
│   ├── solve()
│   ├── LinearSolver
│   ├── BlockSolver
│   ├── NonlinearSolver
│   ├── diffusion_order2()
│   ├── gradient()
│   ├── divergence()
│   ├── identity()
│   └── zero_operator()
└── utils             # Existing: I/O, visualization
```

---

## API Specification

### Python API

#### 1. Simple Implicit Solve

```python
import sampai as sam

# Setup
mesh = sam.mesh.make(box, min_level=4, max_level=8)
u = sam.field.scalar(mesh, "u")
sam.make_dirichlet_bc(u, value=0.0, order=2)

# Define implicit scheme
K = 0.1  # Diffusion coefficient
diff = sam.petsc.diffusion_order2(u, coefficient=K)
id_op = sam.petsc.identity(u)
dt = 0.01

# Backward Euler: (I + dt*K*L) * u^{n+1} = u^n
implicit_scheme = id_op + dt * diff

# One-shot solve
sam.petsc.solve(implicit_scheme, u_next, u)
```

#### 2. Configured Linear Solver

```python
# Create solver
solver = sam.petsc.make_solver(implicit_scheme)
solver.set_unknown(u)

# Optional: Configure PETSc options
ksp = solver.ksp()
ksp.setType("gmres")
ksp.pc.setType("ilu")
ksp.setTolerances(rtol=1e-8, atol=1e-10, div_tol=1e4, max_it=1000)

# Setup and solve
solver.setup()
solver.solve(rhs)

print(f"Converged in {solver.iterations()} iterations")
```

#### 3. Block System (Stokes)

```python
# Coupled velocity-pressure system
v = sam.field.vector(mesh, "velocity", n_components=2, is_soa=True)
p = sam.field.scalar(mesh, "pressure")

# Define block operators
diff_v = sam.petsc.diffusion_order2(v, coefficient=1.0)
grad_p = sam.petsc.gradient(p, v)
div_v = sam.petsc.divergence(v, p)
zero = sam.petsc.zero_operator(p)

# Block operator: [ D  G ]
#                [ D  0 ]
stokes = sam.petsc.make_block_operator(rows=2, cols=2,
                                       op00=diff_v, op01=grad_p,
                                       op10=div_v,  op11=zero)

# Solve
solver = sam.petsc.make_solver(stokes)
solver.set_unknowns(velocity=v, pressure=p)
solver.solve(rhs_v, rhs_p)
```

#### 4. Operator Arithmetic

```python
# Operators support +, -, and scalar *
id = sam.petsc.identity(u)
diff = sam.petsc.diffusion_order2(u, coefficient=0.1)

# Backward Euler: (I - dt*K*L)
backward_euler = id - dt * diff

# Crank-Nicolson: (I + dt/2*K*L)
crank_nicolson = id + (dt / 2.0) * diff

# Combine operators
combined = op1 + 2.0 * op2 - op3
```

#### 5. Nonlinear Solver

```python
# Define nonlinear operator
def nonlinear_op(u_next):
    diff = sam.petsc.diffusion_order2(u_next)
    react = sam.petsc.reaction(u_next)
    return id + dt * diff - dt * react

# Solve with Newton
solver = sam.petsc.make_nonlinear_solver(nonlinear_op)
solver.set_unknown(u)
solver.solve(rhs)

print(f"Newton converged in {solver.iterations()} iterations")
```

### C++ Backend Classes

#### samurai::petsc::LinearSolver<Scheme>

```cpp
template <class Scheme>
class LinearSolver {
public:
    // Set the unknown field
    void set_unknown(Field& unknown);

    // Preallocate and setup
    void setup();

    // Solve Ax = b
    void solve(const Field& rhs);

    // Combined set and solve
    void solve(Field& unknown, const Field& rhs);

    // Access PETSc KSP object
    KSP Ksp();

    // Iteration count
    PetscInt iterations() const;
};
```

#### samurai::petsc::LinearBlockSolver

```cpp
template <bool monolithic, std::size_t rows, std::size_t cols, class... Operators>
class LinearBlockSolver {
public:
    // Set multiple unknowns
    template <class... Fields>
    void set_unknowns(Fields&... fields);

    // Setup
    void setup();

    // Solve block system
    template <class... RHSFields>
    void solve(const RHSFields&... rhs);

    // Configure field split preconditioner
    void set_pc_fieldsplit(PC pc);
};
```

#### Helper Functions

```cpp
// Create solver from scheme
template <class Scheme>
auto make_solver(const Scheme& scheme);

// One-shot solve
template <class Scheme, class Field>
void solve(const Scheme& scheme, Field& unknown, const Field& rhs);

// Create block operator
template <std::size_t rows, std::size_t cols, class... Operators>
auto make_block_operator(Operators&&... ops);
```

---

## Implementation Details

### File Structure

```
src/
├── bindings/
│   ├── main.cpp              [MODIFY] Add petsc submodule init
│   ├── common_types.hpp      [MODIFY] Add PETSc types
│   ├── petsc_bindings.hpp    [NEW]    PETSc binding declarations
│   └── petsc_bindings.cpp    [NEW]    PETSc binding implementation
└── sampai/
    └── __init__.py           [MODIFY] Export petsc module

tests/
└── test_petsc.py             [NEW]    PETSc tests

docs/
└── PETSc_INTEGRATION_DESIGN.md [NEW]  This document

examples/
├── heat_implicit.py          [NEW]    Heat equation
├── stokes_2d.py              [NEW]    Stokes flow
└── nonlinear_heat.py         [NEW]    Nonlinear diffusion
```

### Conditional Compilation

All PETSc code must be conditionally compiled:

```cpp
#ifdef SAMURAI_WITH_PETSC
    // PETSc bindings
#else
    // Stubs or error messages
#endif
```

### Type Erasure for Scheme

The `LinearSolver<Scheme>` is templated on the scheme type. For Python bindings, we use type erasure:

```cpp
// Type-erased interface
class IPyLinearSolver {
public:
    virtual ~IPyLinearSolver() = default;
    virtual void set_unknown(py::object unknown) = 0;
    virtual void setup() = 0;
    virtual void solve(py::object rhs) = 0;
    virtual PetscInt iterations() const = 0;
};

// Concrete implementation
template <class Scheme>
class PyLinearSolver : public IPyLinearSolver {
    samurai::petsc::LinearSolver<Scheme> solver_;
    // ...
};
```

### Memory Layout Handling

Samurai fields support two memory layouts:
- **SOA** (Structure of Arrays): Component-wise storage
- **AOS** (Array of Structures): Cell-wise storage

The bindings must handle both:

```cpp
if (field.is_soa()) {
    // SOA indexing
} else {
    // AOS indexing
}
```

---

## Build System Integration

### meson_options.txt

```meson
option('with_petsc',
    type: 'boolean',
    value: false,
    description: 'Enable PETSc support (requires PETSc library)')
```

### meson.build

```python
# PETSc detection
petsc_dep = dependency('PETSc',
    required: get_option('with_petsc'),
    method: 'pkg-config')

if petsc_dep.found()
    message('Building with PETSc support')
    petsc_found = true
else
    message('PETSc not found - building without PETSc support')
    petsc_found = false
endif

# Conditional compilation
if petsc_found
    petsc_args = ['-DSAMURAI_WITH_PETSC']
else
    petsc_args = []
endif

# Extension module
sampai_pybind = pp.python_extension(
    '_sampai',
    binding_sources,
    dependencies: [
        samurai_dep,
        petsc_dep,
        # ... other deps
    ],
    cpp_args: petsc_args,
    install: true,
    subdir: 'sampai'
)
```

### pyproject.toml

```toml
[project.optional-dependencies]
test = ["pytest >=7.0", "pytest-cov >=3.0", "h5py >=3.0"]
dev = ["sampai[test]", "black >=22.0", "ruff >=0.1.0", "mypy >=1.0"]
viz = ["matplotlib >=3.0", "ipywidgets >=7.0"]
progress = ["tqdm >=4.60"]
petsc = ["petsc >=3.20", "mpi4py >=3.1"]
all = ["sampai[test,dev,viz,progress,petsc]"]
```

### environment.yml

```yaml
name: sampai-dev
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy>=1.20
  - xtensor
  - hdf5
  - highfive>=2.10
  - pugixml
  - fmt
  - cli11
  - pybind11>=2.13
  # PETSc dependencies (optional)
  - petsc  # [petsc]
  - mpich  # [petsc]
  # Python dependencies
  - pip
  - pip:
      - -e .
      - petsc4py  # [petsc]
      - mpi4py   # [petsc]
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_petsc.py
import pytest
import sampai as sam

petsc = pytest.importorskip("samurai.petsc")

def test_simple_diffusion():
    """Test basic implicit diffusion solve"""
    mesh = sam.mesh.make(box=[0, 1], min_level=4, max_level=4)
    u = sam.field.scalar(mesh, "u")
    u.fill(1.0)

    diff = petsc.diffusion_order2(u, coefficient=0.1)
    id_op = petsc.identity(u)
    scheme = id_op + 0.01 * diff

    petsc.solve(scheme, u_next, u)

    # Verify solution
    assert u_next.max() < u.max()

def test_solver_configuration():
    """Test solver customization"""
    solver = petsc.make_solver(scheme)
    ksp = solver.ksp()
    ksp.setType("cg")
    ksp.pc.setType("jacobi")

    solver.setup()
    solver.solve(rhs)

    assert solver.iterations() > 0

def test_block_operator():
    """Test block operator assembly"""
    v = sam.field.vector(mesh, "v", n_components=2)
    p = sam.field.scalar(mesh, "p")

    stokes = petsc.make_block_operator(2, 2,
        diff_v, grad_p,
        div_v,  zero)

    solver = petsc.make_solver(stokes)
    solver.set_unknowns(velocity=v, pressure=p)
    solver.solve(rhs_v, rhs_p)
```

### Integration Tests

Port C++ demos to Python:
- `heat_implicit.py` - Backward Euler heat equation
- `stokes_2d.py` - Lid-driven cavity
- `nonlinear_heat.py` - Newton solver

### CI Testing

```yaml
# .github/workflows/ci.yml
- name: Test with PETSc
  if: matrix.petsc
  run: |
    conda install -c conda-forge petsc mpich
    pip install petsc4py mpi4py
    pytest tests/test_petsc.py -v

env:
  matrix:
    petsc: [true, false]
```

---

## Documentation

### User Documentation

Add to `docs/user_guide.md`:

```markdown
## Implicit Solvers with PETSc

Sampai provides optional PETSc support for implicit solvers.

### Installation

```bash
conda install -c conda-forge petsc mpich
pip install petsc4py mpi4py
pip install sampai[petsc]
```

### Basic Usage

[Examples from API section]
```

### API Reference

Generate with Sphinx/pybind11:

```python
"""PETSc solvers for implicit schemes.

This module provides interfaces to PETSc linear and nonlinear solvers
for Samurai AMR fields.
"""
```

---

## Future Enhancements

### Phase 2 Features (Post-MVP)

| Feature | Description | Complexity |
|---------|-------------|------------|
| Nonlinear solver bindings | SNES interface | Medium |
| Multigrid preconditioner | Geometric multigrid | High |
| Custom matrix assembly | User-provided assembly | Medium |
| Distributed memory | MPI parallelism | Very High |
| Field split options | Advanced block preconditioners | Low |
| Time integration | Built-in time stepping | Medium |

### Potential Optimizations

1. **Matrix caching** - Reuse matrices across time steps
2. **Preconditioner reuse** - Don't rebuild PC when possible
3. **Jacobian reuse** - For nonlinear problems with slow convergence
4. **GPU support** - PETSc with CUDA

---

## Appendix

### A. PETSc Option Database

Users can control solvers via command line or environment variables:

```bash
# Set solver type
python script.py -ksp_type cg -pc_type gamg

# Set tolerances
python script.py -ksp_rtol 1e-8 -ksp_max_it 1000

# Multigrid options
python script.py -pc_type mg -pc_mg_levels 4
```

### B. Compatibility Matrix

| Sampai Version | PETSc Version | Status |
|----------------|---------------|--------|
| 0.30.x | 3.20+ | Supported |
| 0.30.x | 3.19 | Should work |
| 0.30.x | < 3.18 | Not tested |

### C. References

- PETSc Documentation: https://petsc.org/main/
- Samurai PETSc Headers: `subprojects/samurai/include/samurai/petsc/`
- C++ Demos: `subprojects/samurai/demos/FiniteVolume/`

---

**Document Version History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-14 | Initial design document |
