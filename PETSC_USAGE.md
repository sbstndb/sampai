# PETSc Usage with Sampai

This guide explains how to use PETSc (Portable, Extensible Toolkit for Scientific Computation) with Sampai for solving linear systems arising from finite volume and finite difference discretizations on AMR meshes.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Basic PETSc Usage](#basic-petsc-usage)
4. [PETSc with Sampai Fields](#petsc-with-sampai-fields)
5. [Creating Solvers](#creating-solvers)
6. [Advanced Topics](#advanced-topics)
7. [Examples](#examples)
8. [References](#references)

---

## Overview

PETSc provides:
- **KSP**: Krylov subspace methods (CG, GMRES, BiCGStab, etc.)
- **PC**: Preconditioners (Jacobi, ILU, Hypre, Multigrid, etc.)
- **Mat**: Sparse matrix operations
- **Vec**: Vector operations

Sampai's PETSc integration consists of:
1. **C++ bindings** for basic PETSc initialization and configuration
2. **Python utilities** for converting between Sampai fields and PETSc vectors/matrices
3. **Integration with petsc4py** for full PETSc functionality

---

## Installation

### Environment Setup

To use PETSc with Sampai, you need to create a dedicated environment:

```bash
conda env create -f environment_petsc.yml
conda activate sampai-petsc
```

The `environment_petsc.yml` includes:
- PETSc (>= 3.20)
- petsc4py (>= 3.20)
- MPI implementation (mpich)
- External packages: Hypre, Metis, SuperLU_DIST
- Boost with MPI support

### Building Sampai with PETSc

Build Sampai with PETSc enabled:

```bash
pip install -e . --config-settings=setup-args=-Dmpi=true --config-settings=setup-args=-Dpetsc=true
```

Or using Meson directly:

```bash
meson setup builddir -Dmpi=true -Dpetsc=true
meson compile -C builddir
pip install -e .
```

---

## Basic PETSc Usage

### Checking PETSc Status

```python
from sampai import petsc

# Check if PETSc is enabled and initialized
print(f"PETSc version: {petsc.get_version()}")
print(f"PETSc initialized: {petsc.is_initialized()}")
print(f"MPI processes: {petsc.get_world_size()}")
print(f"MPI rank: {petsc.get_world_rank()}")
```

### Setting PETSc Options

PETSc options can be set programmatically (equivalent to command-line arguments):

```python
from sampai import petsc

# Set solver type
petsc.set_option("-ksp_type", "cg")
petsc.set_option("-pc_type", "hypre")

# Set tolerances
petsc.set_option("-ksp_rtol", "1e-8")
petsc.set_option("-ksp_atol", "1e-12")
petsc.set_option("-ksp_max_it", "1000")

# Use options prefix for multiple solvers
petsc.set_options_prefix("solver1_")
petsc.set_option("-ksp_type", "gmres")
petsc.clear_options_prefix()
```

---

## PETSc with Sampai Fields

### Converting Fields to PETSc Vectors

```python
from sampai import field, mesh
from sampai.geometry import box
from sampai.config import MeshConfig
from sampai.petsc import field_to_array, create_petsc_vec_from_field

# Create mesh and field
cfg = MeshConfig()
cfg.level = 4
m = mesh.make(cfg, box([0.0, 0.0], [1.0, 1.0]))
u = field.scalar(m, "u")

# Initialize field (example: u = 1 everywhere)
for cell in m.cells():
    u[cell] = 1.0

# Convert to PETSc vector
vec = create_petsc_vec_from_field(u)

# Or just get the numpy array
arr = field_to_array(u)
```

### Copying Between Fields and PETSc Vectors

```python
from sampai.petsc import copy_field_to_vec, copy_vec_to_field

# Copy field to PETSc RHS vector
rhs_vec = create_petsc_vec_from_field(rhs_field)
copy_field_to_vec(rhs_field, rhs_vec)

# After solving, copy solution back to field
copy_vec_to_field(solution_vec, u)
```

---

## Creating Solvers

### Using Convenience Functions

```python
from sampai.petsc import create_cg_solver, solve_with_ksp
from petsc4py import PETSc

# Create a CG solver with Hypre preconditioning
ksp = create_cg_solver(
    pc_type="hypre",
    rtol=1e-8,
    max_it=1000
)

# Assuming you have A (matrix) and b (RHS vector)
x = solve_with_ksp(ksp, A, b)
```

### Using the KSPSolver Class

```python
from sampai.petsc import KSPSolver

# Create solver with configuration
solver = KSPSolver(
    ksp_type="gmres",
    pc_type="gamg",
    rtol=1e-8,
    max_it=1000
)

# Enable convergence monitoring
solver.set_monitor(enabled=True)

# Solve
x = solver.solve(A, b)

# Check convergence
if solver.is_converged:
    print(f"Converged in {solver.iters} iterations")
else:
    print(f"Solver failed: {solver.converged_reason}")
```

### Available Solver Types

#### KSP (Krylov Subspace) Methods

- `"cg"` - Conjugate Gradient (symmetric positive definite)
- `"gmres"` - Generalized Minimal Residual (non-symmetric)
- `"bcgs"` - Bi-Conjugate Gradient Squared
- `"bicg"` - Bi-Conjugate Gradient
- `"tfqmr"` - Transpose-Free Quasi-Minimal Residual
- `"preonly"` - Preconditioner only (for direct solvers)

#### Preconditioners

- `"jacobi"` - Jacobi (diagonal) preconditioning
- `"ilu"` - Incomplete LU factorization
- `"icc"` - Incomplete Cholesky factorization
- `"hypre"` - Hypre algebraic multigrid (recommended)
- `"gamg"` - Geometric algebraic multigrid
- `"asm"` - Additive Schwarz
- `"bjacobi"` - Block Jacobi
- `"lu"` - Direct LU factorization
- `"sor"` - Successive Over-Relaxation

---

## Advanced Topics

### Matrix Assembly

For manual matrix assembly with PETSc:

```python
from petsc4py import PETSc

# Create a sparse matrix (AIJ format)
m = mesh.nb_cells()  # number of rows
n = mesh.nb_cells()  # number of columns

# Estimate non-zeros per row for preallocation
nnz = [7] * m  # e.g., 7-point stencil in 3D

A = PETSc.Mat().createAIJ((m, n), nnz=nnz)
A.setFromOptions()  # Allow command-line overrides
A.setUp()

# Assemble matrix (example for Laplacian)
for cell in mesh.cells():
    row = cell.index
    # Diagonal entry
    A.setValue(row, row, 4.0, addv=PETSc.InsertMode.ADD_VALUES)
    # Off-diagonal entries (neighbors)
    for neighbor in cell.neighbors:
        A.setValue(row, neighbor.index, -1.0, addv=PETSc.InsertMode.ADD_VALUES)

# Complete assembly
A.assembleBegin()
A.assembleEnd()
```

### Using Samurai's PETSc Integration

Samurai C++ has extensive PETSc integration for matrix assembly. The Python bindings provide utilities to interface with petsc4py:

```python
# For advanced matrix assembly, you can use petsc4py directly
# The field_to_array utility provides access to field data
from sampai.petsc import field_to_array

# Access field data as numpy array for custom assembly
u_data = field_to_array(u)
```

### Multiple Solvers with Options Prefix

```python
from sampai import petsc
from sampai.petsc import KSPSolver

# Configure two different solvers
petsc.set_options_prefix("velocity_")
velocity_solver = KSPSolver(ksp_type="gmres", pc_type="hypre")

petsc.set_options_prefix("pressure_")
pressure_solver = KSPSolver(ksp_type="cg", pc_type="gamg")

petsc.clear_options_prefix()
```

### Monitor Custom Residual

```python
from petsc4py import PETSc

def residual_monitor(ksp, its, rnorm):
    print(f"Iteration {its}: residual norm = {rnorm:.3e}")

ksp = create_cg_solver()
ksp.setMonitor(residual_monitor)
```

---

## Examples

### Example 1: Solving Poisson Equation

```python
"""
Solve -Laplace(u) = f on [0,1]^2 with Dirichlet BCs
using PETSc CG solver with Hypre preconditioning.
"""
from sampai import field, mesh
from sampai.geometry import box
from sampai.config import MeshConfig
from sampai.boundary import dirichlet
from sampai.petsc import create_cg_solver, field_to_array, array_to_field
from petsc4py import PETSc
import numpy as np

# Create mesh
cfg = MeshConfig()
cfg.level = 6
m = mesh.make(cfg, box([0.0, 0.0], [1.0, 1.0]))

# Create field and apply BC
u = field.scalar(m, "u")
bc = dirichert(0.0, on_domain_boundary=lambda cell: cell.is_on_domain_boundary())
bc.apply_to(u)

# Build system matrix A (simplified - use Samurai operators in practice)
# ... matrix assembly code here ...

# Create RHS
rhs = field.scalar(m, "rhs")
# ... fill rhs based on f ...

# Convert to PETSc
b = create_petsc_vec_from_field(rhs)

# Create and configure solver
ksp = create_cg_solver(pc_type="hypre", rtol=1e-8, max_it=1000)

# Solve
x = b.duplicate()
ksp.solve(b, x)

# Copy solution back
array_to_field(x.getArray(), u)

print(f"Converged in {ksp.getIterationNumber()} iterations")
```

### Example 2: System of Equations (Velocity-Pressure)

```python
"""
Solve a small system using block structure.
"""
from sampai.petsc import KSPSolver

# Velocity solver (GMRES)
vel_solver = KSPSolver(ksp_type="gmres", pc_type="hypre")

# Pressure solver (CG for symmetric system)
press_solver = KSPSolver(ksp_type="cg", pc_type="gamg")

# Solve velocity
v = vel_solver.solve(A_vel, b_vel)

# Solve pressure
p = press_solver.solve(A_press, b_press - C @ v)
```

---

## References

### PETSc Documentation
- [PETSc User Manual](https://petsc.org/main/manual/)
- [KSP Solvers](https://petsc.org/main/manualpages/KSP/)
- [PC Preconditioners](https://petsc.org/main/manualpages/PC/)

### petsc4py Documentation
- [petsc4py Documentation](https://petsc4py.readthedocs.io/)

### Samurai PETSc Integration
- Samurai C++ has extensive PETSc integration for matrix assembly
- See `subprojects/samurai/include/samurai/petsc/` for reference

### Command-Line Options

PETSc can be configured via command line:

```bash
python script.py \
    -ksp_type gmres \
    -pc_type gamg \
    -ksp_rtol 1e-8 \
    -ksp_monitor \
    -log_view
```

Use `-ksp_monitor` to see iteration progress and `-log_view` for performance summary.
