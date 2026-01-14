# PETSc Support in Sampai

This document describes how to use PETSc implicit solvers in Sampai.

## What is PETSc?

PETSc (Portable, Extensible Toolkit for Scientific Computation) is a suite of data structures and routines for the scalable (parallel) solution of scientific applications modeled by partial differential equations.

## Installation

### Using Conda (Recommended)

```bash
# Create environment with PETSc
conda env create -f environment-petsc.yml
conda activate sampai-petsc

# Build and install sampai with PETSc
pip install -e . --global-option="-Dwith_petsc=true"
```

### Manual Installation

```bash
# Install PETSc and MPI
conda install -c conda-forge petsc mpich

# Install mpi4py
pip install mpi4py

# Build sampai with PETSc
meson setup builddir -Dwith_petsc=true
meson compile -C builddir
pip install .
```

## Quick Start

### Basic Implicit Solve

```python
import sampai as sam

# Create mesh and field
box = sam.geometry.box([0., 0.], [1., 1.])
mesh = sam.mesh.make(box, min_level=4, max_level=8)
u = sam.field.scalar(mesh, "u")

# Apply boundary conditions
sam.make_dirichlet_bc(u, value=0.0, order=2)

# Define implicit scheme (Backward Euler)
K = 0.1
dt = 0.01
diff = sam.petsc.diffusion_order2(u, coefficient=K)
id_op = sam.petsc.identity(u)
implicit_scheme = id_op + dt * diff

# Solve
solver = sam.petsc.LinearSolver()
solver.set_unknown(u)
solver.setup()
solver.solve(rhs, u)

print(f"Converged in {solver.iterations()} iterations")
```

### Configuring the Solver

```python
solver = sam.petsc.LinearSolver()
solver.set_unknown(u)

# Get KSP object
ksp = solver.ksp()

# Configure solver type
ksp.set_type("gmres")  # Options: "cg", "gmres", "bcgs", etc.

# Configure preconditioner
ksp.pc().set_type("ilu")  # Options: "jacobi", "ilu", "gamg", "hypre"

# Set tolerances
ksp.set_tolerances(rtol=1e-8, atol=1e-10, max_it=1000)

# Setup and solve
solver.setup()
solver.solve(rhs, u)
```

## API Reference

### Operators

#### `diffusion_order2(field, coefficient)`

Create a second-order diffusion operator.

```python
diff = sam.petsc.diffusion_order2(u, coefficient=0.1)
```

#### `identity(field)`

Create an identity operator.

```python
id_op = sam.petsc.identity(u)
```

### Operator Arithmetic

Operators support `+`, `-`, and scalar `*`:

```python
# Backward Euler: (I + dt*L)
backward_euler = id_op + dt * diff

# Crank-Nicolson: (I + dt/2*L)
crank_nicolson = id_op + (dt / 2) * diff
```

### Classes

#### `LinearSolver`

PETSc linear solver for implicit schemes.

**Methods:**
- `set_unknown(field)` - Set the unknown field
- `setup()` - Assemble matrix and configure solver
- `solve(rhs, unknown)` - Solve the linear system
- `ksp()` - Get the KSP object for configuration
- `iterations()` - Get iteration count from last solve
- `is_set_up()` - Check if solver is set up

#### `KSP`

PETSc Krylov Subspace Methods solver.

**Methods:**
- `set_type(type)` - Set solver type ("cg", "gmres", "bcgs", etc.)
- `get_type()` - Get solver type
- `set_tolerances(rtol, atol, div_tol, max_it)` - Set convergence tolerances
- `set_from_options()` - Set options from PETSc database
- `pc()` - Get preconditioner object
- `get_iteration_number()` - Get iteration count
- `get_residual_norm()` - Get residual norm
- `get_converged_reason()` - Get convergence reason

#### `PC`

PETSc Preconditioner.

**Methods:**
- `set_type(type)` - Set preconditioner type
- `get_type()` - Get preconditioner type

## Solver Types

### KSP Types

Common KSP solver types:

| Type | Description | Use Case |
|------|-------------|----------|
| `cg` | Conjugate Gradient | Symmetric positive definite |
| `gmres` | Generalized Minimal Residual | General nonsymmetric |
| `bcgs` | BiCGStab | General nonsymmetric |
| `cgs` | CG Squared | General nonsymmetric |
| `tfqmr` | TFQMR | General nonsymmetric |

### Preconditioner Types

Common PC types:

| Type | Description | Use Case |
|------|-------------|----------|
| `none` | No preconditioning | Testing |
| `jacobi` | Jacobi | Simple diagonal scaling |
| `bjacobi` | Block Jacobi | Parallel |
| `sor` | SOR | Simple iterative |
| `ilu` | Incomplete LU | General problems |
| `gamg` | Geometric AMG | Large sparse systems |
| `hypre` | Hypre preconditioners | Large-scale parallel |

## Command Line Options

PETSc options can be set via command line:

```bash
python script.py -ksp_type cg -pc_type gamg -ksp_rtol 1e-8
```

## Examples

See the `examples/` directory:

- `heat_implicit.py` - Heat equation with implicit solver
- `operator_arithmetic_demo.py` - Operator arithmetic examples

## Troubleshooting

### PETSc not found

```
Error: PETSc was requested but not found
```

**Solution:** Install PETSc:
```bash
conda install -c conda-forge petsc mpich
```

### Import error

```
ImportError: cannot import name 'petsc'
```

**Solution:** Rebuild with PETSc enabled:
```bash
pip install -e . --global-option="-Dwith_petsc=true"
```

### Solver not converging

**Solution:** Try different solver/preconditioner combinations:
```python
ksp.set_type("cg")
ksp.pc().set_type("gamg")
ksp.set_tolerances(rtol=1e-6, max_it=5000)
```

## Performance Tips

1. **Reuse solvers** - Setup is expensive, reuse the solver for multiple time steps
2. **Choose appropriate preconditioner** - `gamg` for large problems, `ilu` for medium
3. **Monitor convergence** - Use `ksp.get_converged_reason()` to diagnose issues
4. **Consider matrix-free** - For very large problems

## References

- [PETSc Documentation](https://petsc.org/main/)
- [Samurai C++ PETSc Examples](../subprojects/samurai/demos/FiniteVolume/)
- [KSP Solvers](https://petsc.org/main/manualpages/KSP/)
- [Preconditioners](https://petsc.org/main/manualpages/PC/)
