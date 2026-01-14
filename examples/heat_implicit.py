"""
Implicit Heat Equation with PETSc
===================================

This example demonstrates solving the heat equation using an implicit
Backward Euler scheme with PETSc solvers.

Equation:
    ∂u/∂t = K ∇²u

Discretization:
    (u^{n+1} - u^n) / Δt = K ∇²u^{n+1}
    (I - Δt * K * L) u^{n+1} = u^n

where L is the discrete Laplacian operator.

Note:
    This example requires PETSc support. Build with:
    meson setup builddir -Dwith_petsc=true
"""

import sampai as sam
import numpy as np


def main():
    """Solve heat equation with implicit scheme."""

    # ============================================================
    # Problem parameters
    # ============================================================

    # Domain
    a, b = 0.0, 1.0  # x and y domain bounds

    # Physics
    K = 0.1          # Diffusion coefficient
    T_final = 0.1    # Final time

    # Discretization
    min_level = 4    # Minimum mesh level
    max_level = 4    # Maximum mesh level (uniform for simplicity)
    dt = 0.01        # Time step

    # ============================================================
    # Create mesh and field
    # ============================================================

    print("Creating mesh...")
    box = sam.geometry.box([a, a], [b, b])
    mesh = sam.mesh.make(box, min_level=min_level, max_level=max_level)

    print(f"Mesh: {mesh.nb_cells()} cells")

    # Create field
    u = sam.field.scalar(mesh, "u")

    # ============================================================
    # Set initial condition
    # ============================================================

    print("Setting initial condition...")

    def initial_condition(x, y):
        """Gaussian initial condition."""
        x0, y0 = 0.5, 0.5
        sigma = 0.1
        r2 = (x - x0)**2 + (y - y0)**2
        return np.exp(-r2 / sigma**2)

    # Apply initial condition
    for cell in mesh.cells():
        x, y = cell.center()
        u[cell] = initial_condition(x, y)

    # ============================================================
    # Boundary conditions
    # ============================================================

    print("Applying boundary conditions...")
    sam.make_dirichlet_bc(u, value=0.0, order=2)

    # ============================================================
    # Define implicit scheme
    # ============================================================

    print("Setting up implicit scheme...")

    # Create operators
    diff = sam.petsc.diffusion_order2(u, coefficient=K)
    id_op = sam.petsc.identity(u)

    # Backward Euler: (I + dt * K * L) u^{n+1} = u^n
    implicit_scheme = id_op + dt * diff

    # ============================================================
    # Setup solver
    # ============================================================

    print("Configuring solver...")
    solver = sam.petsc.LinearSolver()
    solver.set_unknown(u)

    # Configure PETSc KSP
    ksp = solver.ksp()
    ksp.set_type("gmres")
    ksp.pc().set_type("ilu")
    ksp.set_tolerances(rtol=1e-8, atol=1e-10, max_it=1000)

    # Setup (assemble matrix)
    solver.setup()

    # ============================================================
    # Time stepping
    # ============================================================

    t = 0.0
    nt = int(T_final / dt)

    print(f"\nTime stepping: {nt} steps, dt = {dt}")

    for n in range(nt):
        t += dt

        # Save old solution as RHS
        u_old = u.copy()

        # Solve implicit system
        solver.solve(u_old, u)

        # Get solver statistics
        it = solver.iterations()

        if n % 10 == 0:
            umax = u.max()
            umin = u.min()
            print(f"  t = {t:.4f}, iter = {it:3d}, u_max = {umax:.6e}, u_min = {umin:.6e}")

    # ============================================================
    # Final output
    # ============================================================

    print("\nFinal solution:")
    print(f"  u_max = {u.max():.6e}")
    print(f"  u_min = {u.min():.6e}")
    print(f"  Total iterations: {solver.iterations()}")

    # Save to file
    sam.save("u_heat_implicit", u)
    print("\nSolution saved to: u_heat_000000.h5")


if __name__ == "__main__":
    main()
