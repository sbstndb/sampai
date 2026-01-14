"""
PETSc Operator Arithmetic Demo
================================

This example demonstrates the operator arithmetic functionality
for creating implicit schemes.

Note:
    This example requires PETSc support.
"""

import sampai as sam


def main():
    """Demonstrate operator arithmetic."""

    print("=" * 60)
    print("PETSc Operator Arithmetic Demo")
    print("=" * 60)

    # Create mesh and field
    box = sam.geometry.box([0., 0.], [1., 1.])
    mesh = sam.mesh.make(box, min_level=4, max_level=4)
    u = sam.field.scalar(mesh, "u")

    # ============================================================
    # Basic operators
    # ============================================================

    print("\n1. Basic Operators")
    print("-" * 40)

    id_op = sam.petsc.identity(u)
    print(f"Identity: {repr(id_op)}")

    diff = sam.petsc.diffusion_order2(u, coefficient=0.1)
    print(f"Diffusion: {repr(diff)}")

    # ============================================================
    # Scalar multiplication
    # ============================================================

    print("\n2. Scalar Multiplication")
    print("-" * 40)

    scaled = 2.5 * id_op
    print(f"2.5 * Identity: {repr(scaled)}")

    dt = 0.01
    dt_diff = dt * diff
    print(f"{dt} * Diffusion: {repr(dt_diff)}")

    # ============================================================
    # Operator addition
    # ============================================================

    print("\n3. Operator Addition")
    print("-" * 40)

    sum_op = id_op + diff
    print(f"Identity + Diffusion: {repr(sum_op)}")

    # ============================================================
    # Operator subtraction
    # ============================================================

    print("\n4. Operator Subtraction")
    print("-" * 40)

    sub_op = id_op - diff
    print(f"Identity - Diffusion: {repr(sub_op)}")

    # ============================================================
    # Complex expressions (time stepping schemes)
    # ============================================================

    print("\n5. Time Stepping Schemes")
    print("-" * 40)

    # Backward Euler: (I + dt * L) u^{n+1} = u^n
    backward_euler = id_op + dt * diff
    print(f"Backward Euler: {repr(backward_euler)}")

    # Crank-Nicolson: (I + dt/2 * L) u^{n+1} = (I - dt/2 * L) u^n
    crank_nicolson_lhs = id_op + (dt / 2) * diff
    crank_nicolson_rhs = id_op - (dt / 2) * diff
    print(f"Crank-Nicolson LHS: {repr(crank_nicolson_lhs)}")
    print(f"Crank-Nicolson RHS: {repr(crank_nicolson_rhs)}")

    # ============================================================
    # Preconditioning example
    # ============================================================

    print("\n6. Solver Configuration")
    print("-" * 40)

    # Create a solver (when fully implemented)
    print("Solver configuration:")
    print("  ksp.set_type('gmres')")
    print("  ksp.pc().set_type('gamg')")
    print("  ksp.set_tolerances(rtol=1e-8, max_it=1000)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
