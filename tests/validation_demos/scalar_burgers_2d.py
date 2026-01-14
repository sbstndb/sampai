#!/usr/bin/env python3
"""
Validation demo: 2D scalar Burgers equation.

This is a Python replication of subprojects/samurai/demos/FiniteVolume/scalar_burgers_2d.cpp
The goal is to produce identical HDF5 outputs for validation testing.

C++ parameters:
    - Domain: [0, 1] x [0, 1]
    - Final time: Tf = 0.001 (for testing)
    - CFL: 0.5
    - min_level: 2
    - max_level: 6
    - Initial solution: circular condition
    - Scheme: WENO5 convection with RK3 time stepping
    - max_stencil_size: 6
    - BC: Dirichlet(0) on all boundaries
"""

import tempfile
from pathlib import Path

import sampai as sam


def _set_level_field(level_field, mesh):
    """Helper: populate level field from mesh cells."""
    def set_level(cell):
        level_field[cell.index] = cell.level
    sam.algorithms.for_each_cell(mesh, set_level)


def init_circular(u, center=(0.5, 0.5), radius=0.2):
    """Initialize field with a circular condition.

    Args:
        u: ScalarField to initialize
        center: Center of the circle (x, y)
        radius: Radius of the circle
    """
    x_center, y_center = center

    def init_cell(cell):
        cx, cy = cell.center()
        dist_sq = (cx - x_center) * (cx - x_center) + (cy - y_center) * (cy - y_center)
        if dist_sq <= radius * radius:
            u[cell.index] = 1.0
        else:
            u[cell.index] = 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def run_scalar_burgers_2d(
    min_corner=(0.0, 0.0),
    max_corner=(1.0, 1.0),
    Tf=0.001,
    cfl=0.5,
    min_level=2,
    max_level=6,
    output_path=None,
    filename="test_finite_volume_demo_finite-volume-scalar-burgers-2d-0.001",
):
    """Run 2D scalar Burgers simulation.

    Exact replication of C++ scalar_burgers_2d.cpp.

    Args:
        min_corner: Minimum corner of box (default: (0.0, 0.0))
        max_corner: Maximum corner of box (default: (1.0, 1.0))
        Tf: Final time (default: 0.001)
        cfl: CFL number (default: 0.5)
        min_level: Minimum mesh level (default: 2)
        max_level: Maximum mesh level (default: 6)
        output_path: Output directory (default: temp dir)
        filename: Output file prefix (default: "scalar_burgers_2D")

    Returns:
        Path to output directory
    """
    # Create output directory
    if output_path is None:
        output_path = Path(tempfile.mkdtemp())
    else:
        output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Mesh configuration - EXACT replication of C++ code
    # ============================================================

    box = sam.geometry.box(list(min_corner), list(max_corner))

    config = sam.config.make(2)
    config.min_level = min_level
    config.max_level = max_level
    config.max_stencil_size = 6

    mesh = sam.mesh.make(box, config)
    u = sam.field.scalar(mesh, "u", init=0.0)
    u1 = sam.field.scalar(mesh, "u1", init=0.0)
    u2 = sam.field.scalar(mesh, "u2", init=0.0)
    unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

    # ============================================================
    # Initialize field with circular condition
    # ============================================================

    # Note: mesh already created above, just initialize the field
    init_circular(u)

    # ============================================================
    # Boundary conditions
    # ============================================================

    sam.boundary.dirichlet(u, 0.0)

    # ============================================================
    # Convection operator setup
    # ============================================================

    # C++: auto conv = samurai::make_convection_weno5<decltype(u)>();
    # Note: cst = 1 for 2D scalar Burgers (f(u) = u^2)
    cst = 1.0

    # ============================================================
    # Time stepping setup
    # ============================================================

    dx = mesh.min_cell_length
    dt = cfl * dx / (2 ** 2)  # pow(2, dim) for dim=2

    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()

    MRadaptation(mra_config)

    # Save initial state (C++ creates level field fresh for each save)
    init_filepath = str(output_path / f"{filename}_init")
    level = sam.field.scalar(mesh, "level", init=0)
    _set_level_field(level, mesh)
    sam.save(init_filepath, u, level)

    restart_filepath = str(output_path / f"{filename}_restart_init")
    sam.dump(restart_filepath, u)

    # ============================================================
    # Time loop
    # ============================================================

    t = 0.0
    nt = 0

    while t != Tf:
        t += dt
        if t > Tf:
            dt += Tf - t
            t = Tf

        MRadaptation(mra_config)
        u1.resize()
        u2.resize()
        unp1.resize()

        # RK3 time scheme
        conv_result = sam.operators.make_convection_weno5(u)
        u1.assign(u - dt * cst * conv_result)

        conv_result1 = sam.operators.make_convection_weno5(u1)
        u2.assign(0.75 * u + 0.25 * (u1 - dt * cst * conv_result1))

        conv_result2 = sam.operators.make_convection_weno5(u2)
        unp1.assign((1.0/3.0) * u + (2.0/3.0) * (u2 - dt * cst * conv_result2))

        sam.swap_field_arrays_2d(u, unp1)

        # Save at final time
        if t >= Tf:
            # C++: save() function creates a fresh level field for each save
            save_filepath = str(output_path / filename)
            level = sam.field.scalar(mesh, "level", init=0)
            _set_level_field(level, mesh)
            sam.save(save_filepath, u, level)

            restart_filepath_final = str(output_path / f"{filename}_restart")
            sam.dump(restart_filepath_final, u)

        nt += 1

    return output_path


if __name__ == "__main__":
    output_dir = run_scalar_burgers_2d(Tf=0.001)
    print(f"Validation demo complete. Output in: {output_dir}")
