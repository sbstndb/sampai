#!/usr/bin/env python3
"""
Validation demo: 1D Burgers equation.

This is a Python replication of subprojects/samurai/demos/FiniteVolume/burgers.cpp
The goal is to produce identical HDF5 outputs for validation testing.

C++ parameters:
    - Domain: [-1, 1]
    - Final time: Tf = 0.1 (for testing)
    - CFL: 0.95
    - min_level: 0
    - max_level: 5
    - Initial solution: "hat"
    - Scheme: WENO5 convection with RK3 time stepping
    - max_stencil_size: 6
    - BC: Dirichlet(0) on both sides
"""

import tempfile
from pathlib import Path

import sampai as sam


def _set_level_field(level_field, mesh):
    """Helper: populate level field from mesh cells."""
    def set_level(cell):
        level_field[cell.index] = cell.level
    sam.algorithms.for_each_cell(mesh, set_level)


def init_hat(u, max_val=1.0, radius=0.5):
    """Initialize field with a hat function.

    Exact replication of C++ init_sol == "hat" condition.

    Args:
        u: ScalarField to initialize
        max_val: Maximum value at center (default: 1.0)
        radius: Radius of the hat (default: 0.5)
    """
    def init_cell(cell):
        cx = cell.center()[0]
        dist = abs(cx)
        if dist <= radius:
            u[cell.index] = (-max_val / radius) * dist + max_val
        else:
            u[cell.index] = 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def run_burgers_1d(
    left_box=-1.0,
    right_box=1.0,
    Tf=0.1,
    cfl=0.95,
    min_level=0,
    max_level=5,
    output_path=None,
    filename="test_finite_volume_demo_burgers",
):
    """Run 1D Burgers simulation.

    Exact replication of C++ burgers.cpp for dim=1, n_comp=1.

    Args:
        left_box: Left boundary of domain (default: -1.0)
        right_box: Right boundary of domain (default: 1.0)
        Tf: Final time (default: 0.1)
        cfl: CFL number (default: 0.95)
        min_level: Minimum mesh level (default: 0)
        max_level: Maximum mesh level (default: 5)
        output_path: Output directory (default: temp dir)
        filename: Output file prefix (default: "burgers_1D")

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

    box = sam.geometry.box([left_box], [right_box])

    # C++: auto config = samurai::mesh_config<dim>()
    #              .min_level(min_level).max_level(max_level)
    #              .max_stencil_size(6);
    config = sam.config.make(1)
    config.min_level = min_level
    config.max_level = max_level
    config.max_stencil_size = 6

    # C++: auto mesh = samurai::mra::make_empty_mesh(config);
    # C++: auto u = samurai::make_vector_field<n_comp>("u", mesh);
    # For scalar Burgers, n_comp = 1
    mesh = sam.mesh.make(box, config)
    u = sam.field.scalar(mesh, "u", init=0.0)

    # C++: auto u1 = samurai::make_vector_field<n_comp>("u1", mesh);
    # C++: auto u2 = samurai::make_vector_field<n_comp>("u2", mesh);
    # C++: auto unp1 = samurai::make_vector_field<n_comp>("unp1", mesh);
    u1 = sam.field.scalar(mesh, "u1", init=0.0)
    u2 = sam.field.scalar(mesh, "u2", init=0.0)
    unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

    # ============================================================
    # Initialize field with hat function
    # ============================================================

    # C++: mesh = samurai::mra::make_mesh(box, config);
    # C++: u.resize();
    # C++: if (init_sol == "hat") { ... }
    # Note: mesh already created, just initialize the field
    init_hat(u)

    # ============================================================
    # Boundary conditions
    # ============================================================

    # C++: samurai::make_bc<samurai::Dirichlet<3>>(u, 0.0);
    sam.boundary.dirichlet(u, 0.0)

    # C++: u1.copy_bc_from(u);
    # C++: u2.copy_bc_from(u);
    # This is handled automatically in the Python bindings

    # ============================================================
    # Convection operator setup
    # ============================================================

    # C++: double cst = dim == 1 ? 0.5 : 1; // if dim == 1, we want f(u) = (1/2)*u^2
    # C++: auto conv = cst * samurai::make_convection_weno5<decltype(u)>();
    cst = 0.5  # 1D case

    # ============================================================
    # Time stepping setup
    # ============================================================

    # C++: double dx = mesh.min_cell_length();
    # C++: dt = cfl * dx / pow(2, dim);
    dx = mesh.min_cell_length
    dt = cfl * dx

    # C++: auto MRadaptation = samurai::make_MRAdapt(u);
    # C++: auto mra_config = samurai::mra_config();
    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()
    # Note: C++ doesn't set epsilon for burgers, uses default

    # C++: MRadaptation(mra_config);
    # C++: save(...);  // initial save
    MRadaptation(mra_config)

    # Save initial state (C++ creates level field fresh for each save)
    init_filepath = str(output_path / f"{filename}_init")
    level = sam.field.scalar(mesh, "level", init=0)
    _set_level_field(level, mesh)
    sam.save(init_filepath, u, level)

    # Also save restart file
    restart_filepath = str(output_path / f"{filename}_restart_init")
    sam.dump(restart_filepath, u)

    # ============================================================
    # Time loop - EXACT replication of C++ logic
    # ============================================================

    t = 0.0
    nt = 0

    # C++: while (t != Tf)
    while t != Tf:
        # C++: t += dt;
        # C++: if (t > Tf) { dt += Tf - t; t = Tf; }
        t += dt
        if t > Tf:
            dt += Tf - t
            t = Tf

        # C++: std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt) << std::flush;
        # (silently for testing)

        # C++: MRadaptation(mra_config);
        # C++: u1.resize();
        # C++: u2.resize();
        # C++: unp1.resize();
        MRadaptation(mra_config)
        u1.resize()
        u2.resize()
        unp1.resize()

        # ============================================================
        # RK3 time scheme
        # ============================================================

        # C++: u1 = u - dt * conv(u);
        conv_result = sam.operators.make_convection_weno5(u)
        u1.assign(u - dt * cst * conv_result)

        # C++: u2 = 3. / 4 * u + 1. / 4 * (u1 - dt * conv(u1));
        conv_result1 = sam.operators.make_convection_weno5(u1)
        u2.assign(0.75 * u + 0.25 * (u1 - dt * cst * conv_result1))

        # C++: unp1 = 1. / 3 * u + 2. / 3 * (u2 - dt * conv(u2));
        conv_result2 = sam.operators.make_convection_weno5(u2)
        unp1.assign((1.0/3.0) * u + (2.0/3.0) * (u2 - dt * cst * conv_result2))

        # C++: samurai::swap(u, unp1);
        sam.swap_field_arrays_1d(u, unp1)

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
    output_dir = run_burgers_1d(Tf=0.1)
    print(f"Validation demo complete. Output in: {output_dir}")
