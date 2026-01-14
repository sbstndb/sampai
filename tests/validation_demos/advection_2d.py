#!/usr/bin/env python3
"""
Validation demo: 2D Advection equation.

This is a Python replication of subprojects/samurai/demos/FiniteVolume/advection_2d.cpp
The goal is to produce identical HDF5 outputs for validation testing.

C++ parameters:
    - Domain: [0, 1] x [0, 1]
    - Velocity: a = (1, 1)
    - Final time: Tf = 0.01 (for testing)
    - CFL: 0.5
    - min_level: 4
    - max_level: 10
    - epsilon: 2e-4
    - BC: Dirichlet(0) on all boundaries
    - Initial condition: circle at (0.3, 0.3), radius=0.2
"""

import tempfile
from pathlib import Path

import sampai as sam


def init_circular(u, center=(0.3, 0.3), radius=0.2):
    """Initialize field with a circular condition.

    Exact replication of C++ init() function.

    Args:
        u: ScalarField to initialize
        center: Center of the circle (default: (0.3, 0.3))
        radius: Radius of the circle (default: 0.2)
    """
    x_center, y_center = center

    def init_cell(cell):
        cx, cy = cell.center()
        # C++: if (((center[0] - x_center) * (center[0] - x_center) +
        #           (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
        dist_sq = (cx - x_center) * (cx - x_center) + (cy - y_center) * (cy - y_center)
        if dist_sq <= radius * radius:
            u[cell.index] = 1.0
        else:
            u[cell.index] = 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def run_advection_2d(
    min_corner=(0.0, 0.0),
    max_corner=(1.0, 1.0),
    velocity=(1.0, 1.0),
    Tf=0.01,
    cfl=0.5,
    min_level=4,
    max_level=10,
    epsilon=2e-4,
    output_path=None,
    filename="test_finite_volume_demo_finite-volume-advection-2d-0.01",
):
    """Run 2D advection simulation.

    Exact replication of C++ advection_2d.cpp main() function.

    Args:
        min_corner: Minimum corner of box (default: (0.0, 0.0))
        max_corner: Maximum corner of box (default: (1.0, 1.0))
        velocity: Advection velocity (default: (1.0, 1.0))
        Tf: Final time (default: 0.01)
        cfl: CFL number (default: 0.5)
        min_level: Minimum mesh level (default: 4)
        max_level: Maximum mesh level (default: 10)
        epsilon: MRA adaptation threshold (default: 2e-4)
        output_path: Output directory (default: temp dir)
        filename: Output file prefix (default: "FV_advection_2d")

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

    # C++: xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    # C++: xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    # C++: std::array<double, dim> a{{1, 1}};
    box = sam.geometry.box(list(min_corner), list(max_corner))
    a = list(velocity)

    # C++: auto config = samurai::mesh_config<dim>()
    #              .min_level(4).max_level(10).max_stencil_size(2)
    #              .disable_minimal_ghost_width();
    config = sam.config.make(2)
    config.min_level = min_level
    config.max_level = max_level
    config.max_stencil_size = 2

    # C++: auto mesh = samurai::mra::make_empty_mesh(config);
    # C++: auto u = samurai::make_scalar_field<double>("u", mesh);
    mesh = sam.mesh.make(box, config)
    u = sam.field.scalar(mesh, "u", init=0.0)

    # ============================================================
    # Initialize field
    # ============================================================

    # C++: mesh = samurai::mra::make_mesh(box, config);
    # C++: init(u);
    # Note: mesh already created above with sam.mesh.make(box, config)
    init_circular(u)

    # ============================================================
    # Boundary conditions
    # ============================================================

    # C++: samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    sam.boundary.dirichlet(u, 0.0)

    # ============================================================
    # Time stepping setup
    # ============================================================

    # C++: double dt = cfl * mesh.min_cell_length();
    dt = cfl * mesh.min_cell_length

    # C++: const double dt_save = Tf / static_cast<double>(nfiles);
    # With nfiles=1, dt_save = Tf

    # C++: auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    unp1 = sam.field.scalar(mesh, "unp1", init=0.0)

    # ============================================================
    # MRA configuration
    # ============================================================

    # C++: auto MRadaptation = samurai::make_MRAdapt(u);
    # C++: auto mra_config = samurai::mra_config().epsilon(2e-4);
    MRadaptation = sam.adaptation.make_MRAdapt(u)
    mra_config = sam.config.MRAConfig()
    mra_config.epsilon = epsilon

    # C++: MRadaptation(mra_config);
    # C++: save(path, filename, u, "_init");
    MRadaptation(mra_config)

    # Save initial state (C++ creates level field fresh for each save)
    def save_with_level(filepath, field):
        """Helper to save field with level field (like C++ save function)."""
        level = sam.field.scalar(mesh, "level", init=0)
        def set_level(cell):
            level[cell.index] = cell.level
        sam.algorithms.for_each_cell(mesh, set_level)
        sam.save(filepath, field, level)

    init_filepath = str(output_path / f"{filename}_init")
    save_with_level(init_filepath, u)

    # Also save restart file (C++: dump(..., "_restart_init"))
    restart_filepath = str(output_path / f"{filename}_restart_init")
    sam.dump(restart_filepath, u)

    # ============================================================
    # Time loop - EXACT replication of C++ logic
    # ============================================================

    # C++: std::size_t nsave = 1;
    # C++: std::size_t nt = 0;
    nsave = 1
    nt = 0

    t = 0.0

    # C++: while (t != Tf)
    while t != Tf:
        # C++: MRadaptation(mra_config);
        MRadaptation(mra_config)

        # C++: t += dt;
        # C++: if (t > Tf) { dt += Tf - t; t = Tf; }
        t += dt
        if t > Tf:
            dt += Tf - t
            t = Tf

        # C++: std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;
        # (silently for testing)

        # C++: samurai::update_ghost_mr(u);
        sam.adaptation.update_ghost_mr(u)

        # C++: unp1.resize();
        # C++: unp1 = u - dt * samurai::upwind(a, u);
        unp1.resize()
        unp1.assign(u - dt * sam.operators.upwind(u, a))

        # C++: std::swap(u.array(), unp1.array());
        sam.swap_field_arrays_2d(u, unp1)

        # C++: if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        # With nfiles=1, dt_save = Tf, so this is t >= Tf or t == Tf
        if t >= Tf:
            # C++: const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            # With nfiles=1, suffix = ""
            save_filepath = str(output_path / filename)
            save_with_level(save_filepath, u)

            # C++: Also dump restart file at final time
            restart_filepath_final = str(output_path / f"{filename}_restart")
            sam.dump(restart_filepath_final, u)

        nt += 1

    return output_path


if __name__ == "__main__":
    # Run with default parameters for testing
    output_dir = run_advection_2d(Tf=0.01)
    print(f"Validation demo complete. Output in: {output_dir}")
