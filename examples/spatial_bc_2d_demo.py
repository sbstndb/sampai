#!/usr/bin/env python3
"""
2D Finite Volume example demonstrating spatial boundary conditions with wave patterns.

This demo demonstrates:
- New spatial BC functionality: make_spatial_bc() for region-dependent BCs
- New functional BC: make_function_bc() for coordinate-dependent BCs
- 2D convection with sinusoidal "jet" inlet on left boundary
- Wave propagation patterns visualized in Paraview

The linear convection equation: du/dt + v·∇u = 0
Creates wave patterns from spatially-varying boundary conditions.

Usage:
    python spatial_bc_2d_demo.py                    # Run with sinusoidal jet
    python spatial_bc_2d_demo.py --spatial          # Run with discontinuous BC
    python spatial_bc_2d_demo.py --no-plot         # Disable visualization
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sampai as sam
from sampai.utils import viz


# ============================================================
# Boundary Condition Definitions
# ============================================================

def setup_sinusoidal_jet_bc(u, amplitude=1.0, frequency=3.0):
    """Setup sinusoidal jet BC on left boundary.

    Creates a sinusoidal profile along the left edge (x = -1):
    u(-1, y) = amplitude * sin(frequency * pi * (y + 1) / 2)

    This demonstrates make_function_bc() where value depends on space.

    Args:
        u: Scalar field
        amplitude: Amplitude of the sinusoidal profile
        frequency: Number of waves along the boundary
    """
    # 2D sinusoidal profile on left boundary: varies with y
    sam.make_function_bc(
        u,
        lambda coords: amplitude * np.sin(frequency * np.pi * (coords[1] + 1.0) / 2.0),
        order=2
    )


def setup_spatial_shock_bc(u, shock_y=0.0, bottom_val=1.0, top_val=0.0):
    """Setup spatial shock BC on left boundary.

    Creates a discontinuous (shock) profile along the left edge:
    - Bottom half (y < 0): u = bottom_val
    - Top half (y >= 0): u = top_val

    This demonstrates make_spatial_bc() where both WHERE and WHAT depend on space.

    Args:
        u: Scalar field
        shock_y: Y-position of the shock (normalized)
        bottom_val: Value in bottom region
        top_val: Value in top region
    """
    # Bottom half: high value
    sam.make_spatial_bc(
        u,
        lambda coords: coords[1] < shock_y,  # Region: bottom half
        lambda coords: bottom_val,
        order=2
    )

    # Top half: low value
    sam.make_spatial_bc(
        u,
        lambda coords: coords[1] >= shock_y,  # Region: top half
        lambda coords: top_val,
        order=2
    )


def setup_gaussian_jet_bc(u, center_y=0.0, width=0.3, amplitude=1.0):
    """Setup Gaussian jet BC on left boundary.

    Creates a Gaussian profile along the left edge:
    u(-1, y) = amplitude * exp(-(y - center_y)^2 / (2*width^2))

    Args:
        u: Scalar field
        center_y: Center of Gaussian (y-coordinate)
        width: Width of Gaussian
        amplitude: Peak amplitude
    """
    sam.make_function_bc(
        u,
        lambda coords: amplitude * np.exp(-((coords[1] - center_y)**2) / (2 * width**2)),
        order=2
    )


# ============================================================
# Initial Conditions
# ============================================================

def init_zero_2d(u):
    """Initialize 2D field with zeros (BC will drive the solution)."""
    def init_cell(cell):
        u[cell.index] = 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def init_gaussian_pulse_2d(u, center=(0.0, 0.0), width=0.2, amplitude=1.0):
    """Initialize 2D field with Gaussian pulse."""
    def init_cell(cell):
        x, y = cell.center()
        r2 = (x - center[0])**2 + (y - center[1])**2
        u[cell.index] = amplitude * np.exp(-r2 / (2 * width**2))

    sam.algorithms.for_each_cell(u.mesh, init_cell)


# ============================================================
# 2D Demo
# ============================================================

def run_2d_demo(args):
    """Run 2D linear convection with spatial BCs."""

    print("\n" + "=" * 70)
    print("2D Linear Convection - Spatial Boundary Conditions Demo")
    print("=" * 70)

    # Domain and mesh
    box = sam.geometry.box([-1.0, -1.0], [1.0, 1.0])
    config = sam.config.make(2)
    config.min_level = args.min_level
    config.max_level = args.max_level
    config.max_stencil_size = 6  # Required for WENO5
    mesh = sam.mesh.make(box, config)

    # Time parameters
    Tf = args.tf
    velocity = [1.0, 0.5]  # Convection diagonally
    cfl = args.cfl

    # Output parameters
    output_path = Path("./spatial_bc_2d_results")
    nfiles = args.nfiles

    # Setup BC based on selection
    print(f"\nBoundary condition: {args.bc_type}")

    if args.bc_type == "sinusoidal":
        print(f"  Sinusoidal jet on left boundary")
        print(f"  Amplitude: {args.amplitude}, Frequency: {args.frequency} waves")
        bc_setup = lambda u: setup_sinusoidal_jet_bc(u, args.amplitude, args.frequency)
    elif args.bc_type == "spatial":
        print(f"  Discontinuous shock at y = {args.shock_y}")
        print(f"  Bottom: u = {args.bottom_val}, Top: u = {args.top_val}")
        bc_setup = lambda u: setup_spatial_shock_bc(u, args.shock_y, args.bottom_val, args.top_val)
    elif args.bc_type == "gaussian":
        print(f"  Gaussian jet at y = {args.gaussian_center}")
        print(f"  Width: {args.gaussian_width}, Amplitude: {args.amplitude}")
        bc_setup = lambda u: setup_gaussian_jet_bc(u, args.gaussian_center, args.gaussian_width, args.amplitude)
    else:
        # Default: sinusoidal
        print(f"  Sinusoidal jet on left boundary")
        print(f"  Amplitude: {args.amplitude}, Frequency: {args.frequency} waves")
        bc_setup = lambda u: setup_sinusoidal_jet_bc(u, args.amplitude, args.frequency)

    # Create fields
    u = sam.field.zeros(mesh, "u")
    unp1 = sam.field.zeros(mesh, "unp1")

    # Initialize with zeros (BC will drive the solution)
    init_zero_2d(u)

    # Setup BC
    bc_setup(u)

    # Time stepping
    min_cell_length = mesh.min_cell_length
    vel_mag = np.sqrt(sum(v**2 for v in velocity))
    dt = cfl * min_cell_length / vel_mag

    print(f"\nTime stepping parameters:")
    print(f"  Min cell length: {min_cell_length:.6e}")
    print(f"  Time step: {dt:.6e}")
    print(f"  Expected steps: ~{int(Tf/dt)}")
    print(f"  Output: {output_path}/")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"spatial_bc_2d_{args.bc_type}"

    # Save initial condition
    sam.save(str(output_path / f"{filename}_init"), u)
    print(f"  Saved initial condition to {output_path / f'{filename}_init.h5'}")

    # Setup visualization
    plotter = None
    if not args.no_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 8))
        plotter = viz.FieldPlotter(u, ax=ax, cmap='RdBu_r',
                                     vmin=-args.amplitude, vmax=args.amplitude, show_mesh=True)
        plt.pause(0.01)

    # Time loop
    t = 0.0
    iteration = 0
    dt_save = Tf / nfiles
    next_save = dt_save
    nsave = 0
    plot_interval = max(1, int(Tf / dt / 20))

    print(f"\n{'Iter':>6} {'Time':>12} {'dt':>12} {'Cells':>10}")
    print("-" * 42)

    while t < Tf:
        # Update ghost cells (BCs are automatically applied)
        sam.adaptation.update_ghost_mr(u)

        # Compute flux (Linear convection: du/dt + v·∇u = 0)
        flux = sam.operators.convection_weno5(u, velocity)

        # Euler step
        unp1.assign(u - dt * flux)

        # Swap
        u, unp1 = unp1, u
        u.name = "u"

        t += dt
        iteration += 1

        # Save output
        if t >= next_save:
            sam.save(str(output_path / f"{filename}_{nsave:05d}"), u)
            nsave += 1
            next_save += dt_save

        # Update visualization
        if not args.no_plot and plotter is not None and iteration % plot_interval == 0:
            plotter.update(u, title=f"{args.bc_type.capitalize()} BC - t={t:.3f}, cells={mesh.nb_cells}")
            plt.pause(0.01)

        # Print progress
        if iteration % max(1, int(Tf / dt / 10)) == 0:
            print(f"{iteration:6d} {t:12.6f} {dt:12.6f} {mesh.nb_cells:10d}")

    print(f"{iteration:6d} {t:12.6f} {dt:12.6f} {mesh.nb_cells:10d}")

    print()
    print("=" * 70)
    print("Simulation complete!")
    print(f"  Final time: {t:.6f}")
    print(f"  Time steps: {iteration}")
    print(f"  Output files: {nsave + 1} (including initial)")
    print()
    print(f"Generated files in {output_path.absolute()}:")
    print(f"  - {filename}_init.h5/.xdmf    (initial condition)")
    print(f"  - {filename}_*.h5/.xdmf       (time series)")
    print()
    print("To visualize in Paraview:")
    print(f"  paraview {output_path.absolute() / f'{filename}_init.xdmf'}")
    print()

    # Keep matplotlib figure open if visualization was enabled
    if not args.no_plot and plotter is not None:
        plt.ioff()
        print("Close the plot window to exit...")
        plt.show()

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        description="2D spatial boundary conditions demo with wave patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # BC type selection
    parser.add_argument("--bc-type", choices=["sinusoidal", "spatial", "gaussian"],
                        default="sinusoidal",
                        help="Type of boundary condition")

    # Sinusoidal parameters
    parser.add_argument("--amplitude", type=float, default=1.0,
                        help="Amplitude for sinusoidal/gaussian BC")
    parser.add_argument("--frequency", type=float, default=3.0,
                        help="Frequency for sinusoidal BC (number of waves)")

    # Spatial shock parameters
    parser.add_argument("--shock-y", type=float, default=0.0,
                        help="Y-position of shock for spatial BC")
    parser.add_argument("--bottom-val", type=float, default=1.0,
                        help="Value in bottom region for spatial BC")
    parser.add_argument("--top-val", type=float, default=0.0,
                        help="Value in top region for spatial BC")

    # Gaussian parameters
    parser.add_argument("--gaussian-center", type=float, default=0.0,
                        help="Center Y-coordinate for Gaussian BC")
    parser.add_argument("--gaussian-width", type=float, default=0.3,
                        help="Width for Gaussian BC")

    # Mesh parameters
    parser.add_argument("--min-level", type=int, default=3,
                        help="Minimum mesh refinement level")
    parser.add_argument("--max-level", type=int, default=5,
                        help="Maximum mesh refinement level")

    # Time parameters
    parser.add_argument("--tf", type=float, default=2.0,
                        help="Final time")
    parser.add_argument("--cfl", type=float, default=0.5,
                        help="CFL condition")
    parser.add_argument("--nfiles", type=int, default=20,
                        help="Number of output files")

    # Other options
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable visualization")

    args = parser.parse_args()

    # Run demo
    run_2d_demo(args)


if __name__ == "__main__":
    main()
