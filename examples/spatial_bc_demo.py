#!/usr/bin/env python3
"""
Finite Volume example demonstrating spatial boundary conditions with shock-like patterns.

This demo demonstrates:
- New spatial BC functionality: make_spatial_bc() for region-dependent BCs
- New functional BC: make_function_bc() for coordinate-dependent BCs
- 1D convection with different boundary condition types
- Comparison between constant BC, functional BC, and spatial BC
- Real-time visualization

The linear convection equation: du/dt + a*du/dx = 0
Shock-like patterns form from discontinuous boundary conditions.

Usage:
    python spatial_bc_demo.py                  # Run with all BC types
    python spatial_bc_demo.py --spatial        # Only spatial BC
    python spatial_bc_demo.py --no-plot        # Disable visualization
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sampai as sam


# ============================================================
# Boundary Condition Definitions
# ============================================================

def setup_constant_bc(u, value=1.0):
    """Setup constant Dirichlet BC on left boundary.

    Standard BC: u = value everywhere on boundary.

    Args:
        u: Scalar field
        value: Constant value for BC
    """
    sam.make_bc(u, "Dirichlet", value, order=2)


def setup_functional_bc(u, amplitude=1.0, frequency=2.0):
    """Setup function-based Dirichlet BC on left boundary.

    The BC value depends on spatial coordinates:
    - 1D: u(x) = amplitude * sin(frequency * pi * x)

    This demonstrates make_function_bc() where value depends on space.

    Args:
        u: Scalar field
        amplitude: Amplitude of the sinusoidal profile
        frequency: Frequency of the sinusoidal profile
    """
    # 1D sinusoidal profile: u = A * sin(k*pi*x)
    sam.make_function_bc(
        u,
        lambda coords: amplitude * np.sin(frequency * np.pi * coords[0]),
        order=2
    )


def setup_spatial_shock_bc(u, shock_pos=0.0, left_val=1.0, right_val=0.0):
    """Setup spatial shock BC on left boundary.

    Creates a discontinuous (shock) profile on the boundary.
    For 1D on left boundary (x = -1):
    - Apply high value on left portion
    - Apply low value on right portion

    This demonstrates make_spatial_bc() where both WHERE (region) and
    WHAT (value) depend on space.

    Args:
        u: Scalar field
        shock_pos: Position of shock on the boundary
        left_val: Value left of shock
        right_val: Value right of shock
    """
    # On left boundary, create discontinuous profile
    # This is a demonstration - the actual spatial selection depends on coordinates

    # High value region
    sam.make_spatial_bc(
        u,
        lambda coords: coords[0] < shock_pos,  # Region selection
        lambda coords: left_val,  # Value in region
        order=2
    )

    # Low value region
    sam.make_spatial_bc(
        u,
        lambda coords: coords[0] >= shock_pos,  # Other region
        lambda coords: right_val,
        order=2
    )


# ============================================================
# Initial Conditions
# ============================================================

def init_smooth_1d(u):
    """Initialize 1D field with smooth profile (no shock initially)."""
    def init_cell(cell):
        x = cell.center()[0]
        # Smooth sine wave
        u[cell.index] = 0.5 * (1 + np.sin(2 * np.pi * x))

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def init_step_1d(u):
    """Initialize 1D field with step function (shock)."""
    def init_cell(cell):
        x = cell.center()[0]
        # Step function: u = 1 for x < 0, u = 0 for x >= 0
        u[cell.index] = 1.0 if x < 0.0 else 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


def init_zero_1d(u):
    """Initialize 1D field with zeros (BC will drive the solution)."""
    def init_cell(cell):
        u[cell.index] = 0.0

    sam.algorithms.for_each_cell(u.mesh, init_cell)


# ============================================================
# 1D Demo
# ============================================================

def run_1d_demo(args):
    """Run 1D linear convection with different BC types."""

    print("\n" + "=" * 70)
    print("1D Linear Convection - Spatial Boundary Conditions Demo")
    print("=" * 70)

    # Output parameters
    output_path = Path("./spatial_bc_1d_results")
    nfiles = 20  # Number of output files

    # Domain and mesh
    box = sam.geometry.box([-1.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 6
    config.max_level = 6
    config.max_stencil_size = 6  # Required for WENO5
    mesh = sam.mesh.make(box, config)

    # Time parameters
    Tf = 1.0
    velocity = 1.0  # Convection to the right (float for 1D)
    cfl = 0.8

    # Setup for each BC type
    bc_types = []

    if args.constant:
        bc_types.append(("Constant BC (u=1)", lambda u: setup_constant_bc(u, 1.0)))

    if args.functional:
        bc_types.append(("Functional BC (sinusoidal)", lambda u: setup_functional_bc(u, 1.0, 2.0)))

    if args.spatial:
        bc_types.append(("Spatial BC (discontinuous)", lambda u: setup_spatial_shock_bc(u, 0.0, 1.0, 0.0)))

    if not bc_types:
        bc_types.append(("Constant BC (u=1)", lambda u: setup_constant_bc(u, 1.0)))

    # Run simulation for each BC type
    for bc_idx, (bc_name, bc_setup) in enumerate(bc_types):
        print(f"\n--- {bc_name} ---")

        # Create output directory for this BC type
        bc_filename = f"spatial_bc_1d_{bc_idx}"
        bc_output_path = output_path / bc_name.replace(" ", "_").lower()
        bc_output_path.mkdir(parents=True, exist_ok=True)

        # Create new mesh and field for this BC type
        mesh = sam.mesh.make(box, config)
        u = sam.field.zeros(mesh, "u")
        unp1 = sam.field.zeros(mesh, "unp1")

        # Initialize with step function (shock)
        init_step_1d(u)

        # Setup BC
        bc_setup(u)

        # Time stepping
        min_cell_length = mesh.min_cell_length
        dt = cfl * min_cell_length / abs(velocity)

        print(f"  Time step: {dt:.6e}")
        print(f"  Expected steps: ~{int(Tf/dt)}")
        print(f"  Output: {bc_output_path}/{bc_filename}_*.h5/.xdmf")

        # Setup visualization
        if not args.no_plot:
            plt.figure(figsize=(12, 6))
            plt.ion()

        # Save initial condition
        sam.save(str(bc_output_path / f"{bc_filename}_init"), u)
        print(f"  Saved initial condition to {bc_output_path / f'{bc_filename}_init.h5'}")

        # Time loop
        t = 0.0
        iteration = 0
        dt_save = Tf / nfiles
        next_save = dt_save
        nsave = 0
        plot_interval = max(1, int(Tf / dt / 20))

        while t < Tf:
            # Update ghost cells (BCs are automatically applied)
            sam.adaptation.update_ghost_mr(u)

            # Compute flux (Linear convection: du/dt + a*du/dx = 0)
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
                sam.save(str(bc_output_path / f"{bc_filename}_{nsave:05d}"), u)
                nsave += 1
                next_save += dt_save

            # Visualization
            if not args.no_plot and iteration % plot_interval == 0:
                plt.clf()

                # Extract data for plotting
                x_cells = []
                u_values = []

                def collect_data(cell):
                    x_cells.append(cell.center()[0])
                    u_values.append(u[cell.index])

                sam.algorithms.for_each_cell(u.mesh, collect_data)

                # Sort by x
                sorted_indices = np.argsort(x_cells)
                x_sorted = np.array(x_cells)[sorted_indices]
                u_sorted = np.array(u_values)[sorted_indices]

                plt.plot(x_sorted, u_sorted, 'b-', linewidth=2, label='u(x)')
                plt.xlabel('x')
                plt.ylabel('u')
                plt.title(f'{bc_name}\nLinear Convection 1D - t={t:.3f}')
                plt.ylim(-0.2, 1.5)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.pause(0.01)

        print(f"  Final time: {t:.6f}, Steps: {iteration}")
        print(f"  Output files: {nsave + 1} (including initial)")
        print(f"\n  Generated files in {bc_output_path.absolute()}:")
        print(f"    - {bc_filename}_init.h5/.xdmf    (initial condition)")
        print(f"    - {bc_filename}_*.h5/.xdmf       (time series)")

    # After all BC types
    print(f"\n{'='*70}")
    print(f"All outputs saved to: {output_path.absolute()}/")
    print(f"\nTo visualize in Paraview, open one of the .xdmf files, e.g.:")
    for bc_idx, (bc_name, _) in enumerate(bc_types):
        bc_filename = f"spatial_bc_1d_{bc_idx}"
        bc_output_path = output_path / bc_name.replace(" ", "_").lower()
        print(f"  paraview {bc_output_path / f'{bc_filename}_init.xdmf'}")

    if not args.no_plot:
        plt.ioff()
        print("\nClose the plot window to continue...")
        plt.show()


# ============================================================
# Main
# ============================================================

def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        description="Spatial boundary conditions demo with shock-like patterns",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # BC type selection
    parser.add_argument("--constant", action="store_true",
                        help="Run with constant BC")
    parser.add_argument("--functional", action="store_true",
                        help="Run with functional (coordinate-dependent) BC")
    parser.add_argument("--spatial", action="store_true",
                        help="Run with spatial (region-dependent) BC")

    # Other options
    parser.add_argument("--no-plot", action="store_true",
                        help="Disable visualization")

    args = parser.parse_args()

    # Default to all BC types if none selected
    if not args.constant and not args.functional and not args.spatial:
        args.constant = True
        args.functional = True
        args.spatial = True

    # Run demo
    run_1d_demo(args)


if __name__ == "__main__":
    main()
