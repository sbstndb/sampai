#!/usr/bin/env python3

# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Sampai - Development Helper Script

This script provides convenient commands for developing the Python bindings.
It handles building, testing, and installing the package using Meson.

Usage:
    python dev.py build          # Build the module
    python dev.py install        # Install in development mode
    python dev.py test           # Run tests
    python dev.py clean          # Clean build artifacts
    python dev.py reinstall      # Clean and reinstall
    python dev.py all            # Build + install + test
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ANSI colors for terminal output
class Colors:
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


def print_header(msg: str):
    """Print a formatted header message."""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.NC}")
    print(f"{Colors.BLUE}{msg}{Colors.NC}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.NC}\n")


def print_success(msg: str):
    """Print a success message."""
    print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")


def print_warning(msg: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.NC}")


def print_error(msg: str):
    """Print an error message."""
    print(f"{Colors.RED}✗ {msg}{Colors.NC}")


def run_command(cmd: list, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and handle output."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"{Colors.BLUE}Running:{Colors.NC} {cmd_str}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
    )

    if check and result.returncode != 0:
        print_error(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    return result


def check_meson_installed() -> bool:
    """Check if Meson is installed."""
    result = run_command(["meson", "--version"], check=False)
    if result.returncode == 0:
        version = result.stdout.strip().split()[-1]
        print_success(f"Meson {version} found")
        return True

    print_warning("Meson not found!")
    print("Install with: conda install -c conda-forge meson")
    return False


def clean_artifacts(script_dir: Path):
    """Clean build artifacts."""
    print_header("Cleaning Build Artifacts")

    dirs_to_clean = [
        script_dir / "builddir",
        script_dir / "build",
        script_dir / "dist",
        script_dir / "*.egg-info",
        script_dir / "__pycache__",
        script_dir / "src" / "__pycache__",
        script_dir / "src" / "sampai" / "__pycache__",
        script_dir / "tests" / "__pycache__",
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
                print_success(f"Removed {dir_path.name}/")
            else:
                # Handle glob patterns
                for path in script_dir.glob(str(dir_path)):
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                print_success(f"Removed {dir_path.name}")


def build_module(script_dir: Path, build_type: str = "release"):
    """Build the Python module using Meson."""
    print_header("Building Python Module with Meson")

    build_dir = script_dir / "builddir"

    # Configure
    print("Configuring Meson...")
    meson_args = ["setup", str(build_dir)]
    if build_type == "debug":
        meson_args.extend(["-Dbuildtype=debug"])
    else:
        meson_args.extend(["-Dbuildtype=release", "-Doptimization=3"])

    run_command(["meson"] + meson_args)
    print_success("Meson configured")

    # Build
    print("Building...")
    run_command(["meson", "compile", "-C", str(build_dir)])
    print_success("Build complete")


def install_module(script_dir: Path, editable: bool = False):
    """Install the Python module."""
    print_header("Installing Python Module")

    pip_args = ["install"]
    if editable:
        pip_args.append("-e")
    pip_args.append(str(script_dir))

    run_command([sys.executable, "-m", "pip"] + pip_args)
    print_success("Installation complete")


def test_module(script_dir: Path):
    """Run tests."""
    print_header("Running Tests")

    test_dir = script_dir / "tests"
    if not test_dir.exists():
        print_warning(f"Test directory not found: {test_dir}")
        return

    # Run pytest
    run_command([sys.executable, "-m", "pytest", str(test_dir), "-v"])
    print_success("Tests complete")


def reinstall_module(script_dir: Path):
    """Reinstall the module (clean + build + install)."""
    clean_artifacts(script_dir)
    build_module(script_dir)
    install_module(script_dir, editable=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Development helper for Sampai Python bindings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dev.py build              # Build only
  python dev.py install            # Install in editable mode
  python dev.py install --no-edit  # Install normally
  python dev.py test               # Run tests
  python dev.py all                # Build + install + test
  python dev.py reinstall          # Clean + build + install
        """,
    )

    parser.add_argument(
        "command",
        choices=["build", "install", "test", "clean", "reinstall", "all"],
        help="Command to execute",
    )

    parser.add_argument(
        "--no-edit",
        action="store_true",
        help="Install normally (not editable mode)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build in Debug mode",
    )

    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip checking for meson installation",
    )

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.resolve()

    # Check meson unless skipped (samurai is auto-downloaded)
    if not args.skip_check and args.command not in ["clean"]:
        meson_ok = check_meson_installed()
        if not meson_ok:
            print_error("Cannot proceed without required dependencies")
            sys.exit(1)

    # Execute command
    if args.command == "build":
        build_type = "debug" if args.debug else "release"
        build_module(script_dir, build_type)

    elif args.command == "install":
        install_module(script_dir, editable=not args.no_edit)

    elif args.command == "test":
        test_module(script_dir)

    elif args.command == "clean":
        clean_artifacts(script_dir)

    elif args.command == "reinstall":
        reinstall_module(script_dir)

    elif args.command == "all":
        build_type = "debug" if args.debug else "release"
        build_module(script_dir, build_type)
        install_module(script_dir, editable=True)
        test_module(script_dir)

    print_success(f"\n{args.command.capitalize()} completed successfully!")


if __name__ == "__main__":
    main()
