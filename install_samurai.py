#!/usr/bin/env python3
"""
Samurai Dependency Installer

This script automatically installs the samurai C++ library from conda-forge
if it's not already found in the system.

Used by both setup.py (pip install) and CMakeLists.txt (cmake build).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

SAMURAI_VERSION = "0.27.1"
SAMURAI_CONDA_CHANNEL = "conda-forge"


def find_conda() -> Optional[Path]:
    """Find conda or mamba executable."""
    for cmd in ["mamba", "conda", "micromamba"]:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return Path(cmd)
        except FileNotFoundError:
            continue
    return None


def find_samurai_install() -> Optional[Path]:
    """Check if samurai is already installed and findable by CMake."""
    # Check common install locations
    check_paths = []

    # Check CONDA_PREFIX if in conda environment
    if "CONDA_PREFIX" in os.environ:
        check_paths.append(Path(os.environ["CONDA_PREFIX"]))

    # Check CMAKE_PREFIX_PATH environment variable
    if "CMAKE_PREFIX_PATH" in os.environ:
        for path in os.environ["CMAKE_PREFIX_PATH"].split(os.pathsep):
            check_paths.append(Path(path))

    # Check conda environments
    conda_envs = [
        Path.home() / "miniforge3" / "envs",
        Path.home() / "miniconda3" / "envs",
        Path.home() / "anaconda3" / "envs",
        Path.home() / "miniforge3",
        Path.home() / "miniconda3",
        Path.home() / "anaconda3",
    ]

    for env_base in conda_envs:
        if env_base.exists():
            check_paths.append(env_base)

    for check_path in check_paths:
        # Check for samuraiConfig.cmake or samurai-config.cmake
        for config_file in ["samuraiConfig.cmake", "samurai-config.cmake"]:
            config_locations = [
                check_path / "lib" / "cmake" / "samurai" / config_file,
                check_path / "lib" / "cmake" / config_file,
                check_path / "share" / "cmake" / "samurai" / config_file,
            ]
            for loc in config_locations:
                if loc.exists():
                    print(f"Found samurai at: {check_path}")
                    return check_path

        # Check for samurai headers
        header_path = check_path / "include" / "samurai"
        if header_path.exists():
            print(f"Found samurai headers at: {header_path}")
            return check_path

    return None


def install_samurai_conda(conda_cmd: Path, prefix: Optional[Path] = None) -> bool:
    """Install samurai using conda/mamba."""
    cmd = [str(conda_cmd), "install", "-y", "-c", SAMURAI_CONDA_CHANNEL,
           f"samurai={SAMURAI_VERSION}"]

    if prefix:
        cmd.extend(["-p", str(prefix)])

    print(f"Installing samurai {SAMURAI_VERSION} from {SAMURAI_CONDA_CHANNEL}...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("Successfully installed samurai!")
        return True
    else:
        print(f"Failed to install samurai: {result.stderr}")
        return False


def main(install_prefix: Optional[Path] = None, force_install: bool = False) -> int:
    """
    Main entry point.

    Args:
        install_prefix: Optional prefix where to install samurai
        force_install: Force installation even if samurai is found

    Returns:
        0 on success, 1 on failure
    """
    print("=" * 60)
    print("Samurai Dependency Installer")
    print("=" * 60)
    print(f"Required version: {SAMURAI_VERSION}")
    print()

    # Check if already installed
    if not force_install:
        samurai_path = find_samurai_install()
        if samurai_path:
            print(f"Samurai already found at: {samurai_path}")
            print("Skipping installation.")
            return 0

    # Find conda
    conda_cmd = find_conda()
    if not conda_cmd:
        print("ERROR: conda/mamba/micromamba not found!")
        print("Please install conda-forge samurai manually:")
        print(f"  conda install -c {SAMURAI_CONDA_CHANNEL} samurai={SAMURAI_VERSION}")
        return 1

    print(f"Found conda command: {conda_cmd}")

    # Install samurai
    if install_samurai_conda(conda_cmd, install_prefix):
        print()
        print("Samurai installed successfully!")
        return 0
    else:
        print()
        print("Failed to install samurai.")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Install samurai dependency from conda-forge"
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        help="Installation prefix (default: current conda environment)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force installation even if samurai is found"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show required samurai version and exit"
    )

    args = parser.parse_args()

    if args.version:
        print(SAMURAI_VERSION)
        sys.exit(0)

    sys.exit(main(args.prefix, args.force))
