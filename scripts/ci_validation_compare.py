#!/usr/bin/env python3

# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
CI validation script: Compare C++ and Python outputs.
"""
from sampai.utils.io import compare
import sys

def compare_and_exit(name, cpp_file, py_file, tol=1e-10):
    """Compare two HDF5 files and exit on failure."""
    print(f'Comparing {name}...')
    identical = compare.compare_hdf5_files(cpp_file, py_file, tol=tol, verbose=True)
    if not identical:
        print(f'FAILED: {name} outputs differ!')
        sys.exit(1)

def main():
    # Compare 1D advection
    compare_and_exit(
        '1D advection',
        'test_output/cpp_advection_1d.h5',
        'test_output/py_advection_1d.h5'
    )

    # Compare 2D advection
    compare_and_exit(
        '2D advection',
        'test_output/cpp_advection_2d.h5',
        'test_output/py_advection_2d.h5'
    )

    print('SUCCESS: All outputs match!')

if __name__ == '__main__':
    main()
