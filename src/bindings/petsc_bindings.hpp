// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// PETSc Bindings for Sampai
//
// This file contains the PETSc-related Python bindings for linear solvers
// and matrix assembly using the underlying Samurai library's PETSc integration.

#ifndef SAMPAI_PETSC_BINDINGS_HPP
#define SAMPAI_PETSC_BINDINGS_HPP

#include <pybind11/pybind11.h>

namespace sampai::bindings {

// Forward declaration to avoid including heavy headers here
void init_petsc_bindings(pybind11::module_& m);

} // namespace sampai::bindings

#endif // SAMPAI_PETSC_BINDINGS_HPP
