// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// Samurai Python Bindings - CellList bindings
//
// Bindings for CellList and LevelCellList classes used for building
// adaptive mesh refinement (AMR) meshes with hierarchical cell storage.

#ifndef SAMPAI_CELL_LIST_BINDINGS_HPP
#define SAMPAI_CELL_LIST_BINDINGS_HPP

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize cell_list bindings
void init_cell_list_bindings(py::module_& m);

#endif // SAMPAI_CELL_LIST_BINDINGS_HPP
