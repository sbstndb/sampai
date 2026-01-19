// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// Samurai Python Bindings - Subset operations header
//
// Declares the initialization function for subset algebra bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize subset bindings (intersection, union, difference, translate, etc.)
void init_subset_bindings(py::module_& m);
