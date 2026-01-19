// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// Samurai Python Bindings - Algorithm functions header
//
// Declares the initialization function for algorithm bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize algorithm bindings (for_each_interval, etc.)
void init_algorithm_bindings(py::module_& m);
