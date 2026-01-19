// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// Samurai Python Bindings - Timer bindings header
//
// Declares the initialization function for timer bindings

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

// Initialize timer bindings (Timer, etc.)
void init_timer_bindings(py::module_& m);
