// Exception bindings for Sampai
//
// This file declares helper functions to raise Python exceptions from C++ code.
// The exception classes are defined in sampai/exceptions.py.
//
// Usage example in bindings:
//   throw make_mesh_error("Invalid mesh dimension");
//
// This will raise a Python MeshError with the given message.

#pragma once

#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

// Helper functions to raise custom Sampai exceptions from C++
// These functions import the exception classes from sampai.exceptions
// and raise them with the provided message.

py::error_already_set make_sampai_error(const std::string& msg);
py::error_already_set make_mesh_error(const std::string& msg);
py::error_already_set make_field_error(const std::string& msg);
py::error_already_set make_config_error(const std::string& msg);
py::error_already_set make_operator_error(const std::string& msg);
py::error_already_set make_bc_error(const std::string& msg);
py::error_already_set make_adaptation_error(const std::string& msg);
py::error_already_set make_io_error(const std::string& msg);

// Initialization function (called from main.cpp)
void init_exception_bindings(py::module_& m);
