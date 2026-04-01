// Exception bindings implementation for Sampai
//
// This file provides helper functions to raise Python exceptions from C++ code.
// The exception classes are defined in sampai/exceptions.py and are imported
// through the sampai package.

#include "exception_bindings.hpp"

#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

namespace detail {

// Helper to raise an exception from the sampai.exceptions module
static void raise_sampai_exception(const char* exc_name, const std::string& msg)
{
    py::object sampai = py::module_::import("sampai.exceptions");
    py::object exc_class = sampai.attr(exc_name);
    PyErr_SetObject(exc_class.ptr(), py::str(msg).ptr());
    throw py::error_already_set();
}

} // namespace detail

// Public wrapper functions
py::error_already_set make_sampai_error(const std::string& msg)
{
    detail::raise_sampai_exception("SampaiError", msg);
    return py::error_already_set(); // Never reached
}

py::error_already_set make_mesh_error(const std::string& msg)
{
    detail::raise_sampai_exception("MeshError", msg);
    return py::error_already_set(); // Never reached
}

py::error_already_set make_field_error(const std::string& msg)
{
    detail::raise_sampai_exception("FieldError", msg);
    return py::error_already_set(); // Never reached
}

py::error_already_set make_config_error(const std::string& msg)
{
    detail::raise_sampai_exception("ConfigError", msg);
    return py::error_already_set(); // Never reached
}

py::error_already_set make_operator_error(const std::string& msg)
{
    detail::raise_sampai_exception("OperatorError", msg);
    return py::error_already_set(); // Never reached
}

py::error_already_set make_bc_error(const std::string& msg)
{
    detail::raise_sampai_exception("BoundaryConditionError", msg);
    return py::error_already_set(); // Never reached
}

py::error_already_set make_adaptation_error(const std::string& msg)
{
    detail::raise_sampai_exception("AdaptationError", msg);
    return py::error_already_set(); // Never reached
}

py::error_already_set make_io_error(const std::string& msg)
{
    detail::raise_sampai_exception("IOError", msg);
    return py::error_already_set(); // Never reached
}

// Initialize exception bindings (empty - exceptions are Python-side)
void init_exception_bindings(py::module_& m)
{
    // The exceptions are defined in Python (sampai/exceptions.py)
    // This function is a placeholder for potential future C++-side exception registration
    (void)m;
}
