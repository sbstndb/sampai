// Sampai PETSc Bindings - Header
//
// This file provides Python bindings for PETSc solvers in Samurai.
// PETSc support is optional and only compiled when SAMURAI_WITH_PETSC is defined.
//
// Usage:
//   #include "petsc_bindings.hpp"
//   samurai::python::bindings::init_petsc_bindings(module);

#pragma once

#ifdef SAMURAI_WITH_PETSC

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <petscksp.h>

#include "field_bindings.hpp"
#include "common_types.hpp"

namespace py = pybind11;
namespace samurai::python::bindings
{

    // ============================================================
    // Forward declarations for binding initialization
    // ============================================================

    void init_petsc_bindings(py::module_& m);

    // ============================================================
    // Helper function to wrap PETSc initialization
    // ============================================================

    inline void petsc_initialize()
    {
        static bool initialized = false;
        if (!initialized)
        {
            PetscErrorCode ierr = PetscInitializeNoArguments();
            if (ierr) {
                throw std::runtime_error("Failed to initialize PETSc");
            }
            initialized = true;
        }
    }

} // namespace samurai::python::bindings

#endif // SAMURAI_WITH_PETSC
