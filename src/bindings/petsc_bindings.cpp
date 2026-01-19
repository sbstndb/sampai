// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// PETSc Bindings Implementation for Sampai
//
// This file contains the implementation of PETSc-related Python bindings.
// PETSc (Portable, Extensible Toolkit for Scientific Computation) provides
// linear solvers, nonlinear solvers, and time integrators for PDEs.

#include "petsc_bindings.hpp"

// IMPORTANT: Include PETSc headers BEFORE Samurai to avoid macro conflicts
// and multiple definition issues with Samurai's PETSc wrapper functions
#ifdef SAMURAI_WITH_PETSC
#define PETSC_SKIP_CXXCPP_OVERRIDE  // Skip C++ standard library overrides
#include <petscksp.h>  // KSP (Krylov Subspace) solvers
#include <petscmat.h>  // Matrix operations
#include <petscvec.h>  // Vector operations
#endif

// Note: We do NOT include samurai headers here to avoid multiple definition
// issues with Samurai's PETSc wrapper functions. The PETSc bindings
// are independent and only need PETSc headers.

namespace py = pybind11;

//==============================================================================
// BINDING FUNCTIONS
//==============================================================================

namespace sampai::bindings {

void init_petsc_bindings(py::module_& m) {
#ifdef SAMURAI_WITH_PETSC
    //==========================================================================
    // PETSc IS ENABLED - Create full PETSc module
    //==========================================================================

    // Create PETSc submodule
    auto petsc_module = m.def_submodule("petsc",
        "PETSc support for linear solvers and matrix assembly");

    //======================================================================
    // PETSc Query Functions
    //======================================================================

    petsc_module.def("is_initialized",
        []() -> bool {
            PetscBool is_initialized = PETSC_FALSE;
            PetscInitialized(&is_initialized);
            return is_initialized == PETSC_TRUE;
        },
        "Check if PETSc has been initialized");

    petsc_module.def("initialize",
        []() {
            if (PetscInitializeNoArguments() != 0) {
                throw std::runtime_error("Failed to initialize PETSc");
            }
        },
        "Initialize PETSc (usually called automatically on first use)");

    petsc_module.def("finalize",
        []() {
            if (PetscFinalize() != 0) {
                throw std::runtime_error("Failed to finalize PETSc");
            }
        },
        "Finalize PETSc (cleanup, typically called at program exit)");

    //======================================================================
    // PETSc Version Info
    //======================================================================

    petsc_module.def("get_version",
        []() -> std::string {
            char version[256];
            PetscGetVersion(version, sizeof(version));
            return std::string(version);
        },
        "Get PETSc version string");

    //======================================================================
    // PETSc World Communicator Info
    //======================================================================

    petsc_module.def("get_world_size",
        []() -> int {
            // Ensure PETSc is initialized before calling MPI functions
            PetscBool initialized = PETSC_FALSE;
            PetscInitialized(&initialized);
            if (initialized == PETSC_FALSE) {
                PetscInitializeNoArguments();
            }
            PetscInt size;
            MPI_Comm_size(PETSC_COMM_WORLD, &size);
            return static_cast<int>(size);
        },
        "Number of processes in PETSc communicator");

    petsc_module.def("get_world_rank",
        []() -> int {
            // Ensure PETSc is initialized before calling MPI functions
            PetscBool initialized = PETSC_FALSE;
            PetscInitialized(&initialized);
            if (initialized == PETSC_FALSE) {
                PetscInitializeNoArguments();
            }
            PetscInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            return static_cast<int>(rank);
        },
        "Rank of this process in PETSc communicator");

    //======================================================================
    // PETSc Options (command-line like configuration)
    //======================================================================

    petsc_module.def("set_option",
        [](const std::string& name, const std::string& value) {
            PetscOptionsSetValue(NULL, name.c_str(), value.c_str());
        },
        py::arg("name"),
        py::arg("value"),
        "Set a PETSc option (like command-line argument)");

    petsc_module.def("set_options_prefix",
        [](const std::string& prefix) {
            PetscOptionsPrefixPush(NULL, prefix.c_str());
        },
        py::arg("prefix"),
        "Set a prefix for PETSc options (useful for multiple solvers)");

    petsc_module.def("clear_options_prefix",
        []() {
            PetscOptionsPrefixPop(NULL);
        },
        "Clear the current PETSc options prefix");

    //======================================================================
    // PETSc Solver Types (KSP - Krylov Subspace Methods)
    //======================================================================

    // String constants for KSP solver types (safer than enums)
    petsc_module.attr("KSP_RICHARDSON") = "richardson";
    petsc_module.attr("KSP_CHEBYSHEV") = "chebyshev";
    petsc_module.attr("KSP_CG") = "cg";               // Conjugate Gradient
    petsc_module.attr("KSP_GMRES") = "gmres";         // Generalized Minimal Residual
    petsc_module.attr("KSP_TCQMR") = "tcqmr";
    petsc_module.attr("KSP_TFQMR") = "tfqmr";
    petsc_module.attr("KSP_BCGS") = "bcgs";           // Bi-Conjugate Gradient Squared
    petsc_module.attr("KSP_CGS") = "cgs";             // Conjugate Gradient Squared
    petsc_module.attr("KSP_BICG") = "bicg";           // Bi-Conjugate Gradient
    petsc_module.attr("KSP_PREONLY") = "preonly";     // Preconditioner only

    //======================================================================
    // PC (Preconditioner) Types
    //======================================================================

    // String constants for PC types
    petsc_module.attr("PC_NONE") = "none";
    petsc_module.attr("PC_JACOBI") = "jacobi";        // Jacobi (diagonal)
    petsc_module.attr("PC_SOR") = "sor";              // SOR
    petsc_module.attr("PC_LU") = "lu";                // Direct LU
    petsc_module.attr("PC_ILU") = "ilu";              // Incomplete LU
    petsc_module.attr("PC_ICC") = "icc";              // Incomplete Cholesky
    petsc_module.attr("PC_ASM") = "asm";              // Additive Schwarz
    petsc_module.attr("PC_GASM") = "gasm";            // Generalized ASM
    petsc_module.attr("PC_BJACOBI") = "bjacobi";      // Block Jacobi
    petsc_module.attr("PC_MG") = "mg";                // Multigrid
    petsc_module.attr("PC_HYPRE") = "hypre";          // Hypre
    petsc_module.attr("PC_GAMG") = "gamg";            // Geometric algebraic multigrid

    //======================================================================
    // Note: Full Matrix/Vector/Solver bindings
    //======================================================================
    // For a complete PETSc integration, you would want to add:
    // - Vec (PETSc vector) wrapper
    // - Mat (PETSc matrix) wrapper
    // - KSP (Krylov solver) wrapper
    // - SNES (Nonlinear solver) wrapper
    // - TS (Time stepper) wrapper
    // - DM (Data management for mesh/fields) wrapper
    //
    // These require more extensive bindings and should be added incrementally
    // based on actual use cases in Samurai/Sampai applications.
    //
    // The current bindings provide:
    // 1. Basic PETSc initialization/query
    // 2. Access to PETSc options
    // 3. Solver/preconditioner type constants for configuration

#else // !SAMURAI_WITH_PETSC

    //==========================================================================
    // PETSc IS NOT ENABLED - Create stub module
    //==========================================================================

    auto petsc_module = m.def_submodule("petsc",
        "PETSc support (NOT ENABLED - rebuild with -Dmpi=true -Dpetsc=true)");

    petsc_module.def("is_initialized",
        []() { return false; },
        "PETSc is not enabled");

    petsc_module.def("get_version",
        []() -> std::string { return "NOT_ENABLED"; },
        "PETSc version (not enabled)");

    petsc_module.def("get_world_size",
        []() -> int { return 1; },
        "Number of processes (not enabled)");

    petsc_module.def("get_world_rank",
        []() -> int { return 0; },
        "Rank of this process (not enabled)");

    // Stub functions that raise errors
    auto not_implemented = [](const char* name) {
        throw std::runtime_error(
            std::string("PETSc support is not enabled. Rebuild with -Dmpi=true -Dpetsc=true to use ") + name);
    };

    petsc_module.def("initialize",
        [not_implemented]() { not_implemented("PETSc"); });

    petsc_module.def("finalize",
        [not_implemented]() { not_implemented("PETSc"); });

    petsc_module.def("set_option",
        [not_implemented](const std::string&, const std::string&) { not_implemented("PETSc options"); });

    petsc_module.def("set_options_prefix",
        [not_implemented](const std::string&) { not_implemented("PETSc options"); });

    petsc_module.def("clear_options_prefix",
        [not_implemented]() { not_implemented("PETSc options"); });

#endif // SAMURAI_WITH_PETSC
}

} // namespace sampai::bindings
