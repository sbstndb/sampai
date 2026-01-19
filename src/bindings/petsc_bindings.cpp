// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// PETSc Bindings Implementation for Sampai
//
// This file contains the implementation of PETSc-related Python bindings.
// PETSc (Portable, Extensible Toolkit for Scientific Computation) provides
// linear solvers, nonlinear solvers, and time integrators for PDEs.

#include "petsc_bindings.hpp"

// Include PETSc headers if enabled
// Note: PETSc headers MUST be included before Samurai headers due to macro conflicts
#ifdef SAMURAI_WITH_PETSC
#include <petscksp.h>  // KSP (Krylov Subspace) solvers
#include <petscmat.h>  // Matrix operations
#include <petscvec.h>  // Vector operations
#endif

#include "samurai/samurai.hpp"

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

    petsc_module.def_property_readonly_static("version",
        [](py::object&) -> std::string {
            PetscInt major, minor, subminor;
            PetscGetVersion(&major, &minor, &subminor);
            return std::to_string(major) + "." +
                   std::to_string(minor) + "." +
                   std::to_string(subminor);
        },
        "PETSc version string");

    petsc_module.def_property_readonly_static("release_date",
        [](py::object&) -> std::string {
            const char* date;
            PetscGetVersionDate(&date);
            return std::string(date);
        },
        "PETSc release date");

    //======================================================================
    // PETSc World Communicator Info
    //======================================================================

    petsc_module.def_property_readonly_static("world_size",
        [](py::object&) -> int {
            PetscInt size;
            MPI_Comm_size(PETSC_COMM_WORLD, &size);
            return static_cast<int>(size);
        },
        "Number of processes in PETSc communicator");

    petsc_module.def_property_readonly_static("world_rank",
        [](py::object&) -> int {
            PetscInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            return static_cast<int>(rank);
        },
        "Rank of this process in PETSc communicator");

    //======================================================================
    // PETSc Options (command-line like configuration)
    //======================================================================

    petsc_module.def("set_option",
        [](const std::string& name, bool value) {
            PetscOptionsSetValue(NULL, name.c_str(), value ? "true" : "false");
        },
        py::arg("name"),
        py::arg("value"),
        "Set a PETSc option (like command-line argument)");

    petsc_module.def("set_options_prefix",
        [](const std::string& prefix) {
            PetscOptionsPrefixPush(PETSC_COMM_WORLD, prefix.c_str());
        },
        py::arg("prefix"),
        "Set a prefix for PETSc options (useful for multiple solvers)");

    petsc_module.def("clear_options_prefix",
        []() {
            PetscOptionsPrefixPop(PETSC_COMM_WORLD);
        },
        "Clear the current PETSc options prefix");

    //======================================================================
    // PETSc Solver Types (KSP - Krylov Subspace Methods)
    //======================================================================

    // Enum-like class for KSP solver types
    py::enum_<KSPType>(petsc_module, "KSPOption", "Available Krylov solver types")
        .value("RICHARDSON", KSPRICHARDSON)
        .value("CHEBYSHEV", KSPCHEBYSHEV)
        .value("CG", KSPCG)               // Conjugate Gradient (symmetric positive definite)
        .value("GMRES", KSPGMRES)         // Generalized Minimal Residual
        .value("TCQMR", KSPTCQMR)
        .value("TFQMR", KSPTFQMR)
        .value("BCGS", KSPBCGS)           // Bi-Conjugate Gradient Squared
        .value("BCGSL", KSPBCGSL)
        .value("CGNE", KSPCGNE)
        .value("CGS", KSPCGS)             // Conjugate Gradient Squared
        .value("TFQMR", KSPTFQMR)
        .value("CR", KSPCR)
        .value("PIPECG", KSPPIPECG)
        .value("PIPECGRR", KSPPIPECGRR)
        .value("PIPEPRCG", KSPPIPEPRCG)
        .value("PIPECG2", KSPPIPECG2)
        .value("LSQR", KSPLSQR)
        .value("PREONLY", KSPPREONLY)     // Preconditioner only
        .export_values();

    //======================================================================
    // PC (Preconditioner) Types
    //======================================================================

    py::enum_<PCType>(petsc_module, "PCOption", "Available preconditioner types")
        .value("NONE", PCNONE)
        .value("JACOBI", PCJACOBI)        // Jacobi (diagonal) preconditioner
        .value("SOR", PCSOR)              // SOR (Successive Over-Relaxation)
        .value("LU", PCLU)                // Direct LU factorization
        .value("ILU", PCILU)              // Incomplete LU factorization
        .value("ICC", PCICC)              // Incomplete Cholesky factorization
        .value("ASM", PCASM)              // Additive Schwarz
        .value("GASM", PCGASM)            // Generalized ASM
        .value("BJACOBI", PCBJACOBI)      // Block Jacobi
        .value("MG", PCMG)                // Multigrid
        .value("HYPRE", PCHYPRE)          // Hypre preconditioners
        .value("GAMG", PCGAMG)            // Geometric algebraic multigrid
        .export_values();

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
    // 3. Solver/preconditioner type enums for configuration

    message("PETSc bindings: ENABLED (full functionality)");

#else // !SAMURAI_WITH_PETSC

    //==========================================================================
    // PETSc IS NOT ENABLED - Create stub module
    //==========================================================================

    auto petsc_module = m.def_submodule("petsc",
        "PETSc support (NOT ENABLED - rebuild with -Dmpi=true -Dpetsc=true)");

    petsc_module.def("is_initialized",
        []() { return false; },
        "PETSc is not enabled");

    petsc_module.def_property_readonly_static("version",
        [](py::object&) -> std::string { return "NOT_ENABLED"; },
        "PETSc version (not enabled)");

    petsc_module.def_property_readonly_static("enabled",
        [](py::object&) -> bool { return false; },
        "Whether PETSc is enabled");

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
        [not_implemented](const std::string&, bool) { not_implemented("PETSc options"); });

    petsc_module.def("set_options_prefix",
        [not_implemented](const std::string&) { not_implemented("PETSc options"); });

    message("PETSc bindings: STUB (PETSc disabled)");

#endif // SAMURAI_WITH_PETSC
}

} // namespace sampai::bindings
