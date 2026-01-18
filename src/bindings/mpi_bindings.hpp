// MPI Bindings for Sampai
//
// This file contains the MPI-related Python bindings for distributed memory
// parallelization using the underlying Samurai library's MPI implementation.

#ifndef SAMPAI_MPI_BINDINGS_HPP
#define SAMPAI_MPI_BINDINGS_HPP

#include <pybind11/pybind11.h>

namespace sampai::bindings {

// Forward declaration to avoid including heavy headers here
void init_mpi_bindings(pybind11::module_& m);

} // namespace sampai::bindings

#endif // SAMPAI_MPI_BINDINGS_HPP

