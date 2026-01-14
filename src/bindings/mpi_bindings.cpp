// MPI Bindings Implementation for Sampai
//
// This file contains the implementation of MPI-related Python bindings.

#include "mpi_bindings.hpp"

// Include samurai headers BEFORE entering our namespace
// to avoid namespace conflicts with boost::mpi
#ifdef SAMURAI_WITH_MPI
#include <samurai/samurai.hpp>
#endif

namespace py = pybind11;

#ifdef SAMURAI_WITH_MPI

//==============================================================================
// MPI Communicator Wrapper Class
//==============================================================================

// Define at file scope to avoid namespace conflicts
class MPICommWrapper {
public:
    boost::mpi::communicator* comm;
    bool owner;

    MPICommWrapper(boost::mpi::communicator* c = nullptr, bool own = false)
        : comm(c), owner(own)
    {}

    ~MPICommWrapper() {
        if (owner && comm) {
            delete comm;
        }
    }

    // Prevent copying
    MPICommWrapper(const MPICommWrapper&) = delete;
    MPICommWrapper& operator=(const MPICommWrapper&) = delete;

    // Allow moving
    MPICommWrapper(MPICommWrapper&& other) noexcept
        : comm(other.comm), owner(other.owner)
    {
        other.comm = nullptr;
        other.owner = false;
    }

    MPICommWrapper& operator=(MPICommWrapper&& other) noexcept {
        if (this != &other) {
            if (owner && comm) {
                delete comm;
            }
            comm = other.comm;
            owner = other.owner;
            other.comm = nullptr;
            other.owner = false;
        }
        return *this;
    }

    int rank() const {
        return comm ? comm->rank() : 0;
    }

    int size() const {
        return comm ? comm->size() : 1;
    }

    void barrier() const {
        if (comm) {
            comm->barrier();
        }
    }

    static MPICommWrapper world() {
        static boost::mpi::communicator world_comm;
        return MPICommWrapper(&world_comm, false);
    }
};

#endif // SAMURAI_WITH_MPI

//==============================================================================
// BINDING FUNCTIONS
//==============================================================================

namespace sampai::bindings {

void init_mpi_bindings(py::module_& m) {
#ifdef SAMURAI_WITH_MPI
    //==========================================================================
    // MPI IS ENABLED - Create full MPI module
    //==========================================================================

    // Create MPI submodule
    auto mpi_module = m.def_submodule("mpi",
        "MPI support for distributed memory parallelization");

    //======================================================================
    // MPI Communicator Class
    //======================================================================

    py::class_<MPICommWrapper>(mpi_module, "Communicator",
        "MPI communicator wrapper")
        .def(py::init<>())
        .def_property_readonly("rank",
            [](const MPICommWrapper& self) { return self.rank(); })
        .def_property_readonly("size",
            [](const MPICommWrapper& self) { return self.size(); })
        .def("barrier", &MPICommWrapper::barrier)
        .def_property_readonly_static("world",
            [](py::object&) { return MPICommWrapper::world(); });

    //======================================================================
    // MPI Query Functions
    //======================================================================

    mpi_module.def("is_initialized",
        []() { return true; });

    mpi_module.def("rank",
        []() {
            static boost::mpi::communicator world;
            return world.rank();
        });

    mpi_module.def("size",
        []() {
            static boost::mpi::communicator world;
            return world.size();
        });

    mpi_module.def("barrier",
        []() {
            static boost::mpi::communicator world;
            world.barrier();
        });

#else // !SAMURAI_WITH_MPI

    //==========================================================================
    // MPI IS NOT ENABLED - Create stub module
    //==========================================================================

    auto mpi_module = m.def_submodule("mpi",
        "MPI support (NOT ENABLED)");

    mpi_module.def("is_initialized",
        []() { return false; });

    auto not_implemented = [](const char* name) {
        throw std::runtime_error(
            std::string("MPI support is not enabled. Rebuild with -Dmpi=true to use ") + name);
    };

    mpi_module.def("rank",
        [not_implemented]() { not_implemented("mpi.rank()"); });

    mpi_module.def("size",
        [not_implemented]() { not_implemented("mpi.size()"); });

    mpi_module.def("barrier",
        [not_implemented]() { not_implemented("mpi.barrier()"); });

#endif // SAMURAI_WITH_MPI
}

} // namespace sampai::bindings
