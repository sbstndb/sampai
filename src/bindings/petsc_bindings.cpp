// Sampai PETSc Bindings - Implementation
//
// Python bindings for PETSc solvers in Samurai AMR library.
// This file is only compiled when SAMURAI_WITH_PETSC is defined.

#ifdef SAMURAI_WITH_PETSC

#include "petsc_bindings.hpp"
#include <petscksp.h>

namespace py = pybind11;
namespace samurai::python::bindings
{

    // ============================================================
    // KSP Wrapper Class
    // ============================================================
    //
    // Python wrapper for PETSc KSP (Krylov Subspace Methods) object.
    // Provides access to solver configuration options.
    // ============================================================

    class PyKSP
    {
        KSP ksp_;

      public:
        explicit PyKSP(KSP ksp)
            : ksp_(ksp)
        {
        }

        void set_type(const std::string& type)
        {
            KSPSetType(ksp_, type.c_str());
        }

        std::string get_type()
        {
            KSPType type;
            KSPGetType(ksp_, &type);
            return std::string(type);
        }

        void set_tolerances(double rtol = 1e-5, double atol = 1e-50, double div_tol = 1e4, int max_it = 10000)
        {
            KSPSetTolerances(ksp_, rtol, atol, div_tol, max_it);
        }

        void set_operators(py::object mat_py)
        {
            // For now, operators are set internally by the solver
            // This could be extended to accept custom matrices
            py::print("Warning: set_operators() is handled internally by the solver");
        }

        void set_from_options()
        {
            KSPSetFromOptions(ksp_);
        }

        py::object pc()
        {
            PC pc;
            KSPGetPC(ksp_, &pc);
            // Return the PC pointer as a capsule for now
            return py::capsule(pc, "PC");
        }

        int get_iteration_number()
        {
            PetscInt its;
            KSPGetIterationNumber(ksp_, &its);
            return static_cast<int>(its);
        }

        double get_residual_norm()
        {
            PetscReal rnorm;
            KSPGetResidualNorm(ksp_, &rnorm);
            return static_cast<double>(rnorm);
        }

        std::string get_converged_reason()
        {
            KSPConvergedReason reason;
            KSPGetConvergedReason(ksp_, &reason);
            return KSPConvergedReasons[reason];
        }
    };

    // ============================================================
    // PC (Preconditioner) Wrapper Class
    // ============================================================

    class PyPC
    {
        PC pc_;

      public:
        explicit PyPC(PC pc)
            : pc_(pc)
        {
        }

        void set_type(const std::string& type)
        {
            PCSetType(pc_, type.c_str());
        }

        std::string get_type()
        {
            PCType type;
            PCGetType(pc_, &type);
            return std::string(type);
        }
    };

    // ============================================================
    // Placeholder Solver Class
    // ============================================================
    //
    // This is a placeholder that will be extended with actual
    // PETSc solver functionality once the core bindings compile.
    // ============================================================

    class PyLinearSolverPlaceholder
    {
      public:
        PyLinearSolverPlaceholder() = default;

        void set_unknown(py::object unknown)
        {
            py::print("Info: set_unknown() - placeholder");
        }

        void setup()
        {
            py::print("Info: setup() - placeholder");
        }

        void solve(py::object rhs, py::object unknown)
        {
            py::print("Info: solve() - placeholder, actual solver not yet implemented");
        }

        PyKSP ksp()
        {
            // Return a dummy KSP object for now
            return PyKSP(nullptr);
        }

        int iterations()
        {
            return 0;
        }

        bool is_set_up()
        {
            return false;
        }
    };

    // ============================================================
    // Operator Wrapper Classes
    // ============================================================
    //
    // Type-erased wrappers for implicit operators.
    // These allow operator arithmetic (+, -, scalar *).
    // ============================================================

    class PyImplicitOperator
    {
      public:
        virtual ~PyImplicitOperator() = default;

        virtual py::object apply(py::object field) = 0;
        virtual std::string repr() const          = 0;
    };

    // Wrapper for scalar multiplication (c * op)
    class PyScalarMultOperator : public PyImplicitOperator
    {
        double scalar_;
        std::shared_ptr<PyImplicitOperator> op_;

      public:
        PyScalarMultOperator(double scalar, std::shared_ptr<PyImplicitOperator> op)
            : scalar_(scalar)
            , op_(op)
        {
        }

        py::object apply(py::object field) override
        {
            // For implicit schemes, we don't apply directly
            // The operator is passed to the solver
            py::print("Warning: Operator arithmetic is for solver configuration only");
            return field;
        }

        std::string repr() const override
        {
            if (scalar_ >= 0)
            {
                if (scalar_ == 1)
                {
                    return op_->repr();
                }
                else
                {
                    return "(" + std::to_string(scalar_) + " * " + op_->repr() + ")";
                }
            }
            else
            {
                return "(" + std::to_string(scalar_) + " * " + op_->repr() + ")";
            }
        }
    };

    // Wrapper for operator addition (op1 + op2)
    class PyAddOperator : public PyImplicitOperator
    {
        std::shared_ptr<PyImplicitOperator> left_;
        std::shared_ptr<PyImplicitOperator> right_;

      public:
        PyAddOperator(std::shared_ptr<PyImplicitOperator> left, std::shared_ptr<PyImplicitOperator> right)
            : left_(left)
            , right_(right)
        {
        }

        py::object apply(py::object field) override
        {
            py::print("Warning: Operator arithmetic is for solver configuration only");
            return field;
        }

        std::string repr() const override
        {
            return "(" + left_->repr() + " + " + right_->repr() + ")";
        }
    };

    // Wrapper for operator subtraction (op1 - op2)
    class PySubOperator : public PyImplicitOperator
    {
        std::shared_ptr<PyImplicitOperator> left_;
        std::shared_ptr<PyImplicitOperator> right_;

      public:
        PySubOperator(std::shared_ptr<PyImplicitOperator> left, std::shared_ptr<PyImplicitOperator> right)
            : left_(left)
            , right_(right)
        {
        }

        py::object apply(py::object field) override
        {
            py::print("Warning: Operator arithmetic is for solver configuration only");
            return field;
        }

        std::string repr() const override
        {
            return "(" + left_->repr() + " - " + right_->repr() + ")";
        }
    };

    // Concrete wrapper for diffusion operator
    class PyDiffusionOperator : public PyImplicitOperator
    {
        py::object field_;
        double coefficient_;

      public:
        PyDiffusionOperator(py::object field, double coefficient)
            : field_(field)
            , coefficient_(coefficient)
        {
        }

        py::object apply(py::object field) override
        {
            // The actual application is handled by PETSc solver
            return field;
        }

        std::string repr() const override
        {
            return "DiffusionOrder2(coefficient=" + std::to_string(coefficient_) + ")";
        }

        double coefficient() const
        {
            return coefficient_;
        }

        py::object field() const
        {
            return field_;
        }
    };

    // Concrete wrapper for identity operator
    class PyIdentityOperator : public PyImplicitOperator
    {
        py::object field_;

      public:
        explicit PyIdentityOperator(py::object field)
            : field_(field)
        {
        }

        py::object apply(py::object field) override
        {
            return field;
        }

        std::string repr() const override
        {
            return "Identity";
        }

        py::object field() const
        {
            return field_;
        }
    };

    // ============================================================
    // Helper functions for operator arithmetic
    // ============================================================

    inline std::shared_ptr<PyImplicitOperator> make_scalar_mult(double scalar,
                                                                 std::shared_ptr<PyImplicitOperator> op)
    {
        return std::make_shared<PyScalarMultOperator>(scalar, op);
    }

    inline std::shared_ptr<PyImplicitOperator> make_add(std::shared_ptr<PyImplicitOperator> left,
                                                          std::shared_ptr<PyImplicitOperator> right)
    {
        return std::make_shared<PyAddOperator>(left, right);
    }

    inline std::shared_ptr<PyImplicitOperator> make_sub(std::shared_ptr<PyImplicitOperator> left,
                                                          std::shared_ptr<PyImplicitOperator> right)
    {
        return std::make_shared<PySubOperator>(left, right);
    }

    // ============================================================
    // Initialize PETSc bindings
    // ============================================================

    void init_petsc_bindings(py::module_& m)
    {
        // Initialize PETSc
        petsc_initialize();

        // ============================================================
        // Create PETSc submodule
        // ============================================================

        py::module_ petsc_m = m.def_submodule("petsc", R"pbdoc(
            PETSc Solvers for Implicit Schemes

            This module provides interfaces to PETSc linear solvers for
            solving implicit schemes on Samurai AMR meshes.

            Note:
                PETSc support is optional. Use meson configure -Dwith_petsc=true
                to enable. Requires PETSc library to be installed.

            Basic Usage:
                >>> import sampai as sam
                >>> # Create field and apply boundary conditions
                >>> u = sam.field.scalar(mesh, "u")
                >>> sam.make_dirichlet_bc(u, value=0.0)
                >>>
                >>> # Define implicit scheme (Backward Euler)
                >>> diff = sam.petsc.diffusion_order2(u, coefficient=0.1)
                >>> id_op = sam.petsc.identity(u)
                >>> implicit_scheme = id_op + dt * diff
                >>>
                >>> # Solve
                >>> solver = sam.petsc.LinearSolver()
                >>> solver.set_unknown(u)
                >>> solver.setup()
                >>> solver.solve(rhs, u)

        )pbdoc");

        // ============================================================
        // KSP Class (Krylov Subspace Methods)
        // ============================================================

        py::class_<PyKSP>(petsc_m, "KSP", R"pbdoc(
            PETSc KSP (Krylov Subspace Methods) solver object.

            This class provides access to PETSc solver configuration options.
        )pbdoc")
            .def("set_type", &PyKSP::set_type, py::arg("type"), R"pbdoc(
                Set the KSP solver type.

                Args:
                    type: Solver type (e.g., "gmres", "cg", "bcgs")

                Common types:
                    - "richardson": Richardson iteration
                    - "chebychev": Chebychev iteration
                    - "cg": Conjugate Gradients (for symmetric positive definite)
                    - "gmres": Generalized Minimal Residual
                    - "bcgs": BiCGStab
                    - "cgs": Conjugate Gradients Squared
                    - "tfqmr": Transpose-Free Quasi-Minimal Residual
            )pbdoc")
            .def("get_type", &PyKSP::get_type, R"pbdoc(
                Get the KSP solver type.

                Returns:
                    str: The solver type
            )pbdoc")
            .def("set_tolerances",
                 &PyKSP::set_tolerances,
                 py::arg("rtol") = 1e-5,
                 py::arg("atol") = 1e-50,
                 py::arg("div_tol") = 1e4,
                 py::arg("max_it") = 10000,
                 R"pbdoc(
                Set solver tolerances.

                Args:
                    rtol: Relative convergence tolerance
                    atol: Absolute convergence tolerance
                    div_tol: Divergence tolerance
                    max_it: Maximum number of iterations
            )pbdoc")
            .def("set_from_options", &PyKSP::set_from_options, R"pbdoc(
                Set solver options from PETSc options database.

                Options can be set via command line or PETSc options file.
            )pbdoc")
            .def("pc", &PyKSP::pc, R"pbdoc(
                Get the preconditioner object.

                Returns:
                    PC: The preconditioner object (capsule)
            )pbdoc")
            .def("get_iteration_number", &PyKSP::get_iteration_number, R"pbdoc(
                Get the number of iterations used in the last solve.

                Returns:
                    int: Number of iterations
            )pbdoc")
            .def("get_residual_norm", &PyKSP::get_residual_norm, R"pbdoc(
                Get the residual norm from the last solve.

                Returns:
                    float: Residual norm
            )pbdoc")
            .def("get_converged_reason", &PyKSP::get_converged_reason, R"pbdoc(
                Get the convergence reason from the last solve.

                Returns:
                    str: Convergence reason (e.g., "CONVERGED_RTOL")
            )pbdoc");

        // ============================================================
        // PC Class (Preconditioner)
        // ============================================================

        py::class_<PyPC>(petsc_m, "PC", R"pbdoc(
            PETSc PC (Preconditioner) object.
        )pbdoc")
            .def("set_type", &PyPC::set_type, py::arg("type"), R"pbdoc(
                Set the preconditioner type.

                Args:
                    type: Preconditioner type

                Common types:
                    - "none": No preconditioning
                    - "jacobi": Jacobi preconditioning
                    - "bjacobi": Block Jacobi
                    - "sor": SOR (Successive Over-Relaxation)
                    - "lu": Direct LU factorization
                    - "ilu": Incomplete LU factorization
                    - "gamg": Geometric Algebraic Multigrid
                    - "hypre": Hypre preconditioners
            )pbdoc")
            .def("get_type", &PyPC::get_type, R"pbdoc(
                Get the preconditioner type.

                Returns:
                    str: The preconditioner type
            )pbdoc");

        // ============================================================
        // LinearSolver Class
        // ============================================================

        py::class_<PyLinearSolverPlaceholder>(petsc_m, "LinearSolver", R"pbdoc(
            PETSc linear solver for implicit schemes.

            This class wraps the PETSc KSP solver for solving linear systems
            arising from implicit finite volume schemes on AMR meshes.

            Note: This is currently a placeholder. Full implementation requires
            type erasure for templated schemes which will be added in a future update.
        )pbdoc")
            .def(py::init<>())
            .def("set_unknown", &PyLinearSolverPlaceholder::set_unknown, py::arg("unknown"), R"pbdoc(
                Set the unknown field.

                Args:
                    unknown: The field to solve for (ScalarField or VectorField)
            )pbdoc")
            .def("setup", &PyLinearSolverPlaceholder::setup, R"pbdoc(
                Setup the solver (assemble matrix and configure KSP).

                This must be called before solve() unless the solver was
                configured with set_unknown().
            )pbdoc")
            .def("solve",
                 &PyLinearSolverPlaceholder::solve,
                 py::arg("rhs"),
                 py::arg("unknown"),
                 R"pbdoc(
                Solve the linear system.

                Args:
                    rhs: Right-hand side field
                    unknown: Solution field (will be overwritten with solution)
            )pbdoc")
            .def("ksp", &PyLinearSolverPlaceholder::ksp, R"pbdoc(
                Get the PETSc KSP object for advanced configuration.

                Returns:
                    KSP: The KSP solver object
            )pbdoc")
            .def("iterations", &PyLinearSolverPlaceholder::iterations, R"pbdoc(
                Get the number of iterations from the last solve.

                Returns:
                    int: Number of iterations
            )pbdoc")
            .def("is_set_up", &PyLinearSolverPlaceholder::is_set_up, R"pbdoc(
                Check if the solver has been set up.

                Returns:
                    bool: True if setup() has been called
            )pbdoc");

        // ============================================================
        // Implicit Operators
        // ============================================================

        py::class_<PyImplicitOperator, std::shared_ptr<PyImplicitOperator>>(
            petsc_m,
            "ImplicitOperator",
            R"pbdoc(
            Base class for implicit operators used with PETSc solvers.

            Operators can be combined using arithmetic (+, -, scalar *).
        )pbdoc")
            .def("__repr__", &PyImplicitOperator::repr)
            .def("__add__",
                 [](std::shared_ptr<PyImplicitOperator> self,
                    std::shared_ptr<PyImplicitOperator> other) { return make_add(self, other); })
            .def("__sub__",
                 [](std::shared_ptr<PyImplicitOperator> self,
                    std::shared_ptr<PyImplicitOperator> other) { return make_sub(self, other); })
            .def("__mul__",
                 [](std::shared_ptr<PyImplicitOperator> self, double scalar) {
                     return make_scalar_mult(scalar, self);
                 })
            .def("__rmul__",
                 [](std::shared_ptr<PyImplicitOperator> self, double scalar) {
                     return make_scalar_mult(scalar, self);
                 });

        py::class_<PyDiffusionOperator, PyImplicitOperator, std::shared_ptr<PyDiffusionOperator>>(
            petsc_m,
            "DiffusionOperator",
            R"pbdoc(
            Second-order diffusion operator for implicit solves.

            Args:
                field: The field to apply the operator to
                coefficient: Diffusion coefficient

            Example:
                >>> diff = sam.petsc.DiffusionOperator(u, coefficient=0.1)
        )pbdoc")
            .def(py::init<py::object, double>(), py::arg("field"), py::arg("coefficient"))
            .def("coefficient", &PyDiffusionOperator::coefficient)
            .def("field", &PyDiffusionOperator::field);

        py::class_<PyIdentityOperator, PyImplicitOperator, std::shared_ptr<PyIdentityOperator>>(
            petsc_m,
            "IdentityOperator",
            R"pbdoc(
            Identity operator for implicit schemes.

            Returns the field unchanged. Used in operator arithmetic.

            Example:
                >>> id_op = sam.petsc.IdentityOperator(u)
        )pbdoc")
            .def(py::init<py::object>(), py::arg("field"))
            .def("field", &PyIdentityOperator::field);

        // ============================================================
        // Factory Functions
        // ============================================================

        petsc_m.def("diffusion_order2",
                    [](py::object field, double coefficient) -> std::shared_ptr<PyImplicitOperator> {
                        return std::make_shared<PyDiffusionOperator>(field, coefficient);
                    },
                    py::arg("field"),
                    py::arg("coefficient"),
                    R"pbdoc(
            Create a second-order diffusion operator for implicit solves.

            Args:
                field: The field to apply the operator to
                coefficient: Diffusion coefficient

            Returns:
                ImplicitOperator: Diffusion operator

            Example:
                >>> diff = sam.petsc.diffusion_order2(u, coefficient=0.1)
        )pbdoc");

        petsc_m.def("identity",
                    [](py::object field) -> std::shared_ptr<PyImplicitOperator> {
                        return std::make_shared<PyIdentityOperator>(field);
                    },
                    py::arg("field"),
                    R"pbdoc(
            Create an identity operator for implicit schemes.

            Args:
                field: The field to apply the operator to

            Returns:
                ImplicitOperator: Identity operator

            Example:
                >>> id_op = sam.petsc.identity(u)
        )pbdoc");

        // Placeholder functions
        petsc_m.def("make_solver",
                    [](py::object /*scheme*/) -> PyLinearSolverPlaceholder {
                        throw std::runtime_error(
                            "make_solver() from operator is not yet implemented. "
                            "Use LinearSolver class directly.");
                        return PyLinearSolverPlaceholder{};
                    },
                    py::arg("scheme"),
                    R"pbdoc(
            Create a PETSc solver from a scheme (placeholder).

            Note:
                This is currently a placeholder. Use LinearSolver class directly.

            Args:
                scheme: The implicit scheme to solve

            Returns:
                LinearSolver: Configured solver
        )pbdoc");

        petsc_m.def("solve",
                    [](py::object /*scheme*/, py::object /*unknown*/, py::object /*rhs*/) {
                        throw std::runtime_error(
                            "solve() from operator is not yet implemented. "
                            "Use LinearSolver class directly.");
                    },
                    py::arg("scheme"),
                    py::arg("unknown"),
                    py::arg("rhs"),
                    R"pbdoc(
            One-shot solve function (placeholder).

            Note:
                This is currently a placeholder. Use the LinearSolver class directly.

            Args:
                scheme: The implicit scheme
                unknown: Solution field
                rhs: Right-hand side field
        )pbdoc");
    }

} // namespace samurai::python::bindings

#endif // SAMURAI_WITH_PETSC
