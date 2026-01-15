// Samurai Python Bindings - Subset Operations
//
// Bindings for Samurai set algebra operations:
// - intersection, union_, difference (set operations)
// - translate, contraction (geometric operations)
// - projection, prediction (AMR operations)
// - apply_op for operator application on subsets

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <samurai/subset/node.hpp>
#include <samurai/subset/apply.hpp>
#include <samurai/numeric/projection.hpp>
#include <samurai/numeric/prediction.hpp>
#include <samurai/algorithm/update.hpp>

#include "common_types.hpp"

namespace py = pybind11;
using namespace samurai::python::bindings;
using mesh_id_t = samurai::MRMeshId;

// ============================================================
// Materialized Subset Wrapper
// ============================================================
// Wrapper class to materialize subset expressions for Python
// Since Subset<Op, StartEndOp, S...> is a lazy expression template,
// we need to materialize it when crossing the Python boundary

template <std::size_t dim>
class PySubset
{
    using interval_t = typename MRMesh<dim>::interval_t;
    using lca_t      = samurai::LevelCellArray<dim, interval_t>;

    lca_t         m_cells;
    std::size_t   m_level;
    std::string   m_description;

public:
    // Default constructor
    PySubset() : m_cells(0), m_level(0), m_description("empty subset") {}

    // Materialize from a subset expression at a specific level
    template <class SubsetExpr>
    PySubset(SubsetExpr&& expr, std::size_t level, const std::string& desc = "subset")
        : m_cells(level), m_level(level), m_description(desc)
    {
        // Force evaluation by iterating over the subset
        auto subset_on_level = expr.on(level);

        subset_on_level([&](const auto& interval, const auto& index)
                        { m_cells.add_interval_back(interval, index); });
    }

    // Accessors
    bool empty() const { return m_cells.empty(); }
    std::size_t nb_cells() const { return m_cells.nb_cells(); }
    std::size_t level() const { return m_level; }
    std::string description() const { return m_description; }

    // String representation
    std::string repr() const
    {
        std::ostringstream oss;
        oss << "PySubset<" << dim << "D> at level " << m_level << " (" << m_description << ") with "
            << nb_cells() << " cells";
        return oss.str();
    }

    // Get the underlying LevelCellArray (for advanced use)
    const lca_t& cells() const { return m_cells; }
};

// ============================================================
// Subset Operation Factory Functions
// ============================================================

// --- Basic Set Operations ---

template <std::size_t dim>
PySubset<dim> make_intersection(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2, std::size_t level)
{
    auto subset = samurai::intersection(mesh1[mesh_id_t::cells][level], mesh2[mesh_id_t::cells][level]);
    return PySubset<dim>(std::move(subset), level, "intersection");
}

template <std::size_t dim>
PySubset<dim> make_union(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2, std::size_t level)
{
    auto subset = samurai::union_(mesh1[mesh_id_t::cells][level], mesh2[mesh_id_t::cells][level]);
    return PySubset<dim>(std::move(subset), level, "union");
}

template <std::size_t dim>
PySubset<dim> make_difference(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2, std::size_t level)
{
    auto subset = samurai::difference(mesh1[mesh_id_t::cells][level], mesh2[mesh_id_t::cells][level]);
    return PySubset<dim>(std::move(subset), level, "difference");
}

// --- Geometric Operations ---

template <std::size_t dim>
PySubset<dim> make_translate(const MRMesh<dim>& mesh, const std::vector<int>& stencil, std::size_t level)
{
    if (stencil.size() != dim)
    {
        throw std::runtime_error("Stencil dimension must match mesh dimension");
    }

    // Convert to xtensor_fixed
    xt::xtensor_fixed<int, xt::xshape<dim>> stencil_fixed;
    for (std::size_t i = 0; i < dim; ++i)
    {
        stencil_fixed[i] = stencil[i];
    }

    auto subset = samurai::translate(mesh[mesh_id_t::cells][level], stencil_fixed);
    return PySubset<dim>(std::move(subset), level, "translated");
}

template <std::size_t dim>
PySubset<dim> make_contraction(const MRMesh<dim>& mesh, std::size_t level, std::size_t n_cells_to_remove = 1)
{
    auto subset = samurai::contraction(mesh[mesh_id_t::cells][level], n_cells_to_remove);
    return PySubset<dim>(std::move(subset), level, "contraction");
}

// ============================================================
// Field Operations with Subsets (Projection/Prediction)
// ============================================================

// --- Projection ---

template <std::size_t dim>
void apply_projection_single(ScalarField<dim>& field, std::size_t coarse_level)
{
    auto& mesh = field.mesh();

    if (coarse_level < 1 || coarse_level > mesh.max_level())
    {
        throw std::runtime_error("Invalid coarse level for projection");
    }

    // Project from fine to coarse
    auto set_at_coarse = samurai::intersection(mesh[mesh_id_t::cells][coarse_level],
                                               mesh[mesh_id_t::cells][coarse_level + 1])
                             .on(coarse_level);
    set_at_coarse.apply_op(samurai::projection(field));
}

template <std::size_t dim>
void apply_projection_two_fields(ScalarField<dim>& dest, const ScalarField<dim>& src, std::size_t coarse_level)
{
    auto& mesh = dest.mesh();

    if (coarse_level < 1 || coarse_level > mesh.max_level())
    {
        throw std::runtime_error("Invalid coarse level for projection");
    }

    auto set_at_coarse = samurai::intersection(mesh[mesh_id_t::cells][coarse_level],
                                               mesh[mesh_id_t::cells][coarse_level + 1])
                             .on(coarse_level);
    set_at_coarse.apply_op(samurai::projection(dest, src));
}

// --- Prediction ---

template <std::size_t dim>
void apply_prediction_single(ScalarField<dim>& field, std::size_t coarse_level, std::size_t order)
{
    auto& mesh = field.mesh();

    if (coarse_level >= mesh.max_level())
    {
        throw std::runtime_error("Invalid coarse level for prediction");
    }

    // Predict from coarse to fine
    // dest_on_level=false means we predict onto coarse_level+1
    auto expr = samurai::intersection(mesh.domain(),
                                      samurai::difference(mesh[mesh_id_t::cells_and_ghosts][coarse_level + 1],
                                                          mesh.get_union()[coarse_level]))
                    .on(coarse_level + 1);

    switch (order)
    {
        case 0:
            expr.apply_op(samurai::prediction<0, false>(field));
            break;
        case 1:
            expr.apply_op(samurai::prediction<1, false>(field));
            break;
        case 2:
            expr.apply_op(samurai::prediction<2, false>(field));
            break;
        case 3:
            expr.apply_op(samurai::prediction<3, false>(field));
            break;
        case 4:
            expr.apply_op(samurai::prediction<4, false>(field));
            break;
        case 5:
            expr.apply_op(samurai::prediction<5, false>(field));
            break;
        default:
            throw std::runtime_error("Prediction order must be between 0 and 5");
    }
}

template <std::size_t dim>
void apply_prediction_two_fields(ScalarField<dim>& dest,
                                 const ScalarField<dim>& src,
                                 std::size_t coarse_level,
                                 std::size_t order)
{
    auto& mesh = dest.mesh();

    if (coarse_level >= mesh.max_level())
    {
        throw std::runtime_error("Invalid coarse level for prediction");
    }

    auto expr = samurai::intersection(mesh.domain(),
                                      samurai::difference(mesh[mesh_id_t::cells_and_ghosts][coarse_level + 1],
                                                          mesh.get_union()[coarse_level]))
                    .on(coarse_level + 1);

    switch (order)
    {
        case 0:
            expr.apply_op(samurai::prediction<0, false>(dest, src));
            break;
        case 1:
            expr.apply_op(samurai::prediction<1, false>(dest, src));
            break;
        case 2:
            expr.apply_op(samurai::prediction<2, false>(dest, src));
            break;
        case 3:
            expr.apply_op(samurai::prediction<3, false>(dest, src));
            break;
        case 4:
            expr.apply_op(samurai::prediction<4, false>(dest, src));
            break;
        case 5:
            expr.apply_op(samurai::prediction<5, false>(dest, src));
            break;
        default:
            throw std::runtime_error("Prediction order must be between 0 and 5");
    }
}

// --- AMR Update (projection + prediction for all levels) ---

template <std::size_t dim>
void update_ghost_mr(ScalarField<dim>& field)
{
    samurai::update_ghost_mr(field);
}

// ============================================================
// Subset Iteration (for each cell in subset)
// ============================================================

// Helper to iterate over a subset and call a Python function
template <std::size_t dim>
void iterate_subset(const MRMesh<dim>& mesh1,
                    const MRMesh<dim>& mesh2,
                    std::size_t level,
                    const std::string& op,
                    py::function func)
{
    samurai::LevelCellArray<dim> result;

    if (op == "intersection")
    {
        auto subset = samurai::intersection(mesh1[mesh_id_t::cells][level], mesh2[mesh_id_t::cells][level]).on(level);
        subset([&](const auto& interval, const auto& index)
                {
                    func(level, interval.start, interval.end, index);
                });
    }
    else if (op == "union")
    {
        auto subset = samurai::union_(mesh1[mesh_id_t::cells][level], mesh2[mesh_id_t::cells][level]).on(level);
        subset([&](const auto& interval, const auto& index)
                {
                    func(level, interval.start, interval.end, index);
                });
    }
    else if (op == "difference")
    {
        auto subset = samurai::difference(mesh1[mesh_id_t::cells][level], mesh2[mesh_id_t::cells][level]).on(level);
        subset([&](const auto& interval, const auto& index)
                {
                    func(level, interval.start, interval.end, index);
                });
    }
    else
    {
        throw std::runtime_error("Unknown subset operation: " + op);
    }
}

// ============================================================
// Template Binding Functions
// ============================================================

template <std::size_t dim>
void bind_subset_operations(py::module_& m, const std::string& dim_suffix)
{
    std::string subset_name = "Subset" + dim_suffix;
    std::string field_name  = "ScalarField" + dim_suffix;
    std::string mesh_name   = "MRMesh" + dim_suffix;

    // Bind Subset class
    py::class_<PySubset<dim>>(m, subset_name.c_str(), R"pbdoc(
        Materialized subset from set algebra operations.

        Subsets are created by operations like intersection, union, difference.
        They represent a set of cells at a specific mesh level.

        Parameters
        ----------
        level : int
            Mesh level of this subset
        description : str
            Description of the subset operation

        Properties
        ----------
        empty : bool
            True if subset contains no cells
        nb_cells : int
            Number of cells in subset
        level : int
            Mesh level
        description : str
            Description of the subset
    )pbdoc")
        .def(py::init<>(), "Create empty subset")
        .def_property_readonly("empty", &PySubset<dim>::empty, "True if subset contains no cells")
        .def_property_readonly("nb_cells", &PySubset<dim>::nb_cells, "Number of cells in subset")
        .def_property_readonly("level", &PySubset<dim>::level, "Mesh level of this subset")
        .def_property_readonly("description", &PySubset<dim>::description, "Description of the subset")
        .def("__repr__", &PySubset<dim>::repr)
        .def("__str__", &PySubset<dim>::repr);

    // --- Set Operations ---

    m.def("intersection",
          &make_intersection<dim>,
          py::arg("mesh1"),
          py::arg("mesh2"),
          py::arg("level"),
          R"pbdoc(
        Intersection of two meshes at given level.

        Parameters
        ----------
        mesh1 : MRMesh
            First mesh
        mesh2 : MRMesh
            Second mesh
        level : int
            Mesh level to operate on

        Returns
        -------
        Subset
            Materialized subset containing cells in both meshes
    )pbdoc");

    m.def("union_",
          &make_union<dim>,
          py::arg("mesh1"),
          py::arg("mesh2"),
          py::arg("level"),
          R"pbdoc(
        Union of two meshes at given level.

        Parameters
        ----------
        mesh1 : MRMesh
            First mesh
        mesh2 : MRMesh
            Second mesh
        level : int
            Mesh level to operate on

        Returns
        -------
        Subset
            Materialized subset containing cells in either mesh
    )pbdoc");

    m.def("difference",
          &make_difference<dim>,
          py::arg("mesh1"),
          py::arg("mesh2"),
          py::arg("level"),
          R"pbdoc(
        Difference of two meshes at given level.

        Returns cells from mesh1 that are not in mesh2.

        Parameters
        ----------
        mesh1 : MRMesh
            First mesh (minuend)
        mesh2 : MRMesh
            Second mesh (subtrahend)
        level : int
            Mesh level to operate on

        Returns
        -------
        Subset
            Materialized subset containing cells in mesh1 but not in mesh2
    )pbdoc");

    // --- Geometric Operations ---

    m.def("translate",
          &make_translate<dim>,
          py::arg("mesh"),
          py::arg("stencil"),
          py::arg("level"),
          R"pbdoc(
        Translate a mesh by a stencil vector.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to translate
        stencil : list[int]
            Translation vector (e.g., [1, 0] for right in 2D)
        level : int
            Mesh level to operate on

        Returns
        -------
        Subset
            Materialized subset of translated cells
    )pbdoc");

    m.def("contraction",
          &make_contraction<dim>,
          py::arg("mesh"),
          py::arg("level"),
          py::arg("n_cells_to_remove") = 1,
          R"pbdoc(
        Contract mesh by removing cells at boundaries.

        Useful for finding interior cells or for safe projection.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to contract
        level : int
            Mesh level to operate on
        n_cells_to_remove : int, optional
            Number of cells to remove from each boundary (default: 1)

        Returns
        -------
        Subset
            Materialized subset of contracted cells
    )pbdoc");

    // --- Projection ---

    m.def("projection",
          static_cast<void (*)(ScalarField<dim>&, std::size_t)>(&apply_projection_single<dim>),
          py::arg("field"),
          py::arg("coarse_level"),
          R"pbdoc(
        Project field values from fine to coarse level (in-place).

        Averages values from fine cells (coarse_level+1) to coarse cells.

        Parameters
        ----------
        field : ScalarField
            Field to project (modified in-place)
        coarse_level : int
            Coarse level to project onto
    )pbdoc");

    m.def("projection",
          static_cast<void (*)(ScalarField<dim>&, const ScalarField<dim>&, std::size_t)>(
              &apply_projection_two_fields<dim>),
          py::arg("dest"),
          py::arg("src"),
          py::arg("coarse_level"),
          R"pbdoc(
        Project field values from source to destination (fine to coarse).

        Averages values from fine cells (coarse_level+1) to coarse cells.

        Parameters
        ----------
        dest : ScalarField
            Destination field (coarse level, modified)
        src : ScalarField
            Source field (fine level, read-only)
        coarse_level : int
            Coarse level to project onto
    )pbdoc");

    // --- Prediction ---

    m.def("prediction",
          static_cast<void (*)(ScalarField<dim>&, std::size_t, std::size_t)>(&apply_prediction_single<dim>),
          py::arg("field"),
          py::arg("coarse_level"),
          py::arg("order") = 1,
          R"pbdoc(
        Predict field values from coarse to fine level (in-place).

        Uses polynomial interpolation to predict fine cell values from coarse cells.

        Parameters
        ----------
        field : ScalarField
            Field to predict (modified in-place)
        coarse_level : int
            Coarse level to predict from (predicts onto coarse_level+1)
        order : int, optional
            Prediction order (0-5, default: 1)
            Higher orders give better accuracy but may oscillate
    )pbdoc");

    m.def("prediction",
          static_cast<void (*)(ScalarField<dim>&, const ScalarField<dim>&, std::size_t, std::size_t)>(
              &apply_prediction_two_fields<dim>),
          py::arg("dest"),
          py::arg("src"),
          py::arg("coarse_level"),
          py::arg("order") = 1,
          R"pbdoc(
        Predict field values from source to destination (coarse to fine).

        Uses polynomial interpolation to predict fine cell values from coarse cells.

        Parameters
        ----------
        dest : ScalarField
            Destination field (fine level, modified)
        src : ScalarField
            Source field (coarse level, read-only)
        coarse_level : int
            Coarse level to predict from (predicts onto coarse_level+1)
        order : int, optional
            Prediction order (0-5, default: 1)
            Higher orders give better accuracy but may oscillate
    )pbdoc");

    // --- AMR Update ---

    m.def("update_ghost_mr",
          &update_ghost_mr<dim>,
          py::arg("field"),
          R"pbdoc(
        Update ghost cells for multiresolution AMR.

        Performs projection and prediction for all levels to maintain
        consistency between coarse and fine cells.

        Parameters
        ----------
        field : ScalarField
            Field to update
    )pbdoc");

    // --- Iteration ---

    m.def("for_each_subset_cell",
          &iterate_subset<dim>,
          py::arg("mesh1"),
          py::arg("mesh2"),
          py::arg("level"),
          py::arg("operation"),
          py::arg("func"),
          R"pbdoc(
        Iterate over cells in a subset operation.

        Parameters
        ----------
        mesh1 : MRMesh
            First mesh
        mesh2 : MRMesh
            Second mesh
        level : int
            Mesh level to operate on
        operation : str
            One of: 'intersection', 'union', 'difference'
        func : callable
            Python function called as func(level, start, end, index)
    )pbdoc");
}

// ============================================================
// Module Initialization
// ============================================================

void init_subset_bindings(py::module_& m)
{
    // Create subsets submodule
    py::module_ subsets = m.def_submodule("subsets", R"pbdoc(
        Set algebra operations on meshes.

        This module provides operations for working with mesh subsets:
        - Set operations: intersection, union, difference
        - Geometric operations: translate, contraction
        - AMR operations: projection, prediction, update_ghost_mr

        All subset operations are materialized at the Python boundary
        to avoid complex expression template types.

        Examples
        --------
        >>> import sampai as sam
        >>> config = sam.config.MeshConfig2D()
        >>> config.min_level = 2
        >>> config.max_level = 3
        >>> mesh1 = sam.mesh.make_2d(config)
        >>> mesh2 = sam.mesh.make_2d(config)
        >>>
        >>> # Set operations
        >>> inter = sam.subsets.intersection(mesh1, mesh2, level=2)
        >>> print(f"Intersection: {inter.nb_cells} cells")
        >>>
        >>> # Geometric operations
        >>> translated = sam.subsets.translate(mesh1, [1, 0], level=2)
        >>>
        >>> # AMR operations
        >>> field = sam.field.scalar("u", mesh, 1)
        >>> sam.subsets.update_ghost_mr(field)
        >>> sam.subsets.prediction(field, coarse_level=2, order=1)

        Notes
        -----
        - Subset operations return materialized Subset objects
        - All operations require an explicit level parameter
        - For AMR, use update_ghost_mr to maintain level consistency
    )pbdoc");

    // Bind operations for each dimension
    bind_subset_operations<1>(subsets, "1D");
    bind_subset_operations<2>(subsets, "2D");
    bind_subset_operations<3>(subsets, "3D");
}
