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
#include <samurai/reconstruction.hpp>

#include "common_types.hpp"
#include "exception_bindings.hpp"

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

    // Move constructor for LevelCellArray (direct construction for where operations)
    PySubset(lca_t&& cells, std::size_t level, const std::string& desc = "subset")
        : m_cells(std::move(cells)), m_level(level), m_description(desc)
    {}

    // Materialize from a subset expression at a specific level
    // SFINAE: Only enable for types that are NOT LevelCellArray
    template <class SubsetExpr,
              std::enable_if_t<!std::is_same_v<std::decay_t<SubsetExpr>, lca_t>, int> = 0>
    PySubset(SubsetExpr&& expr, std::size_t level, const std::string& desc = "subset")
        : m_cells(level), m_level(level), m_description(desc)
    {
        // Force evaluation by iterating over the subset expression
        samurai::for_each_interval(expr, [&](auto lvl, const auto& interval, const auto& index)
        {
            m_cells.add_interval_back(interval, index);
        });
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

    // Fill a scalar field with a constant value on this subset
    void fill(ScalarField<dim>& field, double value) const
    {
        samurai::for_each_interval(m_cells, [&](auto level, auto& interval, auto& index)
        {
            auto view = field(level, interval, index);
            view.fill(value);
        });
    }

    // Fill a vector field with constant values on this subset
    template <std::size_t n_comp, bool SOA>
    void fill_vector(VectorField<dim, n_comp, SOA>& field, const std::vector<double>& value) const
    {
        if (value.size() != n_comp)
        {
            throw make_field_error("Value size must match vector field components (expected " +
                                    std::to_string(n_comp) + ", got " + std::to_string(value.size()) + ")");
        }

        samurai::for_each_interval(m_cells, [&](auto level, auto& interval, auto& index)
        {
            for (std::size_t comp = 0; comp < n_comp; ++comp)
            {
                // Access vector field component using field(comp, level, interval, index)
                auto view = field(comp, level, interval, index);
                view.fill(value[comp]);
            }
        });
    }
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
        throw make_field_error("Stencil dimension must match mesh dimension");
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

// --- Expand Operations ---

// Helper to handle different width values using std::function to avoid type deduction issues
template <std::size_t dim, typename F>
PySubset<dim> call_expand(F&& func, std::size_t width)
{
    switch (width)
    {
        case 1: return func(std::integral_constant<std::size_t, 1>{});
        case 2: return func(std::integral_constant<std::size_t, 2>{});
        case 3: return func(std::integral_constant<std::size_t, 3>{});
        case 4: return func(std::integral_constant<std::size_t, 4>{});
        case 5: return func(std::integral_constant<std::size_t, 5>{});
        case 6: return func(std::integral_constant<std::size_t, 6>{});
        default: throw make_field_error("Expand width must be between 1 and 6");
    }
}

template <std::size_t dim>
PySubset<dim> make_expand(const MRMesh<dim>& mesh, std::size_t level, std::size_t width = 1)
{
    if (width < 1 || width > 6)
    {
        throw make_field_error("Expand width must be between 1 and 6");
    }
    auto& cells = mesh[mesh_id_t::cells][level];
    using CellsType = std::decay_t<decltype(cells)>;

    auto expand_func = [&](auto width_constant) -> PySubset<dim>
    {
        constexpr std::size_t w = decltype(width_constant)::value;
        auto subset = samurai::expand<CellsType, w>(cells);
        return PySubset<dim>(std::move(subset), level, "expanded");
    };

    return call_expand<dim>(expand_func, width);
}

template <std::size_t dim>
PySubset<dim> make_expand_directional(const MRMesh<dim>& mesh,
                                       std::size_t level,
                                       std::size_t width,
                                       const std::vector<bool>& directions)
{
    if (width < 1 || width > 6)
    {
        throw make_field_error("Expand width must be between 1 and 6");
    }
    if (directions.size() != dim)
    {
        throw make_field_error("Directions array size must match mesh dimension");
    }

    // Convert vector to std::array<bool, dim>
    std::array<bool, dim> dirs_array;
    for (std::size_t i = 0; i < dim; ++i)
    {
        dirs_array[i] = directions[i];
    }

    auto& cells = mesh[mesh_id_t::cells][level];
    using CellsType = std::decay_t<decltype(cells)>;

    auto expand_func = [&](auto width_constant) -> PySubset<dim>
    {
        constexpr std::size_t w = decltype(width_constant)::value;
        auto subset = samurai::expand<CellsType, w>(cells, dirs_array);
        return PySubset<dim>(std::move(subset), level, "expanded_directional");
    };

    return call_expand<dim>(expand_func, width);
}

// --- Contract Operations (modern version with directional control) ---

template <std::size_t dim>
PySubset<dim> make_contract(const MRMesh<dim>& mesh, std::size_t level, std::size_t width = 1)
{
    if (width < 1 || width > 6)
    {
        throw make_field_error("Contract width must be between 1 and 6");
    }
    auto subset = samurai::contract(mesh[mesh_id_t::cells][level], width);
    return PySubset<dim>(std::move(subset), level, "contracted");
}

template <std::size_t dim>
PySubset<dim> make_contract_directional(const MRMesh<dim>& mesh,
                                        std::size_t level,
                                        std::size_t width,
                                        const std::vector<bool>& directions)
{
    if (width < 1 || width > 6)
    {
        throw make_field_error("Contract width must be between 1 and 6");
    }
    if (directions.size() != dim)
    {
        throw make_field_error("Directions array size must match mesh dimension");
    }

    // Convert vector to std::array<bool, dim>
    std::array<bool, dim> dirs_array;
    for (std::size_t i = 0; i < dim; ++i)
    {
        dirs_array[i] = directions[i];
    }

    auto subset = samurai::contract(mesh[mesh_id_t::cells][level], width, dirs_array);
    return PySubset<dim>(std::move(subset), level, "contracted_directional");
}

// --- Self Operation (wrap LevelCellArray for set operations) ---

template <std::size_t dim>
PySubset<dim> make_self(const MRMesh<dim>& mesh, std::size_t level)
{
    // Directly copy the LevelCellArray at the specified level
    using interval_t = typename MRMesh<dim>::interval_t;
    using lca_t = samurai::LevelCellArray<dim, interval_t>;

    const auto& cells = mesh[mesh_id_t::cells];
    if (level < cells.min_level() || level > cells.max_level())
    {
        // Return empty subset if level doesn't exist
        return PySubset<dim>(lca_t(level), level, "self (empty)");
    }

    // Copy the LevelCellArray at this level
    lca_t subset = cells[level];

    return PySubset<dim>(std::move(subset), level, "self");
}

// ============================================================
// Field Operations with Subsets (Projection/Prediction)
// ============================================================

// --- Projection ---

template <std::size_t dim>
void apply_projection_single(ScalarField<dim>& field, std::size_t coarse_level)
{
    auto& mesh = field.mesh();

    if (coarse_level < mesh.min_level() || coarse_level >= mesh.max_level())
    {
        throw std::invalid_argument("Invalid coarse level for projection: must be in [min_level, max_level-1]");
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

    if (coarse_level < mesh.min_level() || coarse_level >= mesh.max_level())
    {
        throw std::invalid_argument("Invalid coarse level for projection: must be in [min_level, max_level-1]");
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

    if (coarse_level < mesh.min_level() || coarse_level >= mesh.max_level())
    {
        throw std::invalid_argument("Invalid coarse level for prediction: must be in [min_level, max_level-1]");
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
            throw make_field_error("Prediction order must be between 0 and 5");
    }
}

template <std::size_t dim>
void apply_prediction_two_fields(ScalarField<dim>& dest,
                                 const ScalarField<dim>& src,
                                 std::size_t coarse_level,
                                 std::size_t order)
{
    auto& mesh = dest.mesh();

    if (coarse_level < mesh.min_level() || coarse_level >= mesh.max_level())
    {
        throw std::invalid_argument("Invalid coarse level for prediction: must be in [min_level, max_level-1]");
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
            throw make_field_error("Prediction order must be between 0 and 5");
    }
}

// --- AMR Update (projection + prediction for all levels) ---

template <std::size_t dim>
void update_ghost_mr(ScalarField<dim>& field)
{
    samurai::update_ghost_mr(field);
}

// ============================================================
// Copy between subsets
// ============================================================

template <std::size_t dim>
void copy_subset(ScalarField<dim>& dst,
                 const PySubset<dim>& dst_subset,
                 const ScalarField<dim>& src,
                 const PySubset<dim>& src_subset)
{
    auto& dst_cells = dst_subset.cells();
    auto& src_cells = src_subset.cells();
    auto dst_level = dst_subset.level();
    auto src_level = src_subset.level();

    if (dst_cells.empty() || src_cells.empty())
    {
        return; // Nothing to copy
    }

    // Simple copy: iterate over both subsets and copy values
    // Note: This assumes the subsets have compatible geometries
    std::size_t copied = 0;
    std::size_t max_copy = std::min(dst_cells.nb_cells(), src_cells.nb_cells());

    samurai::for_each_interval(dst_cells, [&](auto level, auto& interval, auto& index)
    {
        auto dst_view = dst(level, interval, index);
        // For simplicity, copy with a constant offset from source
        // A more sophisticated version would do proper coordinate mapping
        auto src_view = src(level, interval, index);
        dst_view = src_view;
        copied += interval.size();
    });
}

// ============================================================
// Apply function on subset
// ============================================================

template <std::size_t dim>
void apply_function_scalar(ScalarField<dim>& field,
                           const PySubset<dim>& subset,
                           py::function func)
{
    auto& cells = subset.cells();

    samurai::for_each_interval(cells, [&](auto level, auto& interval, auto& index)
    {
        // Get the field view for this interval
        auto view = field(level, interval, index);

        // Iterate over each cell in the interval
        for (auto i = interval.start; i < interval.end; ++i)
        {
            py::object result;
            if constexpr (dim == 1)
            {
                result = func(i, 0, 0, level);
            }
            else if constexpr (dim == 2)
            {
                result = func(i, index[0], 0, level);
            }
            else // dim == 3
            {
                result = func(i, index[0], index[1], level);
            }
            double value = result.cast<double>();
            view[static_cast<std::size_t>(i - interval.start)] = value;
        }
    });
}

// Apply function to vector field (for n_comp components)
template <std::size_t dim, std::size_t n_comp, bool SOA>
void apply_function_vector_impl(VectorField<dim, n_comp, SOA>& field,
                                  const PySubset<dim>& subset,
                                  py::function func)
{
    auto& cells = subset.cells();

    samurai::for_each_interval(cells, [&](auto level, auto& interval, auto& index)
    {
        // Iterate over each cell in the interval
        for (auto i = interval.start; i < interval.end; ++i)
        {
            for (std::size_t comp = 0; comp < n_comp; ++comp)
            {
                py::object result;
                if constexpr (dim == 1)
                {
                    result = func(comp, i, 0, 0, level);
                }
                else if constexpr (dim == 2)
                {
                    result = func(comp, i, index[0], 0, level);
                }
                else // dim == 3
                {
                    result = func(comp, i, index[0], index[1], level);
                }
                double value = result.cast<double>();
                // Access vector field component using field(comp, level, interval, index)
                auto view = field(comp, level, interval, index);
                view[static_cast<std::size_t>(i - interval.start)] = value;
            }
        }
    });
}

// Explicit instantiations for n_comp = 2 and 3, SOA = false
template <std::size_t dim>
void apply_function_vector_2(VectorField<dim, 2, false>& field,
                              const PySubset<dim>& subset,
                              py::function func)
{
    apply_function_vector_impl(field, subset, func);
}

template <std::size_t dim>
void apply_function_vector_3(VectorField<dim, 3, false>& field,
                              const PySubset<dim>& subset,
                              py::function func)
{
    apply_function_vector_impl(field, subset, func);
}

// Copy for vector fields (n_comp components)
template <std::size_t dim, std::size_t n_comp, bool SOA>
void copy_vector_subset(VectorField<dim, n_comp, SOA>& dst,
                       const PySubset<dim>& dst_subset,
                       const VectorField<dim, n_comp, SOA>& src,
                       const PySubset<dim>& src_subset)
{
    auto& dst_cells = dst_subset.cells();
    auto dst_level = dst_subset.level();

    if (dst_cells.empty() || src_subset.cells().empty())
    {
        return;
    }

    samurai::for_each_interval(dst_cells, [&](auto level, auto& interval, auto& index)
    {
        for (std::size_t comp = 0; comp < n_comp; ++comp)
        {
            // Access vector field component using field(comp, level, interval, index)
            auto dst_view = dst(comp, level, interval, index);
            auto src_view = src(comp, level, interval, index);
            dst_view = src_view;
        }
    });
}

// Explicit instantiations for copy_vector
template <std::size_t dim>
void copy_vector_2(VectorField<dim, 2, false>& dst,
                   const PySubset<dim>& dst_subset,
                   const VectorField<dim, 2, false>& src,
                   const PySubset<dim>& src_subset)
{
    copy_vector_subset(dst, dst_subset, src, src_subset);
}

template <std::size_t dim>
void copy_vector_3(VectorField<dim, 3, false>& dst,
                   const PySubset<dim>& dst_subset,
                   const VectorField<dim, 3, false>& src,
                   const PySubset<dim>& src_subset)
{
    copy_vector_subset(dst, dst_subset, src, src_subset);
}

// ============================================================
// Reconstruction Operations (AMR to Uniform Mesh)
// ============================================================

// Reconstruction for scalar fields (MRMesh -> UniformMesh)
template <std::size_t dim>
void reconstruction_to_scalar(ScalarField<dim>& dest,
                              ScalarField<dim>& src)
{
    using mesh_id_t = typename MRMesh<dim>::mesh_id_t;

    // Check that source is MRMesh (AMR) and destination is suitable
    const auto& src_mesh = src.mesh();
    const auto& dest_mesh = dest.mesh();

    // Update ghost cells on source if needed
    samurai::update_ghost_mr_if_needed(src);

    // Get reconstruction level (domain level)
    std::size_t reconstruct_level = src_mesh.domain().level();

    // Loop over all levels in the source AMR field
    std::size_t min_level = src_mesh[mesh_id_t::cells].min_level();
    std::size_t max_level = src_mesh[mesh_id_t::cells].max_level();

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        // Create intersection between source level and destination level
        auto set = samurai::intersection(src_mesh[mesh_id_t::cells][level],
                                        dest_mesh[mesh_id_t::cells][reconstruct_level])
                  .on(level);

        // Apply reconstruction operator
        set.apply_op(samurai::make_reconstruction(reconstruct_level, dest, src));
    }
}

// Reconstruction for 2-component vector fields
template <std::size_t dim>
void reconstruction_to_vector_2(VectorField<dim, 2, false>& dest,
                                 VectorField<dim, 2, false>& src)
{
    using mesh_id_t = typename MRMesh<dim>::mesh_id_t;

    const auto& src_mesh = src.mesh();
    const auto& dest_mesh = dest.mesh();

    samurai::update_ghost_mr_if_needed(src);

    std::size_t reconstruct_level = src_mesh.domain().level();
    std::size_t min_level = src_mesh[mesh_id_t::cells].min_level();
    std::size_t max_level = src_mesh[mesh_id_t::cells].max_level();

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto set = samurai::intersection(src_mesh[mesh_id_t::cells][level],
                                        dest_mesh[mesh_id_t::cells][reconstruct_level])
                  .on(level);

        set.apply_op(samurai::make_reconstruction(reconstruct_level, dest, src));
    }
}

// Reconstruction for 3-component vector fields
template <std::size_t dim>
void reconstruction_to_vector_3(VectorField<dim, 3, false>& dest,
                                 VectorField<dim, 3, false>& src)
{
    using mesh_id_t = typename MRMesh<dim>::mesh_id_t;

    const auto& src_mesh = src.mesh();
    const auto& dest_mesh = dest.mesh();

    samurai::update_ghost_mr_if_needed(src);

    std::size_t reconstruct_level = src_mesh.domain().level();
    std::size_t min_level = src_mesh[mesh_id_t::cells].min_level();
    std::size_t max_level = src_mesh[mesh_id_t::cells].max_level();

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto set = samurai::intersection(src_mesh[mesh_id_t::cells][level],
                                        dest_mesh[mesh_id_t::cells][reconstruct_level])
                  .on(level);

        set.apply_op(samurai::make_reconstruction(reconstruct_level, dest, src));
    }
}

// Note: Previous scalar field support

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
        throw make_field_error("Unknown subset operation: " + op);
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
        .def("__str__", &PySubset<dim>::repr)
        .def("fill", &PySubset<dim>::fill,
             py::arg("field"),
             py::arg("value"),
             R"pbdoc(
            Fill scalar field with constant value on this subset.

            Parameters
            ----------
            field : ScalarField
                Field to fill (modified in-place)
            value : float
                Constant value to assign to all cells in subset

            Examples
            --------
            >>> subset = sam.subsets.intersection(mesh, mesh, level=2)
            >>> subset.fill(field, 3.14)
        )pbdoc")
        // Fill vector field with 2 components
        .def("fill_vector", &PySubset<dim>::template fill_vector<2, false>,
             py::arg("field"),
             py::arg("value"),
             R"pbdoc(
            Fill 2-component vector field with constant values on this subset.

            Parameters
            ----------
            field : VectorField2D_2 or VectorField3D_2
                Vector field to fill (modified in-place)
            value : list[float]
                List of 2 values (one per component)

            Examples
            --------
            >>> subset = sam.subsets.intersection(mesh, mesh, level=2)
            >>> subset.fill_vector(field2, [1.0, 2.0])
        )pbdoc")
        // Fill vector field with 3 components
        .def("fill_vector", &PySubset<dim>::template fill_vector<3, false>,
             py::arg("field"),
             py::arg("value"),
             R"pbdoc(
            Fill 3-component vector field with constant values on this subset.

            Parameters
            ----------
            field : VectorField1D_3, VectorField2D_3, or VectorField3D_3
                Vector field to fill (modified in-place)
            value : list[float]
                List of 3 values (one per component)

            Examples
            --------
            >>> subset = sam.subsets.intersection(mesh, mesh, level=2)
            >>> subset.fill_vector(field3, [1.0, 2.0, 3.0])
        )pbdoc");

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
        Contract mesh by removing cells at boundaries (legacy version).

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

        Note
        ----
        For directional control, use contract() instead.
    )pbdoc");

    // --- Expand Operations ---

    m.def("expand",
          &make_expand<dim>,
          py::arg("mesh"),
          py::arg("level"),
          py::arg("width") = 1,
          R"pbdoc(
        Expand mesh by adding ghost cells in all directions.

        Creates a layer of ghost cells around the mesh boundary.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to expand
        level : int
            Mesh level to operate on
        width : int, optional
            Number of cell layers to add (1-6, default: 1)

        Returns
        -------
        Subset
            Materialized subset of expanded cells

        Examples
        --------
        >>> expanded = sam.subsets.expand(mesh, level=2, width=1)
        >>> # Add 3 layers of ghost cells
        >>> expanded3 = sam.subsets.expand(mesh, level=2, width=3)

        Note
        ----
        For directional control, use expand_dir() instead.
    )pbdoc");

    m.def("expand_dir",
          &make_expand_directional<dim>,
          py::arg("mesh"),
          py::arg("level"),
          py::arg("width"),
          py::arg("directions"),
          R"pbdoc(
        Expand mesh in specific directions only.

        Creates ghost cells only in directions where directions[i] is True.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to expand
        level : int
            Mesh level to operate on
        width : int
            Number of cell layers to add (1-6)
        directions : list[bool]
            Boolean array specifying which directions to expand
            (e.g., [True, False] in 2D expands only in x-direction)

        Returns
        -------
        Subset
            Materialized subset of directionally expanded cells

        Examples
        --------
        >>> # Expand only in x-direction (2D)
        >>> expanded_x = sam.subsets.expand_dir(mesh, level=2, width=1,
        >>>                                    directions=[True, False])
        >>> # Expand only in y-direction (2D)
        >>> expanded_y = sam.subsets.expand_dir(mesh, level=2, width=1,
        >>>                                    directions=[False, True])
    )pbdoc");

    // --- Contract Operations (modern version) ---

    m.def("contract",
          &make_contract<dim>,
          py::arg("mesh"),
          py::arg("level"),
          py::arg("width") = 1,
          R"pbdoc(
        Contract mesh by removing cells at boundaries (modern version).

        Removes boundary layer(s) to get interior cells.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to contract
        level : int
            Mesh level to operate on
        width : int, optional
            Number of cell layers to remove (1-6, default: 1)

        Returns
        -------
        Subset
            Materialized subset of contracted cells

        Note
        ----
        This is the modern version with better control than contraction().
        For directional control, use contract_dir() instead.
    )pbdoc");

    m.def("contract_dir",
          &make_contract_directional<dim>,
          py::arg("mesh"),
          py::arg("level"),
          py::arg("width"),
          py::arg("directions"),
          R"pbdoc(
        Contract mesh in specific directions only.

        Removes boundary cells only in directions where directions[i] is True.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to contract
        level : int
            Mesh level to operate on
        width : int
            Number of cell layers to remove (1-6)
        directions : list[bool]
            Boolean array specifying which directions to contract
            (e.g., [True, False] in 2D contracts only in x-direction)

        Returns
        -------
        Subset
            Materialized subset of directionally contracted cells

        Examples
        --------
        >>> # Contract only in x-direction (2D)
        >>> contracted_x = sam.subsets.contract_dir(mesh, level=2, width=1,
        >>>                                      directions=[True, False])
    )pbdoc");

    // --- Self Operation ---

    m.def("self",
          &make_self<dim>,
          py::arg("mesh"),
          py::arg("level"),
          R"pbdoc(
        Wrap mesh cells as a subset for set algebra operations.

        This is useful for chaining multiple subset operations.

        Parameters
        ----------
        mesh : MRMesh
            Mesh to wrap
        level : int
            Mesh level to operate on

        Returns
        -------
        Subset
            Materialized subset of mesh cells

        Note
        ----
        This is primarily used internally. Most users don't need this
        since intersection/union/difference work directly on meshes.
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

    // --- Copy between subsets ---

    m.def("copy",
          &copy_subset<dim>,
          py::arg("dst"),
          py::arg("dst_subset"),
          py::arg("src"),
          py::arg("src_subset"),
          R"pbdoc(
        Copy field values between subsets.

        Copies field values from source subset to destination subset.
        Both subsets must have compatible geometries.

        Parameters
        ----------
        dst : ScalarField
            Destination field (modified in-place)
        dst_subset : Subset
            Destination subset specifying where to copy to
        src : ScalarField
            Source field (read-only)
        src_subset : Subset
            Source subset specifying where to copy from

        Examples
        --------
        >>> # Copy from one mesh region to another
        >>> subset1 = sam.subsets.intersection(mesh1, mesh1, level=2)
        >>> subset2 = sam.subsets.translate(mesh2, [1, 0], level=2)
        >>> sam.subsets.copy(field2, subset2, field1, subset1)
    )pbdoc");

    // --- Apply function on subset ---

    m.def("apply_function",
          &apply_function_scalar<dim>,
          py::arg("field"),
          py::arg("subset"),
          py::arg("func"),
          R"pbdoc(
        Apply a Python function to scalar field values on a subset.

        The function receives cell indices and level as arguments
        and should return a single float value.

        Parameters
        ----------
        field : ScalarField
            Field to modify (modified in-place)
        subset : Subset
            Subset specifying where to apply the function
        func : callable
            Python function called as func(i, j, k, level)
            Returns the field value for that cell
            Note: j and k are 0 for 1D, k is 0 for 2D

        Examples
        --------
        >>> # Apply function based on cell position
        >>> subset = sam.subsets.intersection(mesh, mesh, level=2)
        >>> sam.subsets.apply_function(field, subset,
        >>>     lambda i, j, k, level: i * 2 + j)
        >>> # Apply sin function based on position
        >>> import math
        >>> sam.subsets.apply_function(field, subset,
        >>>     lambda i, j, k, level: math.sin(i * 0.1))
    )pbdoc");

    // --- Apply function on vector field subset ---

    m.def("apply_function_vector",
          &apply_function_vector_2<dim>,
          py::arg("field"),
          py::arg("subset"),
          py::arg("func"),
          R"pbdoc(
        Apply a Python function to 2-component vector field values on a subset.

        The function receives component index, cell indices and level as arguments
        and should return a single float value.

        Parameters
        ----------
        field : VectorField2D_2 or VectorField3D_2
            Vector field to modify (modified in-place)
        subset : Subset
            Subset specifying where to apply the function
        func : callable
            Python function called as func(comp, i, j, k, level)
            comp: component index (0 or 1)
            Returns the field value for that component and cell

        Examples
        --------
        >>> subset = sam.subsets.intersection(mesh, mesh, level=2)
        >>> sam.subsets.apply_function_vector(field2, subset,
        >>>     lambda comp, i, j, k, level: comp * i + j)
    )pbdoc");

    m.def("apply_function_vector",
          &apply_function_vector_3<dim>,
          py::arg("field"),
          py::arg("subset"),
          py::arg("func"),
          R"pbdoc(
        Apply a Python function to 3-component vector field values on a subset.

        The function receives component index, cell indices and level as arguments
        and should return a single float value.

        Parameters
        ----------
        field : VectorField1D_3, VectorField2D_3, or VectorField3D_3
            Vector field to modify (modified in-place)
        subset : Subset
            Subset specifying where to apply the function
        func : callable
            Python function called as func(comp, i, j, k, level)
            comp: component index (0, 1, or 2)
            Returns the field value for that component and cell

        Examples
        --------
        >>> subset = sam.subsets.intersection(mesh, mesh, level=2)
        >>> sam.subsets.apply_function_vector(field3, subset,
        >>>     lambda comp, i, j, k, level: comp * (i + 1) + j)
    )pbdoc");

    // --- Copy vector field between subsets ---

    m.def("copy_vector",
          &copy_vector_2<dim>,
          py::arg("dst"),
          py::arg("dst_subset"),
          py::arg("src"),
          py::arg("src_subset"),
          R"pbdoc(
        Copy 2-component vector field values between subsets.

        Parameters
        ----------
        dst : VectorField2D_2 or VectorField3D_2
            Destination vector field (modified in-place)
        dst_subset : Subset
            Destination subset specifying where to copy to
        src : VectorField2D_2 or VectorField3D_2
            Source vector field (read-only)
        src_subset : Subset
            Source subset specifying where to copy from

        Examples
        --------
        >>> subset1 = sam.subsets.intersection(mesh, mesh, level=2)
        >>> subset2 = sam.subsets.translate(mesh, [1, 0], level=2)
        >>> sam.subsets.copy_vector(v2_dst, subset2, v2_src, subset1)
    )pbdoc");

    m.def("copy_vector",
          &copy_vector_3<dim>,
          py::arg("dst"),
          py::arg("dst_subset"),
          py::arg("src"),
          py::arg("src_subset"),
          R"pbdoc(
        Copy 3-component vector field values between subsets.

        Parameters
        ----------
        dst : VectorField1D_3, VectorField2D_3, or VectorField3D_3
            Destination vector field (modified in-place)
        dst_subset : Subset
            Destination subset specifying where to copy to
        src : VectorField1D_3, VectorField2D_3, or VectorField3D_3
            Source vector field (read-only)
        src_subset : Subset
            Source subset specifying where to copy from

        Examples
        --------
        >>> subset1 = sam.subsets.intersection(mesh, mesh, level=2)
        >>> subset2 = sam.subsets.translate(mesh, [1, 0], level=2)
        >>> sam.subsets.copy_vector(v3_dst, subset2, v3_src, subset1)
    )pbdoc");

    // --- Reconstruction (AMR to Uniform Mesh) ---

    m.def("reconstruction_to",
          &reconstruction_to_scalar<dim>,
          py::arg("dest"),
          py::arg("src"),
          R"pbdoc(
        Reconstruct an AMR scalar field onto a uniform mesh.

        This operation takes a scalar field defined on an adaptive mesh (MRMesh)
        and reconstructs it onto a uniform mesh (SingleLevelMesh) at the domain level.
        This is useful for visualization or for operations that require uniform meshes.

        The destination mesh MUST be a uniform mesh (min_level == max_level) at the
        same level as the source mesh's domain level.

        Note: The source field may have its ghost cells updated during this operation.

        Parameters
        ----------
        dest : ScalarField1D, ScalarField2D, or ScalarField3D
            Destination scalar field on a uniform mesh (modified in-place)
        src : ScalarField1D, ScalarField2D, or ScalarField3D
            Source scalar field on an adaptive mesh (AMR) (ghosts may be updated)

        Requirements
        ------------
        - Source mesh must be an MRMesh (adaptive mesh with multiple levels)
        - Destination mesh must be a uniform mesh at the domain level
        - Both fields must have the same dimension
        - Destination mesh must have max_stencil_radius >= 2

        Examples
        --------
        >>> # Create AMR field
        >>> amr_config = sam.config.MeshConfig2D(min_level=2, max_level=4)
        >>> amr_mesh = sam.mesh.make(box, amr_config)
        >>> amr_field = sam.field.scalar(amr_mesh, "amr_field")
        >>>
        >>> # Create uniform destination at domain level
        >>> domain_level = amr_mesh.domain().level
        >>> uniform_config = sam.config.MeshConfig2D(min_level=domain_level, max_level=domain_level)
        >>> uniform_mesh = sam.mesh.make(box, uniform_config)
        >>> uniform_field = sam.field.scalar(uniform_mesh, "uniform_field")
        >>>
        >>> # Reconstruct AMR field onto uniform mesh
        >>> sam.subsets.reconstruction_to(uniform_field, amr_field)

        See Also
        --------
        projection : Project field values between specific levels
        prediction : Predict field values between levels
    )pbdoc");

    m.def("reconstruction_to",
          &reconstruction_to_vector_2<dim>,
          py::arg("dest"),
          py::arg("src"),
          R"pbdoc(
        Reconstruct a 2-component AMR vector field onto a uniform mesh.

        Parameters
        ----------
        dest : VectorField1D_2, VectorField2D_2, or VectorField3D_2
            Destination vector field on a uniform mesh (modified in-place)
        src : VectorField1D_2, VectorField2D_2, or VectorField3D_2
            Source vector field on an adaptive mesh (AMR) (ghosts may be updated)

        Examples
        --------
        >>> amr_field = sam.field.vector(amr_mesh, "amr_vel", n_components=2)
        >>> uniform_field = sam.field.vector(uniform_mesh, "uniform_vel", n_components=2)
        >>> sam.subsets.reconstruction_to(uniform_field, amr_field)
    )pbdoc");

    m.def("reconstruction_to",
          &reconstruction_to_vector_3<dim>,
          py::arg("dest"),
          py::arg("src"),
          R"pbdoc(
        Reconstruct a 3-component AMR vector field onto a uniform mesh.

        Parameters
        ----------
        dest : VectorField1D_3, VectorField2D_3, or VectorField3D_3
            Destination vector field on a uniform mesh (modified in-place)
        src : VectorField1D_3, VectorField2D_3, or VectorField3D_3
            Source vector field on an adaptive mesh (AMR) (ghosts may be updated)

        Examples
        --------
        >>> amr_field = sam.field.vector(amr_mesh, "amr_vel", n_components=3)
        >>> uniform_field = sam.field.vector(uniform_mesh, "uniform_vel", n_components=3)
        >>> sam.subsets.reconstruction_to(uniform_field, amr_field)
    )pbdoc");

    // --- Where Operations (Field-based Filtering) ---

    m.def("where",
          [](const ScalarField<dim>& field, const py::function& py_condition, std::size_t level)
          {
              using mesh_id_t = typename MRMesh<dim>::mesh_id_t;
              using interval_t = typename MRMesh<dim>::interval_t;
              using lca_t = samurai::LevelCellArray<dim, interval_t>;

              const auto& mesh = field.mesh();
              lca_t result(level, mesh.origin_point(), mesh.scaling_factor());

              auto& cells = mesh[mesh_id_t::cells];
              if (level < cells.min_level() || level > cells.max_level())
              {
                  return PySubset<dim>(std::move(result), level, "where (empty)");
              }

              // Iterate over intervals at the specified level
              samurai::for_each_interval(cells[level], [&](std::size_t lvl, const auto& interval, const auto& index)
              {
                  // Get field view for this interval
                  auto view = field(lvl, interval, index);

                  // Check each cell in the interval
                  for (std::size_t ii = 0; ii < interval.size(); ++ii)
                  {
                      double value = view[ii];
                      bool keep = py_condition(value).cast<bool>();

                      if (keep)
                      {
                          // Add individual point
                          result.add_point_back(interval.start + ii, index);
                      }
                  }
              });

              return PySubset<dim>(std::move(result), level, "where");
          },
          py::arg("field"),
          py::arg("condition"),
          py::arg("level"),
          R"pbdoc(
        Create a subset where a scalar field satisfies a condition.

        Filters cells at a given level based on a condition applied to field values.
        Only cells where the condition returns True are included in the subset.

        Parameters
        ----------
        field : ScalarField1D, ScalarField2D, or ScalarField3D
            Input scalar field to filter
        condition : callable
            Python function that takes a field value and returns bool
            Example: lambda x: x > 0.0  (keep positive values)
        level : int
            Mesh level to filter

        Returns
        -------
        Subset
            Subset containing cells where the condition is True

        Examples
        --------
        >>> field = sam.field.scalar(mesh, "u")
        >>> # Keep cells where field > 0
        >>> positive_subset = sam.subsets.where(field, lambda x: x > 0.0, level=3)
        >>> # Keep cells where field != 0
        >>> nonzero_subset = sam.subsets.where(field, lambda x: x != 0.0, level=3)
        >>> # Keep cells within a range
        >>> range_subset = sam.subsets.where(field, lambda x: 0.5 < x < 1.5, level=3)

        Notes
        -----
        - The condition is evaluated for each cell individually
        - The result is a single-level subset at the specified level
        - For multi-level filtering, call where() for each level
    )pbdoc");

    // --- Clamp Operations ---

    m.def("clamp",
          [](ScalarField<dim>& field, double min_val, double max_val, std::size_t level)
          {
              using mesh_id_t = typename MRMesh<dim>::mesh_id_t;

              const auto& mesh = field.mesh();
              auto& cells = mesh[mesh_id_t::cells];

              if (level < cells.min_level() || level > cells.max_level())
              {
                  return; // Level doesn't exist, nothing to do
              }

              // Apply clamping to all cells at the specified level
              samurai::for_each_interval(cells[level], [&](auto lvl, const auto& interval, const auto& index)
              {
                  auto view = field(lvl, interval, index);
                  for (std::size_t ii = 0; ii < interval.size(); ++ii)
                  {
                      view[ii] = std::clamp(view[ii], min_val, max_val);
                  }
              });
          },
          py::arg("field"),
          py::arg("min_val"),
          py::arg("max_val"),
          py::arg("level"),
          R"pbdoc(
        Clamp scalar field values to a range [min_val, max_val] at a given level.

        Modifies the field in-place, limiting all values to be within the specified range.

        Parameters
        ----------
        field : ScalarField1D, ScalarField2D, or ScalarField3D
            Input scalar field (modified in-place)
        min_val : float
            Minimum value (values below this are set to min_val)
        max_val : float
            Maximum value (values above this are set to max_val)
        level : int
            Mesh level to apply clamping

        Examples
        --------
        >>> field = sam.field.scalar(mesh, "u")
        >>> # Clamp values to [0, 1]
        >>> sam.subsets.clamp(field, 0.0, 1.0, level=3)
        >>> # Clamp negative values to 0
        >>> sam.subsets.clamp(field, 0.0, float('inf'), level=3)
        >>> # Clamp to a physical range
        >>> sam.subsets.clamp(density, 0.0, max_density, level=2)

        Notes
        -----
        - This operation modifies the field in-place
        - Ghost cells are NOT automatically updated
        - For multi-level fields, call clamp() for each level separately
    )pbdoc");

    // --- Where Operations for Vector Fields ---

    m.def("where_vector",
          [](const VectorField<dim, 2, false>& field,
             const std::string& mode,
             const py::function& py_condition,
             std::size_t level)
          {
              using mesh_id_t = typename MRMesh<dim>::mesh_id_t;
              using interval_t = typename MRMesh<dim>::interval_t;
              using lca_t = samurai::LevelCellArray<dim, interval_t>;

              const auto& mesh = field.mesh();
              lca_t result(level, mesh.origin_point(), mesh.scaling_factor());

              auto& cells = mesh[mesh_id_t::cells];
              if (level < cells.min_level() || level > cells.max_level())
              {
                  return PySubset<dim>(std::move(result), level, "where_vector (empty)");
              }

              samurai::for_each_interval(cells[level], [&](std::size_t lvl, const auto& interval, const auto& index)
              {
                  // Get field views for each component - note order: (comp, level, interval, index)
                  auto view0 = field(0, lvl, interval, index);
                  auto view1 = field(1, lvl, interval, index);

                  for (std::size_t ii = 0; ii < interval.size(); ++ii)
                  {
                      bool keep = false;

                      if (mode == "magnitude")
                      {
                          // Compute magnitude: sqrt(v[0]^2 + v[1]^2)
                          double v0 = view0[ii];
                          double v1 = view1[ii];
                          double magnitude = std::sqrt(v0 * v0 + v1 * v1);
                          keep = py_condition(magnitude).cast<bool>();
                      }
                      else if (mode == "any")
                      {
                          // Keep if ANY component satisfies condition
                          double v0 = view0[ii];
                          double v1 = view1[ii];
                          keep = py_condition(v0).cast<bool>() || py_condition(v1).cast<bool>();
                      }
                      else if (mode == "all")
                      {
                          // Keep if ALL components satisfy condition
                          double v0 = view0[ii];
                          double v1 = view1[ii];
                          keep = py_condition(v0).cast<bool>() && py_condition(v1).cast<bool>();
                      }
                      else
                      {
                          throw make_field_error("Invalid mode. Use: 'magnitude', 'any', or 'all'");
                      }

                      if (keep)
                      {
                          result.add_point_back(interval.start + ii, index);
                      }
                  }
              });

              return PySubset<dim>(std::move(result), level, "where_vector");
          },
          py::arg("field"),
          py::arg("mode") = "magnitude",
          py::arg("condition"),
          py::arg("level"),
          R"pbdoc(
        Create a subset where a 2-component vector field satisfies a condition.

        Parameters
        ----------
        field : VectorField1D_2, VectorField2D_2, or VectorField3D_2
            Input vector field (2 components)
        mode : str
            Comparison mode: 'magnitude', 'any', or 'all'
            - 'magnitude': condition on vector magnitude ||v||
            - 'any': condition on each component separately (true if any component matches)
            - 'all': condition must be true for all components
        condition : callable
            Python function that takes a float and returns bool
        level : int
            Mesh level to filter

        Examples
        --------
        >>> vel = sam.field.vector(mesh, "velocity", n_components=2)
        >>> # Keep cells where velocity magnitude > 1.0
        >>> fast_cells = sam.subsets.where_vector(vel, "magnitude", lambda x: x > 1.0, level=3)
        >>> # Keep cells where ANY component > 0
        >>> any_positive = sam.subsets.where_vector(vel, "any", lambda x: x > 0, level=3)
        >>> # Keep cells where ALL components > 0
        >>> all_positive = sam.subsets.where_vector(vel, "all", lambda x: x > 0, level=3)
    )pbdoc");

    m.def("where_vector",
          [](const VectorField<dim, 3, false>& field,
             const std::string& mode,
             const py::function& py_condition,
             std::size_t level)
          {
              using mesh_id_t = typename MRMesh<dim>::mesh_id_t;
              using interval_t = typename MRMesh<dim>::interval_t;
              using lca_t = samurai::LevelCellArray<dim, interval_t>;

              const auto& mesh = field.mesh();
              lca_t result(level, mesh.origin_point(), mesh.scaling_factor());

              auto& cells = mesh[mesh_id_t::cells];
              if (level < cells.min_level() || level > cells.max_level())
              {
                  return PySubset<dim>(std::move(result), level, "where_vector (empty)");
              }

              samurai::for_each_interval(cells[level], [&](std::size_t lvl, const auto& interval, const auto& index)
              {
                  // Get field views for each component - note order: (comp, level, interval, index)
                  auto view0 = field(0, lvl, interval, index);
                  auto view1 = field(1, lvl, interval, index);
                  auto view2 = field(2, lvl, interval, index);

                  for (std::size_t ii = 0; ii < interval.size(); ++ii)
                  {
                      bool keep = false;

                      if (mode == "magnitude")
                      {
                          // Compute magnitude: sqrt(v[0]^2 + v[1]^2 + v[2]^2)
                          double v0 = view0[ii];
                          double v1 = view1[ii];
                          double v2 = view2[ii];
                          double magnitude = std::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
                          keep = py_condition(magnitude).cast<bool>();
                      }
                      else if (mode == "any")
                      {
                          double v0 = view0[ii];
                          double v1 = view1[ii];
                          double v2 = view2[ii];
                          keep = py_condition(v0).cast<bool>() ||
                                 py_condition(v1).cast<bool>() ||
                                 py_condition(v2).cast<bool>();
                      }
                      else if (mode == "all")
                      {
                          double v0 = view0[ii];
                          double v1 = view1[ii];
                          double v2 = view2[ii];
                          keep = py_condition(v0).cast<bool>() &&
                                 py_condition(v1).cast<bool>() &&
                                 py_condition(v2).cast<bool>();
                      }
                      else
                      {
                          throw make_field_error("Invalid mode. Use: 'magnitude', 'any', or 'all'");
                      }

                      if (keep)
                      {
                          result.add_point_back(interval.start + ii, index);
                      }
                  }
              });

              return PySubset<dim>(std::move(result), level, "where_vector");
          },
          py::arg("field"),
          py::arg("mode") = "magnitude",
          py::arg("condition"),
          py::arg("level"),
          R"pbdoc(
        Create a subset where a 3-component vector field satisfies a condition.

        Parameters
        ----------
        field : VectorField1D_3, VectorField2D_3, or VectorField3D_3
            Input vector field (3 components)
        mode : str
            Comparison mode: 'magnitude', 'any', or 'all'
        condition : callable
            Python function that takes a float and returns bool
        level : int
            Mesh level to filter

        Examples
        --------
        >>> vel = sam.field.vector(mesh, "velocity", n_components=3)
        >>> # Keep cells where velocity magnitude > 1.0
        >>> fast_cells = sam.subsets.where_vector(vel, "magnitude", lambda x: x > 1.0, level=2)
    )pbdoc");

    // --- Clamp Operations for Vector Fields ---

    m.def("clamp_vector",
          [](VectorField<dim, 2, false>& field, double min_val, double max_val, std::size_t level)
          {
              using mesh_id_t = typename MRMesh<dim>::mesh_id_t;

              const auto& mesh = field.mesh();
              auto& cells = mesh[mesh_id_t::cells];

              if (level < cells.min_level() || level > cells.max_level())
              {
                  return;
              }

              // Apply clamping to all cells at the specified level
              samurai::for_each_interval(cells[level], [&](auto lvl, const auto& interval, const auto& index)
              {
                  // Get views for both components
                  auto view0 = field(0, lvl, interval, index);
                  auto view1 = field(1, lvl, interval, index);

                  for (std::size_t ii = 0; ii < interval.size(); ++ii)
                  {
                      view0[ii] = std::clamp(view0[ii], min_val, max_val);
                      view1[ii] = std::clamp(view1[ii], min_val, max_val);
                  }
              });
          },
          py::arg("field"),
          py::arg("min_val"),
          py::arg("max_val"),
          py::arg("level"),
          R"pbdoc(
        Clamp 2-component vector field values to [min_val, max_val].

        Each component is clamped independently.

        Parameters
        ----------
        field : VectorField with 2 components
            Input vector field (modified in-place)
        min_val : float
            Minimum value
        max_val : float
            Maximum value
        level : int
            Mesh level to apply clamping

        Examples
        --------
        >>> vel = sam.field.vector(mesh, "velocity", n_components=2)
        >>> sam.subsets.clamp_vector(vel, -10.0, 10.0, level=3)
    )pbdoc");

    m.def("clamp_vector",
          [](VectorField<dim, 3, false>& field, double min_val, double max_val, std::size_t level)
          {
              using mesh_id_t = typename MRMesh<dim>::mesh_id_t;

              const auto& mesh = field.mesh();
              auto& cells = mesh[mesh_id_t::cells];

              if (level < cells.min_level() || level > cells.max_level())
              {
                  return;
              }

              // Apply clamping to all cells at the specified level
              samurai::for_each_interval(cells[level], [&](auto lvl, const auto& interval, const auto& index)
              {
                  // Get views for all three components
                  auto view0 = field(0, lvl, interval, index);
                  auto view1 = field(1, lvl, interval, index);
                  auto view2 = field(2, lvl, interval, index);

                  for (std::size_t ii = 0; ii < interval.size(); ++ii)
                  {
                      view0[ii] = std::clamp(view0[ii], min_val, max_val);
                      view1[ii] = std::clamp(view1[ii], min_val, max_val);
                      view2[ii] = std::clamp(view2[ii], min_val, max_val);
                  }
              });
          },
          py::arg("field"),
          py::arg("min_val"),
          py::arg("max_val"),
          py::arg("level"),
          R"pbdoc(
        Clamp 3-component vector field values to [min_val, max_val].

        Each component is clamped independently.
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
