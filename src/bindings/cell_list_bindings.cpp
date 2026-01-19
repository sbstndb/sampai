// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// Samurai Python Bindings - CellList bindings
//
// Bindings for CellList and LevelCellList classes used for building
// adaptive mesh refinement (AMR) meshes with hierarchical cell storage.

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/cell.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/interval.hpp>
#include <samurai/level_cell_list.hpp>
#include <samurai/list_of_intervals.hpp>
#include "cell_list_bindings.hpp"

namespace py = pybind11;

// ============================================================================
// Type aliases for CellList bindings
// ============================================================================

// Use int interval type for CellList (matching Samurai default config)
// Note: Use signed long long int to match samurai::default_config::interval_t
using cell_interval = samurai::Interval<int, signed long long int>;

template <std::size_t dim>
using LevelCellList = samurai::LevelCellList<dim, samurai::Interval<int, signed long long int>>;

template <std::size_t dim>
using CellList = samurai::CellList<dim, samurai::Interval<int, signed long long int>>;

// ListOfIntervals is not dimension-specific
using ListOfIntervals = samurai::ListOfIntervals<int, signed long long int>;

using cell_coord_index_t = typename cell_interval::coord_index_t;

// Cell type aliases
template <std::size_t dim>
using Cell = samurai::Cell<dim, cell_interval>;

// ============================================================================
// Helper functions for coordinate conversion (must be before bind_cell)
// ============================================================================

// Convert Python tuple/list to xtensor_fixed coordinate (for yz indexing)
template <std::size_t dim>
xt::xtensor_fixed<cell_coord_index_t, xt::xshape<dim - 1>>
convert_to_index_yz(const py::tuple& t)
{
    static_assert(dim > 1, "1D has no yz coordinates");
    xt::xtensor_fixed<cell_coord_index_t, xt::xshape<dim - 1>> index;
    for (std::size_t i = 0; i < dim - 1; ++i)
    {
        index[i] = t[i].cast<cell_coord_index_t>();
    }
    return index;
}

// Convert Python tuple/list to xtensor_fixed coordinate (for origin point)
template <std::size_t dim>
xt::xtensor_fixed<double, xt::xshape<dim>>
convert_to_origin_point(const py::object& obj)
{
    xt::xtensor_fixed<double, xt::xshape<dim>> origin;
    if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj))
    {
        auto t = obj.cast<py::sequence>();
        for (std::size_t i = 0; i < dim; ++i)
        {
            origin[i] = t[i].cast<double>();
        }
    }
    else if (py::isinstance<py::array>(obj))
    {
        auto a = obj.cast<py::array_t<double>>();
        for (std::size_t i = 0; i < dim; ++i)
        {
            origin[i] = a.at(i);
        }
    }
    else
    {
        throw std::runtime_error("origin_point must be tuple, list, or array");
    }
    return origin;
}

// Convert xtensor_fixed to Python tuple
template <std::size_t dim>
py::tuple convert_to_tuple(const xt::xtensor_fixed<double, xt::xshape<dim>>& arr)
{
    py::tuple t(dim);
    for (std::size_t i = 0; i < dim; ++i)
    {
        t[i] = arr[i];
    }
    return t;
}

// ============================================================================
// Cell class bindings
// ============================================================================

template <std::size_t dim>
void bind_cell(py::module_& m, const std::string& name)
{
    using Cell_t = Cell<dim>;
    using indices_t = typename Cell_t::indices_t;
    using coords_t = typename Cell_t::coords_t;
    using index_t = typename Cell_t::index_t;

    py::class_<Cell_t>(m, name.c_str(), R"pbdoc(
        Cell in adaptive mesh refinement.

        A cell is defined by its level, integer coordinates (indices),
        and optional origin_point and scaling_factor.
    )pbdoc")
        // Full constructor
        .def(py::init(
            [](const coords_t& origin_point,
               double scaling_factor,
               std::size_t level,
               const py::object& indices_obj,
               index_t index)
            {
                indices_t indices;
                if (py::isinstance<py::tuple>(indices_obj) || py::isinstance<py::list>(indices_obj))
                {
                    auto seq = indices_obj.cast<py::sequence>();
                    if (seq.size() != dim)
                    {
                        throw std::runtime_error("indices must have " + std::to_string(dim) + " elements");
                    }
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        indices[i] = seq[i].cast<typename indices_t::value_type>();
                    }
                }
                else if (py::isinstance<py::array>(indices_obj))
                {
                    auto arr = indices_obj.cast<py::array_t<typename indices_t::value_type>>();
                    if (arr.size() != dim)
                    {
                        throw std::runtime_error("indices must have " + std::to_string(dim) + " elements");
                    }
                    auto buf = arr.request();
                    auto* ptr = static_cast<typename indices_t::value_type*>(buf.ptr);
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        indices[i] = ptr[i];
                    }
                }
                else
                {
                    throw std::runtime_error("indices must be tuple, list, or array");
                }
                return Cell_t(origin_point, scaling_factor, level, indices, index);
            }),
            py::arg("origin_point"),
            py::arg("scaling_factor"),
            py::arg("level"),
            py::arg("indices"),
            py::arg("index") = 0,
            "Create a Cell with all parameters")

        // Simplified constructor (inferred origin_point)
        .def(py::init(
            [](std::size_t level,
               const py::object& indices_obj,
               index_t index,
               const py::object& origin_obj,
               double scaling_factor)
            {
                coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<dim>(origin_obj);

                indices_t indices;
                if (py::isinstance<py::tuple>(indices_obj) || py::isinstance<py::list>(indices_obj))
                {
                    auto seq = indices_obj.cast<py::sequence>();
                    if (seq.size() != dim)
                    {
                        throw std::runtime_error("indices must have " + std::to_string(dim) + " elements");
                    }
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        indices[i] = seq[i].cast<typename indices_t::value_type>();
                    }
                }
                else if (py::isinstance<py::array>(indices_obj))
                {
                    auto arr = indices_obj.cast<py::array_t<typename indices_t::value_type>>();
                    if (arr.size() != dim)
                    {
                        throw std::runtime_error("indices must have " + std::to_string(dim) + " elements");
                    }
                    auto buf = arr.request();
                    auto* ptr = static_cast<typename indices_t::value_type*>(buf.ptr);
                    for (std::size_t i = 0; i < dim; ++i)
                    {
                        indices[i] = ptr[i];
                    }
                }
                else
                {
                    throw std::runtime_error("indices must be tuple, list, or array");
                }
                return Cell_t(origin, scaling_factor, level, indices, index);
            }),
            py::arg("level"),
            py::arg("indices"),
            py::arg("index") = 0,
            py::arg("origin_point") = py::none(),
            py::arg("scaling_factor") = 1.0,
            "Create a Cell (origin_point and scaling_factor are optional)")

        // Properties
        .def_property_readonly("level",
                              [](const Cell_t& c) -> std::size_t
                              {
                                  return c.level;
                              },
                              "Refinement level")
        .def_property_readonly("index",
                              [](const Cell_t& c)
                              {
                                  return static_cast<std::size_t>(c.index);
                              },
                              "Index in data array")
        .def_property_readonly("length",
                              [](const Cell_t& c) -> double
                              {
                                  return c.length;
                              },
                              "Physical cell size")
        .def_property_readonly("indices",
                              [](const Cell_t& c) -> py::tuple
                              {
                                  py::tuple t(dim);
                                  for (std::size_t i = 0; i < dim; ++i)
                                  {
                                      t[i] = c.indices[i];
                                  }
                                  return t;
                              },
                              "Integer coordinates as tuple")
        .def_property_readonly("origin_point",
                              [](const Cell_t& c) -> py::tuple
                              {
                                  return convert_to_tuple<dim>(c.origin_point);
                              },
                              "Origin point as tuple")

        // Methods
        .def("center",
             [](const Cell_t& c) -> py::tuple
             {
                 auto cent = c.center();
                 py::tuple t(dim);
                 for (std::size_t i = 0; i < dim; ++i)
                 {
                     t[i] = cent[i];
                 }
                 return t;
             },
             "Cell center as tuple")
        .def("center",
             [](const Cell_t& c, std::size_t i) -> double
             {
                 return c.center(i);
             },
             py::arg("i"),
             "Center coordinate in dimension i")
        .def("corner",
             [](const Cell_t& c) -> py::tuple
             {
                 auto corn = c.corner();
                 py::tuple t(dim);
                 for (std::size_t i = 0; i < dim; ++i)
                 {
                     t[i] = corn[i];
                 }
                 return t;
             },
             "Cell corner (min point) as tuple")
        .def("corner",
             [](const Cell_t& c, std::size_t i) -> double
             {
                 return c.corner(i);
             },
             py::arg("i"),
             "Corner coordinate in dimension i")

        // String representation
        .def("__repr__",
             [name](const Cell_t& c)
             {
                 std::ostringstream oss;
                 oss << name << "(level=" << c.level << ", index=";
                 oss << static_cast<std::size_t>(c.index) << ")";
                 return oss.str();
             })
        .def("__str__",
             [name](const Cell_t& c)
             {
                 std::ostringstream oss;
                 oss << name << " at level " << c.level;
                 return oss.str();
             });
}

// ============================================================================
// ListOfIntervals binding (used by LevelCellList)
// ============================================================================

void bind_list_of_intervals(py::module_& m)
{
    using interval_t = ListOfIntervals::interval_t;
    using value_t = ListOfIntervals::value_t;

    py::class_<ListOfIntervals>(m, "ListOfIntervals", R"pbdoc(
        Forward list of intervals along x-axis.

        Used internally by LevelCellList to store interval data.
        Automatically merges overlapping or adjacent intervals.
    )pbdoc")
        .def(py::init<>(), "Create empty ListOfIntervals")
        .def("add_point",
             [](ListOfIntervals& list, value_t point)
             {
                 list.add_point(point);
             },
             py::arg("point"),
             "Add a single point as interval [point, point+1)")
        .def("add_interval",
             [](ListOfIntervals& list, const interval_t& interval)
             {
                 list.add_interval(interval);
             },
             py::arg("interval"),
             "Add interval (merges overlapping intervals)")
        .def_property_readonly("size",
                              &ListOfIntervals::size,
                              "Number of intervals in list")
        .def("__len__",
             &ListOfIntervals::size,
             "Number of intervals in list")
        .def("__iter__",
             [](const ListOfIntervals& list)
             {
                 return py::make_iterator(list.begin(), list.end());
             },
             py::keep_alive<0, 1>())
        .def("__repr__",
             [](const ListOfIntervals& list)
             {
                 std::ostringstream oss;
                 oss << "<ListOfIntervals size=" << list.size() << ">";
                 return oss.str();
             });
}

// ============================================================================
// LevelCellList binding - wrapper for nested map structure
// ============================================================================

template <std::size_t dim>
void bind_level_cell_list(py::module_& m, const std::string& name)
{
    using LCL = LevelCellList<dim>;
    using index_yz_t = typename LCL::index_yz_t;
    using coords_t = typename LCL::coords_t;

    py::class_<LCL>(m, name.c_str(), R"pbdoc(
        Level-specific cell list for adaptive mesh refinement.

        Stores cells at a single refinement level using interval-based
        sparse representation with nested map structure for efficiency.

        For {1}D, access intervals directly: lcl[{}].add_interval(...)
        For {2}D, access by y-coordinate: lcl[{y}].add_interval(...)
        For {3}D, access by y,z coordinates: lcl[{y, z}].add_interval(...)
    )pbdoc")

        // Default constructor
        .def(py::init<>(), "Create empty LevelCellList at level 0")

        // Constructor with level
        .def(py::init<std::size_t>(),
             py::arg("level"),
             "Create LevelCellList at specified level")

        // Constructor with level and geometry - wrapper to accept Python tuples
        .def(py::init([](std::size_t level, const py::object& origin_obj, double scaling_factor)
             {
                 coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<dim>(origin_obj);
                 return LCL(level, origin, scaling_factor);
             }),
             py::arg("level"),
             py::arg("origin_point"),
             py::arg("scaling_factor") = 1.0,
             "Create LevelCellList with geometry parameters")

        // Properties
        .def_property_readonly("level",
                              &LCL::level,
                              "Refinement level")
        .def_property_readonly("empty",
                              &LCL::empty,
                              "Check if cell list is empty")
        .def_property_readonly("origin_point",
                              [](const LCL& lcl)
                              {
                                  return convert_to_tuple<dim>(lcl.origin_point());
                              },
                              "Origin point in physical coordinates")
        .def_property_readonly("scaling_factor",
                              &LCL::scaling_factor,
                              "Scaling factor for coordinate transformation")

        // Methods
        // Note: clear() method has C++ implementation issues with forward_list inheritance
        // .def("clear", &LCL::clear, "Clear all cells from this level")

        .def("add_cell",
             [](LCL& lcl, const Cell<dim>& cell)
             {
                 lcl.add_cell(cell);
             },
             py::arg("cell"),
             "Add a cell to this level")

        // String representations
        .def("__repr__",
             [name](const LCL& lcl)
             {
                 std::ostringstream oss;
                 oss << name << "(level=" << lcl.level();
                 if (lcl.empty())
                 {
                     oss << ", empty";
                 }
                 else
                 {
                     oss << ", non_empty";
                 }
                 oss << ")";
                 return oss.str();
             })
        .def("__str__",
             [name](const LCL& lcl)
             {
                 std::ostringstream oss;
                 oss << name << " at level " << lcl.level();
                 if (lcl.empty())
                 {
                     oss << " (empty)";
                 }
                 return oss.str();
             })
        .def("__bool__",
             &LCL::empty,
             "False if empty")

        // Nested access operator[] for adding intervals
        // For 1D: no key needed (returns ListOfIntervals directly)
        // For 2D: takes y-coordinate, returns ListOfIntervals
        // For 3D: takes y,z coordinates as tuple, returns ListOfIntervals
        .def("__getitem__",
             [](LCL& lcl, const py::object& key_obj)
             {
                 if constexpr (dim == 1)
                 {
                     // 1D: Return the grid_yz (ListOfIntervals) directly
                     // The key is ignored for 1D - we always return the intervals
                     return const_cast<typename LCL::list_interval_t&>(lcl.grid_yz());
                 }
                 else if constexpr (dim == 2)
                 {
                     // 2D: key is y-coordinate
                     using coord_index_t = typename LCL::coord_index_t;
                     coord_index_t y = key_obj.cast<coord_index_t>();
                     auto& grid = const_cast<typename LCL::grid_t&>(lcl.grid_yz());
                     return grid[y];
                 }
                 else  // dim == 3
                 {
                     // 3D: key is (y, z) tuple
                     auto t = key_obj.cast<py::sequence>();
                     using coord_index_t = typename LCL::coord_index_t;
                     coord_index_t y = t[0].cast<coord_index_t>();
                     coord_index_t z = t[1].cast<coord_index_t>();
                     auto& grid = const_cast<typename LCL::grid_t&>(lcl.grid_yz());
                     return grid[y][z];
                 }
             },
             py::return_value_policy::reference_internal,
             "Access ListOfIntervals at given yz-index to add intervals")

        // For convenience, also expose add_interval/add_point methods directly on LevelCellList
        // This allows: cl[0].add_interval(...) instead of cl[0][()].add_interval(...)
        .def("add_interval",
             [](LCL& lcl, const py::object& interval_obj)
             {
                 if constexpr (dim == 1)
                 {
                     // 1D: Add to the grid (which is ListOfIntervals)
                     using interval_t = typename LCL::interval_t;
                     auto interval = interval_obj.cast<interval_t>();
                     auto& grid = const_cast<typename LCL::list_interval_t&>(lcl.grid_yz());
                     grid.add_interval(interval);
                 }
                 else
                 {
                     throw std::runtime_error("add_interval requires yz-index for dim > 1. Use: cl[level][y_or_tuple].add_interval(...)");
                 }
             },
             py::arg("interval"),
             "Add interval to level (only for 1D, use indexing for higher dims)")

        .def("add_point",
             [](LCL& lcl, py::object point_obj)
             {
                 if constexpr (dim == 1)
                 {
                     using index_t = typename LCL::index_t;
                     auto point = point_obj.cast<index_t>();
                     auto& grid = const_cast<typename LCL::list_interval_t&>(lcl.grid_yz());
                     grid.add_point(point);
                 }
                 else
                 {
                     throw std::runtime_error("add_point requires yz-index for dim > 1. Use: cl[level][y_or_tuple].add_point(...)");
                 }
             },
             py::arg("point"),
             "Add point as interval to level (only for 1D, use indexing for higher dims)")

        // Also keep grid_yz as read-only property for advanced use
        .def_property_readonly("grid_yz",
                               &LCL::grid_yz,
                               "Underlying sparse grid structure (nested maps)");
}

// ============================================================================
// CellList binding
// ============================================================================

template <std::size_t dim>
void bind_cell_list(py::module_& m, const std::string& name)
{
    using CL = CellList<dim>;
    using lcl_type = typename CL::lcl_type;
    using coords_t = typename CL::coords_t;

    py::class_<CL>(m, name.c_str(), R"pbdoc(
        Hierarchical cell container for building adaptive meshes.

        CellList stores cells organized by refinement levels, with each level
        represented by a LevelCellList containing interval-based sparse data.

        This is a builder structure used during mesh construction. For efficient
        computation, convert to CellArray via mesh.Mesh.from_cell_list().

        Examples
        --------
        >>> import sampai as sam
        >>> cl = sam.cell.CellList2D()
        >>> cl[0][0].add_interval(sam.Interval(0, 4))  # Level 0, y=0, x=[0,4)
        >>> cl[1][2].add_interval(sam.Interval(2, 6))  # Level 1, y=2, x=[2,6]
    )pbdoc")

        // Default constructor
        .def(py::init<>(), "Create empty CellList (all levels initialized)")

        // Constructor with geometry - wrapper to accept Python tuples
        .def(py::init([](const py::object& origin_obj, double scaling_factor)
             {
                 coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<dim>(origin_obj);
                 return CL(origin, scaling_factor);
             }),
             py::arg("origin_point") = py::none(),
             py::arg("scaling_factor") = 1.0,
             "Create CellList with origin point and scaling factor")

        // Access LevelCellList at level
        .def("__getitem__",
             [](CL& cl, std::size_t level) -> lcl_type&
             {
                 if (level > CL::max_size)
                 {
                     throw py::index_error("Level out of range");
                 }
                 return cl[level];
             },
             py::arg("level"),
             py::return_value_policy::reference_internal,
             "Access LevelCellList at given refinement level")

        // Properties
        .def_property_readonly("max_level",
                              [](const CL&)
                              {
                                  return CL::max_size;
                              },
                              "Maximum refinement level")
        .def_property_readonly("dim",
                              [](const CL&)
                              {
                                  return CL::dim;
                              },
                              "Spatial dimension")
        .def_property_readonly("origin_point",
                              [](const CL& cl)
                              {
                                  return convert_to_tuple<dim>(cl.origin_point());
                              },
                              "Origin point in physical coordinates")
        .def_property_readonly("scaling_factor",
                              &CL::scaling_factor,
                              "Scaling factor for coordinate transformation")

        // Methods
        // Note: clear() method has C++ implementation issues with forward_list inheritance
        // .def("clear", &CL::clear, "Clear all levels")

        .def("empty",
             [](const CL& cl)
             {
                 // CellList doesn't have empty(), check all levels
                 for (std::size_t level = 0; level <= CL::max_size; ++level)
                 {
                     if (!cl[level].empty())
                     {
                         return false;
                     }
                 }
                 return true;
             },
             "Check if all levels are empty")

        .def("to_stream",
             [](const CL& cl, std::ostream& os)
             {
                 cl.to_stream(os);
             },
             py::arg("os"),
             "Output to stream (for debugging)")

        // String representations
        .def("__repr__",
             [name](const CL& cl)
             {
                 std::ostringstream oss;
                 oss << name << "(dim=" << CL::dim << ", max_level=" << CL::max_size << ")";
                 return oss.str();
             })
        .def("__str__",
             [name](const CL& cl)
             {
                 std::ostringstream oss;
                 oss << name << " [" << CL::dim << "D, max L=" << CL::max_size << "]";
                 return oss.str();
             });
}

// ============================================================================
// Factory function for dimension-agnostic creation
// ============================================================================

// Infer dimension from origin_point tuple/list length
std::size_t infer_dim_from_origin(const py::object& origin_obj)
{
    if (origin_obj.is_none())
    {
        throw std::runtime_error("Cannot infer dimension: origin_point is None. "
                                 "Please specify 'dim' parameter explicitly.");
    }

    if (py::isinstance<py::tuple>(origin_obj))
    {
        auto t = origin_obj.cast<py::tuple>();
        return t.size();
    }
    else if (py::isinstance<py::list>(origin_obj))
    {
        auto l = origin_obj.cast<py::list>();
        return l.size();
    }
    else if (py::isinstance<py::array>(origin_obj))
    {
        auto a = origin_obj.cast<py::array_t<double>>();
        return a.ndim();
    }
    else
    {
        throw std::runtime_error("Cannot infer dimension from origin_point type. "
                                 "Use tuple/list/array or specify 'dim' parameter.");
    }
}

// Main factory function with smart dimension inference
py::object cell_list_factory(py::object dim_obj, py::object origin_obj, double scaling_factor)
{
    std::size_t dim = 0;

    // Determine dimension
    if (!dim_obj.is_none())
    {
        // Explicit dim parameter
        dim = dim_obj.cast<std::size_t>();
    }
    else if (!origin_obj.is_none())
    {
        // Infer from origin_point
        dim = infer_dim_from_origin(origin_obj);
    }
    else
    {
        throw std::runtime_error("Cannot determine dimension. "
                                 "Please specify either 'dim' or 'origin_point' parameter.");
    }

    // Validate dimension
    if (dim < 1 || dim > 3)
    {
        throw std::runtime_error("Invalid dimension: must be 1, 2, or 3");
    }

    // Create appropriate CellList
    if (dim == 1)
    {
        using CL = CellList<1>;
        using coords_t = typename CL::coords_t;
        coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<1>(origin_obj);
        return py::cast(CL(origin, scaling_factor));
    }
    else if (dim == 2)
    {
        using CL = CellList<2>;
        using coords_t = typename CL::coords_t;
        coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<2>(origin_obj);
        return py::cast(CL(origin, scaling_factor));
    }
    else  // dim == 3
    {
        using CL = CellList<3>;
        using coords_t = typename CL::coords_t;
        coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<3>(origin_obj);
        return py::cast(CL(origin, scaling_factor));
    }
}

// Infer dimension from indices tuple/list length
std::size_t infer_dim_from_indices(const py::object& indices_obj)
{
    if (py::isinstance<py::tuple>(indices_obj))
    {
        auto t = indices_obj.cast<py::tuple>();
        return t.size();
    }
    else if (py::isinstance<py::list>(indices_obj))
    {
        auto l = indices_obj.cast<py::list>();
        return l.size();
    }
    else if (py::isinstance<py::array>(indices_obj))
    {
        auto a = indices_obj.cast<py::array_t<int>>();
        return a.size();
    }
    else
    {
        throw std::runtime_error("Cannot infer dimension from indices type. "
                                 "Use tuple/list/array or specify 'dim' parameter.");
    }
}

// Factory function for Cell with dimension inference
py::object cell_factory(py::object dim_obj,
                        std::size_t level,
                        const py::object& indices_obj,
                        long long int index,
                        const py::object& origin_obj,
                        double scaling_factor)
{
    std::size_t dim = 0;

    // Determine dimension
    if (!dim_obj.is_none())
    {
        // Explicit dim parameter
        dim = dim_obj.cast<std::size_t>();
    }
    else
    {
        // Infer from indices
        dim = infer_dim_from_indices(indices_obj);
    }

    // Validate dimension
    if (dim < 1 || dim > 3)
    {
        throw std::runtime_error("Invalid dimension: must be 1, 2, or 3");
    }

    // Create appropriate Cell
    if (dim == 1)
    {
        using Cell_t = Cell<1>;
        using coords_t = typename Cell_t::coords_t;
        using indices_t = typename Cell_t::indices_t;
        coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<1>(origin_obj);

        indices_t indices;
        if (py::isinstance<py::tuple>(indices_obj) || py::isinstance<py::list>(indices_obj))
        {
            auto seq = indices_obj.cast<py::sequence>();
            if (seq.size() != 1)
            {
                throw std::runtime_error("indices must have 1 element for 1D");
            }
            indices[0] = seq[0].cast<typename indices_t::value_type>();
        }
        else if (py::isinstance<py::array>(indices_obj))
        {
            auto arr = indices_obj.cast<py::array_t<typename indices_t::value_type>>();
            if (arr.size() != 1)
            {
                throw std::runtime_error("indices must have 1 element for 1D");
            }
            auto buf = arr.request();
            auto* ptr = static_cast<typename indices_t::value_type*>(buf.ptr);
            indices[0] = ptr[0];
        }
        else
        {
            throw std::runtime_error("indices must be tuple, list, or array");
        }
        return py::cast(Cell_t(origin, scaling_factor, level, indices, static_cast<signed long long int>(index)));
    }
    else if (dim == 2)
    {
        using Cell_t = Cell<2>;
        using coords_t = typename Cell_t::coords_t;
        using indices_t = typename Cell_t::indices_t;
        coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<2>(origin_obj);

        indices_t indices;
        if (py::isinstance<py::tuple>(indices_obj) || py::isinstance<py::list>(indices_obj))
        {
            auto seq = indices_obj.cast<py::sequence>();
            if (seq.size() != 2)
            {
                throw std::runtime_error("indices must have 2 elements for 2D");
            }
            indices[0] = seq[0].cast<typename indices_t::value_type>();
            indices[1] = seq[1].cast<typename indices_t::value_type>();
        }
        else if (py::isinstance<py::array>(indices_obj))
        {
            auto arr = indices_obj.cast<py::array_t<typename indices_t::value_type>>();
            if (arr.size() != 2)
            {
                throw std::runtime_error("indices must have 2 elements for 2D");
            }
            auto buf = arr.request();
            auto* ptr = static_cast<typename indices_t::value_type*>(buf.ptr);
            indices[0] = ptr[0];
            indices[1] = ptr[1];
        }
        else
        {
            throw std::runtime_error("indices must be tuple, list, or array");
        }
        return py::cast(Cell_t(origin, scaling_factor, level, indices, static_cast<signed long long int>(index)));
    }
    else  // dim == 3
    {
        using Cell_t = Cell<3>;
        using coords_t = typename Cell_t::coords_t;
        using indices_t = typename Cell_t::indices_t;
        coords_t origin = origin_obj.is_none() ? coords_t{} : convert_to_origin_point<3>(origin_obj);

        indices_t indices;
        if (py::isinstance<py::tuple>(indices_obj) || py::isinstance<py::list>(indices_obj))
        {
            auto seq = indices_obj.cast<py::sequence>();
            if (seq.size() != 3)
            {
                throw std::runtime_error("indices must have 3 elements for 3D");
            }
            indices[0] = seq[0].cast<typename indices_t::value_type>();
            indices[1] = seq[1].cast<typename indices_t::value_type>();
            indices[2] = seq[2].cast<typename indices_t::value_type>();
        }
        else if (py::isinstance<py::array>(indices_obj))
        {
            auto arr = indices_obj.cast<py::array_t<typename indices_t::value_type>>();
            if (arr.size() != 3)
            {
                throw std::runtime_error("indices must have 3 elements for 3D");
            }
            auto buf = arr.request();
            auto* ptr = static_cast<typename indices_t::value_type*>(buf.ptr);
            indices[0] = ptr[0];
            indices[1] = ptr[1];
            indices[2] = ptr[2];
        }
        else
        {
            throw std::runtime_error("indices must be tuple, list, or array");
        }
        return py::cast(Cell_t(origin, scaling_factor, level, indices, static_cast<signed long long int>(index)));
    }
}

// ============================================================================
// Module initialization
// ============================================================================

void init_cell_list_bindings(py::module_& m)
{
    // ============================================================
    // Create cell submodule for organized API access
    // ============================================================
    py::module_ cell = m.def_submodule("cell",
                                         "Cell construction utilities for Samurai AMR simulations\n\n"
                                         "Factory Functions:\n"
                                         "  CellList(dim=None, origin_point=None, scaling_factor=1.0) - Create CellList\n"
                                         "    Dimension is inferred from origin_point if not specified.\n\n"
                                         "  Cell(dim=None, level, indices, index=0, origin_point=None, scaling_factor=1.0) - Create Cell\n"
                                         "    Dimension is inferred from indices if not specified.\n\n"
                                         "  Interval(start, end, index=0) - Create an Interval\n\n"
                                         "Classes (for advanced use):\n"
                                         "  Cell1D, Cell2D, Cell3D - Individual cell objects\n"
                                         "  CellList1D, CellList2D, CellList3D - Hierarchical cell containers\n"
                                         "  LevelCellList1D, LevelCellList2D, LevelCellList3D - Single-level cell lists\n"
                                         "  ListOfIntervals - Forward list of intervals along x-axis\n\n"
                                         "Examples:\n"
                                         "    >>> import sampai as sam\n"
                                         "    >>> # Create a cell\n"
                                         "    >>> cell = sam.cell.Cell(level=2, indices=(3, 7))\n"
                                         "    >>> # Add cell to LevelCellList\n"
                                         "    >>> cl = sam.cell.CellList(dim=2)\n"
                                         "    >>> cl[2].add_cell(cell)\n");

    // Bind Cell classes for each dimension
    bind_cell<1>(cell, "Cell1D");
    bind_cell<2>(cell, "Cell2D");
    bind_cell<3>(cell, "Cell3D");

    // Bind ListOfIntervals (used by LevelCellList)
    bind_list_of_intervals(cell);

    // Bind LevelCellList for each dimension (optional, for advanced users)
    bind_level_cell_list<1>(cell, "LevelCellList1D");
    bind_level_cell_list<2>(cell, "LevelCellList2D");
    bind_level_cell_list<3>(cell, "LevelCellList3D");

    // Bind CellList for each dimension (optional, for advanced users)
    bind_cell_list<1>(cell, "CellList1D");
    bind_cell_list<2>(cell, "CellList2D");
    bind_cell_list<3>(cell, "CellList3D");

    // Main factory function - named CellList for API consistency
    // Works like sam.geometry.box() - infers dimension from origin_point
    cell.def("CellList",
             &cell_list_factory,
             py::arg("dim") = py::none(),
             py::arg("origin_point") = py::none(),
             py::arg("scaling_factor") = 1.0,
             R"pbdoc(Create CellList for adaptive mesh construction.

    The dimension is automatically inferred from the origin_point length
    if not explicitly specified. This provides a similar API to box().

    Parameters
    ----------
    dim : int, optional
        Spatial dimension (1, 2, or 3). If None, inferred from origin_point.
    origin_point : tuple of float, optional
        Physical coordinates of origin point. Dimension inferred from length.
        If None and dim is specified, uses default origin (0,)*dim.
    scaling_factor : float, optional
        Scaling factor for cell sizes (default: 1.0)

    Returns
    -------
    CellList1D, CellList2D, or CellList3D
        CellList object for the specified or inferred dimension

    Raises
    ------
    RuntimeError
        If dimension cannot be determined or is invalid

    Examples
    --------
    >>> import sampai as sam
    >>> # Dimension inferred from origin_point (2D)
    >>> cl = sam.cell.CellList(origin_point=(0., 0.))
    >>> # Explicit dimension with default origin
    >>> cl = sam.cell.CellList(dim=2)
    >>> # With all parameters
    >>> cl = sam.cell.CellList(dim=3, origin_point=(1., 2., 3.), scaling_factor=0.5)
    >>> # Access level 0
    >>> level_0 = cl[0]
    >>> level_0.level
    0
    )pbdoc");

    // Factory function for Cell
    cell.def("Cell",
             &cell_factory,
             py::arg("dim") = py::none(),
             py::arg("level"),
             py::arg("indices"),
             py::arg("index") = 0,
             py::arg("origin_point") = py::none(),
             py::arg("scaling_factor") = 1.0,
             R"pbdoc(Create a Cell for adding to LevelCellList.

    The dimension is automatically inferred from the indices length
    if not explicitly specified.

    Parameters
    ----------
    dim : int, optional
        Spatial dimension (1, 2, or 3). If None, inferred from indices.
    level : int
        Refinement level of the cell
    indices : tuple of int
        Integer coordinates (i, j, k) for the cell
    index : int, optional
        Index in data array (default: 0)
    origin_point : tuple of float, optional
        Physical origin (default: (0,)*dim)
    scaling_factor : float, optional
        Scaling factor (default: 1.0)

    Returns
    -------
    Cell1D, Cell2D, or Cell3D
        Cell object for the specified or inferred dimension

    Raises
    ------
    RuntimeError
        If dimension cannot be determined or is invalid

    Examples
    --------
    >>> import sampai as sam
    >>> # Dimension inferred from indices (2D)
    >>> cell = sam.cell.Cell(level=2, indices=(3, 7))
    >>> # Explicit dimension
    >>> cell = sam.cell.Cell(dim=2, level=2, indices=(3, 7))
    >>> # Add cell to LevelCellList
    >>> cl = sam.cell.CellList(dim=2)
    >>> cl[2].add_cell(cell)
    )pbdoc");

    // Also expose Interval for convenience
    cell.def("Interval",
             [](int start, int end, long long int index)
             {
                 return cell_interval(start, end, index);
             },
             py::arg("start"),
             py::arg("end"),
             py::arg("index") = 0,
             "Create an Interval [start, end) with optional index");
}
