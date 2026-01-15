// Samurai Python Bindings - HDF5 I/O
//
// Bindings for save(), dump(), and load() functions for fields and meshes
// to enable Paraview visualization and checkpoint/restart functionality

#include <filesystem>
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mesh_config.hpp>
#include <samurai/mr/mesh.hpp>
#include "common_types.hpp"

namespace py = pybind11;

// Use centralized type aliases from common_types.hpp
using namespace samurai::python::bindings;

// ============================================================
// Constants for bulk field operations
// ============================================================

constexpr std::size_t MAX_BULK_FIELDS = 9;

// ============================================================
// Helper to convert Python path/string to fs::path
// Supports pathlib.Path objects and PathLike protocol
// ============================================================

inline std::filesystem::path to_fs_path(const py::object& path_obj)
{
    if (path_obj.is_none())
    {
        return std::filesystem::current_path();
    }

    // Try os.PathLike protocol first (supports pathlib.Path)
    if (py::hasattr(path_obj, "__fspath__"))
    {
        auto fspath_result = path_obj.attr("__fspath__")();
        return std::filesystem::path(py::str(fspath_result));
    }

    // Fallback to string conversion
    return std::filesystem::path(py::str(path_obj));
}

// ============================================================
// Helper to extract mesh from field (for single field case)
// ============================================================

template <std::size_t dim>
const MRMesh<dim>& extract_mesh(const ScalarField<dim>& field)
{
    return field.mesh();
}

// ============================================================
// Unified filepath parsing helper
// ============================================================

// Parse unified filepath into directory and filename
struct FilePathParts
{
    std::filesystem::path directory;
    std::string basename;
};

inline FilePathParts parse_unified_filepath(const py::object& filepath_obj)
{
    std::filesystem::path filepath = to_fs_path(filepath_obj);

    // Extract directory and basename
    std::filesystem::path directory = filepath.parent_path();
    std::string filename = filepath.filename().string();

    // Remove .h5 extension if present (only at the end)
    // Note: can't use stem() because it removes everything after the first dot
    // e.g., "test-0.1.h5" -> stem() = "test-0", but we want "test-0.1"
    std::string basename = filename;
    if (basename.size() >= 3 && basename.substr(basename.size() - 3) == ".h5")
    {
        basename = basename.substr(0, basename.size() - 3);
    }

    // If no directory, use current directory
    if (directory.empty())
    {
        directory = std::filesystem::current_path();
    }

    return {directory, basename};
}

// ============================================================
// Unified save() function wrappers - 1D
// ============================================================

// Save single field (1D) - unified filepath
void save_1d(const py::object& filepath_obj, const ScalarField<1>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

// Save two fields (1D) - unified filepath
void save_1d_2fields(const py::object& filepath_obj, const ScalarField<1>& field1, const ScalarField<1>& field2)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2);
}

// Save three fields (1D) - unified filepath
void save_1d_3fields(const py::object& filepath_obj, const ScalarField<1>& field1, const ScalarField<1>& field2, const ScalarField<1>& field3)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// Unified save() function wrappers - 2D
// ============================================================

void save_2d(const py::object& filepath_obj, const ScalarField<2>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

void save_2d_2fields(const py::object& filepath_obj, const ScalarField<2>& field1, const ScalarField<2>& field2)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2);
}

void save_2d_3fields(const py::object& filepath_obj, const ScalarField<2>& field1, const ScalarField<2>& field2, const ScalarField<2>& field3)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// Unified save() function wrappers - 3D
// ============================================================

void save_3d(const py::object& filepath_obj, const ScalarField<3>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

void save_3d_2fields(const py::object& filepath_obj, const ScalarField<3>& field1, const ScalarField<3>& field2)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2);
}

void save_3d_3fields(const py::object& filepath_obj, const ScalarField<3>& field1, const ScalarField<3>& field2, const ScalarField<3>& field3)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field1.mesh(), field1, field2, field3);
}

// ============================================================
// Unified save() function wrappers - VectorField (2D and 3D)
// ============================================================

void save_2d_vector(const py::object& filepath_obj, const VectorField2D_2& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

void save_3d_vector(const py::object& filepath_obj, const VectorField3D_3& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - 1D
// ============================================================

void dump_1d(const py::object& filepath_obj, const ScalarField<1>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - 2D
// ============================================================

void dump_2d(const py::object& filepath_obj, const ScalarField<2>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - 3D
// ============================================================

void dump_3d(const py::object& filepath_obj, const ScalarField<3>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified dump() function wrappers - VectorField (2D and 3D)
// ============================================================

void dump_2d_vector(const py::object& filepath_obj, const VectorField2D_2& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

void dump_3d_vector(const py::object& filepath_obj, const VectorField3D_3& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified load() function wrappers - 1D
// ============================================================

void load_1d(const py::object& filepath_obj, ScalarField<1>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified load() function wrappers - 2D
// ============================================================

void load_2d(const py::object& filepath_obj, ScalarField<2>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// Unified load() function wrappers - 3D
// ============================================================

void load_3d(const py::object& filepath_obj, ScalarField<3>& field)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// open_h5py() helper function - Open HDF5 files with h5py
// ============================================================

py::object open_h5py_wrapper(const py::object& filename_obj, const std::string& mode)
{
    // Convert filename to string, add .h5 extension if needed
    std::string filename = py::str(filename_obj);
    if (filename.size() < 3 || filename.substr(filename.size() - 3) != ".h5")
    {
        filename = filename + ".h5";
    }

    // Import h5py
    auto h5py = py::module_::import("h5py");
    auto File = h5py.attr("File");

    // Return h5py.File object
    return File(filename, mode);
}

// ============================================================
// Field method wrappers for save()
// ============================================================

template <std::size_t dim>
void field_method_save(const ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_save_vector(const VectorField2D_2& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_save_vector3d(const VectorField3D_3& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::save(directory, basename, field.mesh(), field);
}

// ============================================================
// Field method wrappers for dump()
// ============================================================

template <std::size_t dim>
void field_method_dump(const ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_dump_vector(const VectorField2D_2& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

template <std::size_t dim>
void field_method_dump_vector3d(const VectorField3D_3& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::dump(directory, basename, field.mesh(), field);
}

// ============================================================
// Field method wrappers for load()
// ============================================================

template <std::size_t dim>
void field_method_load(ScalarField<dim>& field, const py::object& filepath_obj)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    samurai::load(directory, basename, field.mesh(), field);
}

// ============================================================
// Bulk save() - support arbitrary number of fields (up to 8)
// Uses py::args to capture variable-length field lists
// ============================================================

// Helper to validate const fields and extract mesh (for save() and dump())
// Returns dimension and mesh pointer
template <std::size_t dim>
const MRMesh<dim>* try_validate_and_get_mesh(py::args fields, size_t& out_n)
{
    out_n = len(fields);
    if (out_n == 0)
    {
        return nullptr;
    }

    try
    {
        const auto& first_field = fields[0].cast<const ScalarField<dim>&>();
        const auto& mesh = &first_field.mesh();

        // Validate all other fields have the same mesh
        for (size_t i = 1; i < out_n; ++i)
        {
            const auto& field = fields[i].cast<const ScalarField<dim>&>();
            if (&field.mesh() != mesh)
            {
                throw std::runtime_error("All fields must share the same mesh");
            }
        }

        return mesh;
    }
    catch (const py::cast_error&)
    {
        return nullptr;
    }
}

// Helper to validate non-const fields and extract mesh (for load())
// Returns dimension and mesh pointer
template <std::size_t dim>
MRMesh<dim>* try_validate_and_get_mesh_nonconst(py::args fields, size_t& out_n)
{
    out_n = len(fields);
    if (out_n == 0)
    {
        return nullptr;
    }

    try
    {
        auto& first_field = fields[0].cast<ScalarField<dim>&>();
        auto* mesh = &first_field.mesh();

        // Validate all other fields have the same mesh
        for (size_t i = 1; i < out_n; ++i)
        {
            auto& field = fields[i].cast<ScalarField<dim>&>();
            if (&field.mesh() != mesh)
            {
                throw std::runtime_error("All fields must share the same mesh");
            }
        }

        return mesh;
    }
    catch (const py::cast_error&)
    {
        return nullptr;
    }
}

// Single bulk save function that handles all dimensions
void save_bulk(const py::object& filepath_obj, py::args fields)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    size_t n = len(fields);

    if (n == 0)
    {
        throw std::runtime_error("At least one field must be provided");
    }

    // Try to detect dimension by attempting to cast the first field
    const MRMesh<1>* mesh_1d = nullptr;
    const MRMesh<2>* mesh_2d = nullptr;
    const MRMesh<3>* mesh_3d = nullptr;

    size_t dummy_n;
    mesh_1d = try_validate_and_get_mesh<1>(fields, dummy_n);
    if (mesh_1d)
    {
        // 1D fields
        switch (n)
        {
            case 1:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0);
                return;
            }
            case 2:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1);
                return;
            }
            case 3:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1, f2);
                return;
            }
            case 4:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                const auto& f6 = fields[6].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                const auto& f6 = fields[6].cast<const ScalarField<1>&>();
                const auto& f7 = fields[7].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                const auto& f6 = fields[6].cast<const ScalarField<1>&>();
                const auto& f7 = fields[7].cast<const ScalarField<1>&>();
                const auto& f8 = fields[8].cast<const ScalarField<1>&>();
                samurai::save(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 1D bulk save, got {}", MAX_BULK_FIELDS, n));
        }
    }

    mesh_2d = try_validate_and_get_mesh<2>(fields, dummy_n);
    if (mesh_2d)
    {
        // 2D fields
        switch (n)
        {
            case 1:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0);
                return;
            }
            case 2:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1);
                return;
            }
            case 3:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1, f2);
                return;
            }
            case 4:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                const auto& f6 = fields[6].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                const auto& f6 = fields[6].cast<const ScalarField<2>&>();
                const auto& f7 = fields[7].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                const auto& f6 = fields[6].cast<const ScalarField<2>&>();
                const auto& f7 = fields[7].cast<const ScalarField<2>&>();
                const auto& f8 = fields[8].cast<const ScalarField<2>&>();
                samurai::save(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 2D bulk save, got {}", MAX_BULK_FIELDS, n));
        }
    }

    mesh_3d = try_validate_and_get_mesh<3>(fields, dummy_n);
    if (mesh_3d)
    {
        // 3D fields
        switch (n)
        {
            case 1:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0);
                return;
            }
            case 2:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1);
                return;
            }
            case 3:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1, f2);
                return;
            }
            case 4:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                const auto& f6 = fields[6].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                const auto& f6 = fields[6].cast<const ScalarField<3>&>();
                const auto& f7 = fields[7].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                const auto& f6 = fields[6].cast<const ScalarField<3>&>();
                const auto& f7 = fields[7].cast<const ScalarField<3>&>();
                const auto& f8 = fields[8].cast<const ScalarField<3>&>();
                samurai::save(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 3D bulk save, got {}", MAX_BULK_FIELDS, n));
        }
    }

    throw std::runtime_error("Unable to determine field dimension or invalid field types");
}

// ============================================================
// Bulk dump() - support arbitrary number of fields (up to 8)
// Uses py::args to capture variable-length field lists
// ============================================================

void dump_bulk(const py::object& filepath_obj, py::args fields)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    size_t n = len(fields);

    if (n == 0)
    {
        throw std::runtime_error("At least one field must be provided");
    }

    // Try to detect dimension by attempting to cast the first field
    const MRMesh<1>* mesh_1d = nullptr;
    const MRMesh<2>* mesh_2d = nullptr;
    const MRMesh<3>* mesh_3d = nullptr;

    size_t dummy_n;
    mesh_1d = try_validate_and_get_mesh<1>(fields, dummy_n);
    if (mesh_1d)
    {
        // 1D fields
        switch (n)
        {
            case 1:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0);
                return;
            }
            case 2:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1);
                return;
            }
            case 3:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1, f2);
                return;
            }
            case 4:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                const auto& f6 = fields[6].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                const auto& f6 = fields[6].cast<const ScalarField<1>&>();
                const auto& f7 = fields[7].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                const auto& f0 = fields[0].cast<const ScalarField<1>&>();
                const auto& f1 = fields[1].cast<const ScalarField<1>&>();
                const auto& f2 = fields[2].cast<const ScalarField<1>&>();
                const auto& f3 = fields[3].cast<const ScalarField<1>&>();
                const auto& f4 = fields[4].cast<const ScalarField<1>&>();
                const auto& f5 = fields[5].cast<const ScalarField<1>&>();
                const auto& f6 = fields[6].cast<const ScalarField<1>&>();
                const auto& f7 = fields[7].cast<const ScalarField<1>&>();
                const auto& f8 = fields[8].cast<const ScalarField<1>&>();
                samurai::dump(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 1D bulk dump, got {}", MAX_BULK_FIELDS, n));
        }
    }

    mesh_2d = try_validate_and_get_mesh<2>(fields, dummy_n);
    if (mesh_2d)
    {
        // 2D fields
        switch (n)
        {
            case 1:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0);
                return;
            }
            case 2:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1);
                return;
            }
            case 3:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1, f2);
                return;
            }
            case 4:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                const auto& f6 = fields[6].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                const auto& f6 = fields[6].cast<const ScalarField<2>&>();
                const auto& f7 = fields[7].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                const auto& f0 = fields[0].cast<const ScalarField<2>&>();
                const auto& f1 = fields[1].cast<const ScalarField<2>&>();
                const auto& f2 = fields[2].cast<const ScalarField<2>&>();
                const auto& f3 = fields[3].cast<const ScalarField<2>&>();
                const auto& f4 = fields[4].cast<const ScalarField<2>&>();
                const auto& f5 = fields[5].cast<const ScalarField<2>&>();
                const auto& f6 = fields[6].cast<const ScalarField<2>&>();
                const auto& f7 = fields[7].cast<const ScalarField<2>&>();
                const auto& f8 = fields[8].cast<const ScalarField<2>&>();
                samurai::dump(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 2D bulk dump, got {}", MAX_BULK_FIELDS, n));
        }
    }

    mesh_3d = try_validate_and_get_mesh<3>(fields, dummy_n);
    if (mesh_3d)
    {
        // 3D fields
        switch (n)
        {
            case 1:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0);
                return;
            }
            case 2:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1);
                return;
            }
            case 3:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1, f2);
                return;
            }
            case 4:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                const auto& f6 = fields[6].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                const auto& f6 = fields[6].cast<const ScalarField<3>&>();
                const auto& f7 = fields[7].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                const auto& f0 = fields[0].cast<const ScalarField<3>&>();
                const auto& f1 = fields[1].cast<const ScalarField<3>&>();
                const auto& f2 = fields[2].cast<const ScalarField<3>&>();
                const auto& f3 = fields[3].cast<const ScalarField<3>&>();
                const auto& f4 = fields[4].cast<const ScalarField<3>&>();
                const auto& f5 = fields[5].cast<const ScalarField<3>&>();
                const auto& f6 = fields[6].cast<const ScalarField<3>&>();
                const auto& f7 = fields[7].cast<const ScalarField<3>&>();
                const auto& f8 = fields[8].cast<const ScalarField<3>&>();
                samurai::dump(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 3D bulk dump, got {}", MAX_BULK_FIELDS, n));
        }
    }

    throw std::runtime_error("Unable to determine field dimension or invalid field types");
}

// ============================================================
// Bulk load() - support arbitrary number of fields (up to 8)
// Uses py::args to capture variable-length field lists
// ============================================================

void load_bulk(const py::object& filepath_obj, py::args fields)
{
    auto [directory, basename] = parse_unified_filepath(filepath_obj);
    size_t n = len(fields);

    if (n == 0)
    {
        throw std::runtime_error("At least one field must be provided");
    }

    // Try to detect dimension by attempting to cast the first field (non-const for load)
    MRMesh<1>* mesh_1d = nullptr;
    MRMesh<2>* mesh_2d = nullptr;
    MRMesh<3>* mesh_3d = nullptr;

    size_t dummy_n;
    mesh_1d = try_validate_and_get_mesh_nonconst<1>(fields, dummy_n);
    if (mesh_1d)
    {
        // 1D fields
        switch (n)
        {
            case 1:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0);
                return;
            }
            case 2:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1);
                return;
            }
            case 3:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                auto& f2 = fields[2].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1, f2);
                return;
            }
            case 4:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                auto& f2 = fields[2].cast<ScalarField<1>&>();
                auto& f3 = fields[3].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                auto& f2 = fields[2].cast<ScalarField<1>&>();
                auto& f3 = fields[3].cast<ScalarField<1>&>();
                auto& f4 = fields[4].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                auto& f2 = fields[2].cast<ScalarField<1>&>();
                auto& f3 = fields[3].cast<ScalarField<1>&>();
                auto& f4 = fields[4].cast<ScalarField<1>&>();
                auto& f5 = fields[5].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                auto& f2 = fields[2].cast<ScalarField<1>&>();
                auto& f3 = fields[3].cast<ScalarField<1>&>();
                auto& f4 = fields[4].cast<ScalarField<1>&>();
                auto& f5 = fields[5].cast<ScalarField<1>&>();
                auto& f6 = fields[6].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                auto& f2 = fields[2].cast<ScalarField<1>&>();
                auto& f3 = fields[3].cast<ScalarField<1>&>();
                auto& f4 = fields[4].cast<ScalarField<1>&>();
                auto& f5 = fields[5].cast<ScalarField<1>&>();
                auto& f6 = fields[6].cast<ScalarField<1>&>();
                auto& f7 = fields[7].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                auto& f0 = fields[0].cast<ScalarField<1>&>();
                auto& f1 = fields[1].cast<ScalarField<1>&>();
                auto& f2 = fields[2].cast<ScalarField<1>&>();
                auto& f3 = fields[3].cast<ScalarField<1>&>();
                auto& f4 = fields[4].cast<ScalarField<1>&>();
                auto& f5 = fields[5].cast<ScalarField<1>&>();
                auto& f6 = fields[6].cast<ScalarField<1>&>();
                auto& f7 = fields[7].cast<ScalarField<1>&>();
                auto& f8 = fields[8].cast<ScalarField<1>&>();
                samurai::load(directory, basename, *mesh_1d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 1D bulk load, got {}", MAX_BULK_FIELDS, n));
        }
    }

    mesh_2d = try_validate_and_get_mesh_nonconst<2>(fields, dummy_n);
    if (mesh_2d)
    {
        // 2D fields
        switch (n)
        {
            case 1:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0);
                return;
            }
            case 2:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1);
                return;
            }
            case 3:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                auto& f2 = fields[2].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1, f2);
                return;
            }
            case 4:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                auto& f2 = fields[2].cast<ScalarField<2>&>();
                auto& f3 = fields[3].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                auto& f2 = fields[2].cast<ScalarField<2>&>();
                auto& f3 = fields[3].cast<ScalarField<2>&>();
                auto& f4 = fields[4].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                auto& f2 = fields[2].cast<ScalarField<2>&>();
                auto& f3 = fields[3].cast<ScalarField<2>&>();
                auto& f4 = fields[4].cast<ScalarField<2>&>();
                auto& f5 = fields[5].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                auto& f2 = fields[2].cast<ScalarField<2>&>();
                auto& f3 = fields[3].cast<ScalarField<2>&>();
                auto& f4 = fields[4].cast<ScalarField<2>&>();
                auto& f5 = fields[5].cast<ScalarField<2>&>();
                auto& f6 = fields[6].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                auto& f2 = fields[2].cast<ScalarField<2>&>();
                auto& f3 = fields[3].cast<ScalarField<2>&>();
                auto& f4 = fields[4].cast<ScalarField<2>&>();
                auto& f5 = fields[5].cast<ScalarField<2>&>();
                auto& f6 = fields[6].cast<ScalarField<2>&>();
                auto& f7 = fields[7].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                auto& f0 = fields[0].cast<ScalarField<2>&>();
                auto& f1 = fields[1].cast<ScalarField<2>&>();
                auto& f2 = fields[2].cast<ScalarField<2>&>();
                auto& f3 = fields[3].cast<ScalarField<2>&>();
                auto& f4 = fields[4].cast<ScalarField<2>&>();
                auto& f5 = fields[5].cast<ScalarField<2>&>();
                auto& f6 = fields[6].cast<ScalarField<2>&>();
                auto& f7 = fields[7].cast<ScalarField<2>&>();
                auto& f8 = fields[8].cast<ScalarField<2>&>();
                samurai::load(directory, basename, *mesh_2d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 2D bulk load, got {}", MAX_BULK_FIELDS, n));
        }
    }

    mesh_3d = try_validate_and_get_mesh_nonconst<3>(fields, dummy_n);
    if (mesh_3d)
    {
        // 3D fields
        switch (n)
        {
            case 1:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0);
                return;
            }
            case 2:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1);
                return;
            }
            case 3:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                auto& f2 = fields[2].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1, f2);
                return;
            }
            case 4:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                auto& f2 = fields[2].cast<ScalarField<3>&>();
                auto& f3 = fields[3].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1, f2, f3);
                return;
            }
            case 5:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                auto& f2 = fields[2].cast<ScalarField<3>&>();
                auto& f3 = fields[3].cast<ScalarField<3>&>();
                auto& f4 = fields[4].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1, f2, f3, f4);
                return;
            }
            case 6:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                auto& f2 = fields[2].cast<ScalarField<3>&>();
                auto& f3 = fields[3].cast<ScalarField<3>&>();
                auto& f4 = fields[4].cast<ScalarField<3>&>();
                auto& f5 = fields[5].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5);
                return;
            }
            case 7:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                auto& f2 = fields[2].cast<ScalarField<3>&>();
                auto& f3 = fields[3].cast<ScalarField<3>&>();
                auto& f4 = fields[4].cast<ScalarField<3>&>();
                auto& f5 = fields[5].cast<ScalarField<3>&>();
                auto& f6 = fields[6].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6);
                return;
            }
            case 8:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                auto& f2 = fields[2].cast<ScalarField<3>&>();
                auto& f3 = fields[3].cast<ScalarField<3>&>();
                auto& f4 = fields[4].cast<ScalarField<3>&>();
                auto& f5 = fields[5].cast<ScalarField<3>&>();
                auto& f6 = fields[6].cast<ScalarField<3>&>();
                auto& f7 = fields[7].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6, f7);
                return;
            }
            case 9:
            {
                auto& f0 = fields[0].cast<ScalarField<3>&>();
                auto& f1 = fields[1].cast<ScalarField<3>&>();
                auto& f2 = fields[2].cast<ScalarField<3>&>();
                auto& f3 = fields[3].cast<ScalarField<3>&>();
                auto& f4 = fields[4].cast<ScalarField<3>&>();
                auto& f5 = fields[5].cast<ScalarField<3>&>();
                auto& f6 = fields[6].cast<ScalarField<3>&>();
                auto& f7 = fields[7].cast<ScalarField<3>&>();
                auto& f8 = fields[8].cast<ScalarField<3>&>();
                samurai::load(directory, basename, *mesh_3d, f0, f1, f2, f3, f4, f5, f6, f7, f8);
                return;
            }
            default:
                throw std::runtime_error(fmt::format("Maximum {} fields supported for 3D bulk load, got {}", MAX_BULK_FIELDS, n));
        }
    }

    throw std::runtime_error("Unable to determine field dimension or invalid field types");
}

// ============================================================
// Module initialization
// ============================================================

void init_io_bindings(py::module_& m)
{
    // ============================================================
    // save() function bindings - unified filepath API
    //
    // Note: The old fixed-arity overloads (1, 2, 3 fields) have been replaced
    // by a single bulk save function that supports an arbitrary number of fields:
    // - 1D: 1-9 fields
    // - 2D: 1-9 fields
    // - 3D: 1-9 fields
    //
    // The function automatically detects the dimension from the fields.
    // All fields must share the same mesh. Field names are taken from each
    // field's name attribute.
    // ============================================================

    m.def("save",
          &save_bulk,
          py::arg("filepath"),
          R"pbdoc(
            Save mesh and multiple fields to HDF5 + XDMF.

            Supports 1-9 fields for all dimensions (1D, 2D, 3D).
            All fields must share the same mesh. Field names are taken from each
            field's name attribute. The dimension is automatically detected from
            the fields provided.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "results/solution.h5")
            *fields : ScalarField1D, ScalarField2D, or ScalarField3D
                Variable number of scalar fields (1-9)

            Examples
            --------
            >>> samurai.save("results/solution.h5", field1)
            >>> samurai.save("results/solution.h5", field1, field2, field3)
            >>> samurai.save("results/solution.h5", rho, qx, qy, e, p, t, vel, mag)

            Notes
            -----
            This is the recommended way to save multiple fields in simulations
            with many variables (e.g., fluid dynamics with density, velocity,
            energy, pressure, etc.).

            Maximum field count: 9 fields for all dimensions (1D, 2D, 3D).
        )pbdoc");

    // ============================================================
    // dump() function bindings - unified filepath API
    //
    // Note: The old fixed-arity overloads have been replaced
    // by a single bulk dump function that supports an arbitrary number of fields:
    // - 1D: 1-9 fields
    // - 2D: 1-9 fields
    // - 3D: 1-9 fields
    //
    // The function automatically detects the dimension from the fields.
    // All fields must share the same mesh. Field names are taken from each
    // field's name attribute.
    // ============================================================

    m.def("dump",
          &dump_bulk,
          py::arg("filepath"),
          R"pbdoc(
            Dump mesh and multiple fields to HDF5 for checkpoint/restart.

            Supports 1-9 fields for all dimensions (1D, 2D, 3D).
            All fields must share the same mesh. Field names are taken from each
            field's name attribute. The dimension is automatically detected from
            the fields provided.

            Creates HDF5-only file (no XDMF metadata) for efficient
            checkpointing and restarting simulations.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "checkpoints/solution.h5")
            *fields : ScalarField1D, ScalarField2D, or ScalarField3D
                Variable number of scalar fields (1-9)

            Creates
            -------
            {directory}/{basename}.h5 - HDF5 restart file

            Examples
            --------
            >>> import sampai as sam
            >>> samurai.dump("checkpoints/solution.h5", field1)
            >>> samurai.dump("checkpoints/solution.h5", field1, field2, field3)

            Notes
            -----
            This is the recommended way to dump multiple fields for checkpointing.

            Maximum field count: 9 fields for all dimensions (1D, 2D, 3D).
        )pbdoc");

    // ============================================================
    // VectorField save() bindings - unified filepath API
    // ============================================================

    m.def("save",
          &save_2d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Save 2D vector field (2 components) mesh and data to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field (VectorField2D).");

    m.def("save",
          &save_3d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Save 3D vector field (3 components) mesh and data to HDF5 + XDMF. "
          "Parameters: filepath (str/Path), field (VectorField3D).");

    // ============================================================
    // VectorField dump() bindings - unified filepath API
    // ============================================================

    m.def("dump",
          &dump_2d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Dump 2D vector field (2 components) mesh and data to HDF5 for checkpoint/restart. "
          "Parameters: filepath (str/Path), field (VectorField2D).");

    m.def("dump",
          &dump_3d_vector,
          py::arg("filepath"),
          py::arg("field"),
          "Dump 3D vector field (3 components) mesh and data to HDF5 for checkpoint/restart. "
          "Parameters: filepath (str/Path), field (VectorField3D).");

    // ============================================================
    // load() function bindings - unified filepath API
    //
    // Note: The old fixed-arity overloads have been replaced
    // by a single bulk load function that supports an arbitrary number of fields:
    // - 1D: 1-9 fields
    // - 2D: 1-9 fields
    // - 3D: 1-9 fields
    //
    // The function automatically detects the dimension from the fields.
    // All fields must share the same mesh. Field names must match the names
    // used when creating the restart file.
    // ============================================================

    m.def("load",
          &load_bulk,
          py::arg("filepath"),
          R"pbdoc(
            Load mesh and multiple fields from HDF5 restart file.

            Supports 1-9 fields for all dimensions (1D, 2D, 3D).
            All fields must share the same mesh. The dimension is automatically
            detected from the fields provided.

            Parameters
            ----------
            filepath : str or Path
                Unified file path (e.g., "checkpoints/solution.h5")
            *fields : ScalarField1D, ScalarField2D, or ScalarField3D
                Variable number of scalar fields to load data into (1-9)

            Reads
            ------
            {directory}/{basename}.h5 - HDF5 restart file

            Note
            ----
            The mesh and field objects will have their data replaced
            with the contents of the restart file. The field names
            must match the names used when creating the restart file.

            Examples
            --------
            >>> import sampai as sam
            >>> samurai.load("checkpoints/solution.h5", field1)
            >>> samurai.load("checkpoints/solution.h5", field1, field2, field3)

            Notes
            -----
            This is the recommended way to load multiple fields for restarting.

            Maximum field count: 9 fields for all dimensions (1D, 2D, 3D).
        )pbdoc");

    // ============================================================
    // open_h5py() function binding
    // ============================================================

    m.def("open_h5py",
          &open_h5py_wrapper,
          py::arg("filename"),
          py::arg("mode") = "r",
          R"pbdoc(
            Open HDF5 file created by Samurai using h5py.

            This function opens an HDF5 file created by Samurai's save() or dump()
            functions and returns an h5py.File object for direct data access.

            Parameters
            ----------
            filename : str or Path
                File path to open (with or without .h5 extension)
            mode : str, default: 'r'
                File access mode:
                - 'r': Read-only (default)
                - 'r+': Read and write
                - 'w': Write (truncate existing file)

            Returns
            -------
            h5py.File
                h5py File object for direct HDF5 data access

            Examples
            --------
            >>> import sampai as sam
            >>> # Save a field
            >>> field.save("results/solution.h5")
            >>> # Open with h5py
            >>> with samurai.open_h5py("results/solution.h5") as f:
            ...     data = f["mesh/fields/u"][:]
            ...     points = f["mesh/points"][:]
            ...     print(f"Field min: {data.min()}, max: {data.max()}")

            Notes
            -----
            Requires h5py to be installed (pip install h5py).

            The HDF5 file structure is:
            /mesh/points - Cell coordinates (N x 3)
            /mesh/connectivity - Cell connectivity (N_cells x 2^dim)
            /mesh/fields/{field_name} - Field data

            See Also
            --------
            Field.save : Save field to HDF5
            Field.load : Load field from HDF5
        )pbdoc");
}
