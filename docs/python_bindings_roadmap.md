# Python Bindings Roadmap
## Samurai → Python Compatibility Specification

This document specifies **exactly what needs to be added** to the Python wrapper (`src/bindings/`) to achieve full compatibility between Samurai C++ capabilities and our proposed Python Simulation API.

---

## Priority Legend

- **🔴 CRITICAL**: Blocking for common use cases (heat equation, vector calculus)
- **🟠 HIGH**: Major functionality gaps (regional operations, level iteration)
- **🟡 MEDIUM**: Important enhancements (time-dependent BCs, diagnostics)
- **🟢 LOW**: Nice-to-have features (metadata, compression)

---

## Part I: Numerical Operators (🔴 CRITICAL)

### 1.1 Diffusion Operators

**C++ Location**: `subprojects/samurai/include/samurai/schemes/fv/operators/diffusion.hpp`

**C++ Signature**:
```cpp
template <class Field>
auto make_diffusion_order2(double coefficient);

template <class Field, std::size_t dim>
auto make_diffusion_order2(DiffCoeff<dim> tensor);

template <class Field>
auto make_laplacian_order2();
```

**Python Binding Required**:
```python
# In src/bindings/operator_bindings.cpp

// Scalar diffusion coefficient
template <std::size_t dim>
ScalarField<dim> diffusion_order2(const ScalarField<dim>& field, double k)
{
    auto scheme = samurai::make_diffusion_order2<Field>(k);
    return apply_scheme(scheme, field);
}

// Diagonal tensor (for anisotropic diffusion)
template <std::size_t dim>
ScalarField<dim> diffusion_order2_tensor(const ScalarField<dim>& field, xt::xtensor<double, 1> K)
{
    DiffCoeff<dim> coeff;
    for (std::size_t d = 0; d < dim; ++d) {
        coeff[d] = K[d];
    }
    auto scheme = samurai::make_diffusion_order2<Field>(coeff);
    return apply_scheme(scheme, field);
}

// Laplacian (k=1)
template <std::size_t dim>
ScalarField<dim> laplacian_order2(const ScalarField<dim>& field)
{
    auto scheme = samurai::make_laplacian_order2<Field>();
    return apply_scheme(scheme, field);
}
```

**Module Exports** (add to `operator_bindings.cpp`):
```cpp
// 1D
m.def("diffusion", &diffusion_order2<1>, "Compute diffusion (1D)",
      py::arg("field"), py::arg("coefficient"));
m.def("laplacian", &laplacian_order2<1>, "Compute Laplacian (1D)",
      py::arg("field"));

// 2D
m.def("diffusion", &diffusion_order2<2>, "Compute diffusion (2D)",
      py::arg("field"), py::arg("coefficient"));
m.def("diffusion_tensor", &diffusion_order2_tensor<2>, "Compute anisotropic diffusion (2D)",
      py::arg("field"), py::arg("K"));  // K = [kx, ky]
m.def("laplacian", &laplacian_order2<2>, "Compute Laplacian (2D)",
      py::arg("field"));

// 3D
m.def("diffusion", &diffusion_order2<3>, "Compute diffusion (3D)",
      py::arg("field"), py::arg("coefficient"));
m.def("diffusion_tensor", &diffusion_order2_tensor<3>, "Compute anisotropic diffusion (3D)",
      py::arg("field"), py::arg("K"));  // K = [kx, ky, kz]
m.def("laplacian", &laplacian_order2<3>, "Compute Laplacian (3D)",
      py::arg("field"));
```

**Usage Example**:
```python
import sampai as sam

# Heat equation: du/dt = k * laplacian(u)
k = 0.1
flux = sam.laplacian(u)  # or sam.diffusion(u, k)
unp1.assign(u + dt * k * flux)

# Anisotropic diffusion
K = [0.1, 0.5]  # Different coefficients in x, y
flux = sam.diffusion_tensor(u, K)
```

**Why Critical**: Heat equation is one of the most fundamental PDEs. Currently **impossible** to solve in pure Python.

---

### 1.2 Gradient Operator

**C++ Location**: `subprojects/samurai/include/samurai/schemes/fv/operators/gradient.hpp`

**C++ Signature**:
```cpp
template <class Field>
auto make_gradient_order2();
// Input: ScalarField<dim>
// Output: VectorField<dim, dim> (gradient vector)
```

**Python Binding Required**:
```cpp
template <std::size_t dim>
VectorField<dim, dim> gradient_order2(const ScalarField<dim>& field)
{
    auto scheme = samurai::make_gradient_order2<ScalarField<dim>>();
    // Apply and return vector field
    return apply_gradient_scheme(scheme, field);
}
```

**Module Exports**:
```cpp
m.def("gradient", &gradient_order2<1>, "Compute gradient (1D)", py::arg("field"));
m.def("gradient", &gradient_order2<2>, "Compute gradient (2D)", py::arg("field"));
m.def("gradient", &gradient_order2<3>, "Compute gradient (3D)", py::arg("field"));
```

**Usage Example**:
```python
# Compute gradient of scalar field
grad_u = sam.gradient(u)  # Returns VectorField
# grad_u[0] = ∂u/∂x, grad_u[1] = ∂u/∂y, ...
```

---

### 1.3 Divergence Operator

**C++ Location**: `subprojects/samurai/include/samurai/schemes/fv/operators/divergence.hpp`

**C++ Signature**:
```cpp
template <class Field>
auto make_divergence_order2();
// Input: VectorField<dim, dim> (velocity field)
// Output: ScalarField<dim> (divergence)
```

**Python Binding Required**:
```cpp
template <std::size_t dim>
ScalarField<dim> divergence_order2(const VectorField<dim, dim>& field)
{
    auto scheme = samurai::make_divergence_order2<VectorField<dim, dim>>();
    return apply_divergence_scheme(scheme, field);
}
```

**Module Exports**:
```cpp
m.def("divergence", &divergence_order2<1>, "Compute divergence (1D)", py::arg("field"));
m.def("divergence", &divergence_order2<2>, "Compute divergence (2D)", py::arg("field"));
m.def("divergence", &divergence_order2<3>, "Compute divergence (3D)", py::arg("field"));
```

**Usage Example**:
```python
# Incompressible flow: div(v) = 0
div_v = sam.divergence(velocity)  # Returns ScalarField
```

---

### 1.4 Cell-Based Diffusion (Alternative)

**C++ Location**: `subprojects/samurai/include/samurai/schemes/fv/operators/diffusion_cell_based.hpp`

**C++ Signature**:
```cpp
template <class Field>
auto make_diffusion_cell_based();
// Star-stencil Laplacian (more intuitive for some applications)
```

**Python Binding**:
```cpp
template <std::size_t dim>
ScalarField<dim> diffusion_cell_based(const ScalarField<dim>& field)
{
    auto scheme = samurai::make_diffusion_cell_based<ScalarField<dim>>();
    return apply_scheme(scheme, field);
}
```

**Module Exports**:
```cpp
m.def("diffusion_cell_based", &diffusion_cell_based<2>, "Star-stencil diffusion (2D)", py::arg("field"));
```

---

## Part II: Subset Operations (🟠 HIGH)

### 2.1 Set Algebra Operations

**C++ Location**: `subprojects/samurai/include/samurai/subset/node.hpp`

**C++ Signature**:
```cpp
template <typename... Sets>
auto intersection(Sets&&... sets);

template <typename... Sets>
auto union_(Sets&&... sets);

template <typename... Sets>
auto difference(Sets&&... sets);

template <typename Set, typename Stencil>
auto translate(const Set& set, const Stencil& stencil);

template <typename Set, std::size_t width>
auto expand(const Set& set);

template <typename Set, std::size_t width>
auto contract(const Set& set);
```

**Python Binding Required**:
```cpp
// New file: src/bindings/subset_bindings.cpp

// Subset wrapper class
template <std::size_t dim>
class PySubset {
    using subset_t = samurai::Subset<dim, interval_t>;
    subset_t m_subset;

public:
    PySubset(subset_t subset) : m_subset(subset) {}

    // Project to specific level
    PySubset on_level(std::size_t level) {
        return PySubset(m_subset.on(level));
    }

    // Check if empty
    bool empty() const {
        return m_subset.empty();
    }

    // Iterate over intervals
    void for_each_interval(std::function<void(std::size_t, interval_t, xt::xtensor_fixed<std::size_t, xt::xshape<dim-1>>)> func) {
        samurai::for_each_interval(m_subset, func);
    }

    // String representation
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Subset<" << dim << "D>";
        return oss.str();
    }
};

// Set algebra functions
template <std::size_t dim>
PySubset<dim> intersection_mesh(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2) {
    auto subset = samurai::intersection(mesh1[mesh1.min_level()], mesh2[mesh2.min_level()]);
    return PySubset<dim>(subset);
}

template <std::size_t dim>
PySubset<dim> difference_mesh(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2) {
    auto subset = samurai::difference(mesh1[mesh1.min_level()], mesh2[mesh2.min_level()]);
    return PySubset<dim>(subset);
}

template <std::size_t dim>
PySubset<dim> union_mesh(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2) {
    auto subset = samurai::union_(mesh1[mesh1.min_level()], mesh2[mesh2.min_level()]);
    return PySubset<dim>(subset);
}
```

**Module Definition**:
```cpp
// In main.cpp or subset_bindings.cpp
py::module_ m = ...; // Get or create submodule

py::class_<PySubset<1>>(m, "Subset1D")
    .def("on_level", &PySubset<1>::on_level)
    .def("empty", &PySubset<1>::empty)
    .def("for_each_interval", &PySubset<1>::for_each_interval)
    .def("__repr__", &PySubset<1>::to_string);

py::class_<PySubset<2>>(m, "Subset2D")
    .def("on_level", &PySubset<2>::on_level)
    .def("empty", &PySubset<2>::empty)
    .def("for_each_interval", &PySubset<2>::for_each_interval)
    .def("__repr__", &PySubset<2>::to_string);

py::class_<PySubset<3>>(m, "Subset3D")
    // ... same ...

// Module-level functions
m.def("intersection", &intersection_mesh<1>, "Intersection of two meshes (1D)");
m.def("intersection", &intersection_mesh<2>, "Intersection of two meshes (2D)");
m.def("difference", &difference_mesh<1>, "Difference of two meshes (1D)");
m.def("difference", &difference_mesh<2>, "Difference of two meshes (2D)");
m.def("union_", &union_mesh<1>, "Union of two meshes (1D)");
m.def("union_", &union_mesh<2>, "Union of two meshes (2D)");
```

**Alternative Design (simpler)**:
```python
# Instead of full subset class, provide field-based operations:

def apply_to_region(field, region_func, value):
    """Apply value to cells where region_func(cell) returns True."""
    def setter(cell):
        if region_func(cell):
            field[cell.index] = value
    sam.algorithms.for_each_cell(field.mesh, setter)

# Usage:
def left_boundary(cell):
    x, _ = cell.center()
    return x < 0.01

apply_to_region(u, left_boundary, 0.0)  # Dirichlet on left
```

**Why High Priority**: Enables efficient regional operations (BCs on specific boundaries, localized refinement, etc.).

---

### 2.2 Level-Based Iteration

**C++ Location**: `subprojects/samurai/include/samurai/algorithm.hpp:50-66`

**C++ Signature**:
```cpp
template <class Mesh, class Func>
void for_each_level(const Mesh& mesh, Func&& func, bool include_empty_levels = false);
```

**Python Binding**:
```cpp
// Add to algorithm_bindings.cpp

template <std::size_t dim>
void for_each_level_py(const MRMesh<dim>& mesh,
                       std::function<void(std::size_t)> func,
                       bool include_empty_levels = false)
{
    samurai::for_each_level(mesh, func, include_empty_levels);
}
```

**Module Exports**:
```cpp
m.def("for_each_level", &for_each_level_py<1>, "Iterate over refinement levels (1D)",
      py::arg("mesh"), py::arg("func"), py::arg("include_empty_levels")=false);
m.def("for_each_level", &for_each_level_py<2>, "Iterate over refinement levels (2D)",
      py::arg("mesh"), py::arg("func"), py::arg("include_empty_levels")=false);
m.def("for_each_level", &for_each_level_py<3>, "Iterate over refinement levels (3D)",
      py::arg("mesh"), py::arg("func"), py::arg("include_empty_levels")=false);
```

**Usage Example**:
```python
# Process each level separately
def process_level(level):
    cells_at_level = mesh.nb_cells_at_level(level)
    print(f"Level {level}: {cells_at_level} cells")

sam.algorithms.for_each_level(mesh, process_level)

# Filter cells by level
coarse_cells = []
def collect_coarse(cell):
    if cell.level <= 3:
        coarse_cells.append(cell.index)

sam.algorithms.for_each_cell(mesh, collect_coarse)
```

---

### 2.3 Cell Finding

**C++ Location**: `subprojects/samurai/include/samurai/algorithm.hpp:484-531`

**C++ Signature**:
```cpp
template <class Mesh>
auto find_cell(const Mesh& mesh, const xt::xtensor_fixed<double, xt::xshape<dim>>& coords);
// Returns Cell with length=0 if not found
```

**Python Binding**:
```cpp
template <std::size_t dim>
CellWrapper<dim> find_cell_py(const MRMesh<dim>& mesh,
                               const xt::xtensor<double, 1>& coords)
{
    xt::xtensor_fixed<double, xt::xshape<dim>> coords_fixed;
    std::copy(coords.begin(), coords.end(), coords_fixed.begin());

    auto cell = samurai::find_cell(mesh, coords_fixed);
    return CellWrapper<dim>(cell);
}
```

**Module Exports**:
```cpp
m.def("find_cell", &find_cell_py<1>, "Find cell from coordinates (1D)",
      py::arg("mesh"), py::arg("coords"));
m.def("find_cell", &find_cell_py<2>, "Find cell from coordinates (2D)",
      py::arg("mesh"), py::arg("coords"));
m.def("find_cell", &find_cell_py<3>, "Find cell from coordinates (3D)",
      py::arg("mesh"), py::arg("coords"));
```

**Usage Example**:
```python
# Find cell at specific coordinates
cell = sam.algorithms.find_cell(mesh, [0.5, 0.3])
if cell.length > 0:  # Valid cell
    u[cell.index] = 1.0
else:
    print("Coordinates outside mesh")
```

---

## Part III: Boundary Conditions (🟡 MEDIUM)

### 3.1 FunctionBc for Time-Dependent BCs

**C++ Location**: `subprojects/samurai/include/samurai/bc/bc.hpp`

**C++ Signature**:
```cpp
template <class Field>
class FunctionBc : public BcValue<Field> {
public:
    using function_t = std::function<value_t(const direction_t&, const cell_t&, const coords_t&)>;

    explicit FunctionBc(const function_t& f);

    value_t get_value(const direction_t& d, const cell_t& cell_in, const coords_t& coords) const override;
};
```

**Python Binding Required**:
```cpp
// Add to bc_bindings.cpp

template <std::size_t dim, std::size_t order>
auto make_function_bc_dirichlet(
    ScalarField<dim>& field,
    std::function<double(double t, const xt::xtensor<double, 1>& coords)> func,
    std::function<bool(double t)> time_depended_fn = nullptr)
{
    // Wrap Python function with time capture
    using DirectionVector = xt::xtensor_fixed<int, xt::xshape<dim>>;
    using coords_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

    auto bc_func = [func](const DirectionVector& dir, const Cell<dim, interval_t>& cell,
                          const coords_t& coords) -> double
    {
        // Extract time from capture (would need to be stored somewhere)
        // For now, use coords only
        xt::xtensor<double, 1> coords_arr = coords;
        return func(0.0, coords_arr);  // t=0 for now
    };

    return samurai::make_bc<samurai::Dirichlet<order>>(field, bc_func);
}
```

**Simplified Approach** (using Python callbacks):
```python
# In Python, provide a helper function:

def make_time_dependent_bc(field, func):
    """
    Create a time-dependent boundary condition.

    Args:
        field: ScalarField to apply BC to
        func: callable(t, coords) -> value

    Example:
        def inlet_bc(t, coords):
            x, y = coords
            return np.sin(2*np.pi*5*t) * np.exp(-10*((x-0.5)**2 + y**2))

        sam.boundary.dirichlet(u, inlet_bc, time_dependent=True)
    """
    # Store function and call with current time during BC application
    # This requires modifying the bc application logic
    ...
```

**Alternative Design** (hook-based):
```python
# Use Simulation API hooks instead of native FunctionBc

@sim.before_step
def update_time_dependent_bcs(u, t, iteration):
    # Update boundary values manually
    def update_boundary(cell):
        x, y = cell.center()
        if x < 0.01:  # Left boundary
            u[cell.index] = np.sin(2*np.pi*5*t)

    sam.algorithms.for_each_cell(u.mesh, update_boundary)
```

---

## Part IV: Diagnostics and Statistics (🟡 MEDIUM)

### 4.1 L2 Error Computation

**C++ Location**: `subprojects/samurai/include/samurai/numeric/error.hpp`

**C++ Signature**:
```cpp
template <class Field, class ExactSolution>
double L2_error(const Field& approximate, const ExactSolution& exact);

template <class Field, class ExactSolution>
double L2_error_relative(const Field& approximate, const ExactSolution& exact);
```

**Python Binding**:
```cpp
// New file: src/bindings/numeric_bindings.cpp

template <std::size_t dim>
double l2_error_func(
    const ScalarField<dim>& approximate,
    std::function<double(xt::xtensor_fixed<double, xt::xshape<dim>>)> exact_func)
{
    auto error = samurai::L2_error(approximate, exact_func);
    return error;
}

template <std::size_t dim>
double l2_error_field(
    const ScalarField<dim>& approximate,
    const ScalarField<dim>& exact)
{
    auto error = samurai::L2_error(approximate,
        [&](const auto& coords) { return exact(coords); });
    return error;
}
```

**Module Definition**:
```cpp
py::module_ m = ...; // Create sam.numeric submodule

m.def("l2_error", &l2_error_func<1>, "Compute L2 error norm (1D)",
      py::arg("approximate"), py::arg("exact_function"));
m.def("l2_error", &l2_error_func<2>, "Compute L2 error norm (2D)",
      py::arg("approximate"), py::arg("exact_function"));
m.def("l2_error", &l2_error_field<1>, "Compute L2 error against reference field (1D)",
      py::arg("approximate"), py::arg("exact"));
m.def("l2_error", &l2_error_field<2>, "Compute L2 error against reference field (2D)",
      py::arg("approximate"), py::arg("exact"));
```

**Usage Example**:
```python
import sampai as sam
import numpy as np

# Exact solution function
def exact_solution(coords):
    x, y = coords
    return np.sin(np.pi*x) * np.cos(np.pi*y)

# Compute error
error = sam.numeric.l2_error(u, exact_solution)

# Or against reference field
u_ref = sam.load("reference.h5")
error = sam.numeric.l2_error(u, u_ref)
```

---

### 4.2 Timer Integration

**C++ Location**: `subprojects/samurai/include/samurai/timers.hpp`

**C++ API**:
```cpp
samurai::times::timers.start("name");
samurai::times::timers.stop("name");
double elapsed = samurai::times::timers.getElapsedTime("name");
samurai::times::timers.print();  // Called automatically with --timers flag
```

**Python Binding**:
```cpp
// Add to a new timers_bindings.cpp or existing bindings

class PyTimers {
public:
    static void start(const std::string& name) {
        samurai::times::timers.start(name);
    }

    static void stop(const std::string& name) {
        samurai::times::timers.stop(name);
    }

    static double get_elapsed(const std::string& name) {
        return samurai::times::timers.getElapsedTime(name);
    }

    static void print_timers() {
        samurai::times::timers.print();
    }

    static std::map<std::string, double> get_all_timers() {
        std::map<std::string, double> result;
        // Access internal timer data (would need to expose from Timers class)
        return result;
    }
};
```

**Module Exports**:
```cpp
py::module_ m = ...; // Create sam.utils.timer submodule

py::class_<PyTimers>(m, "Timers")
    .def_static("start", &PyTimers::start, "Start named timer")
    .def_static("stop", &PyTimers::stop, "Stop named timer")
    .def_static("get_elapsed", &PyTimers::get_elapsed, "Get elapsed time for timer")
    .def_static("print", &PyTimers::print_timers, "Print all timers")
    .def_static("get_all", &PyTimers::get_all_timers, "Get all timer data as dict");
```

**Usage Example**:
```python
from sampai.utils import timer

timer.Timers.start("adaptation")
MRadaptation(config)
timer.Timers.stop("adaptation")

print(f"Adaptation took {timer.Timers.get_elapsed('adaptation')}s")

# At program end or with --timers flag
timer.Timers.print()
```

---

### 4.3 Memory Usage Tracking

**C++ Location**: `subprojects/samurai/include/samurai/memory.hpp`

**C++ Signature**:
```cpp
std::size_t memory_usage(const Mesh& mesh, bool verbose = false);
```

**Python Binding**:
```cpp
template <std::size_t dim>
std::size_t memory_usage_mesh(const MRMesh<dim>& mesh, bool verbose = false)
{
    return samurai::memory_usage(mesh, verbose);
}
```

**Module Exports**:
```cpp
m.def("memory_usage", &memory_usage_mesh<1>, "Get mesh memory usage (1D)",
      py::arg("mesh"), py::arg("verbose")=false);
m.def("memory_usage", &memory_usage_mesh<2>, "Get mesh memory usage (2D)",
      py::arg("mesh"), py::arg("verbose")=false);
```

---

## Part V: I/O Enhancements (🟢 LOW)

### 5.1 HDF5 Version Metadata

**Required Changes** (modify `io_bindings.cpp`):

```cpp
// Add to save() functions
void save_with_metadata(const std::string& path, const Field& field, ...)
{
    // Existing save code...

    // Add metadata
    hid_t file_id = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t root = H5Gopen(file_id, "/", H5P_DEFAULT);

    // Write version
    H5LTset_attribute_string(root, ".", "samurai_version", SAMURAI_VERSION);
    H5LTset_attribute_string(root, ".", "format_version", "1.0");
    H5LTset_attribute_string(root, ".", "creation_date", get_timestamp().c_str());

    // Write mesh info
    H5LTset_attribute_int(root, "/mesh", "dim", dim);
    // ... other attributes ...

    H5Gclose(root);
    H5Fclose(file_id);
}
```

**Python Helper**:
```python
def save_with_metadata(field, path, **kwargs):
    """Save field with version metadata."""
    # Current save
    sam.save(path, field, **kwargs)

    # Add metadata using h5py
    import h5py
    from sampai import __version__

    with h5py.File(path, 'a') as f:
        f.attrs['sampai_version'] = __version__
        f.attrs['format_version'] = '1.0'
        f.attrs['creation_date'] = datetime.datetime.now().isoformat()
```

---

### 5.2 HDF5 Compression

**Required Changes**:
```cpp
// Add compression option to save()
void save_compressed(const std::string& path, const Field& field,
                    int compression_level = 6)  // gzip 0-9
{
    hid_t file_id = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // Create property list for compression
    hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_deflate(plist, compression_level);  // gzip compression

    // Write datasets with compression
    // ... existing write code with plist ...

    H5Pclose(plist);
    H5Fclose(file_id);
}
```

**Python Binding**:
```cpp
m.def("save_compressed", &save_compressed<1>, "Save with gzip compression (1D)",
      py::arg("path"), py::arg("field"), py::arg("compression_level")=6);
```

---

### 5.3 Appendable Time Series

**Design**:
```python
def save_timeseries(filepath, field, t, mode='a'):
    """
    Append field to HDF5 file as new timestep.

    File structure:
    /timesteps
        /000
            /mesh/...
            /fields/...
        /001
            ...
    """
    import h5py

    if mode == 'w':
        # Create new file or truncate
        with h5py.File(filepath, 'w') as f:
            f.attrs['n_timesteps'] = 1
            # Create first timestep group
            save_to_group(f, 'timesteps/000', field)
            f['timesteps/000'].attrs['time'] = t
    else:
        # Append
        with h5py.File(filepath, 'a') as f:
            n = f.attrs['n_timesteps']
            group_name = f'timesteps/{n:03d}'
            save_to_group(f, group_name, field)
            f[group_name].attrs['time'] = t
            f.attrs['n_timesteps'] = n + 1
```

---

## Part VI: Integration with Simulation API

### 6.1 Required Bindings for Simulation API

The Simulation API requires these bindings to function properly:

| Feature | Required Binding | Priority |
|---------|------------------|----------|
| `upwind` operator | ✅ Already exposed | - |
| `convection_weno5` | ✅ Already exposed | - |
| `diffusion` | ❌ Must add (Part I.1) | 🔴 CRITICAL |
| `gradient` | ❌ Must add (Part I.2) | 🟡 MEDIUM |
| `divergence` | ❌ Must add (Part I.3) | 🟡 MEDIUM |
| `laplacian` | ❌ Must add (Part I.1) | 🔴 CRITICAL |
| `for_each_cell` | ✅ Already exposed | - |
| `for_each_level` | ❌ Should add (Part II.2) | 🟠 HIGH |
| `find_cell` | ❌ Should add (Part II.3) | 🟠 HIGH |
| `intersection/difference` | ❌ Should add (Part II.1) | 🟠 HIGH |
| `L2_error` | ❌ Should add (Part IV.1) | 🟡 MEDIUM |
| `timers` | ❌ Should add (Part IV.2) | 🟡 MEDIUM |
| `memory_usage` | ❌ Should add (Part IV.3) | 🟢 LOW |

---

## Part VII: Implementation Plan

### Phase 1: CRITICAL Operators (Weeks 1-2)

**Files to Create/Modify**:
1. `src/bindings/operator_bindings.cpp` - Add diffusion, gradient, divergence
2. `src/bindings/operator_bindings.hpp` - Add function declarations
3. `tests/test_operators_diffusion.py` - Test new operators

**Tests Required**:
```python
def test_diffusion_scalar():
    """Test scalar diffusion operator."""
    # Solve du/dt = k*laplacian(u) with known solution
    # Compare against analytical solution

def test_diffusion_tensor():
    """Test anisotropic diffusion."""

def test_gradient():
    """Test gradient of scalar field."""

def test_divergence():
    """Test divergence of vector field."""

def test_laplacian():
    """Test Laplacian operator."""
```

### Phase 2: Subset Operations (Weeks 3-4)

**Files to Create/Modify**:
1. `src/bindings/subset_bindings.cpp` - New file for subset operations
2. `src/bindings/subset_bindings.hpp` - Header file
3. `tests/test_subset_operations.py` - Test subset operations

**Tests Required**:
```python
def test_intersection():
    """Test mesh intersection."""

def test_difference():
    """Test mesh difference."""

def test_union():
    """Test mesh union."""

def test_for_each_level():
    """Test level-based iteration."""

def test_find_cell():
    """Test cell finding from coordinates."""
```

### Phase 3: Diagnostics (Week 5)

**Files to Create/Modify**:
1. `src/bindings/numeric_bindings.cpp` - New file for L2 error
2. `src/bindings/timer_bindings.cpp` - New file for timers
3. `tests/test_diagnostics.py` - Test diagnostics

### Phase 4: I/O Enhancements (Week 6)

**Files to Create/Modify**:
1. `src/bindings/io_bindings.cpp` - Add compression, metadata
2. `src/sampai/utils/io/enhanced.py` - Python helpers for I/O

---

## Part VIII: Quick Reference

### Summary of Required Bindings

```cpp
// ========== DIFFUSION OPERATORS (CRITICAL) ==========

// Scalar diffusion
template <std::size_t dim>
ScalarField<dim> diffusion(const ScalarField<dim>& field, double k);

// Anisotropic diffusion
template <std::size_t dim>
ScalarField<dim> diffusion_tensor(const ScalarField<dim>& field, xt::xtensor<double, 1> K);

// Laplacian
template <std::size_t dim>
ScalarField<dim> laplacian(const ScalarField<dim>& field);

// ========== VECTOR CALCULUS ==========

// Gradient
template <std::size_t dim>
VectorField<dim, dim> gradient(const ScalarField<dim>& field);

// Divergence
template <std::size_t dim>
ScalarField<dim> divergence(const VectorField<dim, dim>& field);

// ========== SUBSET OPERATIONS ==========

template <std::size_t dim>
PySubset<dim> intersection(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2);

template <std::size_t dim>
PySubset<dim> difference(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2);

template <std::size_t dim>
PySubset<dim> union_(const MRMesh<dim>& mesh1, const MRMesh<dim>& mesh2);

// ========== ITERATION ==========

template <std::size_t dim>
void for_each_level(const MRMesh<dim>& mesh,
                   std::function<void(std::size_t)> func,
                   bool include_empty_levels = false);

template <std::size_t dim>
CellWrapper<dim> find_cell(const MRMesh<dim>& mesh,
                           const xt::xtensor<double, 1>& coords);

// ========== DIAGNOSTICS ==========

template <std::size_t dim>
double l2_error(const ScalarField<dim>& approximate,
               std::function<double(xt::xtensor_fixed<double, xt::xshape<dim>>)> exact);

template <std::size_t dim>
std::size_t memory_usage(const MRMesh<dim>& mesh, bool verbose = false);

// Timer methods (class-based)
class PyTimers {
    static void start(const std::string& name);
    static void stop(const std::string& name);
    static double get_elapsed(const std::string& name);
    static void print();
};
```

---

## Part IX: Validation Strategy

### C++ Reference Tests

For each new binding, create a C++ test that validates against the Python version:

```cpp
// tests/cpp/test_diffusion_bindings.cpp
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv/operators/diffusion.hpp>

int main() {
    // Setup mesh
    using Config = samurai::mara::config<1>;
    samurai::Box<double, 1> box{-1, 1};
    auto mesh = samurai::mra::make_mesh(box, Config{});

    // Create field
    auto u = samurai::make_scalar_field<double>("u", mesh);

    // Apply diffusion
    auto result = samurai::make_diffusion_order2(0.1);
    // ... compute and save reference ...
}
```

```python
# tests/python/test_diffusion_bindings.py
import sampai as sam

def test_diffusion_matches_cpp():
    """Python diffusion produces same result as C++."""
    # Setup identical mesh
    mesh = sam.mesh.make(box, config)
    u = sam.field.scalar(mesh, "u")

    # Apply Python diffusion
    result = sam.diffusion(u, 0.1)

    # Compare against C++ reference
    ref = sam.load("cpp_reference.h5")
    error = sam.numeric.l2_error(result, ref)
    assert error < 1e-14
```

---

## Part X: Documentation Requirements

### For Each New Binding:

1. **Docstring** in Python binding
2. **Example** in `examples/` directory
3. **Test** in `tests/` directory
4. **Update** `CLAUDE.md` if new pattern introduced

---

## Appendix: File Locations Reference

### C++ Headers to Bind:

| Operator | Header File |
|----------|-------------|
| Diffusion | `subprojects/samurai/include/samurai/schemes/fv/operators/diffusion.hpp` |
| Gradient | `subprojects/samurai/include/samurai/schemes/fv/operators/gradient.hpp` |
| Divergence | `subprojects/samurai/include/samurai/schemes/fv/operators/divergence.hpp` |
| Subset ops | `subprojects/samurai/include/samurai/subset/node.hpp` |
| Algorithms | `subprojects/samurai/include/samurai/algorithm.hpp` |
| L2 Error | `subprojects/samurai/include/samurai/numeric/error.hpp` |
| Timers | `subprojects/samurai/include/samurai/timers.hpp` |
| Memory | `subprojects/samurai/include/samurai/memory.hpp` |

### Python Binding Files:

| File | Purpose |
|------|---------|
| `src/bindings/operator_bindings.cpp` | Operator bindings |
| `src/bindings/operator_bindings.hpp` | Operator declarations |
| `src/bindings/subset_bindings.cpp` | **NEW** Subset operations |
| `src/bindings/numeric_bindings.cpp` | **NEW** L2 error, etc. |
| `src/bindings/timer_bindings.cpp` | **NEW** Timers |
| `src/bindings/algorithm_bindings.cpp` | Algorithm iteration |

---

**Version**: 1.0
**Last Updated**: 2025-01-15
**Status**: Ready for implementation
