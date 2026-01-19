# Samurai Python Bindings - Meson Build Configuration
#
# This meson.build builds the Python bindings using samurai from source.
# Samurai is automatically downloaded from GitHub via Meson subprojects.
# Conda-forge is used for dependencies (xtensor, fmt, CLI, hdf5).
#
# Build:
#   pip install .               # Build from source (samurai auto-downloaded)
#   pip install -e .            # Editable install (development)
#   meson setup builddir        # Configure build
#   meson compile -C builddir   # Build
#
# Requirements:
#   - samurai is auto-downloaded from GitHub (main branch)
#   - conda-forge for dependencies (xtensor, fmt, CLI, hdf5)
#   - Python >= 3.9
#   - pybind11 >= 2.13.0
#   - C++20 compiler

project('sampai', 'cpp', 'c',
    version: '0.30.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'optimization=3',
        'b_ndebug=if-release',
        'b_lto=true',
    ],
    meson_version: '>=0.64.0'
)

# ============================================================================
# OPTIONS
# ============================================================================

build_python_tests = get_option('build_python_tests')
enable_mpi = get_option('mpi')

message('=== Sampai - Python Interface for Samurai AMR Library ===')
message('')

# ============================================================================
# FIND DEPENDENCIES
# ============================================================================

# Find Python 3.9+
py_mod = import('python')
py = py_mod.find_installation(pure: false, required: true)
py_version = py.language_version()

# Check Python version
if py_version.version_compare('< 3.9')
    error('Python >= 3.9 is required, found @0@'.format(py_version))
endif

message('Python executable: @0@'.format(py.get_variable('PYTHON')))
message('Python include dirs: @0@'.format(py.get_variable('INCLUDEPY')))

# Find pybind11
pybind11_dep = dependency('pybind11', version: '>=2.13.0', required: false)

if not pybind11_dep.found()
    # Try to find pybind11 in conda environment
    conda_prefix = run_command('printenv', 'CONDA_PREFIX', check: false).stdout().strip()
    if conda_prefix != ''
        pybind11_include = include_directories(conda_prefix / 'include')
        message('Using conda pybind11 from: @0@'.format(conda_prefix / 'include'))
    else
        error('pybind11 not found. Install with: conda install -c conda-forge pybind11')
    endif
else
    message('pybind11 found: @0@'.format(pybind11_dep.version()))
    pybind11_include = include_directories(pybind11_dep.get_variable('INCLUDEDIR', default_value: ''))
endif

# ============================================================================
# FIND SAMURAI C++ LIBRARY (via subproject)
# ============================================================================
# Samurai is a header-only library downloaded via the samurai-headers subproject.
# The subproject handles fetching Samurai from GitHub and exposing the headers.
#
# You can control which Samurai version is used with Meson options:
#
#   meson setup builddir                                    # Uses main (default)
#   meson setup builddir -Dsamurai_commit=v0.27.1            # Uses release tag
#   meson setup builddir -Dsamurai_commit=b39857a278...      # Uses specific commit
#   meson setup builddir -Dsamurai_local_path=/path/to/samurai  # Uses local copy
# ============================================================================

# Forward Samurai options to the subproject
samurai_commit = get_option('samurai_commit')
samurai_depth = get_option('samurai_depth')
samurai_local_path = get_option('samurai_local_path')

# Use the samurai-headers subproject to fetch Samurai
samurai_subproj = subproject('samurai-headers',
                             required: true)

# Get the samurai dependency from the subproject
samurai_base_dep = samurai_subproj.get_variable('samurai_dep')

# Store conda prefix for dependency includes (but NOT for samurai headers)
# We'll add these separately to avoid conda's samurai headers taking precedence
conda_prefix = run_command('printenv', 'CONDA_PREFIX', check: false).stdout().strip()
conda_dep_include_dirs = []

if conda_prefix != ''
    message('  Conda prefix: ' + conda_prefix + ' (for dependencies)')
    # Extract only the dependency subdirectories, NOT the entire include dir
    # This avoids picking up conda's samurai headers
    conda_include_root = conda_prefix / 'include'

    # Check for specific dependency subdirectories
    dep_subdirs = ['xtensor', 'xtl', 'nlohmann', 'fmt', 'CLI', 'highfive', 'hdf5', 'pugixml']

    foreach subdir : dep_subdirs
        subdir_path = conda_include_root / subdir
        if run_command('test', '-d', subdir_path, check: false).returncode() == 0
            conda_dep_include_dirs += include_directories(subdir_path)
        endif
    endforeach

    # Also add the root include for dependencies that don't have subdirs
    # but mark it as NON-system to avoid overriding our samurai headers
    # The key is that samurai headers from subproject will be searched FIRST
    conda_dep_include_dirs += include_directories(conda_include_root)
endif

# Use samurai from subproject WITHOUT conda include
# This ensures our downloaded samurai headers are found first
samurai_dep = samurai_base_dep

message('')

# Additional dependencies from samurai (via conda)
# These are pulled in by samurai but we list them for completeness
xtensor_dep = dependency('xtensor', required: false)
fmt_dep = dependency('fmt', required: false)
cli11_dep = dependency('cli11', required: false)
highfive_dep = dependency('highfive', version: '>=2.10', required: false)
hdf5_dep = dependency('hdf5', required: false)
pugixml_dep = dependency('pugixml', required: false)

# ============================================================================
# MPI DETECTION (optional)
# ============================================================================
# MPI detection via pkg-config (works with system MPI and conda-forge)
# Note: Set PKG_CONFIG_PATH if MPI is in a non-standard location
mpi_dep = dependency('mpi', required: false)

if not mpi_dep.found() and enable_mpi
    # Try OpenMPI/MPICH pkg-config names explicitly (conda-forge)
    ompi_cxx_dep = dependency('ompi-cxx', method: 'pkg-config', required: false)
    ompi_c_dep = dependency('ompi-c', method: 'pkg-config', required: false)

    if ompi_cxx_dep.found()
        mpi_dep = ompi_cxx_dep
        message('Found MPI via pkg-config: ompi-cxx')
    elif ompi_c_dep.found()
        mpi_dep = ompi_c_dep
        message('Found MPI via pkg-config: ompi-c')
    endif

    if not mpi_dep.found()
        mpich_cxx_dep = dependency('mpich-cxx', method: 'pkg-config', required: false)
        mpich_c_dep = dependency('mpich', method: 'pkg-config', required: false)

        if mpich_cxx_dep.found()
            mpi_dep = mpich_cxx_dep
            message('Found MPI via pkg-config: mpich-cxx')
        elif mpich_c_dep.found()
            mpi_dep = mpich_c_dep
            message('Found MPI via pkg-config: mpich')
        endif
    endif
endif

# If still not found, provide helpful error message
if enable_mpi and not mpi_dep.found()
    error('MPI requested but not found via pkg-config. Install one of:\n' +
          '  - conda install -c conda-forge mpich\n' +
          '  - conda install -c conda-forge openmpi\n' +
          '  - Set PKG_CONFIG_PATH to your MPI installation')
endif

# Find Boost.MPI via pkg-config
boost_mpi_dep = dependency('boost_mpi', required: false)

# Fallback to find_library for Boost (some systems don't provide pkg-config)
if not boost_mpi_dep.found() and enable_mpi
    cpp = meson.get_compiler('cpp')
    conda_prefix = run_command('printenv', 'CONDA_PREFIX', check: false).stdout().strip()

    boost_search_dirs = []
    if conda_prefix != ''
        boost_search_dirs += [
            join_paths(conda_prefix, 'lib'),
            join_paths(conda_prefix, 'lib64'),
        ]
    endif

    boost_mpi_lib = cpp.find_library('boost_mpi', dirs: boost_search_dirs, required: false)
    boost_serialization_lib = cpp.find_library('boost_serialization', dirs: boost_search_dirs, required: false)

    if boost_mpi_lib.found() and boost_serialization_lib.found()
        boost_mpi_dep = declare_dependency(
            dependencies: [boost_mpi_lib, boost_serialization_lib]
        )
        message('Found Boost.MPI via find_library')
    endif
endif

has_boost_mpi = boost_mpi_dep.found()

if enable_mpi and not has_boost_mpi
    error('MPI support requires Boost.MPI; install libboost-mpi or conda-forge boost')
endif

has_mpi = enable_mpi and mpi_dep.found()

# Collect cpp_args for conditional compilation
extra_cpp_args = []

if has_mpi
    message('')
    message('=== MPI Support Enabled ===')
    message('MPI library: @0@'.format(mpi_dep.name()))
    if has_boost_mpi
        message('Boost.MPI: found')
    else
        warning('Boost.MPI not found - MPI support may be limited')
    endif

    # Define SAMURAI_WITH_MPI for conditional compilation
    extra_cpp_args += '-DSAMURAI_WITH_MPI'
endif

message('')
message('=== Dependencies Summary ===')
message('Python: @0@'.format(py_version))
message('pybind11: found')
message('samurai: from GitHub (auto-downloaded via subproject)')
if xtensor_dep.found()
    message('xtensor: @0@'.format(xtensor_dep.version()))
endif
if hdf5_dep.found()
    message('hdf5: @0@'.format(hdf5_dep.version()))
endif
if has_mpi
    message('MPI: ENABLED (@0@)'.format(mpi_dep.name()))
else
    message('MPI: DISABLED (use -Dmpi=true to enable)')
endif
message('')

# ============================================================================
# COLLECT SOURCE FILES
# ============================================================================

# Core binding sources (always included)
binding_sources = files(
    'src/bindings/main.cpp',
    'src/bindings/box_bindings.cpp',
    'src/bindings/domain_builder_bindings.cpp',
    'src/bindings/mesh_config_bindings.cpp',
    'src/bindings/mesh_bindings.cpp',
    'src/bindings/field_bindings.cpp',
    'src/bindings/interval_bindings.cpp',
    'src/bindings/algorithm_bindings.cpp',
    'src/bindings/operator_bindings.cpp',
    'src/bindings/bc_bindings.cpp',
    'src/bindings/mra_config_bindings.cpp',
    'src/bindings/adapt_bindings.cpp',
    'src/bindings/subset_bindings.cpp',
    'src/bindings/io_bindings.cpp',
    'src/bindings/mpi_bindings.cpp',
)

if has_mpi
    message('MPI bindings: ENABLED')
else
    message('MPI bindings: STUB (MPI disabled)')
endif

message('Binding source files: @0@'.format(binding_sources.length()))

# ============================================================================
# BUILD PYTHON EXTENSION MODULE
# ============================================================================

# Get Python extension suffix and module directory
py_abi_flags = py.get_variable('ABIFLAGS', '')
py_soabi = py.get_variable('SOABI', '')
py_ext_suffix = py.get_variable('EXT_SUFFIX', '')
py_debug = py.get_variable('Py_DEBUG', false)

# Build the pybind11 extension module
# Note: We use _sampai (private) so the Python package can import it
# The extension module will be installed as _sampai.so alongside sampai/ package

# Collect all dependencies
all_dependencies = [
    samurai_dep,
    fmt_dep,
    cli11_dep,
    hdf5_dep,
    highfive_dep,
    pugixml_dep,
]

# Add MPI dependencies if enabled
if has_mpi
    all_dependencies += [mpi_dep]
    all_dependencies += [boost_mpi_dep]
endif

# Collect all cpp_args
all_cpp_args = [
    '-DNDEBUG',
]

# Add MPI-specific cpp_args
if has_mpi
    all_cpp_args += extra_cpp_args
endif

pybind11_module = py.extension_module(
    '_sampai',
    binding_sources,
    dependencies: all_dependencies,
    include_directories: [
        pybind11_include,
        include_directories('src'),
    ] + conda_dep_include_dirs,  # Add conda dependency includes (but NOT samurai headers from conda)
    install: true,
    install_tag: 'runtime',
    # Optimization flags for better performance
    cpp_args: all_cpp_args,
    # Link with additional dependencies
    link_with: [],
)

message('Extension module: _sampai@0@'.format(py_ext_suffix))
message('')

# ============================================================================
# PYTHON UTILITIES
# ============================================================================

# Install pure Python utilities using meson-python's py.install_sources()
# This is the correct way to install Python files with meson-python

# Install __init__.py at package root
py.install_sources(
    'src/sampai/__init__.py',
    subdir: 'sampai',
)

# Install utils package
py.install_sources(
    'src/sampai/utils/__init__.py',
    subdir: 'sampai/utils',
)

# Install io subpackage
py.install_sources(
    'src/sampai/utils/io/__init__.py',
    'src/sampai/utils/io/read_hdf5_1d.py',
    'src/sampai/utils/io/compare.py',
    'src/sampai/utils/io/mesh_stats.py',
    subdir: 'sampai/utils/io',
)

# Install viz subpackage
py.install_sources(
    'src/sampai/utils/viz/__init__.py',
    'src/sampai/utils/viz/samplotlib.py',
    subdir: 'sampai/utils/viz',
)

# Install progress subpackage
py.install_sources(
    'src/sampai/utils/progress/__init__.py',
    'src/sampai/utils/progress/core.py',
    'src/sampai/utils/progress/stats.py',
    subdir: 'sampai/utils/progress',
)

# Install MPI subpackage (if enabled)
if has_mpi
    py.install_sources(
        'src/sampai/mpi/__init__.py',
        'src/sampai/mpi/comm.py',
        subdir: 'sampai/mpi',
    )
endif

# ============================================================================
# COPY PYTHON UTILITIES TO BUILD DIRECTORY (for development)
# ============================================================================

# Use a post-build script to copy utilities
configure_file(
    input: 'src/sampai/__init__.py',
    output: '__init__.py',
    copy: true,
    install: false,
)

# ============================================================================
# SUMMARY
# ============================================================================

message('')
message('=== Sampai Configuration Summary ===')
message('  Build type:          @0@'.format(get_option('buildtype')))
message('  C++ standard:        C++20')
message('  Python version:      @0@'.format(py_version))
message('  Module name:         sampai (from _sampai extension)')
message('  Samurai source:      auto-downloaded from GitHub (header-only)')
message('  Build tests:         @0@'.format(build_python_tests))
message('=========================================')
message('')
