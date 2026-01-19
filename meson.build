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
enable_petsc = get_option('petsc')

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
# FIND SAMURAI C++ LIBRARY (auto-downloaded from GitHub)
# ============================================================================
# IMPORTANT: We use samurai from source (auto-downloaded) because conda version is outdated.
# Samurai is a header-only library, so no compiled library is needed.
# The conda package is only used for dependencies (xtensor, fmt, CLI, hdf5, etc.).
# Samurai is automatically cloned from GitHub into subprojects/samurai

# Define samurai download location (relative path for Meson)
samurai_subproj_dir = 'subprojects/samurai'
samurai_headers_check = run_command('test', '-d', samurai_subproj_dir / 'include/samurai', check: false)

# Check if samurai is already downloaded, if not clone it
if samurai_headers_check.returncode() != 0
    message('Samurai not found locally, downloading from GitHub...')
    # Clone samurai repository (shallow clone for speed)
    samurai_clone = run_command('git', 'clone', '--depth=1', '--single-branch',
                               'https://github.com/hpc-maths/samurai.git', samurai_subproj_dir,
                               check: true)
    message('Samurai downloaded successfully!')
endif

# Samurai is header-only - just need headers (use relative path)
samurai_include = include_directories(samurai_subproj_dir / 'include')

message('=== Using Samurai (header-only) from GitHub ===')
message('  Source: https://github.com/hpc-maths/samurai.git (main branch)')
message('  Location: subprojects/samurai')

# Add conda include for dependencies (xtensor, fmt, CLI, hdf5, etc.)
# These are needed by samurai headers but we don't want conda samurai headers
conda_prefix = run_command('printenv', 'CONDA_PREFIX', check: false).stdout().strip()
if conda_prefix != ''
    # Add conda include for dependencies
    conda_include = include_directories(conda_prefix / 'include')
    message('  Conda prefix: ' + conda_prefix + ' (for dependencies)')
    # Combine samurai headers with conda dependency headers
    samurai_dep = declare_dependency(
        include_directories: [samurai_include, conda_include],
    )
else
    samurai_dep = declare_dependency(
        include_directories: samurai_include,
    )
endif

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
# Get C++ compiler for finding libraries
cpp = meson.get_compiler('cpp')

# Try to find MPI via dependency() first (works with system MPI)
mpi_dep = dependency('mpi', required: false)

# If not found, try OpenMPI/MPICH pkg-config names explicitly (conda-forge)
if not mpi_dep.found() and enable_mpi
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

# If not found, try finding MPI libraries explicitly (works with conda-forge)
if not mpi_dep.found() and enable_mpi
    # Get conda prefix if available
    conda_prefix = run_command('printenv', 'CONDA_PREFIX', check: false).stdout().strip()

    if conda_prefix != ''
        # Try to find MPI in conda environment
        mpi_lib_dirs = [
            join_paths(conda_prefix, 'lib'),
            join_paths(conda_prefix, 'lib/openmpi'),
            join_paths(conda_prefix, 'lib64'),
            join_paths(conda_prefix, 'lib64/openmpi'),
        ]

        mpi_lib = cpp.find_library('mpi', dirs: mpi_lib_dirs, required: false)
        mpi_cxx_lib = cpp.find_library('mpi_cxx', dirs: mpi_lib_dirs, required: false)

        if mpi_lib.found()
            mpi_libs = [mpi_lib]
            mpi_include_dirs = []

            mpi_include_path = join_paths(conda_prefix, 'include')
            mpi_include_openmpi = join_paths(conda_prefix, 'include/openmpi')

            if run_command('test', '-d', mpi_include_path, check: false).returncode() == 0
                mpi_include_dirs += [include_directories(mpi_include_path)]
            endif
            if run_command('test', '-d', mpi_include_openmpi, check: false).returncode() == 0
                mpi_include_dirs += [include_directories(mpi_include_openmpi)]
            endif

            if mpi_cxx_lib.found()
                mpi_libs += [mpi_cxx_lib]
                message('Found MPI C++ library in conda')
            else
                message('MPI C++ library not found (using C bindings only)')
            endif

            mpi_dep = declare_dependency(
                include_directories: mpi_include_dirs,
                dependencies: mpi_libs,
            )
            message('Found MPI libraries via conda environment')
        endif
    endif

    # Fallback to system-wide search if conda detection failed
    if not mpi_dep.found()
        mpi_lib = cpp.find_library('mpi', required: false)
        if mpi_lib.found()
            mpi_dep = declare_dependency(dependencies: mpi_lib)
            message('Found MPI libraries via system search')
        endif
    endif
endif

# Still not found and MPI is required?
if enable_mpi and not mpi_dep.found()
    error('MPI requested but not found. Install system MPI or conda-forge openmpi')
endif

# Find boost-mpi and boost-serialization libraries explicitly
# Using find_library is more reliable than dependency() for MPI
boost_search_dirs = []
if conda_prefix != ''
    boost_search_dirs += [
        join_paths(conda_prefix, 'lib'),
        join_paths(conda_prefix, 'lib64'),
    ]
endif

boost_mpi_lib = cpp.find_library('boost_mpi', dirs: boost_search_dirs, required: false)
boost_serialization_lib = cpp.find_library('boost_serialization', dirs: boost_search_dirs, required: false)

has_boost_mpi = boost_mpi_lib.found() and boost_serialization_lib.found()

# Combine into a dependency-like object
if has_boost_mpi
    boost_mpi_dep = declare_dependency(
        dependencies: [boost_mpi_lib, boost_serialization_lib]
    )
    message('Found boost-mpi and boost-serialization libraries')
else
    boost_mpi_dep = declare_dependency()
    if enable_mpi
        warning('boost-mpi or boost-serialization not found - MPI support may be limited')
    endif
endif

has_mpi = enable_mpi and mpi_dep.found()

# Boost.MPI is required for Samurai's MPI features, but NOT for PETSc
# PETSc has its own MPI implementation and doesn't need Boost.MPI
if enable_mpi and not has_boost_mpi and not enable_petsc
    error('MPI support requires Boost.MPI and Boost.Serialization; install libboost-mpi and libboost-serialization')
endif

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

# ============================================================================
# PETSc DETECTION (optional, requires MPI)
# ============================================================================
# PETSc is detected only if MPI is enabled
petsc_dep = dependency('PETSc', required: false)

if not petsc_dep.found() and enable_petsc and has_mpi
    # Try pkg-config for PETSc (works with conda-forge)
    petsc_pkgconfig = dependency('PETSc', method: 'pkg-config', required: false)
    if petsc_pkgconfig.found()
        petsc_dep = petsc_pkgconfig
        message('Found PETSc via pkg-config')
    endif
endif

# If not found, try finding PETSc libraries explicitly (conda-forge)
if not petsc_dep.found() and enable_petsc and has_mpi
    if conda_prefix != ''
        petsc_lib_dirs = [
            join_paths(conda_prefix, 'lib'),
            join_paths(conda_prefix, 'lib64'),
        ]

        petsc_lib = cpp.find_library('petsc', dirs: petsc_lib_dirs, required: false)

        if petsc_lib.found()
            petsc_include_path = join_paths(conda_prefix, 'include')
            petsc_include_dirs = []

            if run_command('test', '-d', petsc_include_path, check: false).returncode() == 0
                petsc_include_dirs += [include_directories(petsc_include_path)]
            endif

            petsc_dep = declare_dependency(
                include_directories: petsc_include_dirs,
                dependencies: petsc_lib,
            )
            message('Found PETSc libraries via conda environment')
        endif
    endif
endif

# PETSc requires MPI - ensure this is enforced
if enable_petsc and not has_mpi
    error('PETSc support requires MPI to be enabled (use -Dmpi=true)')
endif

has_petsc = enable_petsc and petsc_dep.found()

if enable_petsc and not has_petsc
    error('PETSc requested but not found. Install conda-forge PETSc: conda install -c conda-forge petsc')
endif

if has_petsc
    message('')
    message('=== PETSc Support Enabled ===')
    message('PETSc library: @0@'.format(petsc_dep.name()))

    # Define SAMURAI_WITH_PETSC for conditional compilation
    extra_cpp_args += '-DSAMURAI_WITH_PETSC'
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
if has_petsc
    message('PETSc: ENABLED (@0@)'.format(petsc_dep.name()))
else
    message('PETSc: DISABLED (use -Dmpi=true -Dpetsc=true to enable)')
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
    'src/bindings/petsc_bindings.cpp',
)

if has_mpi
    message('MPI bindings: ENABLED')
else
    message('MPI bindings: STUB (MPI disabled)')
endif

if has_petsc
    message('PETSc bindings: ENABLED')
else
    message('PETSc bindings: STUB (PETSc disabled)')
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
    # Always link boost_serialization with MPI if found (Samurai needs it)
    if boost_serialization_lib.found()
        all_dependencies += [boost_serialization_lib]
    endif
    # Always link boost_mpi if found (needed by mpi_bindings.cpp)
    if has_boost_mpi
        all_dependencies += [boost_mpi_dep]
    endif
endif

# Add boost_serialization if found (required by Samurai for Interval types)
# even without MPI
if not has_mpi and boost_serialization_lib.found()
    boost_serialization_dep = declare_dependency(
        dependencies: [boost_serialization_lib],
        link_args: ['-lboost_serialization']
    )
    all_dependencies += [boost_serialization_dep]
endif

# Add PETSc dependencies if enabled
if has_petsc
    all_dependencies += [petsc_dep]
endif

# Collect all cpp_args
all_cpp_args = [
    '-DNDEBUG',
]

# Add MPI-specific cpp_args
if has_mpi
    all_cpp_args += extra_cpp_args
endif

# Add PETSc-specific cpp_args (includes SAMURAI_WITH_PETSC)
if has_petsc
    all_cpp_args += extra_cpp_args
endif

pybind11_module = py.extension_module(
    '_sampai',
    binding_sources,
    dependencies: all_dependencies,
    include_directories: [
        pybind11_include,
        include_directories('src'),
    ],
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

# Install PETSc subpackage (if enabled)
if has_petsc
    py.install_sources(
        'src/sampai/petsc/__init__.py',
        subdir: 'sampai/petsc',
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
