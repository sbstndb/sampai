# Samurai Python Bindings - Meson Build Configuration
#
# This meson.build builds the Python bindings using samurai from source.
# Samurai conda package is only used for dependencies (xtensor, fmt, CLI, hdf5).
#
# Build:
#   pip install .               # Build from source
#   pip install -e .            # Editable install (development)
#   meson setup builddir        # Configure build
#   meson compile -C builddir   # Build
#
# Requirements:
#   - samurai source code next to sampai (../samurai)
#   - samurai built locally (../samurai/build)
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
# FIND SAMURAI C++ LIBRARY (header-only from source)
# ============================================================================
# IMPORTANT: We MUST use samurai from source because conda version is outdated.
# Samurai is a header-only library, so no compiled library is needed.
# The conda package is only used for dependencies (xtensor, fmt, CLI, hdf5, etc.).

# Verify local samurai source exists
samurai_source_check = run_command('test', '-d', '../samurai/include/samurai', check: false)
if samurai_source_check.returncode() != 0
    error('Samurai source not found at ../samurai/include. ' +
          'Please clone samurai next to sampai: git clone https://github.com/hpc-maths/samurai.git ../samurai')
endif

# Samurai is header-only - just need headers
samurai_include_dirs = [
    include_directories('../samurai/include'),
]

message('=== Using Samurai (header-only) from source ===')
message('  Source: ../samurai')
message('  Headers: ../samurai/include')

# Add conda include for dependencies (xtensor, fmt, CLI, hdf5, etc.)
# These are needed by samurai headers but we don't want conda samurai headers
conda_prefix = run_command('printenv', 'CONDA_PREFIX', check: false).stdout().strip()
if conda_prefix != ''
    # Add conda include for dependencies only
    # Note: We add this AFTER samurai source so samurai headers take precedence
    samurai_include_dirs += include_directories(conda_prefix / 'include')
    message('  Conda prefix: ' + conda_prefix + ' (for dependencies)')
endif

# Create samurai dependency (header-only, no link args needed)
samurai_dep = declare_dependency(
    include_directories: samurai_include_dirs,
)

message('')

# Additional dependencies from samurai (via conda)
# These are pulled in by samurai but we list them for completeness
xtensor_dep = dependency('xtensor', required: false)
fmt_dep = dependency('fmt', required: false)
cli11_dep = dependency('cli11', required: false)
highfive_dep = dependency('highfive', version: '>=2.10', required: false)
hdf5_dep = dependency('hdf5', required: false)
pugixml_dep = dependency('pugixml', required: false)

message('')
message('=== Dependencies Summary ===')
message('Python: @0@'.format(py_version))
message('pybind11: found')
message('samurai: from source (local)')
if xtensor_dep.found()
    message('xtensor: @0@'.format(xtensor_dep.version()))
endif
if hdf5_dep.found()
    message('hdf5: @0@'.format(hdf5_dep.version()))
endif
message('')

# ============================================================================
# COLLECT SOURCE FILES
# ============================================================================

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
    'src/bindings/io_bindings.cpp',
)

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
pybind11_module = py.extension_module(
    '_sampai',
    binding_sources,
    dependencies: [
        samurai_dep,
        fmt_dep,
        cli11_dep,
        hdf5_dep,
        highfive_dep,
        pugixml_dep,
    ],
    include_directories: [
        pybind11_include,
        include_directories('src'),
    ],
    install: true,
    install_tag: 'runtime',
    # Optimization flags for better performance
    cpp_args: [
        '-DNDEBUG',
    ],
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
message('  Samurai source:      local (header-only)')
message('  Build tests:         @0@'.format(build_python_tests))
message('=========================================')
message('')
