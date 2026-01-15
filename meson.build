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
        error('pybind11 not found. Install with: pip install pybind11 or conda install -c conda-forge pybind11')
    endif
else
    message('pybind11 found: @0@'.format(pybind11_dep.version()))
    # When pybind11 is found via cmake/pkg-config, get the include dir correctly
    # Try different variable names that pybind11-config might provide
    pybind11_include_dirs = pybind11_dep.get_variable('pybind11_INCLUDE_DIRS', default_value: '')
    if pybind11_include_dirs == ''
        pybind11_include_dirs = pybind11_dep.get_variable('INCLUDEDIR', default_value: '')
    endif
    if pybind11_include_dirs != ''
        pybind11_include = include_directories(pybind11_include_dirs)
    else
        # Fallback: let pybind11_dep handle includes via dependencies
        pybind11_include = include_directories()
    endif
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

# Find xtensor include dirs (if not found via pkg-config/cmake)
xtensor_include_dirs = []
if not xtensor_dep.found()
    # xtensor might be installed via pip (in site-packages)
    # Use Python to find xtensor include path directly
    find_xtensor = run_command(py, '-c', '''
import sysconfig
import os

# Check purelib (pip install location)
purelib = sysconfig.get_path("purelib")
xtensor_path = os.path.join(purelib, "xtensor", "include")
if os.path.isdir(xtensor_path):
    print(xtensor_path)
else:
    # Try platlib as well
    platlib = sysconfig.get_path("platlib")
    xtensor_path = os.path.join(platlib, "xtensor", "include")
    if os.path.isdir(xtensor_path):
        print(xtensor_path)
    else:
        print("")
''', check: false)

    if find_xtensor.returncode() == 0
        xtensor_path = find_xtensor.stdout().strip()
        if xtensor_path != ''
            xtensor_include_dirs += include_directories(xtensor_path)
            message('Using xtensor from pip: @0@'.format(xtensor_path))
        else
            warning('xtensor not found via pip. Ensure xtensor is installed.')
        endif
    else
        warning('Failed to search for xtensor in site-packages')
    endif
endif

message('')
message('=== Dependencies Summary ===')
message('Python: @0@'.format(py_version))
message('pybind11: found')
message('samurai: from GitHub (auto-downloaded via subproject)')
if xtensor_dep.found()
    message('xtensor: @0@'.format(xtensor_dep.version()))
elif xtensor_include_dirs.length() > 0
    message('xtensor: from pip (site-packages)')
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

# Build include directories list
module_includes = [include_directories('src')]
if xtensor_include_dirs.length() > 0
    module_includes += xtensor_include_dirs
endif

pybind11_module = py.extension_module(
    '_sampai',
    binding_sources,
    dependencies: [
        pybind11_dep,  # pybind11 dependency (adds includes automatically)
        samurai_dep,
        fmt_dep,
        cli11_dep,
        hdf5_dep,
        highfive_dep,
        pugixml_dep,
    ],
    include_directories: module_includes,
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
message('  Samurai source:      auto-downloaded from GitHub (header-only)')
message('  Build tests:         @0@'.format(build_python_tests))
message('=========================================')
message('')
