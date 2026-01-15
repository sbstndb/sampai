#!/bin/bash
# Helper script to setup C++ dependencies for wheel building
# This is used by cibuildwheel's before-all hook

set -e

echo "=== Setting up C++ dependencies for wheel build ==="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"

    # Install base tools
    echo "Installing base build tools..."
    yum install -y epel-release 2>/dev/null || true
    yum install -y meson ninja-build git wget

    # Install CMake 3.x (CMake 2.8 from yum is too old)
    echo "Installing CMake 3.x..."
    if ! command -v cmake &> /dev/null || [ "$(cmake --version | grep -oP 'cmake version \K[0-9.]+' | cut -d. -f1)" -lt 3 ]; then
        curl -LO "https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz"
        tar --strip-components=1 -xzvf cmake-3.26.4-linux-x86_64.tar.gz -C /usr/local
        rm cmake-3.26.4-linux-x86_64.tar.gz
        echo "CMake 3.26.4 installed"
    fi
    cmake --version

    # Install dependencies available via yum
    echo "Installing dependencies via yum..."
    yum install -y \
        hdf5-devel \
        pugixml-devel \
        fmt-devel \
        openssl-devel \
        bzip2-devel \
        zlib-devel || true

    # Install xtensor (header-only, via pip)
    echo "Installing xtensor (header-only)..."
    pip install --upgrade xtensor xsim 2>/dev/null || true

    # Install CLI11 from source
    echo "Installing CLI11..."
    CLI11_VER="2.6.1"
    if [ ! -f "/usr/local/include/cli/cli.hpp" ]; then
        # Use GitHub archive URL instead of releases
        curl -LO "https://github.com/CLIUtils/CLI11/archive/refs/tags/v${CLI11_VER}.tar.gz"
        tar -xzf "v${CLI11_VER}.tar.gz"
        cd "CLI11-${CLI11_VER}"
        cmake -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCLI11_BUILD_TESTS=OFF \
            -DCLI11_BUILD_EXAMPLES=OFF \
            -DCMAKE_INSTALL_PREFIX=/usr/local
        cmake --build build --parallel
        cmake --install build
        cd ..
        rm -rf "CLI11-${CLI11_VER}" "v${CLI11_VER}.tar.gz"
        echo "CLI11 installed"
    else
        echo "CLI11 already installed"
    fi

    # Install HighFive from source
    echo "Installing HighFive..."
    HIGHFIVE_VER="2.10.0"
    if [ ! -f "/usr/local/include/highfive/H5Easy.hpp" ]; then
        curl -LO "https://github.com/BlueBrain/HighFive/archive/refs/tags/v${HIGHFIVE_VER}.tar.gz"
        tar -xzf "v${HIGHFIVE_VER}.tar.gz"
        cd "HighFive-${HIGHFIVE_VER}"
        cmake -B build \
            -DHIGHFIVE_BUILD_EXAMPLES=OFF \
            -DHIGHFIVE_BUILD_TESTS=OFF \
            -DHIGHFIVE_USE_BOOST=OFF \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        cmake --build build --parallel
        cmake --install build
        cd ..
        rm -rf "HighFive-${HIGHFIVE_VER}" "v${HIGHFIVE_VER}.tar.gz"
        echo "HighFive installed"
    else
        echo "HighFive already installed"
    fi

    # Update pkg-config path
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"

    # Update brew and install dependencies
    echo "Installing dependencies via brew..."
    brew update || true
    brew install meson ninja hdf5 pugixml fmt cmake

    # Install xtensor (header-only, via pip)
    echo "Installing xtensor (header-only)..."
    pip install --upgrade xtensor xsim 2>/dev/null || true

    # Install CLI11 via brew (or from source if not available)
    echo "Installing CLI11..."
    brew install cli11 2>/dev/null || true

    # Check if CLI11 was installed, if not, install from source
    if [ ! -f "/usr/local/include/cli/cli.hpp" ] && [ ! -f "$(brew --prefix)/include/cli/cli.hpp" ]; then
        CLI11_VER="2.6.1"
        # Use GitHub archive URL instead of releases
        curl -LO "https://github.com/CLIUtils/CLI11/archive/refs/tags/v${CLI11_VER}.tar.gz"
        tar -xzf "v${CLI11_VER}.tar.gz"
        cd "CLI11-${CLI11_VER}"
        cmake -B build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCLI11_BUILD_TESTS=OFF \
            -DCLI11_BUILD_EXAMPLES=OFF \
            -DCMAKE_INSTALL_PREFIX=/usr/local
        cmake --build build --parallel
        cmake --install build
        cd ..
        rm -rf "CLI11-${CLI11_VER}" "v${CLI11_VER}.tar.gz"
        echo "CLI11 installed from source"
    else
        echo "CLI11 already installed"
    fi

    # Install HighFive from source (brew version might be old)
    echo "Installing HighFive..."
    HIGHFIVE_VER="2.10.0"
    if [ ! -f "/usr/local/include/highfive/H5Easy.hpp" ]; then
        curl -LO "https://github.com/BlueBrain/HighFive/archive/refs/tags/v${HIGHFIVE_VER}.tar.gz"
        tar -xzf "v${HIGHFIVE_VER}.tar.gz"
        cd "HighFive-${HIGHFIVE_VER}"
        cmake -B build \
            -DHIGHFIVE_BUILD_EXAMPLES=OFF \
            -DHIGHFIVE_BUILD_TESTS=OFF \
            -DHIGHFIVE_USE_BOOST=OFF \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        cmake --build build --parallel
        cmake --install build
        cd ..
        rm -rf "HighFive-${HIGHFIVE_VER}" "v${HIGHFIVE_VER}.tar.gz"
        echo "HighFive installed"
    else
        echo "HighFive already installed"
    fi

    # Update pkg-config path
    export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
fi

echo "=== C++ dependencies setup complete ==="
