#!/bin/bash
# Helper script to build samurai-python conda package locally
#
# Usage:
#   ./build.sh              # Build for current platform
#   ./build.sh --test       # Build and test
#   ./build.sh --install    # Build and install locally
#   ./build.sh --upload     # Build and prepare for upload

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${GREEN}=== Samurai Python Conda Build Helper ===${NC}"
echo ""

# Parse arguments
DO_TEST=false
DO_INSTALL=false
DO_UPLOAD=false
BUILD_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            DO_TEST=true
            shift
            ;;
        --install)
            DO_INSTALL=true
            shift
            ;;
        --upload)
            DO_UPLOAD=true
            shift
            ;;
        --no-test)
            BUILD_ARGS="--no-test"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Check if samurai C++ package is installed
echo -e "${YELLOW}Checking for samurai C++ package...${NC}"
if ! conda list samurai 2>/dev/null | grep -q "samurai"; then
    echo -e "${RED}Error: samurai C++ package not found!${NC}"
    echo -e "Please install it first:"
    echo -e "  conda install -c conda-forge samurai"
    echo -e "Or build it from source before building the Python bindings."
    exit 1
fi
echo -e "${GREEN}Found samurai C++ package${NC}"
echo ""

# Build the package
echo -e "${YELLOW}Building samurai-python conda package...${NC}"
cd "$PYTHON_DIR"

# Build with conda build
conda build \
    --no-anaconda-upload \
    $BUILD_ARGS \
    conda-recipe/

BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit $BUILD_STATUS
fi

echo -e "${GREEN}Build successful!${NC}"
echo ""

# Get the output path
OUTPUT_PATH=$(conda build --output conda-recipe/ 2>/dev/null | tail -n 1)
echo -e "${GREEN}Package location: ${OUTPUT_PATH}${NC}"
echo ""

# Test the package if requested
if [ "$DO_TEST" = true ]; then
    echo -e "${YELLOW}Testing package...${NC}"
    # Note: conda build already runs tests if --no-test is not used
    echo -e "${GREEN}Tests completed during build${NC}"
fi

# Install locally if requested
if [ "$DO_INSTALL" = true ]; then
    echo -e "${YELLOW}Installing package locally...${NC}"
    conda install --use-local samurai-python
    echo -e "${GREEN}Package installed!${NC}"
    echo ""
    echo -e "Test the installation:"
    echo -e "  python -c 'import samurai_python; print(samurai_python.__version__)'"
fi

# Prepare for upload if requested
if [ "$DO_UPLOAD" = true ]; then
    echo -e "${YELLOW}Preparing package for upload...${NC}"
    echo -e "${YELLOW}Note: Manual upload to anaconda.org is required${NC}"
    echo -e ""
    echo -e "To upload to anaconda.org:"
    echo -e "  anaconda upload $OUTPUT_PATH"
    echo -e ""
    echo -e "Or create a PR to conda-forge:"
    echo -e "  https://github.com/conda-forge/staged-recipes"
fi

echo ""
echo -e "${GREEN}=== Build Complete ===${NC}"
