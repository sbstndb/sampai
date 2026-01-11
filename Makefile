# Samurai Python Bindings - Makefile
#
# Convenience Makefile for building and developing the Python bindings.
# This is a thin wrapper around the dev.py script.
#
# Usage:
#   make build          # Build the module
#   make install        # Install in editable mode
#   make test           # Run tests
#   make clean          # Clean build artifacts
#   make reinstall      # Clean + build + install
#   make all            # Build + install + test
#   make help           # Show all targets

.PHONY: all build install test clean reinstall help fmt lint check

# Default target
all: build install test

# Build the Python module
build:
	@echo "Building Python module..."
	python3 dev.py build

# Install in editable mode (development)
install:
	@echo "Installing Python module in editable mode..."
	python3 dev.py install

# Run tests
test:
	@echo "Running tests..."
	python3 dev.py test

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	python3 dev.py clean

# Reinstall (clean + build + install)
reinstall:
	@echo "Reinstalling Python module..."
	python3 dev.py reinstall

# Format code with black
fmt:
	@echo "Formatting Python code..."
	@black src/ tests/ --line-length 100
	@isort src/ tests/ --profile black

# Lint with ruff
lint:
	@echo "Linting Python code..."
	@ruff check src/ tests/

# Type check with mypy
check:
	@echo "Type checking with mypy..."
	@mypy src/ --ignore-missing-imports

# Run all checks
check-all: lint check
	@echo "All checks passed!"

# Build conda package
conda:
	@echo "Building conda package..."
	@cd conda-recipe && bash build.sh

# Build wheel
wheel:
	@echo "Building wheel package..."
	python3 -m build

# Install from wheel
install-wheel: wheel
	@echo "Installing from wheel..."
	pip3 install dist/*.whl

# Show help
help:
	@echo "Samurai Python Bindings - Makefile targets"
	@echo ""
	@echo "Build targets:"
	@echo "  make build          - Build the Python module"
	@echo "  make install        - Install in editable mode (development)"
	@echo "  make reinstall      - Clean + build + install"
	@echo "  make wheel          - Build wheel package"
	@echo "  make install-wheel  - Install from wheel"
	@echo "  make conda          - Build conda package"
	@echo ""
	@echo "Testing targets:"
	@echo "  make test           - Run tests"
	@echo ""
	@echo "Code quality targets:"
	@echo "  make fmt            - Format code with black/isort"
	@echo "  make lint           - Lint with ruff"
	@echo "  make check          - Type check with mypy"
	@echo "  make check-all      - Run all checks"
	@echo ""
	@echo "Maintenance targets:"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make all            - Build + install + test"
	@echo "  make help           - Show this message"
	@echo ""
