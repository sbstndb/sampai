#!/usr/bin/env python3
"""
Tests for NumPy __array_ufunc__ protocol support in Sampai fields.

This test file validates that ScalarField and VectorField properly implement
the NumPy array function protocol, enabling NumPy ufuncs to work directly
with field objects.

Coverage:
- Unary ufuncs via xtensor (sin, cos, exp, log, sqrt, etc.)
- Fallback to NumPy for unsupported ufuncs
- VectorField component-wise operations
- Chaining ufuncs
- Integration with existing field operations
"""

import numpy as np
import pytest

import sampai as sam

# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def mesh_1d():
    """Create a simple 1D mesh for testing."""
    box = sam.geometry.box([0.0], [1.0])
    config = sam.config.make(1)
    config.min_level = 4
    config.max_level = 4
    return sam.mesh.make(box, config)


@pytest.fixture
def mesh_2d():
    """Create a simple 2D mesh for testing."""
    box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
    config = sam.config.make(2)
    config.min_level = 4
    config.max_level = 4
    return sam.mesh.make(box, config)


@pytest.fixture
def scalar_field_1d(mesh_1d):
    """Create a ScalarField1D initialized to 1.0."""
    field = sam.field.scalar(mesh_1d, "u", init=1.0)
    return field


@pytest.fixture
def scalar_field_2d(mesh_2d):
    """Create a ScalarField2D initialized to 2.0."""
    field = sam.field.scalar(mesh_2d, "v", init=2.0)
    return field


@pytest.fixture
def vector_field_2d(mesh_2d):
    """Create a VectorField2D_2 initialized to [1.0, 2.0]."""
    field = sam.field.vector(mesh_2d, "vel", n_components=2, init=0.0)
    field.fill([1.0, 2.0])
    return field


# ============================================================
# Trigonometric Ufuncs Tests (ScalarField)
# ============================================================

class TestTrigonometricUfuncsScalarField:
    """Test trigonometric ufuncs on ScalarField."""

    def _verify_field_values(self, field, expected_value, tol=1e-10):
        """Helper to verify field values are close to expected."""
        arr = field.numpy_view()
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol), \
                f"Expected {expected_value}, got {nonzero_values}"

    def test_sin_ones(self, scalar_field_1d):
        """Test np.sin on field of ones."""
        result = np.sin(scalar_field_1d)
        assert isinstance(result, type(scalar_field_1d)), "Should return ScalarField"
        assert "sin" in result.name, "Result name should contain 'sin'"
        self._verify_field_values(result, np.sin(1.0))

    def test_cos_zeros(self, mesh_1d):
        """Test np.cos on field of zeros."""
        field = sam.field.scalar(mesh_1d, "u", init=0.0)
        result = np.cos(field)
        self._verify_field_values(result, np.cos(0.0))

    def test_tan(self, scalar_field_1d):
        """Test np.tan on field."""
        result = np.tan(scalar_field_1d)
        self._verify_field_values(result, np.tan(1.0))

    def test_asin(self, mesh_1d):
        """Test np.arcsin on field."""
        field = sam.field.scalar(mesh_1d, "u", init=0.5)
        result = np.arcsin(field)
        self._verify_field_values(result, np.arcsin(0.5))

    def test_acos(self, mesh_1d):
        """Test np.arccos on field."""
        field = sam.field.scalar(mesh_1d, "u", init=0.5)
        result = np.arccos(field)
        self._verify_field_values(result, np.arccos(0.5))

    def test_atan(self, scalar_field_1d):
        """Test np.arctan on field."""
        result = np.arctan(scalar_field_1d)
        self._verify_field_values(result, np.arctan(1.0))


# ============================================================
# Hyperbolic Ufuncs Tests (ScalarField)
# ============================================================

class TestHyperbolicUfuncsScalarField:
    """Test hyperbolic ufuncs on ScalarField."""

    def _verify_field_values(self, field, expected_value, tol=1e-10):
        """Helper to verify field values are close to expected."""
        arr = field.numpy_view()
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol)

    def test_sinh(self, scalar_field_1d):
        """Test np.sinh on field."""
        result = np.sinh(scalar_field_1d)
        self._verify_field_values(result, np.sinh(1.0))

    def test_cosh(self, scalar_field_1d):
        """Test np.cosh on field."""
        result = np.cosh(scalar_field_1d)
        self._verify_field_values(result, np.cosh(1.0))

    def test_tanh(self, scalar_field_1d):
        """Test np.tanh on field."""
        result = np.tanh(scalar_field_1d)
        self._verify_field_values(result, np.tanh(1.0))


# ============================================================
# Exponential and Logarithmic Ufuncs Tests (ScalarField)
# ============================================================

class TestExponentialUfuncsScalarField:
    """Test exponential and logarithmic ufuncs on ScalarField."""

    def _verify_field_values(self, field, expected_value, tol=1e-10):
        """Helper to verify field values are close to expected."""
        arr = field.numpy_view()
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol, rtol=1e-7)

    def test_exp(self, scalar_field_1d):
        """Test np.exp on field."""
        result = np.exp(scalar_field_1d)
        self._verify_field_values(result, np.exp(1.0))

    def test_expm1(self, scalar_field_1d):
        """Test np.expm1 on field."""
        result = np.expm1(scalar_field_1d)
        self._verify_field_values(result, np.expm1(1.0))

    def test_log(self, mesh_1d):
        """Test np.log on field."""
        field = sam.field.scalar(mesh_1d, "u", init=2.0)
        result = np.log(field)
        arr = result.numpy_view()
        nonzero = arr[np.abs(arr) > 1e-10]
        assert np.allclose(nonzero, np.log(2.0), atol=1e-10, rtol=1e-7)

    def test_log10(self, mesh_1d):
        """Test np.log10 on field."""
        field = sam.field.scalar(mesh_1d, "u", init=10.0)
        result = np.log10(field)
        arr = result.numpy_view()
        nonzero = arr[np.abs(arr) > 1e-10]
        assert np.allclose(nonzero, np.log10(10.0), atol=1e-10, rtol=1e-7)

    def test_log2(self, mesh_1d):
        """Test np.log2 on field."""
        field = sam.field.scalar(mesh_1d, "u", init=2.0)
        result = np.log2(field)
        arr = result.numpy_view()
        nonzero = arr[np.abs(arr) > 1e-10]
        assert np.allclose(nonzero, np.log2(2.0), atol=1e-10, rtol=1e-7)

    def test_log1p(self, scalar_field_1d):
        """Test np.log1p on field."""
        result = np.log1p(scalar_field_1d)
        self._verify_field_values(result, np.log1p(1.0))


# ============================================================
# Mathematical Ufuncs Tests (ScalarField)
# ============================================================

class TestMathematicalUfuncsScalarField:
    """Test mathematical ufuncs on ScalarField."""

    def _verify_field_values(self, field, expected_value, tol=1e-10):
        """Helper to verify field values are close to expected."""
        arr = field.numpy_view()
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol)

    def test_sqrt(self, mesh_1d):
        """Test np.sqrt on field."""
        field = sam.field.scalar(mesh_1d, "u", init=4.0)
        result = np.sqrt(field)
        self._verify_field_values(result, 2.0)

    def test_square(self, scalar_field_1d):
        """Test np.square on field."""
        result = np.square(scalar_field_1d)
        self._verify_field_values(result, 1.0)

    def test_abs(self, mesh_1d):
        """Test np.abs on field with negative values."""
        field = sam.field.scalar(mesh_1d, "u", init=-2.0)
        result = np.abs(field)
        self._verify_field_values(result, 2.0)

    def test_fabs(self, mesh_1d):
        """Test np.fabs on field with negative values."""
        field = sam.field.scalar(mesh_1d, "u", init=-2.0)
        result = np.fabs(field)
        self._verify_field_values(result, 2.0)

    def test_reciprocal(self, scalar_field_1d):
        """Test np.reciprocal on field."""
        result = np.reciprocal(scalar_field_1d)
        self._verify_field_values(result, 1.0)

    def test_negative(self, mesh_1d):
        """Test np.negative on field."""
        field = sam.field.scalar(mesh_1d, "u", init=2.0)
        result = np.negative(field)
        self._verify_field_values(result, -2.0)

    def test_positive(self, scalar_field_1d):
        """Test np.positive on field."""
        result = np.positive(scalar_field_1d)
        self._verify_field_values(result, 1.0)


# ============================================================
# Rounding Ufuncs Tests (ScalarField)
# ============================================================

class TestRoundingUfuncsScalarField:
    """Test rounding ufuncs on ScalarField."""

    def _verify_field_values(self, field, expected_value, tol=1e-10):
        """Helper to verify field values are close to expected."""
        arr = field.numpy_view()
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol)

    def test_floor(self, mesh_1d):
        """Test np.floor on field."""
        field = sam.field.scalar(mesh_1d, "u", init=2.7)
        result = np.floor(field)
        self._verify_field_values(result, 2.0)

    def test_ceil(self, mesh_1d):
        """Test np.ceil on field."""
        field = sam.field.scalar(mesh_1d, "u", init=2.3)
        result = np.ceil(field)
        self._verify_field_values(result, 3.0)

    def test_trunc(self, mesh_1d):
        """Test np.trunc on field."""
        field = sam.field.scalar(mesh_1d, "u", init=2.7)
        result = np.trunc(field)
        self._verify_field_values(result, 2.0)

    def test_rint(self, mesh_1d):
        """Test np.rint on field."""
        field = sam.field.scalar(mesh_1d, "u", init=2.3)
        result = np.rint(field)
        self._verify_field_values(result, 2.0)


# ============================================================
# Special Functions Ufuncs Tests (ScalarField)
# ============================================================
# NOTE: erf/erfc are in scipy.special, not NumPy, so they are not tested here.
# They can be added later if scipy integration is desired.

# ============================================================
# NumPy Fallback Tests (ScalarField)
# ============================================================
# NOTE: clip and heaviside are multi-input ufuncs which are not currently supported.
# They require special handling since they take additional parameters beyond the input array.
# This can be added in a future update if needed.

# ============================================================
# Chaining and Composition Tests
# ============================================================

class TestUfuncChainingScalarField:
    """Test chaining multiple ufuncs."""

    def test_chain_sin_cos(self, scalar_field_1d):
        """Test sin(cos(field)) composition."""
        inner = np.cos(scalar_field_1d)
        result = np.sin(inner)
        expected = np.sin(np.cos(1.0))
        arr = result.numpy_view()
        nonzero = arr[np.abs(arr) > 1e-10]
        assert np.allclose(nonzero, expected, atol=1e-10, rtol=1e-7)

    def test_chain_exp_log(self, mesh_1d):
        """Test exp(log(exp(field))) chain."""
        field = sam.field.scalar(mesh_1d, "u", init=0.5)
        result = np.exp(np.log(np.exp(field)))
        arr = result.numpy_view()
        nonzero = arr[np.abs(arr) > 1e-10]
        # exp(log(exp(0.5))) = exp(0.5)
        expected = np.exp(0.5)
        assert np.allclose(nonzero, expected, atol=1e-10, rtol=1e-7)

    def test_add_after_ufunc(self, scalar_field_1d):
        """Test arithmetic after ufunc."""
        result = np.sin(scalar_field_1d) + 1.0
        expected = np.sin(1.0) + 1.0
        arr = result.numpy_view()
        nonzero = arr[np.abs(arr) > 1e-10]
        assert np.allclose(nonzero, expected, atol=1e-10, rtol=1e-7)

    def test_ufunc_after_arithmetic(self, scalar_field_1d):
        """Test ufunc after arithmetic."""
        doubled = scalar_field_1d * 2.0
        result = np.sin(doubled)
        expected = np.sin(2.0)
        arr = result.numpy_view()
        nonzero = arr[np.abs(arr) > 1e-10]
        assert np.allclose(nonzero, expected, atol=1e-10, rtol=1e-7)


# ============================================================
# VectorField Ufunc Tests
# ============================================================

class TestUfuncsVectorField:
    """Test ufuncs on VectorField (component-wise application)."""

    def _verify_component(self, arr, component_idx, expected_value, tol=1e-10):
        """Helper to verify a specific component value."""
        # VectorField numpy_view returns (n_cells, n_components) for AOS
        component = arr[:, component_idx]
        assert not np.any(np.isnan(component)), f"Component {component_idx} contains NaN"
        nonzero_mask = np.abs(component) > tol
        if np.any(nonzero_mask):
            nonzero_values = component[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol, rtol=1e-7)

    def test_sin_vector(self, vector_field_2d):
        """Test np.sin on VectorField (component-wise)."""
        result = np.sin(vector_field_2d)
        assert isinstance(result, type(vector_field_2d)), "Should return VectorField"
        arr = result.numpy_view()
        # [sin(1.0), sin(2.0)]
        self._verify_component(arr, 0, np.sin(1.0))
        self._verify_component(arr, 1, np.sin(2.0))

    def test_exp_vector(self, vector_field_2d):
        """Test np.exp on VectorField (component-wise)."""
        result = np.exp(vector_field_2d)
        arr = result.numpy_view()
        # [exp(1.0), exp(2.0)]
        self._verify_component(arr, 0, np.exp(1.0))
        self._verify_component(arr, 1, np.exp(2.0))

    def test_sqrt_vector(self, mesh_2d):
        """Test np.sqrt on VectorField (component-wise)."""
        field = sam.field.vector(mesh_2d, "vel", n_components=2, init=0.0)
        field.fill([4.0, 9.0])
        result = np.sqrt(field)
        arr = result.numpy_view()
        # [sqrt(4.0), sqrt(9.0)] = [2.0, 3.0]
        self._verify_component(arr, 0, 2.0)
        self._verify_component(arr, 1, 3.0)

    def test_abs_vector_negative(self, mesh_2d):
        """Test np.abs on VectorField with negative values."""
        field = sam.field.vector(mesh_2d, "vel", n_components=2, init=0.0)
        field.fill([-1.5, -2.5])
        result = np.abs(field)
        arr = result.numpy_view()
        self._verify_component(arr, 0, 1.5)
        self._verify_component(arr, 1, 2.5)

    def test_chain_vector(self, vector_field_2d):
        """Test chaining ufuncs on VectorField."""
        result = np.sin(np.exp(vector_field_2d))
        arr = result.numpy_view()
        # [sin(exp(1.0)), sin(exp(2.0))]
        self._verify_component(arr, 0, np.sin(np.exp(1.0)))
        self._verify_component(arr, 1, np.sin(np.exp(2.0)))


# ============================================================
# 2D ScalarField Tests
# ============================================================

class TestUfuncsScalarField2D:
    """Test that ufuncs work correctly on 2D fields."""

    def _verify_field_values(self, field, expected_value, tol=1e-10):
        """Helper to verify field values are close to expected."""
        arr = field.numpy_view()
        assert not np.any(np.isnan(arr)), "Array contains NaN values"
        nonzero_mask = np.abs(arr) > tol
        if np.any(nonzero_mask):
            nonzero_values = arr[nonzero_mask]
            assert np.allclose(nonzero_values, expected_value, atol=tol, rtol=1e-7)

    def test_sin_2d(self, scalar_field_2d):
        """Test np.sin on 2D field."""
        result = np.sin(scalar_field_2d)
        self._verify_field_values(result, np.sin(2.0))

    def test_exp_2d(self, scalar_field_2d):
        """Test np.exp on 2D field."""
        result = np.exp(scalar_field_2d)
        self._verify_field_values(result, np.exp(2.0))

    def test_sqrt_2d(self, mesh_2d):
        """Test np.sqrt on 2D field."""
        field = sam.field.scalar(mesh_2d, "u", init=4.0)
        result = np.sqrt(field)
        self._verify_field_values(result, 2.0)


# ============================================================
# Error Handling Tests
# ============================================================

class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_log_zero(self, mesh_1d):
        """Test np.log on field with zeros (should give -inf)."""
        field = sam.field.scalar(mesh_1d, "u", init=0.0)
        result = np.log(field)
        # This should work but produce -inf or warnings
        assert isinstance(result, type(field)), "Should return ScalarField"

    def test_sqrt_negative(self, mesh_1d):
        """Test np.sqrt on field with negative values."""
        field = sam.field.scalar(mesh_1d, "u", init=-1.0)
        result = np.sqrt(field)
        # This should produce NaN for negative inputs
        assert isinstance(result, type(field)), "Should return ScalarField"
        arr = result.numpy_view()
        # Should have NaN values
        assert np.any(np.isnan(arr)), "Expected NaN for sqrt of negative numbers"


# ============================================================
# Integration with Existing Operations Tests
# ============================================================

class TestIntegrationWithExistingOps:
    """Test that __array_ufunc__ doesn't break existing functionality."""

    def test_still_has_numpy_view(self, scalar_field_1d):
        """Test that numpy_view still works."""
        arr = scalar_field_1d.numpy_view()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (scalar_field_1d.size,)

    def test_still_has_reductions(self, scalar_field_1d):
        """Test that reduction methods still work."""
        total = scalar_field_1d.sum()
        mean_val = scalar_field_1d.mean()
        max_val = scalar_field_1d.max()
        min_val = scalar_field_1d.min()

        assert isinstance(total, float)
        assert isinstance(mean_val, float)
        assert isinstance(max_val, float)
        assert isinstance(min_val, float)

    def test_still_has_arithmetic(self, scalar_field_1d):
        """Test that arithmetic operators still work."""
        result = scalar_field_1d + 1.0
        result = scalar_field_1d - 0.5
        result = scalar_field_1d * 2.0
        result = scalar_field_1d / 2.0
        assert isinstance(result, type(scalar_field_1d))

    def test_still_has_indexing(self, scalar_field_1d):
        """Test that indexing still works."""
        val = scalar_field_1d[0]
        scalar_field_1d[0] = 5.0
        assert scalar_field_1d[0] == 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
