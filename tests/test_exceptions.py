"""
Tests for Sampai exception hierarchy.

This module tests the custom exception types and their relationships.
"""

import pytest

import sampai
from sampai import (
    SampaiError,
    MeshError,
    FieldError,
    ConfigError,
    OperatorError,
    BoundaryConditionError,
    AdaptationError,
    IOError,
)


class TestExceptionHierarchy:
    """Test that all exceptions are properly defined and inherit correctly."""

    def test_all_exceptions_exported(self):
        """Test that all exceptions are exported from sampai package."""
        assert "SampaiError" in sampai.__all__
        assert "MeshError" in sampai.__all__
        assert "FieldError" in sampai.__all__
        assert "ConfigError" in sampai.__all__
        assert "OperatorError" in sampai.__all__
        assert "BoundaryConditionError" in sampai.__all__
        assert "AdaptationError" in sampai.__all__
        assert "IOError" in sampai.__all__

    def test_base_exception(self):
        """Test SampaiError is the base exception."""
        assert issubclass(SampaiError, Exception)
        assert SampaiError.__name__ == "SampaiError"

    def test_mesh_error_inheritance(self):
        """Test MeshError inherits from SampaiError."""
        assert issubclass(MeshError, SampaiError)
        assert issubclass(MeshError, Exception)
        assert MeshError.__name__ == "MeshError"

    def test_field_error_inheritance(self):
        """Test FieldError inherits from SampaiError."""
        assert issubclass(FieldError, SampaiError)
        assert issubclass(FieldError, Exception)
        assert FieldError.__name__ == "FieldError"

    def test_config_error_inheritance(self):
        """Test ConfigError inherits from SampaiError."""
        assert issubclass(ConfigError, SampaiError)
        assert issubclass(ConfigError, Exception)
        assert ConfigError.__name__ == "ConfigError"

    def test_operator_error_inheritance(self):
        """Test OperatorError inherits from SampaiError."""
        assert issubclass(OperatorError, SampaiError)
        assert issubclass(OperatorError, Exception)
        assert OperatorError.__name__ == "OperatorError"

    def test_boundary_condition_error_inheritance(self):
        """Test BoundaryConditionError inherits from SampaiError."""
        assert issubclass(BoundaryConditionError, SampaiError)
        assert issubclass(BoundaryConditionError, Exception)
        assert BoundaryConditionError.__name__ == "BoundaryConditionError"

    def test_adaptation_error_inheritance(self):
        """Test AdaptationError inherits from SampaiError."""
        assert issubclass(AdaptationError, SampaiError)
        assert issubclass(AdaptationError, Exception)
        assert AdaptationError.__name__ == "AdaptationError"

    def test_io_error_inheritance(self):
        """Test IOError inherits from SampaiError."""
        assert issubclass(IOError, SampaiError)
        assert issubclass(IOError, Exception)
        assert IOError.__name__ == "IOError"


class TestExceptionInstances:
    """Test exception instance creation and catching."""

    def test_sampai_error_instance(self):
        """Test creating and catching SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise SampaiError("Test error")
        assert str(exc_info.value) == "Test error"

    def test_mesh_error_instance(self):
        """Test creating and catching MeshError."""
        with pytest.raises(MeshError) as exc_info:
            raise MeshError("Test mesh error")
        assert str(exc_info.value) == "Test mesh error"

    def test_field_error_instance(self):
        """Test creating and catching FieldError."""
        with pytest.raises(FieldError) as exc_info:
            raise FieldError("Test field error")
        assert str(exc_info.value) == "Test field error"

    def test_catch_mesh_error_as_sampai_error(self):
        """Test that MeshError can be caught as SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise MeshError("Test mesh error")
        assert isinstance(exc_info.value, MeshError)
        assert str(exc_info.value) == "Test mesh error"

    def test_catch_field_error_as_sampai_error(self):
        """Test that FieldError can be caught as SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise FieldError("Test field error")
        assert isinstance(exc_info.value, FieldError)
        assert str(exc_info.value) == "Test field error"

    def test_catch_config_error_as_sampai_error(self):
        """Test that ConfigError can be caught as SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise ConfigError("Test config error")
        assert isinstance(exc_info.value, ConfigError)
        assert str(exc_info.value) == "Test config error"

    def test_catch_operator_error_as_sampai_error(self):
        """Test that OperatorError can be caught as SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise OperatorError("Test operator error")
        assert isinstance(exc_info.value, OperatorError)
        assert str(exc_info.value) == "Test operator error"

    def test_catch_bc_error_as_sampai_error(self):
        """Test that BoundaryConditionError can be caught as SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise BoundaryConditionError("Test BC error")
        assert isinstance(exc_info.value, BoundaryConditionError)
        assert str(exc_info.value) == "Test BC error"

    def test_catch_adaptation_error_as_sampai_error(self):
        """Test that AdaptationError can be caught as SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise AdaptationError("Test adaptation error")
        assert isinstance(exc_info.value, AdaptationError)
        assert str(exc_info.value) == "Test adaptation error"

    def test_catch_io_error_as_sampai_error(self):
        """Test that IOError can be caught as SampaiError."""
        with pytest.raises(SampaiError) as exc_info:
            raise IOError("Test IO error")
        assert isinstance(exc_info.value, IOError)
        assert str(exc_info.value) == "Test IO error"

    def test_catch_any_sampai_error_as_exception(self):
        """Test that any SampaiError can be caught as Exception."""
        for exc_cls in [MeshError, FieldError, ConfigError, OperatorError,
                        BoundaryConditionError, AdaptationError, IOError]:
            with pytest.raises(Exception) as exc_info:
                raise exc_cls(f"Test {exc_cls.__name__}")
            assert isinstance(exc_info.value, SampaiError)

    def test_exception_isinstance_checks(self):
        """Test isinstance checks for all exceptions."""
        mesh_err = MeshError("test")
        field_err = FieldError("test")
        config_err = ConfigError("test")
        operator_err = OperatorError("test")
        bc_err = BoundaryConditionError("test")
        adapt_err = AdaptationError("test")
        io_err = IOError("test")

        # All should be instances of SampaiError
        assert isinstance(mesh_err, SampaiError)
        assert isinstance(field_err, SampaiError)
        assert isinstance(config_err, SampaiError)
        assert isinstance(operator_err, SampaiError)
        assert isinstance(bc_err, SampaiError)
        assert isinstance(adapt_err, SampaiError)
        assert isinstance(io_err, SampaiError)

        # All should be instances of Exception
        assert isinstance(mesh_err, Exception)
        assert isinstance(field_err, Exception)
        assert isinstance(config_err, Exception)
        assert isinstance(operator_err, Exception)
        assert isinstance(bc_err, Exception)
        assert isinstance(adapt_err, Exception)
        assert isinstance(io_err, Exception)

    def test_exception_does_not_inherit_from_wrong_type(self):
        """Test that exceptions don't inherit from each other."""
        mesh_err = MeshError("test")
        field_err = FieldError("test")

        # MeshError should not be instance of FieldError
        assert not isinstance(mesh_err, FieldError)
        assert not isinstance(field_err, MeshError)
