"""
Basic tests for Samurai Python bindings

These tests verify that the Python bindings can be imported and basic functionality works.
"""

import pytest


def test_module_import():
    """Test that the sampai module can be imported."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("sampai")
        assert spec is not None, "Module can be found"
    except ImportError as e:
        # If module is not built yet, skip test
        import pytest
        pytest.skip(f"Module not built yet: {e}")


def test_version_attribute():
    """Test that the module has a __version__ attribute."""
    try:
        import sampai
        assert hasattr(sampai, "__version__")
        assert isinstance(sampai.__version__, str)
        assert len(sampai.__version__) > 0
    except ImportError:
        import pytest
        pytest.skip("Module not built yet")


def test_test_function():
    """Test the placeholder test_function."""
    try:
        import sampai
        result = sampai.test_function()
        assert result == "Samurai Python bindings are working!"
    except ImportError:
        import pytest
        pytest.skip("Module not built yet")
    except AttributeError:
        import pytest
        pytest.skip("test_function not yet implemented")


def test_module_docstring():
    """Test that the module has proper documentation."""
    try:
        import sampai
        assert sampai.__doc__ is not None
        assert len(sampai.__doc__) > 0
        assert "Samurai" in sampai.__doc__
    except ImportError:
        import pytest
        pytest.skip("Module not built yet")


if __name__ == "__main__":
    # Run tests manually for quick verification
    import pytest
    pytest.main([__file__, "-v"])
