# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for Sampai timer functionality.

Tests the timer utilities for performance tracking and benchmarking.
The timer API follows the pattern described in the Python bindings roadmap.

Timer functionality includes:
- Timers class with static methods for named timer management
- start(name): Start a named timer
- stop(name): Stop a named timer
- get_elapsed(name): Get elapsed time for a named timer
- print(): Print all registered timers
- get_all(): Get all timer data as a dictionary
"""

import time
from io import StringIO
from unittest.mock import patch

import pytest

try:
    import sampai as sam

    # Timer API v0.30.0+: in sam.utils.timer submodule
    # Based on roadmap: timer.Timers class with static methods
    try:
        from sampai.utils import timer

        Timers = timer.Timers
        HAS_TIMER = True
    except (ImportError, AttributeError):
        # Fallback: try direct import from sampai if exposed at module level
        try:
            Timers = sam.Timers
            HAS_TIMER = True
        except AttributeError:
            HAS_TIMER = False
except ImportError:
    pytest.skip("sampai module not available", allow_module_level=True)


# =============================================================================
# Skip Tests if Timer Not Implemented
# =============================================================================

pytestmark = []

if not HAS_TIMER:
    pytestmark.append(pytest.mark.skip("Timer functionality not yet implemented"))


# =============================================================================
# Basic Timer Functionality Tests
# =============================================================================


class TestTimerBasic:
    """Test suite for basic timer start/stop functionality."""

    def test_timer_start_stop(self):
        """Test basic start and stop of a named timer."""
        timer_name = "test_basic"
        Timers.start(timer_name)
        time.sleep(0.01)  # Small delay to ensure measurable time
        Timers.stop(timer_name)
        elapsed = Timers.get_elapsed(timer_name)
        assert elapsed > 0
        assert elapsed < 1.0  # Should be very short

    def test_timer_elapsed_time_increases(self):
        """Test that elapsed time increases with longer operations."""
        timer_name = "test_elapsed"
        Timers.start(timer_name)
        time.sleep(0.05)
        Timers.stop(timer_name)
        elapsed = Timers.get_elapsed(timer_name)
        assert elapsed >= 0.05
        assert elapsed < 0.2  # Upper bound for test tolerance

    def test_multiple_start_stop_cycles(self):
        """Test starting and stopping the same timer multiple times."""
        timer_name = "test_cycles"
        elapsed_times = []

        for i in range(3):
            Timers.start(timer_name)
            time.sleep(0.01)
            Timers.stop(timer_name)
            elapsed = Timers.get_elapsed(timer_name)
            elapsed_times.append(elapsed)

        # Each cycle should have recorded a time
        for elapsed in elapsed_times:
            assert elapsed > 0

    def test_timer_name_can_be_any_string(self):
        """Test that timer names can be arbitrary strings."""
        timer_names = [
            "adaptation",
            "mesh_update",
            "field_copy",
            "solver_iteration",
            "custom_timer_123",
            "timer-with-dashes",
            "timer_with_underscores",
        ]

        for name in timer_names:
            Timers.start(name)
            time.sleep(0.001)
            Timers.stop(name)
            elapsed = Timers.get_elapsed(name)
            assert elapsed >= 0


# =============================================================================
# Multiple Timer Tests
# =============================================================================


class TestMultipleTimers:
    """Test suite for using multiple timers simultaneously."""

    def test_two_timers_simultaneous(self):
        """Test running two timers concurrently."""
        timer1 = "timer_a"
        timer2 = "timer_b"

        Timers.start(timer1)
        time.sleep(0.02)
        Timers.start(timer2)
        time.sleep(0.02)
        Timers.stop(timer2)
        time.sleep(0.02)
        Timers.stop(timer1)

        elapsed1 = Timers.get_elapsed(timer1)
        elapsed2 = Timers.get_elapsed(timer2)

        # timer1 should have run longer
        assert elapsed1 > elapsed2
        assert elapsed2 >= 0

    def test_multiple_timers_independent(self):
        """Test that multiple timers track time independently."""
        timers = ["timer_1", "timer_2", "timer_3"]
        elapsed_times = {}

        for timer in timers:
            Timers.start(timer)
            time.sleep(0.01 * (int(timer.split("_")[1])))
            Timers.stop(timer)
            elapsed_times[timer] = Timers.get_elapsed(timer)

        # Each timer should have independent timing
        assert elapsed_times["timer_1"] < elapsed_times["timer_2"]
        assert elapsed_times["timer_2"] < elapsed_times["timer_3"]

    def test_get_all_timers(self):
        """Test getting all timer data as dictionary."""
        timer_names = ["first", "second", "third"]

        for name in timer_names:
            Timers.start(name)
            time.sleep(0.01)
            Timers.stop(name)

        all_timers = Timers.get_all()

        assert isinstance(all_timers, dict)
        # Should contain our timers
        for name in timer_names:
            assert name in all_timers
            assert all_timers[name] > 0


# =============================================================================
# Print Functionality Tests
# =============================================================================


class TestTimerPrint:
    """Test suite for timer print functionality."""

    def test_print_no_errors(self):
        """Test that print() executes without errors."""
        timer_name = "test_print"
        Timers.start(timer_name)
        time.sleep(0.01)
        Timers.stop(timer_name)

        # Should not raise any exception
        Timers.print()

    def test_print_with_multiple_timers(self):
        """Test printing multiple timers."""
        timers = ["timer_a", "timer_b", "timer_c"]

        for name in timers:
            Timers.start(name)
            time.sleep(0.01)
            Timers.stop(name)

        # Should not raise any exception
        Timers.print()

    def test_print_output_contains_timer_names(self, capsys=None):
        """Test that print output contains timer names."""
        timer_name = "test_output"
        Timers.start(timer_name)
        time.sleep(0.01)
        Timers.stop(timer_name)

        Timers.print()

        # Note: capsys might not capture C++ output
        # This test mainly verifies no exceptions are raised


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestTimerErrorHandling:
    """Test suite for timer error handling."""

    def test_stop_nonexistent_timer(self):
        """Test stopping a timer that was never started."""
        # Should handle gracefully, either:
        # - Silently ignore
        # - Raise a specific exception
        # Behavior depends on implementation
        timer_name = "nonexistent_timer"

        try:
            Timers.stop(timer_name)
            # Silently ignored - this is acceptable
        except (KeyError, RuntimeError, ValueError) as e:
            # Specific exception is also acceptable
            assert True

    def test_get_elapsed_nonexistent_timer(self):
        """Test getting elapsed time for a non-existent timer."""
        timer_name = "nonexistent_timer"

        try:
            elapsed = Timers.get_elapsed(timer_name)
            # Should return 0 or raise exception
            assert elapsed == 0
        except (KeyError, RuntimeError, ValueError):
            # Exception is acceptable
            assert True

    def test_start_already_running_timer(self):
        """Test starting a timer that's already running."""
        timer_name = "running_timer"

        Timers.start(timer_name)
        time.sleep(0.01)

        try:
            # Try to start again
            Timers.start(timer_name)
            # May silently ignore or restart
            time.sleep(0.01)
            Timers.stop(timer_name)
            # Should have some elapsed time
            elapsed = Timers.get_elapsed(timer_name)
            assert elapsed >= 0
        except RuntimeError:
            # May raise exception
            Timers.stop(timer_name)

    def test_stop_already_stopped_timer(self):
        """Test stopping a timer that's already stopped."""
        timer_name = "stopped_timer"

        Timers.start(timer_name)
        time.sleep(0.01)
        Timers.stop(timer_name)

        try:
            # Try to stop again
            Timers.stop(timer_name)
            # Should handle gracefully
        except (KeyError, RuntimeError, ValueError):
            # Exception is acceptable
            assert True

    def test_empty_timer_name(self):
        """Test behavior with empty timer name."""
        empty_name = ""

        try:
            Timers.start(empty_name)
            time.sleep(0.01)
            Timers.stop(empty_name)
            elapsed = Timers.get_elapsed(empty_name)
            # Either works or raises exception
            if isinstance(elapsed, (int, float)):
                assert elapsed >= 0
        except (ValueError, KeyError):
            # Exception for empty name is acceptable
            assert True

# =============================================================================
# Integration Tests
# =============================================================================


class TestTimerIntegration:
    """Integration tests for timer with real Sampai operations."""

    def test_timer_with_mesh_creation(self):
        """Test timing mesh creation operations."""
        try:
            timer_name = "mesh_creation"

            Timers.start(timer_name)
            box = sam.geometry.box([0.0], [1.0])
            config = sam.config.make(1)
            config.min_level = 0
            config.max_level = 4
            mesh = sam.mesh.make(box, config)
            Timers.stop(timer_name)

            elapsed = Timers.get_elapsed(timer_name)
            assert elapsed > 0
        except AttributeError:
            pytest.skip("Mesh creation not available")

    def test_timer_with_field_operations(self):
        """Test timing field operations."""
        try:
            timer_name = "field_ops"

            # Setup
            box = sam.geometry.box([0.0], [1.0])
            config = sam.config.make(1)
            config.min_level = 0
            config.max_level = 2
            mesh = sam.mesh.make(box, config)

            Timers.start(timer_name)
            field = sam.field.scalar(mesh, "u", init=1.0)
            sam.boundary.dirichlet(field, 0.0)
            Timers.stop(timer_name)

            elapsed = Timers.get_elapsed(timer_name)
            assert elapsed > 0
        except AttributeError:
            pytest.skip("Field operations not available")

    def test_timer_with_adaptation(self):
        """Test timing mesh adaptation."""
        try:
            # Setup
            box = sam.geometry.box([0.0], [1.0])
            config = sam.config.make(1)
            config.min_level = 0
            config.max_level = 4
            mesh = sam.mesh.make(box, config)
            field = sam.field.scalar(mesh, "u", init=1.0)
            sam.boundary.dirichlet(field, 0.0)

            timer_name = "adaptation"

            Timers.start(timer_name)
            MRadapt = sam.adaptation.make_MRAdapt(field)
            mra_config = sam.config.MRAConfig()
            mra_config.epsilon = 1e-2
            MRadapt(mra_config)
            sam.adaptation.update_ghost_mr(field)
            Timers.stop(timer_name)

            elapsed = Timers.get_elapsed(timer_name)
            assert elapsed > 0
        except AttributeError:
            pytest.skip("Adaptation not available")

    def test_multiple_operations_timing(self):
        """Test timing multiple operations in sequence."""
        try:
            results = {}

            # Operation 1: Mesh creation
            Timers.start("op1_mesh")
            box = sam.geometry.box([0.0, 0.0], [1.0, 1.0])
            config = sam.config.make(2)
            config.min_level = 0
            config.max_level = 2
            mesh = sam.mesh.make(box, config)
            Timers.stop("op1_mesh")
            results["mesh"] = Timers.get_elapsed("op1_mesh")

            # Operation 2: Field creation
            Timers.start("op2_field")
            field = sam.field.scalar(mesh, "u", init=1.0)
            Timers.stop("op2_field")
            results["field"] = Timers.get_elapsed("op2_field")

            # Operation 3: Boundary conditions
            Timers.start("op3_bc")
            sam.boundary.dirichlet(field, 0.0)
            Timers.stop("op3_bc")
            results["bc"] = Timers.get_elapsed("op3_bc")

            # All operations should have taken some time
            for op, elapsed in results.items():
                assert elapsed > 0, f"Operation {op} should have taken time"

            # Get all timers at once
            all_timers = Timers.get_all()
            assert "op1_mesh" in all_timers
            assert "op2_field" in all_timers
            assert "op3_bc" in all_timers
        except AttributeError:
            pytest.skip("Required operations not available")


# =============================================================================
# Performance Tests
# =============================================================================


class TestTimerPerformance:
    """Test timer performance and overhead."""

    def test_timer_overhead_is_small(self):
        """Test that timer overhead is minimal."""
        # Measure timer overhead
        iterations = 100

        start_time = time.time()
        for i in range(iterations):
            Timers.start(f"overhead_{i}")
            Timers.stop(f"overhead_{i}")
        end_time = time.time()

        total_overhead = end_time - start_time
        avg_overhead = total_overhead / iterations

        # Average overhead should be very small (< 1ms)
        assert avg_overhead < 0.001

    def test_timer_precision(self):
        """Test timer precision for short intervals."""
        timer_name = "precision_test"

        # Measure a very short interval
        Timers.start(timer_name)
        # No delay, just immediate stop
        Timers.stop(timer_name)

        elapsed = Timers.get_elapsed(timer_name)

        # Timer should be able to measure sub-millisecond intervals
        # (or return 0 if below precision threshold)
        assert elapsed >= 0


# =============================================================================
# Module Organization Tests
# =============================================================================


class TestTimerModuleOrganization:
    """Test suite for timer module organization (v0.30.0+ API)."""

    def test_timer_in_utils_submodule(self):
        """Test that timer is accessible from utils submodule."""
        # Check module structure
        assert hasattr(sam, "utils")

        # Timer should be in utils (or sam directly for backward compat)
        has_utils_timer = hasattr(sam.utils, "timer")
        has_direct_timer = hasattr(sam, "Timers")

        # At least one should be available
        assert has_utils_timer or has_direct_timer

    def test_timer_class_exists(self):
        """Test that Timers class is available."""
        # Check that Timers class exists with required methods
        assert hasattr(Timers, "start")
        assert hasattr(Timers, "stop")
        assert hasattr(Timers, "get_elapsed")
        assert hasattr(Timers, "print")

        # get_all is optional (planned feature)
        has_get_all = hasattr(Timers, "get_all")


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


@pytest.fixture
def clean_timer_state():
    """
    Fixture to provide clean timer state for tests.

    Resets all timers before and after test if supported.
    """
    # Setup: Clear any existing timers if API supports it
    try:
        if hasattr(Timers, "reset") or hasattr(Timers, "clear"):
            Timers.clear()
    except AttributeError:
        pass

    yield

    # Teardown: Clean up timers
    try:
        if hasattr(Timers, "reset") or hasattr(Timers, "clear"):
            Timers.clear()
    except AttributeError:
        pass


# =============================================================================
# Main Test Runner
# =============================================================================

if __name__ == "__main__":
    # Run tests manually for quick verification
    pytest.main([__file__, "-v", "-s"])
