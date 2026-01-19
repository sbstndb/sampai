# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for timer bindings.
"""

import pytest
import time


def test_timers_basic():
    """Test basic timer functionality."""
    import sampai as sam

    # Test start and stop
    sam.Timers.start("test_timer")
    time.sleep(0.1)  # Sleep for 100ms
    sam.Timers.stop("test_timer")

    # Test get_elapsed
    elapsed = sam.Timers.get_elapsed("test_timer")
    assert elapsed >= 0.1, f"Expected at least 0.1s, got {elapsed}s"
    assert elapsed < 0.5, f"Expected less than 0.5s, got {elapsed}s"


def test_timers_multiple():
    """Test multiple timers."""
    import sampai as sam

    # Create multiple timers
    sam.Timers.start("timer1")
    time.sleep(0.05)
    sam.Timers.stop("timer1")

    sam.Timers.start("timer2")
    time.sleep(0.05)
    sam.Timers.stop("timer2")

    # Check both timers
    elapsed1 = sam.Timers.get_elapsed("timer1")
    elapsed2 = sam.Timers.get_elapsed("timer2")

    assert elapsed1 >= 0.05, f"Timer1: expected at least 0.05s, got {elapsed1}s"
    assert elapsed2 >= 0.05, f"Timer2: expected at least 0.05s, got {elapsed2}s"


def test_timers_accumulate():
    """Test that timers accumulate time across multiple start/stop cycles."""
    import sampai as sam

    # Run timer twice
    sam.Timers.start("accumulate")
    time.sleep(0.05)
    sam.Timers.stop("accumulate")

    sam.Timers.start("accumulate")
    time.sleep(0.05)
    sam.Timers.stop("accumulate")

    elapsed = sam.Timers.get_elapsed("accumulate")
    assert elapsed >= 0.1, f"Expected at least 0.1s (2 x 0.05s), got {elapsed}s"


def test_timers_print():
    """Test that print() works without errors."""
    import sampai as sam

    # Create a timer
    sam.Timers.start("print_test")
    time.sleep(0.05)
    sam.Timers.stop("print_test")

    # Print timers - just verify it doesn't raise an exception
    # Note: Output goes to C++ stdout, not captured by Python's capsys
    sam.Timers.print()


def test_timers_get_all():
    """Test get_all() returns all timer data."""
    import sampai as sam

    # Create multiple timers
    sam.Timers.start("first")
    time.sleep(0.02)
    sam.Timers.stop("first")

    sam.Timers.start("second")
    time.sleep(0.02)
    sam.Timers.stop("second")

    # Get all timers
    all_timers = sam.Timers.get_all()

    # Should contain our timers
    assert isinstance(all_timers, dict)
    assert "first" in all_timers
    assert "second" in all_timers
    assert all_timers["first"] >= 0.02
    assert all_timers["second"] >= 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
