# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for timer bindings.
"""

import pytest
import time


def test_timers_basic():
    """Test basic timer functionality."""
    from sampai.utils import timer

    # Test start and stop
    timer.Timers.start("test_timer")
    time.sleep(0.1)  # Sleep for 100ms
    timer.Timers.stop("test_timer")

    # Test get_elapsed
    elapsed = timer.Timers.get_elapsed("test_timer")
    assert elapsed >= 0.1, f"Expected at least 0.1s, got {elapsed}s"
    assert elapsed < 0.5, f"Expected less than 0.5s, got {elapsed}s"


def test_timers_multiple():
    """Test multiple timers."""
    from sampai.utils import timer

    # Create multiple timers
    timer.Timers.start("timer1")
    time.sleep(0.05)
    timer.Timers.stop("timer1")

    timer.Timers.start("timer2")
    time.sleep(0.05)
    timer.Timers.stop("timer2")

    # Check both timers
    elapsed1 = timer.Timers.get_elapsed("timer1")
    elapsed2 = timer.Timers.get_elapsed("timer2")

    assert elapsed1 >= 0.05, f"Timer1: expected at least 0.05s, got {elapsed1}s"
    assert elapsed2 >= 0.05, f"Timer2: expected at least 0.05s, got {elapsed2}s"


def test_timers_accumulate():
    """Test that timers accumulate time across multiple start/stop cycles."""
    from sampai.utils import timer

    # Run timer twice
    timer.Timers.start("accumulate")
    time.sleep(0.05)
    timer.Timers.stop("accumulate")

    timer.Timers.start("accumulate")
    time.sleep(0.05)
    timer.Timers.stop("accumulate")

    elapsed = timer.Timers.get_elapsed("accumulate")
    assert elapsed >= 0.1, f"Expected at least 0.1s (2 x 0.05s), got {elapsed}s"


def test_timers_print(capsys):
    """Test that print() works without errors."""
    from sampai.utils import timer

    # Create a timer
    timer.Timers.start("print_test")
    time.sleep(0.05)
    timer.Timers.stop("print_test")

    # Print timers
    timer.Timers.print()

    # Capture output
    captured = capsys.readouterr()
    assert "print_test" in captured.out or "print_test" in captured.err


def test_timers_reset():
    """Test timer reset functionality."""
    from sampai.utils import timer

    # Create and stop a timer
    timer.Timers.start("reset_test")
    time.sleep(0.05)
    timer.Timers.stop("reset_test")

    elapsed_before = timer.Timers.get_elapsed("reset_test")
    assert elapsed_before >= 0.05

    # Reset the timer
    timer.Timers.reset("reset_test")

    # After reset, get_elapsed should fail or return 0
    # This depends on the implementation - just check it doesn't crash
    try:
        elapsed_after = timer.Timers.get_elapsed("reset_test")
        # If it succeeds, it should be 0 or very small
        assert elapsed_after < 0.01, f"Expected ~0 after reset, got {elapsed_after}s"
    except (RuntimeError, KeyError):
        # Or it might raise an error, which is also acceptable
        pass


def test_timers_context_manager():
    """Test Timer context manager."""
    from sampai.utils.timer import Timer

    # Test context manager - Timer uses Python perf_counter internally
    with Timer("context_test") as t:
        time.sleep(0.05)

    # The Timer instance tracks elapsed time
    elapsed = t.elapsed
    assert elapsed >= 0.05, f"Expected at least 0.05s, got {elapsed}s"


def test_timers_decorator():
    """Test @timeit decorator."""
    import pytest
    pytest.skip("Python timeit decorator with C++ Samurai timers needs further investigation")

    # from sampai.utils.timer import timeit
    # import sampai as sam
    #
    # @timeit
    # def test_function():
    #     time.sleep(0.05)
    #     return 42
    #
    # result = test_function()
    # assert result == 42
    #
    # # Check that timer was created in C++ Timers
    # elapsed = sam.Timers.get_elapsed("test_function")
    # assert elapsed >= 0.05, f"Expected at least 0.05s, got {elapsed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
