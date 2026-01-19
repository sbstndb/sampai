# Copyright 2026 Sebastien Dubois (sbstndbs)
# SPDX-License-Identifier: Apache-2.0

"""
Timer utilities for Samurai simulations.

This module provides convenient timing utilities for performance profiling
and benchmarking of adaptive mesh refinement simulations.

The module provides both a low-level API for manual timing and high-level
context managers for automatic timing of code blocks.

Example usage:
    >>> from sampai.utils import timer
    >>>
    >>> # Context manager for timing code blocks
    >>> with timer.Timer("mesh_adaptation"):
    ...     MRadaptation(config)
    >>>
    >>> # Manual timing with start/stop
    >>> timer.Timers.start("solve")
    >>> solve_system()
    >>> timer.Timers.stop("solve")
    >>>
    >>> # Get elapsed time
    >>> elapsed = timer.Timers.get_elapsed("solve")
    >>> print(f"Solve took {elapsed:.3f}s")
    >>>
    >>> # Print all timers
    >>> timer.Timers.print()
    >>>
    >>> # Helper for timing functions
    >>> result, elapsed = timer.time_function(my_function, arg1, arg2)
"""

import sys
import time
from contextlib import contextmanager
from typing import Dict, Optional, Callable, Any


class Timers:
    """Timer management class wrapping C++ timer functionality.

    This class provides a static interface for managing named timers.
    It automatically uses C++ timers if available (via the _sampai module),
    or falls back to a pure Python implementation.

    The class supports creating, starting, stopping, and querying timers
    by name, making it easy to track performance of different parts of
    a simulation.

    Example:
        >>> timer.Timers.start("adaptation")
        >>> MRadaptation(config)
        >>> timer.Timers.stop("adaptation")
        >>> elapsed = timer.Timers.get_elapsed("adaptation")
    """

    # Try to import C++ timers
    _cpp_timers: Optional[Any] = None
    _use_cpp: bool = False

    # Python fallback storage
    _python_timers: Dict[str, Dict[str, float]] = {}

    @classmethod
    def _init_cpp_timers(cls) -> None:
        """Initialize C++ timer bindings if available.

        This method attempts to import the C++ timer bindings from the
        _sampai module. If successful, all timing operations will use
        the high-precision C++ implementation.
        """
        if cls._cpp_timers is not None:
            return

        # Disable C++ timers for now - they have stability issues
        cls._use_cpp = False
        return

        try:
            import _sampai
            if hasattr(_sampai, 'Timers'):
                cls._cpp_timers = _sampai.Timers
                cls._use_cpp = True
        except (ImportError, AttributeError):
            cls._use_cpp = False

    @classmethod
    def start(cls, name: str) -> None:
        """Start a named timer.

        If the timer doesn't exist, it will be created. If it already exists
        and is running, this will reset the start time.

        Args:
            name: Name of the timer to start

        Example:
            >>> Timers.start("mesh_adaptation")
            >>> MRadaptation(config)
            >>> Timers.stop("mesh_adaptation")
        """
        cls._init_cpp_timers()

        if cls._use_cpp and cls._cpp_timers is not None:
            cls._cpp_timers.start(name)
        else:
            # Python fallback
            if name not in cls._python_timers:
                cls._python_timers[name] = {'start': 0.0, 'elapsed': 0.0, 'running': False}
            cls._python_timers[name]['start'] = time.perf_counter()
            cls._python_timers[name]['running'] = True

    @classmethod
    def stop(cls, name: str) -> float:
        """Stop a named timer and return elapsed time.

        Args:
            name: Name of the timer to stop

        Returns:
            Elapsed time in seconds since the timer was started

        Raises:
            KeyError: If the timer doesn't exist

        Example:
            >>> Timers.start("computation")
            >>> perform_computation()
            >>> elapsed = Timers.stop("computation")
            >>> print(f"Computation took {elapsed:.3f}s")
        """
        cls._init_cpp_timers()

        if cls._use_cpp and cls._cpp_timers is not None:
            try:
                cls._cpp_timers.stop(name)
                return cls.get_elapsed(name)
            except (KeyError, RuntimeError, AttributeError):
                # C++ timer might not exist or handle errors differently
                # Fall through to Python implementation
                pass

        # Python fallback
        if name not in cls._python_timers:
            raise KeyError(f"Timer '{name}' not found")

        timer_data = cls._python_timers[name]
        if not timer_data['running']:
            raise RuntimeError(f"Timer '{name}' is not running")

        elapsed = time.perf_counter() - timer_data['start']
        timer_data['elapsed'] += elapsed
        timer_data['running'] = False
        return elapsed

    @classmethod
    def get_elapsed(cls, name: str) -> float:
        """Get the elapsed time for a named timer.

        Returns the total accumulated time for the timer. If the timer
        is currently running, returns the time elapsed so far plus any
        previously accumulated time.

        Args:
            name: Name of the timer

        Returns:
            Elapsed time in seconds

        Raises:
            KeyError: If the timer doesn't exist

        Example:
            >>> Timers.start("solver")
            >>> solve_system()
            >>> print(f"Solver: {Timers.get_elapsed('solver'):.3f}s")
        """
        cls._init_cpp_timers()

        if cls._use_cpp and cls._cpp_timers is not None:
            try:
                return cls._cpp_timers.get_elapsed(name)
            except (KeyError, RuntimeError, AttributeError):
                # C++ timer might not exist or handle errors differently
                # Fall through to Python implementation
                pass

        # Python fallback
        if name not in cls._python_timers:
            raise KeyError(f"Timer '{name}' not found")

        timer_data = cls._python_timers[name]
        if timer_data['running']:
            current_elapsed = time.perf_counter() - timer_data['start']
            return timer_data['elapsed'] + current_elapsed
        return timer_data['elapsed']

    @classmethod
    def get_all(cls) -> Dict[str, float]:
        """Get all timers and their elapsed times.

        Returns a dictionary mapping timer names to their accumulated
        elapsed times in seconds.

        Returns:
            Dictionary of timer names to elapsed times

        Example:
            >>> timers = Timers.get_all()
            >>> for name, elapsed in sorted(timers.items()):
            ...     print(f"{name}: {elapsed:.3f}s")
        """
        cls._init_cpp_timers()

        if cls._use_cpp and cls._cpp_timers is not None:
            return cls._cpp_timers.get_all()
        else:
            # Python fallback
            result = {}
            for name, timer_data in cls._python_timers.items():
                result[name] = cls.get_elapsed(name)
            return result

    @classmethod
    def print(cls, file=None) -> None:
        """Print all timers and their elapsed times.

        Args:
            file: File object to write to (default: sys.stdout)

        Example:
            >>> Timers.print()
            Timer            Time (s)    Percentage
            ----------------------------------------------
            mesh_adaptation  1.234       45.2%
            solve            1.001       36.7%
            io               0.496       18.2%
        """
        cls._init_cpp_timers()

        if cls._use_cpp and cls._cpp_timers is not None:
            cls._cpp_timers.print()
        else:
            # Python fallback
            if file is None:
                file = sys.stdout

            timers = cls.get_all()
            if not timers:
                print("No timers recorded", file=file)
                return

            # Calculate total time
            total = sum(timers.values())

            # Print header
            print(f"\n{'Timer':<20} {'Time (s)':>12} {'Percentage':>12}", file=file)
            print("-" * 46, file=file)

            # Print each timer
            for name, elapsed in sorted(timers.items()):
                percentage = (elapsed / total * 100) if total > 0 else 0
                print(f"{name:<20} {elapsed:>12.3f} {percentage:>11.1f}%", file=file)

            print("-" * 46, file=file)
            print(f"{'Total':<20} {total:>12.3f} {100.0:>11.1f}%", file=file)


class Timer:
    """Context manager for timing code blocks.

    This is the most convenient way to time a block of code. The timer
    automatically starts when entering the context and stops when exiting.

    Args:
        name: Name for the timer (used in reports)
        silent: If True, don't print timing info on exit

    Example:
        >>> with Timer("mesh_adaptation"):
        ...     MRadaptation(config)
        >>> # Timer automatically stopped here
        >>>
        >>> # Silent mode (no automatic printing)
        >>> with Timer("computation", silent=True):
        ...     result = expensive_computation()
        >>> elapsed = Timers.get_elapsed("computation")
    """

    def __init__(self, name: str, silent: bool = False):
        """Initialize timer context manager.

        Args:
            name: Name for the timer
            silent: If True, don't print timing info on exit
        """
        self.name = name
        self.silent = silent
        self._elapsed: Optional[float] = None

    def __enter__(self) -> 'Timer':
        """Enter context and start timer.

        Returns:
            self, allowing access to timer properties
        """
        Timers.start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and stop timer.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised
        """
        self._elapsed = Timers.stop(self.name)
        if not self.silent:
            print(f"{self.name}: {self._elapsed:.3f}s")

    @property
    def elapsed(self) -> Optional[float]:
        """Get the elapsed time for this timer.

        Returns None if the timer is still running.

        Returns:
            Elapsed time in seconds, or None if timer is running
        """
        return self._elapsed


@contextmanager
def timer_context(name: str, silent: bool = False):
    """Context manager function for timing code blocks.

    This is an alternative to the Timer class that uses a functional style.
    Both approaches are equivalent; choose the one you prefer.

    Args:
        name: Name for the timer
        silent: If True, don't print timing info on exit

    Yields:
        Timer object that can be queried for elapsed time

    Example:
        >>> with timer_context("adaptation"):
        ...     MRadaptation(config)
        >>>
        >>> # Alternative style
        >>> with timer_context("solve", silent=True) as t:
        ...     solve_system()
        >>> print(f"Solve took {t.elapsed:.3f}s")
    """
    with Timer(name, silent=silent) as t:
        yield t


def time_function(func: Callable, *args, name: Optional[str] = None, **kwargs) -> tuple:
    """Time a single function call.

    This is a convenience function for timing a single function call
    without modifying the function itself.

    Args:
        func: Function to call
        *args: Positional arguments to pass to function
        name: Name for the timer (default: function name)
        **kwargs: Keyword arguments to pass to function

    Returns:
        Tuple of (result, elapsed_time)

    Example:
        >>> result, elapsed = time_function(expensive_computation, arg1, arg2)
        >>> print(f"Result: {result}, took {elapsed:.3f}s")
        >>>
        >>> # With custom name
        >>> result, elapsed = time_function(
        ...     solve_system,
        ...     name="linear_solver"
        ... )
    """
    timer_name = name or func.__name__
    with Timer(timer_name, silent=True) as t:
        result = func(*args, **kwargs)
    return result, t.elapsed


def benchmark(
    func: Callable,
    *args,
    repeat: int = 10,
    warmup: int = 3,
    name: Optional[str] = None,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function with multiple runs.

    Runs the function multiple times and returns statistics about the
    execution time. This is useful for understanding the variability
    in execution time.

    Args:
        func: Function to benchmark
        *args: Positional arguments to pass to function
        repeat: Number of times to run the function (including warmup)
        warmup: Number of warmup runs (not included in statistics)
        name: Name for the benchmark (default: function name)
        **kwargs: Keyword arguments to pass to function

    Returns:
        Dictionary with statistics: min, max, mean, median, std

    Example:
        >>> stats = benchmark(
        ...     solve_system,
        ...     repeat=10,
        ...     warmup=2
        ... )
        >>> print(f"Mean: {stats['mean']:.3f}s")
        >>> print(f"Std:  {stats['std']:.3f}s")
    """
    import statistics

    timer_name = name or func.__name__
    times = []

    for i in range(repeat):
        with Timer(f"{timer_name}_bench", silent=True) as t:
            result = func(*args, **kwargs)

        if i >= warmup:
            times.append(t.elapsed)

    if not times:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0, 'std': 0.0}

    return {
        'min': min(times),
        'max': max(times),
        'mean': statistics.mean(times),
        'median': statistics.median(times),
        'std': statistics.stdev(times) if len(times) > 1 else 0.0
    }
