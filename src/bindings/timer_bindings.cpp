// Copyright 2026 Sebastien Dubois (sbstndbs)
// SPDX-License-Identifier: Apache-2.0

// Samurai Python Bindings - Timer utilities
//
// Bindings for samurai::times::timers global instance
// Provides timing functionality for performance profiling and benchmarking

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samurai/timers.hpp>
#include <chrono>
#include <map>
#include <set>
#include <mutex>

namespace py = pybind11;

// ============================================================
// Global timer name tracking
// ============================================================
// Since Samurai C++ doesn't expose a way to get all timers,
// we track timer names manually in the bindings.

static std::set<std::string> g_timer_names;
static std::mutex g_timer_names_mutex;

// ============================================================
// Helper functions to handle MPI/non-MPI return type differences
// ============================================================

#ifdef SAMURAI_WITH_MPI
    // In MPI mode, getElapsedTime() returns double (seconds)
    inline double get_elapsed_seconds(const std::string& name)
    {
        return samurai::times::timers.getElapsedTime(name);
    }
#else
    // In non-MPI mode, getElapsedTime() returns microseconds, convert to seconds
    inline double get_elapsed_seconds(const std::string& name)
    {
        auto microseconds = samurai::times::timers.getElapsedTime(name);
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(microseconds);
        return duration.count();
    }
#endif

// ============================================================
// Timers wrapper class
// ============================================================
// Note: We create a wrapper class because samurai::times::timers is a global
// instance and we can't directly bind it. The wrapper provides static methods
// that delegate to the global instance.

class PyTimers
{
public:
    static void start(const std::string& name)
    {
        try
        {
            // Track timer name for get_all()
            {
                std::lock_guard<std::mutex> lock(g_timer_names_mutex);
                g_timer_names.insert(name);
            }
            samurai::times::timers.start(name);
        }
        catch (const std::exception& e)
        {
            std::string msg = "Failed to start timer '" + name + "': " + e.what();
            throw std::runtime_error(msg);
        }
    }

    static void stop(const std::string& name)
    {
        try
        {
            samurai::times::timers.stop(name);
        }
        catch (const std::exception& e)
        {
            std::string msg = "Failed to stop timer '" + name + "': " + e.what();
            throw std::runtime_error(msg);
        }
    }

    static double get_elapsed(const std::string& name)
    {
        try
        {
            return get_elapsed_seconds(name);
        }
        catch (const std::exception& e)
        {
            std::string msg = "Failed to get elapsed time for '" + name + "': " + e.what();
            throw std::runtime_error(msg);
        }
    }

    static void print()
    {
        try
        {
            samurai::times::timers.print();
        }
        catch (const std::exception& e)
        {
            std::string msg = "Failed to print timers: " + std::string(e.what());
            throw std::runtime_error(msg);
        }
    }

    static std::map<std::string, double> get_all()
    {
        std::map<std::string, double> result;

        // Get all tracked timer names
        std::lock_guard<std::mutex> lock(g_timer_names_mutex);

        for (const auto& name : g_timer_names)
        {
            try
            {
                // Try to get elapsed time for each tracked timer
                // Note: This may throw if timer was never stopped
                result[name] = get_elapsed_seconds(name);
            }
            catch (...)
            {
                // Timer might exist but never stopped - skip it
                // or set to 0 if preferred
                result[name] = 0.0;
            }
        }

        return result;
    }
};

// ============================================================
// Module initialization function for Timer bindings
// ============================================================

void init_timer_bindings(py::module_& m)
{
    // ============================================================
    // Bind Timers class with static methods
    // ============================================================
    py::class_<PyTimers>(m, "Timers", R"pbdoc(
        Timer management class for Samurai AMR simulations.

        This class provides a static interface for managing named timers.
        Use it to track performance of different parts of a simulation.

        Timers are global and can be accessed from anywhere in the code.

        Examples
        --------
        >>> import sampai as sam
        >>> # Start a timer
        >>> sam.Timers.start('adaptation')
        >>> # ... do work ...
        >>> sam.Timers.stop('adaptation')
        >>> # Get elapsed time in seconds
        >>> elapsed = sam.Timers.get_elapsed('adaptation')
        >>> print(f'Adaptation took {elapsed:.3f} seconds')
        >>> # Print all timers
        >>> sam.Timers.print()

        Notes
        -----
        - All methods are static, call them on the class itself
        - Timers are global and persist across the entire application
        - In MPI mode, print() shows statistics across all processes
    )pbdoc")

        .def_static("start", &PyTimers::start, py::arg("name"), R"pbdoc(
            Start a named timer.

            If the timer does not exist, it will be created. If it already exists,
            it will be reset and started again.

            Parameters
            ----------
            name : str
                Name of the timer to start

            Examples
            --------
            >>> import sampai as sam
            >>> sam.Timers.start('adaptation')

            Notes
            -----
            - Timers are global and can be accessed from anywhere in the code
            - Starting an existing timer resets it
            - Always call stop() after start() to record the elapsed time
        )pbdoc")

        .def_static("stop", &PyTimers::stop, py::arg("name"), R"pbdoc(
            Stop a named timer and record the elapsed time.

            The timer must have been previously started with start().

            Parameters
            ----------
            name : str
                Name of the timer to stop

            Raises
            ------
            RuntimeError
                If the timer has not been started

            Examples
            --------
            >>> import sampai as sam
            >>> sam.Timers.start('adaptation')
            >>> # ... do work ...
            >>> sam.Timers.stop('adaptation')

            Notes
            -----
            - Stopping a timer accumulates the elapsed time
            - Multiple start/stop cycles can be used to measure cumulative time
        )pbdoc")

        .def_static("get_elapsed", &PyTimers::get_elapsed, py::arg("name"), R"pbdoc(
            Get the elapsed time for a named timer in seconds.

            The timer must have been started and stopped at least once.

            Parameters
            ----------
            name : str
                Name of the timer

            Returns
            -------
            float
                Elapsed time in seconds

            Raises
            ------
            RuntimeError
                If the timer has not been started

            Examples
            --------
            >>> import sampai as sam
            >>> sam.Timers.start('adaptation')
            >>> # ... do work ...
            >>> sam.Timers.stop('adaptation')
            >>> elapsed = sam.Timers.get_elapsed('adaptation')
            >>> print(f'Adaptation took {elapsed:.3f} seconds')

            Notes
            -----
            - Returns the accumulated time across all start/stop cycles
            - Time is measured in seconds with microsecond precision
        )pbdoc")

        .def_static("print", &PyTimers::print, R"pbdoc(
            Print all timers to stdout.

            Displays a formatted table of all timers with their elapsed times.
            In MPI mode, shows min/max/average/std deviation across all processes.
            In non-MPI mode, shows elapsed time and percentage of total.

            Examples
            --------
            >>> import sampai as sam
            >>> sam.Timers.start('adaptation')
            >>> # ... do work ...
            >>> sam.Timers.stop('adaptation')
            >>> sam.Timers.start('prediction')
            >>> # ... do work ...
            >>> sam.Timers.stop('prediction')
            >>> sam.Timers.print()

            Notes
            -----
            - Timers are sorted by elapsed time (descending)
            - In MPI mode, only rank 0 prints the results
            - The 'total runtime' timer (if present) is used to calculate percentages
        )pbdoc")

        .def_static("get_all", &PyTimers::get_all, R"pbdoc(
            Get all timers and their elapsed times.

            Returns a dictionary mapping timer names to their accumulated
            elapsed times in seconds. Only includes timers that have been
            started via this Python interface.

            Returns
            -------
            dict
                Dictionary of timer names to elapsed times

            Examples
            --------
            >>> import sampai as sam
            >>> sam.Timers.start('adaptation')
            >>> # ... do work ...
            >>> sam.Timers.stop('adaptation')
            >>> sam.Timers.start('prediction')
            >>> # ... do work ...
            >>> sam.Timers.stop('prediction')
            >>> timers = sam.Timers.get_all()
            >>> for name, elapsed in sorted(timers.items()):
            ...     print(f"{name}: {elapsed:.3f}s")

            Notes
            -----
            - Only includes timers started through this Python interface
            - Timers that were started but not stopped will show as 0.0
            - For a formatted table output, use print() instead
        )pbdoc");
}
