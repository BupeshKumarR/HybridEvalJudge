"""
Profiling utilities for performance optimization.

This module provides utilities for profiling code execution, identifying
bottlenecks, and tracking performance metrics.
"""

import cProfile
import functools
import io
import logging
import pstats
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """
    Result from profiling a code block or function.

    Attributes:
        name: Name of the profiled operation
        duration: Total duration in seconds
        call_count: Number of times the operation was called
        cumulative_time: Cumulative time including subcalls
        stats: Detailed profiling statistics
    """

    name: str
    duration: float
    call_count: int = 1
    cumulative_time: float = 0.0
    stats: Optional[str] = None


class Profiler:
    """
    Profiler for tracking performance metrics and identifying bottlenecks.

    This class provides utilities for profiling code execution with support
    for both manual timing and cProfile-based detailed profiling.

    Example:
        >>> profiler = Profiler()
        >>> with profiler.profile("model_loading"):
        ...     model = load_model()
        >>> print(profiler.get_summary())
    """

    def __init__(self):
        """Initialize the Profiler."""
        self._results: Dict[str, List[ProfileResult]] = {}
        self._active_timers: Dict[str, float] = {}

    @contextmanager
    def profile(self, name: str, detailed: bool = False):
        """
        Context manager for profiling a code block.

        Args:
            name: Name of the operation being profiled
            detailed: If True, use cProfile for detailed profiling

        Yields:
            None

        Example:
            >>> profiler = Profiler()
            >>> with profiler.profile("data_processing"):
            ...     process_data()
        """
        if detailed:
            # Use cProfile for detailed profiling
            profiler = cProfile.Profile()
            profiler.enable()

        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time

            if detailed:
                profiler.disable()

                # Capture stats
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s)
                ps.sort_stats("cumulative")
                ps.print_stats(20)  # Top 20 functions
                stats_str = s.getvalue()
            else:
                stats_str = None

            # Record result
            result = ProfileResult(
                name=name,
                duration=duration,
                cumulative_time=duration,
                stats=stats_str,
            )

            if name not in self._results:
                self._results[name] = []
            self._results[name].append(result)

            logger.debug(f"Profiled '{name}': {duration:.3f}s")

    def start_timer(self, name: str):
        """
        Start a named timer.

        Args:
            name: Name of the timer

        Example:
            >>> profiler = Profiler()
            >>> profiler.start_timer("inference")
            >>> # ... do work ...
            >>> profiler.stop_timer("inference")
        """
        self._active_timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and record the result.

        Args:
            name: Name of the timer

        Returns:
            Duration in seconds

        Raises:
            ValueError: If timer was not started

        Example:
            >>> profiler = Profiler()
            >>> profiler.start_timer("inference")
            >>> # ... do work ...
            >>> duration = profiler.stop_timer("inference")
            >>> print(f"Inference took {duration:.3f}s")
        """
        if name not in self._active_timers:
            raise ValueError(f"Timer '{name}' was not started")

        start_time = self._active_timers.pop(name)
        duration = time.time() - start_time

        # Record result
        result = ProfileResult(
            name=name,
            duration=duration,
            cumulative_time=duration,
        )

        if name not in self._results:
            self._results[name] = []
        self._results[name].append(result)

        return duration

    def get_results(self, name: Optional[str] = None) -> Dict[str, List[ProfileResult]]:
        """
        Get profiling results.

        Args:
            name: Optional name to filter results

        Returns:
            Dictionary mapping operation names to lists of ProfileResult objects

        Example:
            >>> profiler = Profiler()
            >>> # ... run profiled operations ...
            >>> results = profiler.get_results("model_loading")
            >>> for result in results["model_loading"]:
            ...     print(f"Duration: {result.duration:.3f}s")
        """
        if name:
            return {name: self._results.get(name, [])}
        return self._results.copy()

    def get_summary(self) -> str:
        """
        Get a human-readable summary of profiling results.

        Returns:
            Formatted string with profiling statistics

        Example:
            >>> profiler = Profiler()
            >>> # ... run profiled operations ...
            >>> print(profiler.get_summary())
        """
        if not self._results:
            return "No profiling data available"

        lines = ["Profiling Summary", "=" * 80]

        for name, results in sorted(self._results.items()):
            total_time = sum(r.duration for r in results)
            avg_time = total_time / len(results)
            min_time = min(r.duration for r in results)
            max_time = max(r.duration for r in results)

            lines.append(f"\n{name}:")
            lines.append(f"  Calls: {len(results)}")
            lines.append(f"  Total: {total_time:.3f}s")
            lines.append(f"  Average: {avg_time:.3f}s")
            lines.append(f"  Min: {min_time:.3f}s")
            lines.append(f"  Max: {max_time:.3f}s")

        return "\n".join(lines)

    def get_bottlenecks(self, top_n: int = 5) -> List[tuple[str, float]]:
        """
        Identify the top bottlenecks by total time.

        Args:
            top_n: Number of top bottlenecks to return

        Returns:
            List of (operation_name, total_time) tuples sorted by time

        Example:
            >>> profiler = Profiler()
            >>> # ... run profiled operations ...
            >>> bottlenecks = profiler.get_bottlenecks(3)
            >>> for name, time in bottlenecks:
            ...     print(f"{name}: {time:.3f}s")
        """
        totals = []
        for name, results in self._results.items():
            total_time = sum(r.duration for r in results)
            totals.append((name, total_time))

        # Sort by time descending
        totals.sort(key=lambda x: x[1], reverse=True)

        return totals[:top_n]

    def reset(self):
        """
        Reset all profiling data.

        Example:
            >>> profiler = Profiler()
            >>> # ... run profiled operations ...
            >>> profiler.reset()
            >>> # Start fresh profiling
        """
        self._results.clear()
        self._active_timers.clear()
        logger.info("Profiler reset")


def profile_function(name: Optional[str] = None, detailed: bool = False):
    """
    Decorator for profiling a function.

    Args:
        name: Optional name for the profiled operation (defaults to function name)
        detailed: If True, use cProfile for detailed profiling

    Returns:
        Decorated function

    Example:
        >>> @profile_function("data_processing")
        ... def process_data():
        ...     # ... processing logic ...
        ...     pass
    """

    def decorator(func: Callable) -> Callable:
        operation_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create profiler from first argument if it has one
            profiler = None
            if args and hasattr(args[0], "_profiler"):
                profiler = args[0]._profiler

            if profiler:
                with profiler.profile(operation_name, detailed=detailed):
                    return func(*args, **kwargs)
            else:
                # No profiler available, just run the function
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global profiler instance for convenience
_global_profiler = Profiler()


def get_global_profiler() -> Profiler:
    """
    Get the global profiler instance.

    Returns:
        Global Profiler instance

    Example:
        >>> profiler = get_global_profiler()
        >>> with profiler.profile("operation"):
        ...     do_work()
    """
    return _global_profiler


def reset_global_profiler():
    """
    Reset the global profiler.

    Example:
        >>> reset_global_profiler()
    """
    _global_profiler.reset()
