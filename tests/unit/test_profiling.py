"""
Unit tests for profiling utilities.

This module tests the profiling functionality used for performance optimization.
"""

import time
import unittest

from llm_judge_auditor.utils.profiling import (
    Profiler,
    get_global_profiler,
    profile_function,
    reset_global_profiler,
)


class TestProfiler(unittest.TestCase):
    """Test cases for the Profiler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.profiler = Profiler()

    def test_profile_context_manager(self):
        """Test profiling with context manager."""
        # Profile a simple operation
        with self.profiler.profile("test_operation"):
            time.sleep(0.01)  # Simulate work

        # Check results
        results = self.profiler.get_results("test_operation")
        self.assertIn("test_operation", results)
        self.assertEqual(len(results["test_operation"]), 1)

        result = results["test_operation"][0]
        self.assertEqual(result.name, "test_operation")
        self.assertGreater(result.duration, 0.01)
        self.assertLess(result.duration, 0.1)  # Should be quick

    def test_multiple_profiles(self):
        """Test profiling multiple operations."""
        # Profile multiple operations
        with self.profiler.profile("operation_1"):
            time.sleep(0.01)

        with self.profiler.profile("operation_2"):
            time.sleep(0.02)

        with self.profiler.profile("operation_1"):
            time.sleep(0.01)

        # Check results
        results = self.profiler.get_results()
        self.assertIn("operation_1", results)
        self.assertIn("operation_2", results)

        # operation_1 should have 2 calls
        self.assertEqual(len(results["operation_1"]), 2)

        # operation_2 should have 1 call
        self.assertEqual(len(results["operation_2"]), 1)

    def test_start_stop_timer(self):
        """Test manual timer start/stop."""
        # Start timer
        self.profiler.start_timer("manual_timer")
        time.sleep(0.01)
        duration = self.profiler.stop_timer("manual_timer")

        # Check duration
        self.assertGreater(duration, 0.01)
        self.assertLess(duration, 0.1)

        # Check results
        results = self.profiler.get_results("manual_timer")
        self.assertIn("manual_timer", results)
        self.assertEqual(len(results["manual_timer"]), 1)

    def test_stop_timer_not_started(self):
        """Test stopping a timer that wasn't started."""
        with self.assertRaises(ValueError) as context:
            self.profiler.stop_timer("nonexistent_timer")

        self.assertIn("was not started", str(context.exception))

    def test_get_summary(self):
        """Test getting profiling summary."""
        # Profile some operations
        with self.profiler.profile("op1"):
            time.sleep(0.01)

        with self.profiler.profile("op1"):
            time.sleep(0.02)

        with self.profiler.profile("op2"):
            time.sleep(0.01)

        # Get summary
        summary = self.profiler.get_summary()

        # Check summary contains expected information
        self.assertIn("Profiling Summary", summary)
        self.assertIn("op1", summary)
        self.assertIn("op2", summary)
        self.assertIn("Calls:", summary)
        self.assertIn("Total:", summary)
        self.assertIn("Average:", summary)

    def test_get_summary_empty(self):
        """Test getting summary with no profiling data."""
        summary = self.profiler.get_summary()
        self.assertEqual(summary, "No profiling data available")

    def test_get_bottlenecks(self):
        """Test identifying bottlenecks."""
        # Profile operations with different durations
        with self.profiler.profile("fast_op"):
            time.sleep(0.01)

        with self.profiler.profile("slow_op"):
            time.sleep(0.05)

        with self.profiler.profile("medium_op"):
            time.sleep(0.03)

        # Get bottlenecks
        bottlenecks = self.profiler.get_bottlenecks(3)

        # Check results
        self.assertEqual(len(bottlenecks), 3)

        # Should be sorted by time (descending)
        self.assertEqual(bottlenecks[0][0], "slow_op")
        self.assertEqual(bottlenecks[1][0], "medium_op")
        self.assertEqual(bottlenecks[2][0], "fast_op")

        # Check times are reasonable
        self.assertGreater(bottlenecks[0][1], 0.05)
        self.assertGreater(bottlenecks[1][1], 0.03)
        self.assertGreater(bottlenecks[2][1], 0.01)

    def test_get_bottlenecks_limit(self):
        """Test limiting number of bottlenecks returned."""
        # Profile multiple operations
        for i in range(5):
            with self.profiler.profile(f"op_{i}"):
                time.sleep(0.01)

        # Get top 3 bottlenecks
        bottlenecks = self.profiler.get_bottlenecks(3)
        self.assertEqual(len(bottlenecks), 3)

    def test_reset(self):
        """Test resetting profiler."""
        # Profile some operations
        with self.profiler.profile("test_op"):
            time.sleep(0.01)

        # Verify data exists
        results = self.profiler.get_results()
        self.assertIn("test_op", results)

        # Reset
        self.profiler.reset()

        # Verify data is cleared
        results = self.profiler.get_results()
        self.assertEqual(len(results), 0)

        summary = self.profiler.get_summary()
        self.assertEqual(summary, "No profiling data available")

    def test_nested_profiling(self):
        """Test nested profiling contexts."""
        with self.profiler.profile("outer"):
            time.sleep(0.01)

            with self.profiler.profile("inner"):
                time.sleep(0.01)

            time.sleep(0.01)

        # Check results
        results = self.profiler.get_results()
        self.assertIn("outer", results)
        self.assertIn("inner", results)

        # Outer should take longer than inner
        outer_time = results["outer"][0].duration
        inner_time = results["inner"][0].duration

        self.assertGreater(outer_time, inner_time)


class TestProfileFunction(unittest.TestCase):
    """Test cases for the profile_function decorator."""

    def test_profile_function_decorator(self):
        """Test profiling a function with decorator."""

        class MockClass:
            def __init__(self):
                self._profiler = Profiler()

            @profile_function("test_method")
            def test_method(self):
                time.sleep(0.01)
                return "result"

        # Create instance and call method
        obj = MockClass()
        result = obj.test_method()

        # Check result
        self.assertEqual(result, "result")

        # Check profiling data
        results = obj._profiler.get_results("test_method")
        self.assertIn("test_method", results)
        self.assertEqual(len(results["test_method"]), 1)

    def test_profile_function_without_profiler(self):
        """Test decorator works without profiler attribute."""

        @profile_function("standalone_function")
        def standalone_function():
            time.sleep(0.01)
            return "result"

        # Should work without error
        result = standalone_function()
        self.assertEqual(result, "result")

    def test_profile_function_default_name(self):
        """Test decorator uses function name by default."""

        class MockClass:
            def __init__(self):
                self._profiler = Profiler()

            @profile_function()
            def my_method(self):
                time.sleep(0.01)
                return "result"

        # Create instance and call method
        obj = MockClass()
        result = obj.my_method()

        # Check profiling data uses function name
        results = obj._profiler.get_results("my_method")
        self.assertIn("my_method", results)


class TestGlobalProfiler(unittest.TestCase):
    """Test cases for global profiler functions."""

    def setUp(self):
        """Set up test fixtures."""
        reset_global_profiler()

    def tearDown(self):
        """Clean up after tests."""
        reset_global_profiler()

    def test_get_global_profiler(self):
        """Test getting global profiler instance."""
        profiler = get_global_profiler()
        self.assertIsInstance(profiler, Profiler)

        # Should return same instance
        profiler2 = get_global_profiler()
        self.assertIs(profiler, profiler2)

    def test_global_profiler_usage(self):
        """Test using global profiler."""
        profiler = get_global_profiler()

        # Profile an operation
        with profiler.profile("global_test"):
            time.sleep(0.01)

        # Check results
        results = profiler.get_results("global_test")
        self.assertIn("global_test", results)

    def test_reset_global_profiler(self):
        """Test resetting global profiler."""
        profiler = get_global_profiler()

        # Profile an operation
        with profiler.profile("test_op"):
            time.sleep(0.01)

        # Verify data exists
        results = profiler.get_results()
        self.assertIn("test_op", results)

        # Reset
        reset_global_profiler()

        # Verify data is cleared
        results = profiler.get_results()
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
