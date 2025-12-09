"""
Unit tests for the PluginRegistry component.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import pytest

from llm_judge_auditor.components.plugin_registry import (
    PluginMetadata,
    PluginRegistry,
)


# Mock verifier for testing
class MockVerifier:
    """Mock verifier for testing plugin registration."""

    def verify_statement(
        self, statement: str, context: str, passages: Optional[List[Any]] = None
    ) -> Any:
        return {"label": "SUPPORTED", "confidence": 0.9}

    def batch_verify(
        self,
        statements: List[str],
        contexts: List[str],
        passages_list: Optional[List[List[Any]]] = None,
    ) -> List[Any]:
        return [{"label": "SUPPORTED", "confidence": 0.9}] * len(statements)


# Mock judge for testing
class MockJudge:
    """Mock judge for testing plugin registration."""

    def evaluate(
        self, source_text: str, candidate_output: str, retrieved_context: str = ""
    ) -> Any:
        return {"score": 85.0, "reasoning": "Good quality"}


# Mock aggregator for testing
def mock_aggregator(scores: List[float]) -> float:
    """Mock aggregator that returns the geometric mean."""
    if not scores:
        return 0.0
    product = 1.0
    for score in scores:
        product *= score
    return product ** (1.0 / len(scores))


class TestPluginRegistry:
    """Test suite for PluginRegistry."""

    def test_initialization(self):
        """Test basic initialization of PluginRegistry."""
        registry = PluginRegistry()
        assert registry is not None
        plugins = registry.list_plugins()
        assert plugins == {"verifiers": [], "judges": [], "aggregators": []}

    def test_register_verifier(self):
        """Test registering a verifier plugin."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        registry.register_verifier(
            "mock_verifier",
            load_verifier,
            version="1.0.0",
            description="A mock verifier for testing",
        )

        plugins = registry.list_plugins()
        assert "mock_verifier" in plugins["verifiers"]

    def test_register_duplicate_verifier_raises_error(self):
        """Test that registering a duplicate verifier raises an error."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        registry.register_verifier("mock_verifier", load_verifier)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_verifier("mock_verifier", load_verifier)

    def test_get_verifier(self):
        """Test retrieving a registered verifier."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        registry.register_verifier("mock_verifier", load_verifier)
        verifier = registry.get_verifier("mock_verifier")

        assert verifier is not None
        assert isinstance(verifier, MockVerifier)

    def test_get_nonexistent_verifier_raises_error(self):
        """Test that getting a non-existent verifier raises an error."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_verifier("nonexistent")

    def test_unregister_verifier(self):
        """Test unregistering a verifier."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        registry.register_verifier("mock_verifier", load_verifier)
        assert "mock_verifier" in registry.list_plugins()["verifiers"]

        registry.unregister_verifier("mock_verifier")
        assert "mock_verifier" not in registry.list_plugins()["verifiers"]

    def test_unregister_nonexistent_verifier_raises_error(self):
        """Test that unregistering a non-existent verifier raises an error."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister_verifier("nonexistent")

    def test_register_judge(self):
        """Test registering a judge plugin."""
        registry = PluginRegistry()

        def load_judge():
            return MockJudge()

        registry.register_judge(
            "mock_judge",
            load_judge,
            version="1.0.0",
            description="A mock judge for testing",
        )

        plugins = registry.list_plugins()
        assert "mock_judge" in plugins["judges"]

    def test_register_duplicate_judge_raises_error(self):
        """Test that registering a duplicate judge raises an error."""
        registry = PluginRegistry()

        def load_judge():
            return MockJudge()

        registry.register_judge("mock_judge", load_judge)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_judge("mock_judge", load_judge)

    def test_get_judge(self):
        """Test retrieving a registered judge."""
        registry = PluginRegistry()

        def load_judge():
            return MockJudge()

        registry.register_judge("mock_judge", load_judge)
        judge = registry.get_judge("mock_judge")

        assert judge is not None
        assert isinstance(judge, MockJudge)

    def test_get_nonexistent_judge_raises_error(self):
        """Test that getting a non-existent judge raises an error."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_judge("nonexistent")

    def test_unregister_judge(self):
        """Test unregistering a judge."""
        registry = PluginRegistry()

        def load_judge():
            return MockJudge()

        registry.register_judge("mock_judge", load_judge)
        assert "mock_judge" in registry.list_plugins()["judges"]

        registry.unregister_judge("mock_judge")
        assert "mock_judge" not in registry.list_plugins()["judges"]

    def test_unregister_nonexistent_judge_raises_error(self):
        """Test that unregistering a non-existent judge raises an error."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister_judge("nonexistent")

    def test_register_aggregator(self):
        """Test registering an aggregator plugin."""
        registry = PluginRegistry()

        registry.register_aggregator(
            "geometric_mean",
            mock_aggregator,
            version="1.0.0",
            description="Geometric mean aggregator",
        )

        plugins = registry.list_plugins()
        assert "geometric_mean" in plugins["aggregators"]

    def test_register_duplicate_aggregator_raises_error(self):
        """Test that registering a duplicate aggregator raises an error."""
        registry = PluginRegistry()

        registry.register_aggregator("geometric_mean", mock_aggregator)

        with pytest.raises(ValueError, match="already registered"):
            registry.register_aggregator("geometric_mean", mock_aggregator)

    def test_get_aggregator(self):
        """Test retrieving a registered aggregator."""
        registry = PluginRegistry()

        registry.register_aggregator("geometric_mean", mock_aggregator)
        aggregator = registry.get_aggregator("geometric_mean")

        assert aggregator is not None
        assert callable(aggregator)

        # Test the aggregator function
        result = aggregator([4.0, 9.0, 16.0])
        expected = (4.0 * 9.0 * 16.0) ** (1.0 / 3.0)
        assert abs(result - expected) < 0.001

    def test_get_nonexistent_aggregator_raises_error(self):
        """Test that getting a non-existent aggregator raises an error."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_aggregator("nonexistent")

    def test_unregister_aggregator(self):
        """Test unregistering an aggregator."""
        registry = PluginRegistry()

        registry.register_aggregator("geometric_mean", mock_aggregator)
        assert "geometric_mean" in registry.list_plugins()["aggregators"]

        registry.unregister_aggregator("geometric_mean")
        assert "geometric_mean" not in registry.list_plugins()["aggregators"]

    def test_unregister_nonexistent_aggregator_raises_error(self):
        """Test that unregistering a non-existent aggregator raises an error."""
        registry = PluginRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister_aggregator("nonexistent")

    def test_list_plugins(self):
        """Test listing all registered plugins."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        def load_judge():
            return MockJudge()

        registry.register_verifier("verifier1", load_verifier)
        registry.register_verifier("verifier2", load_verifier)
        registry.register_judge("judge1", load_judge)
        registry.register_aggregator("agg1", mock_aggregator)

        plugins = registry.list_plugins()

        assert len(plugins["verifiers"]) == 2
        assert "verifier1" in plugins["verifiers"]
        assert "verifier2" in plugins["verifiers"]
        assert len(plugins["judges"]) == 1
        assert "judge1" in plugins["judges"]
        assert len(plugins["aggregators"]) == 1
        assert "agg1" in plugins["aggregators"]

    def test_get_plugin_info(self):
        """Test retrieving plugin metadata."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        registry.register_verifier(
            "mock_verifier",
            load_verifier,
            version="2.0.0",
            description="Test verifier",
            author="Test Author",
        )

        info = registry.get_plugin_info("mock_verifier")

        assert info is not None
        assert isinstance(info, PluginMetadata)
        assert info.name == "mock_verifier"
        assert info.version == "2.0.0"
        assert info.plugin_type == "verifier"
        assert info.description == "Test verifier"
        assert info.author == "Test Author"

    def test_get_plugin_info_nonexistent(self):
        """Test getting info for non-existent plugin returns None."""
        registry = PluginRegistry()
        info = registry.get_plugin_info("nonexistent")
        assert info is None

    def test_check_compatibility(self):
        """Test version compatibility checking."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        registry.register_verifier(
            "mock_verifier",
            load_verifier,
            compatible_versions=["1.0.0", "1.1.0"],
        )

        assert registry.check_compatibility("mock_verifier", "1.0.0") is True
        assert registry.check_compatibility("mock_verifier", "1.1.0") is True
        assert registry.check_compatibility("mock_verifier", "2.0.0") is False

    def test_check_compatibility_no_constraints(self):
        """Test compatibility when no version constraints are specified."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        registry.register_verifier("mock_verifier", load_verifier)

        # Should be compatible with any version when no constraints
        assert registry.check_compatibility("mock_verifier", "1.0.0") is True
        assert registry.check_compatibility("mock_verifier", "99.0.0") is True

    def test_clear_all(self):
        """Test clearing all registered plugins."""
        registry = PluginRegistry()

        def load_verifier():
            return MockVerifier()

        def load_judge():
            return MockJudge()

        registry.register_verifier("verifier1", load_verifier)
        registry.register_judge("judge1", load_judge)
        registry.register_aggregator("agg1", mock_aggregator)

        plugins = registry.list_plugins()
        assert len(plugins["verifiers"]) == 1
        assert len(plugins["judges"]) == 1
        assert len(plugins["aggregators"]) == 1

        registry.clear_all()

        plugins = registry.list_plugins()
        assert len(plugins["verifiers"]) == 0
        assert len(plugins["judges"]) == 0
        assert len(plugins["aggregators"]) == 0

    def test_discover_plugins_empty_directory(self):
        """Test plugin discovery with an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = PluginRegistry()
            discovered = registry.discover_plugins(tmpdir)

            assert discovered == {"verifiers": 0, "judges": 0, "aggregators": 0}

    def test_discover_plugins_nonexistent_directory(self):
        """Test plugin discovery with a non-existent directory."""
        registry = PluginRegistry()
        discovered = registry.discover_plugins("/nonexistent/path")

        assert discovered == {"verifiers": 0, "judges": 0, "aggregators": 0}

    def test_discover_plugins_with_valid_plugin(self):
        """Test plugin discovery with a valid plugin module."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin file
            plugin_file = Path(tmpdir) / "test_plugin.py"
            plugin_code = """
def register_plugin(registry):
    def load_verifier():
        class TestVerifier:
            def verify_statement(self, statement, context, passages=None):
                return {"label": "SUPPORTED"}
            def batch_verify(self, statements, contexts, passages_list=None):
                return [{"label": "SUPPORTED"}] * len(statements)
        return TestVerifier()
    
    registry.register_verifier("test_verifier", load_verifier, version="1.0.0")
"""
            plugin_file.write_text(plugin_code)

            registry = PluginRegistry()
            discovered = registry.discover_plugins(tmpdir)

            assert discovered["verifiers"] == 1
            assert "test_verifier" in registry.list_plugins()["verifiers"]

    def test_discover_plugins_skips_private_modules(self):
        """Test that plugin discovery skips modules starting with underscore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a private plugin file
            plugin_file = Path(tmpdir) / "_private_plugin.py"
            plugin_code = """
def register_plugin(registry):
    def load_verifier():
        return None
    registry.register_verifier("private_verifier", load_verifier)
"""
            plugin_file.write_text(plugin_code)

            registry = PluginRegistry()
            discovered = registry.discover_plugins(tmpdir)

            assert discovered["verifiers"] == 0
            assert "private_verifier" not in registry.list_plugins()["verifiers"]

    def test_discover_plugins_handles_invalid_module(self):
        """Test that plugin discovery handles invalid modules gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid plugin file
            plugin_file = Path(tmpdir) / "invalid_plugin.py"
            plugin_code = """
# This module has a syntax error
def register_plugin(registry)
    pass  # Missing colon
"""
            plugin_file.write_text(plugin_code)

            registry = PluginRegistry()
            # Should not raise an exception
            discovered = registry.discover_plugins(tmpdir)

            assert discovered["verifiers"] == 0

    def test_discover_plugins_handles_missing_register_function(self):
        """Test that plugin discovery handles modules without register_plugin."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin file without register_plugin
            plugin_file = Path(tmpdir) / "no_register.py"
            plugin_code = """
def some_other_function():
    pass
"""
            plugin_file.write_text(plugin_code)

            registry = PluginRegistry()
            discovered = registry.discover_plugins(tmpdir)

            assert discovered["verifiers"] == 0

    def test_plugin_loader_failure(self):
        """Test that plugin loader failures are handled gracefully."""
        registry = PluginRegistry()

        def failing_loader():
            raise RuntimeError("Loader failed")

        registry.register_verifier("failing_verifier", failing_loader)

        with pytest.raises(RuntimeError, match="Failed to load verifier"):
            registry.get_verifier("failing_verifier")

    def test_initialization_with_plugins_dir(self):
        """Test initialization with automatic plugin discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a plugin file
            plugin_file = Path(tmpdir) / "auto_plugin.py"
            plugin_code = """
def register_plugin(registry):
    def load_judge():
        class TestJudge:
            def evaluate(self, source_text, candidate_output, retrieved_context=""):
                return {"score": 90.0}
        return TestJudge()
    
    registry.register_judge("auto_judge", load_judge)
"""
            plugin_file.write_text(plugin_code)

            # Initialize registry with plugins directory
            registry = PluginRegistry(plugins_dir=tmpdir)

            # Plugin should be auto-discovered
            assert "auto_judge" in registry.list_plugins()["judges"]
