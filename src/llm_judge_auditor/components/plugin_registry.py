"""
Plugin Registry for the LLM Judge Auditor toolkit.

This module provides the PluginRegistry class that allows users to register
custom models and components without modifying core code. It supports plugin
discovery from a plugins/ directory and provides version compatibility checking.
"""

import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Type

from llm_judge_auditor.components.aggregation_engine import AggregationStrategy

logger = logging.getLogger(__name__)


class VerifierProtocol(Protocol):
    """Protocol defining the interface for custom verifier plugins."""

    def verify_statement(
        self, statement: str, context: str, passages: Optional[List[Any]] = None
    ) -> Any:
        """Verify a single statement against context and passages."""
        ...

    def batch_verify(
        self,
        statements: List[str],
        contexts: List[str],
        passages_list: Optional[List[List[Any]]] = None,
    ) -> List[Any]:
        """Verify multiple statements in batch."""
        ...


class JudgeProtocol(Protocol):
    """Protocol defining the interface for custom judge plugins."""

    def evaluate(
        self, source_text: str, candidate_output: str, retrieved_context: str = ""
    ) -> Any:
        """Evaluate a candidate output against source text."""
        ...


class AggregatorProtocol(Protocol):
    """Protocol defining the interface for custom aggregator plugins."""

    def aggregate(self, scores: List[float]) -> float:
        """Aggregate multiple scores into a single consensus score."""
        ...


class PluginMetadata:
    """
    Metadata for a registered plugin.

    Attributes:
        name: Plugin name
        version: Plugin version string
        plugin_type: Type of plugin (verifier, judge, aggregator)
        loader: Callable that loads/creates the plugin instance
        description: Optional description of the plugin
        author: Optional author information
        compatible_versions: Optional list of compatible toolkit versions
    """

    def __init__(
        self,
        name: str,
        version: str,
        plugin_type: str,
        loader: Callable,
        description: str = "",
        author: str = "",
        compatible_versions: Optional[List[str]] = None,
    ):
        self.name = name
        self.version = version
        self.plugin_type = plugin_type
        self.loader = loader
        self.description = description
        self.author = author
        self.compatible_versions = compatible_versions or []

    def __repr__(self) -> str:
        return (
            f"PluginMetadata(name='{self.name}', version='{self.version}', "
            f"type='{self.plugin_type}')"
        )


class PluginRegistry:
    """
    Registry for managing custom plugins.

    This class allows users to register custom verifiers, judges, and aggregators
    without modifying core code. It supports plugin discovery from a plugins/
    directory and provides version compatibility checking.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register_verifier("my_verifier", my_verifier_loader)
        >>> verifier = registry.get_verifier("my_verifier")
    """

    def __init__(self, plugins_dir: Optional[str] = None):
        """
        Initialize the PluginRegistry.

        Args:
            plugins_dir: Optional path to plugins directory for auto-discovery
        """
        self._verifiers: Dict[str, PluginMetadata] = {}
        self._judges: Dict[str, PluginMetadata] = {}
        self._aggregators: Dict[str, PluginMetadata] = {}
        self.plugins_dir = plugins_dir

        logger.info("PluginRegistry initialized")

        # Auto-discover plugins if directory is provided
        if plugins_dir and os.path.exists(plugins_dir):
            self.discover_plugins(plugins_dir)

    def register_verifier(
        self,
        name: str,
        loader: Callable,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        compatible_versions: Optional[List[str]] = None,
    ) -> None:
        """
        Register a custom verifier plugin.

        The loader should be a callable that returns an object implementing
        the VerifierProtocol interface (verify_statement, batch_verify methods).

        Args:
            name: Unique name for the verifier
            loader: Callable that loads/creates the verifier instance
            version: Version string for the plugin
            description: Optional description of the verifier
            author: Optional author information
            compatible_versions: Optional list of compatible toolkit versions

        Raises:
            ValueError: If a verifier with the same name is already registered

        Example:
            >>> def load_my_verifier():
            ...     return MyCustomVerifier()
            >>> registry.register_verifier("my_verifier", load_my_verifier)
        """
        if name in self._verifiers:
            raise ValueError(
                f"Verifier '{name}' is already registered. "
                f"Use unregister_verifier() first to replace it."
            )

        metadata = PluginMetadata(
            name=name,
            version=version,
            plugin_type="verifier",
            loader=loader,
            description=description,
            author=author,
            compatible_versions=compatible_versions,
        )

        self._verifiers[name] = metadata
        logger.info(f"Registered verifier plugin: {name} (version {version})")

    def register_judge(
        self,
        name: str,
        loader: Callable,
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        compatible_versions: Optional[List[str]] = None,
    ) -> None:
        """
        Register a custom judge plugin.

        The loader should be a callable that returns an object implementing
        the JudgeProtocol interface (evaluate method).

        Args:
            name: Unique name for the judge
            loader: Callable that loads/creates the judge instance
            version: Version string for the plugin
            description: Optional description of the judge
            author: Optional author information
            compatible_versions: Optional list of compatible toolkit versions

        Raises:
            ValueError: If a judge with the same name is already registered

        Example:
            >>> def load_my_judge():
            ...     return MyCustomJudge()
            >>> registry.register_judge("my_judge", load_my_judge)
        """
        if name in self._judges:
            raise ValueError(
                f"Judge '{name}' is already registered. "
                f"Use unregister_judge() first to replace it."
            )

        metadata = PluginMetadata(
            name=name,
            version=version,
            plugin_type="judge",
            loader=loader,
            description=description,
            author=author,
            compatible_versions=compatible_versions,
        )

        self._judges[name] = metadata
        logger.info(f"Registered judge plugin: {name} (version {version})")

    def register_aggregator(
        self,
        name: str,
        aggregator: Callable[[List[float]], float],
        version: str = "1.0.0",
        description: str = "",
        author: str = "",
        compatible_versions: Optional[List[str]] = None,
    ) -> None:
        """
        Register a custom aggregation strategy.

        The aggregator should be a callable that takes a list of scores
        and returns a single aggregated score.

        Args:
            name: Unique name for the aggregation strategy
            aggregator: Callable that aggregates scores
            version: Version string for the plugin
            description: Optional description of the aggregator
            author: Optional author information
            compatible_versions: Optional list of compatible toolkit versions

        Raises:
            ValueError: If an aggregator with the same name is already registered

        Example:
            >>> def harmonic_mean(scores):
            ...     return len(scores) / sum(1/s for s in scores if s > 0)
            >>> registry.register_aggregator("harmonic_mean", harmonic_mean)
        """
        if name in self._aggregators:
            raise ValueError(
                f"Aggregator '{name}' is already registered. "
                f"Use unregister_aggregator() first to replace it."
            )

        metadata = PluginMetadata(
            name=name,
            version=version,
            plugin_type="aggregator",
            loader=aggregator,  # For aggregators, the loader is the function itself
            description=description,
            author=author,
            compatible_versions=compatible_versions,
        )

        self._aggregators[name] = metadata
        logger.info(f"Registered aggregator plugin: {name} (version {version})")

    def unregister_verifier(self, name: str) -> None:
        """
        Unregister a verifier plugin.

        Args:
            name: Name of the verifier to unregister

        Raises:
            KeyError: If the verifier is not registered
        """
        if name not in self._verifiers:
            raise KeyError(f"Verifier '{name}' is not registered")

        del self._verifiers[name]
        logger.info(f"Unregistered verifier plugin: {name}")

    def unregister_judge(self, name: str) -> None:
        """
        Unregister a judge plugin.

        Args:
            name: Name of the judge to unregister

        Raises:
            KeyError: If the judge is not registered
        """
        if name not in self._judges:
            raise KeyError(f"Judge '{name}' is not registered")

        del self._judges[name]
        logger.info(f"Unregistered judge plugin: {name}")

    def unregister_aggregator(self, name: str) -> None:
        """
        Unregister an aggregator plugin.

        Args:
            name: Name of the aggregator to unregister

        Raises:
            KeyError: If the aggregator is not registered
        """
        if name not in self._aggregators:
            raise KeyError(f"Aggregator '{name}' is not registered")

        del self._aggregators[name]
        logger.info(f"Unregistered aggregator plugin: {name}")

    def get_verifier(self, name: str) -> Any:
        """
        Get a verifier plugin instance.

        Args:
            name: Name of the verifier

        Returns:
            Verifier instance created by the loader

        Raises:
            KeyError: If the verifier is not registered
        """
        if name not in self._verifiers:
            raise KeyError(
                f"Verifier '{name}' is not registered. "
                f"Available verifiers: {list(self._verifiers.keys())}"
            )

        metadata = self._verifiers[name]
        try:
            verifier = metadata.loader()
            logger.debug(f"Loaded verifier plugin: {name}")
            return verifier
        except Exception as e:
            logger.error(f"Failed to load verifier '{name}': {e}")
            raise RuntimeError(f"Failed to load verifier '{name}': {e}") from e

    def get_judge(self, name: str) -> Any:
        """
        Get a judge plugin instance.

        Args:
            name: Name of the judge

        Returns:
            Judge instance created by the loader

        Raises:
            KeyError: If the judge is not registered
        """
        if name not in self._judges:
            raise KeyError(
                f"Judge '{name}' is not registered. "
                f"Available judges: {list(self._judges.keys())}"
            )

        metadata = self._judges[name]
        try:
            judge = metadata.loader()
            logger.debug(f"Loaded judge plugin: {name}")
            return judge
        except Exception as e:
            logger.error(f"Failed to load judge '{name}': {e}")
            raise RuntimeError(f"Failed to load judge '{name}': {e}") from e

    def get_aggregator(self, name: str) -> Callable[[List[float]], float]:
        """
        Get an aggregator plugin function.

        Args:
            name: Name of the aggregator

        Returns:
            Aggregator function

        Raises:
            KeyError: If the aggregator is not registered
        """
        if name not in self._aggregators:
            raise KeyError(
                f"Aggregator '{name}' is not registered. "
                f"Available aggregators: {list(self._aggregators.keys())}"
            )

        metadata = self._aggregators[name]
        logger.debug(f"Retrieved aggregator plugin: {name}")
        return metadata.loader

    def list_plugins(self) -> Dict[str, List[str]]:
        """
        List all registered plugins by type.

        Returns:
            Dictionary mapping plugin types to lists of plugin names

        Example:
            >>> registry = PluginRegistry()
            >>> plugins = registry.list_plugins()
            >>> print(plugins)
            {'verifiers': ['custom_verifier'], 'judges': ['custom_judge'], ...}
        """
        return {
            "verifiers": list(self._verifiers.keys()),
            "judges": list(self._judges.keys()),
            "aggregators": list(self._aggregators.keys()),
        }

    def get_plugin_info(self, name: str) -> Optional[PluginMetadata]:
        """
        Get metadata for a specific plugin.

        Args:
            name: Name of the plugin

        Returns:
            PluginMetadata if found, None otherwise
        """
        # Check all plugin types
        if name in self._verifiers:
            return self._verifiers[name]
        elif name in self._judges:
            return self._judges[name]
        elif name in self._aggregators:
            return self._aggregators[name]
        else:
            return None

    def discover_plugins(self, plugins_dir: str) -> Dict[str, int]:
        """
        Discover and load plugins from a directory.

        This method scans the plugins directory for Python modules and attempts
        to load them. Each plugin module should define a `register_plugin(registry)`
        function that registers the plugin with the provided registry.

        Args:
            plugins_dir: Path to the plugins directory

        Returns:
            Dictionary with counts of discovered plugins by type

        Example:
            Plugin module structure (plugins/my_plugin.py):
            ```python
            def register_plugin(registry):
                def load_my_verifier():
                    return MyVerifier()
                registry.register_verifier("my_verifier", load_my_verifier)
            ```
        """
        if not os.path.exists(plugins_dir):
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return {"verifiers": 0, "judges": 0, "aggregators": 0}

        plugins_path = Path(plugins_dir)
        discovered = {"verifiers": 0, "judges": 0, "aggregators": 0}

        logger.info(f"Discovering plugins in: {plugins_dir}")

        # Track initial counts
        initial_verifiers = len(self._verifiers)
        initial_judges = len(self._judges)
        initial_aggregators = len(self._aggregators)

        # Scan for Python files
        for plugin_file in plugins_path.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue  # Skip private modules

            try:
                # Load the module
                module_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(
                    module_name, plugin_file
                )
                if spec is None or spec.loader is None:
                    logger.warning(f"Could not load spec for {plugin_file}")
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # Look for register_plugin function
                if hasattr(module, "register_plugin"):
                    register_func = getattr(module, "register_plugin")
                    if callable(register_func):
                        # Call the registration function
                        register_func(self)
                        logger.info(f"Loaded plugin module: {module_name}")
                    else:
                        logger.warning(
                            f"register_plugin in {module_name} is not callable"
                        )
                else:
                    logger.warning(
                        f"Plugin module {module_name} does not define register_plugin()"
                    )

            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")

        # Calculate discovered counts
        discovered["verifiers"] = len(self._verifiers) - initial_verifiers
        discovered["judges"] = len(self._judges) - initial_judges
        discovered["aggregators"] = len(self._aggregators) - initial_aggregators

        total = sum(discovered.values())
        logger.info(
            f"Plugin discovery complete: {total} plugins loaded "
            f"({discovered['verifiers']} verifiers, {discovered['judges']} judges, "
            f"{discovered['aggregators']} aggregators)"
        )

        return discovered

    def check_compatibility(
        self, plugin_name: str, toolkit_version: str
    ) -> bool:
        """
        Check if a plugin is compatible with the current toolkit version.

        Args:
            plugin_name: Name of the plugin
            toolkit_version: Current toolkit version string

        Returns:
            True if compatible or no version constraints specified, False otherwise
        """
        metadata = self.get_plugin_info(plugin_name)
        if metadata is None:
            return False

        # If no compatible versions specified, assume compatible
        if not metadata.compatible_versions:
            return True

        # Check if toolkit version is in the compatible list
        # This is a simple string match; more sophisticated version
        # comparison could be implemented using packaging.version
        return toolkit_version in metadata.compatible_versions

    def clear_all(self) -> None:
        """
        Clear all registered plugins.

        This is useful for testing or resetting the registry.
        """
        self._verifiers.clear()
        self._judges.clear()
        self._aggregators.clear()
        logger.info("Cleared all registered plugins")
