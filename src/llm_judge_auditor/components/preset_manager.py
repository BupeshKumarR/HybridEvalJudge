"""
Preset Manager for the LLM Judge Auditor toolkit.

This module provides the PresetManager class for loading and managing
pre-configured evaluation presets (fast, balanced, etc.).
"""

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from llm_judge_auditor.config import ToolkitConfig


class PresetInfo:
    """Information about an available preset."""

    def __init__(self, name: str, description: str, config: ToolkitConfig):
        """
        Initialize preset information.

        Args:
            name: Preset name (e.g., "fast", "balanced")
            description: Human-readable description of the preset
            config: ToolkitConfig instance for this preset
        """
        self.name = name
        self.description = description
        self.config = config

    def __repr__(self) -> str:
        return f"PresetInfo(name='{self.name}', description='{self.description}')"


class PresetManager:
    """
    Manager for loading and listing evaluation presets.

    The PresetManager provides access to pre-configured evaluation modes
    optimized for different use cases (speed, accuracy, resource usage).
    """

    # Built-in preset definitions
    _BUILTIN_PRESETS = {
        "fast": {
            "description": "Minimal processing for quick evaluations (no retrieval, single lightweight judge)",
            "config": {
                "verifier_model": "MiniCheck/flan-t5-base-finetuned",
                "judge_models": ["microsoft/Phi-3-mini-4k-instruct"],
                "quantize": True,
                "enable_retrieval": False,
                "aggregation_strategy": "mean",
                "batch_size": 1,
                "max_length": 512,
            },
        },
        "balanced": {
            "description": "Good accuracy with reasonable resource usage (retrieval enabled, 2 judges)",
            "config": {
                "verifier_model": "MiniCheck/flan-t5-large-finetuned",
                "judge_models": ["meta-llama/Llama-3-8B", "mistralai/Mistral-7B-v0.1"],
                "quantize": True,
                "enable_retrieval": True,
                "aggregation_strategy": "mean",
                "batch_size": 1,
                "max_length": 512,
            },
        },
    }

    def __init__(self, preset_dir: Optional[Path] = None):
        """
        Initialize the PresetManager.

        Args:
            preset_dir: Optional directory containing custom preset YAML files.
                       If None, only built-in presets are available.
        """
        self.preset_dir = preset_dir
        self._custom_presets: Dict[str, Dict] = {}

        # Load custom presets from directory if provided
        if preset_dir and preset_dir.exists():
            self._load_custom_presets()

    def _load_custom_presets(self) -> None:
        """Load custom presets from YAML files in preset_dir."""
        if not self.preset_dir or not self.preset_dir.exists():
            return

        for yaml_file in self.preset_dir.glob("*.yaml"):
            preset_name = yaml_file.stem
            # Skip if it would override a built-in preset
            if preset_name not in self._BUILTIN_PRESETS:
                try:
                    with open(yaml_file, "r") as f:
                        config_dict = yaml.safe_load(f)
                    self._custom_presets[preset_name] = {
                        "description": config_dict.pop("description", f"Custom preset: {preset_name}"),
                        "config": config_dict,
                    }
                except Exception as e:
                    # Log warning but don't fail - just skip this preset
                    print(f"Warning: Failed to load preset from {yaml_file}: {e}")

    def load_preset(self, name: str) -> ToolkitConfig:
        """
        Load a preset configuration by name.

        Args:
            name: Name of the preset to load (e.g., "fast", "balanced")

        Returns:
            ToolkitConfig instance with preset values

        Raises:
            ValueError: If preset name is not found
        """
        # Check built-in presets first
        if name in self._BUILTIN_PRESETS:
            preset_data = self._BUILTIN_PRESETS[name]
            return ToolkitConfig(**preset_data["config"])

        # Check custom presets
        if name in self._custom_presets:
            preset_data = self._custom_presets[name]
            return ToolkitConfig(**preset_data["config"])

        # Preset not found
        available = list(self._BUILTIN_PRESETS.keys()) + list(self._custom_presets.keys())
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")

    def list_presets(self) -> List[PresetInfo]:
        """
        List all available presets (built-in and custom).

        Returns:
            List of PresetInfo objects describing available presets
        """
        presets = []

        # Add built-in presets
        for name, data in self._BUILTIN_PRESETS.items():
            config = ToolkitConfig(**data["config"])
            presets.append(PresetInfo(name, data["description"], config))

        # Add custom presets
        for name, data in self._custom_presets.items():
            config = ToolkitConfig(**data["config"])
            presets.append(PresetInfo(name, data["description"], config))

        return presets

    def get_preset_names(self) -> List[str]:
        """
        Get list of available preset names.

        Returns:
            List of preset names
        """
        return list(self._BUILTIN_PRESETS.keys()) + list(self._custom_presets.keys())

    def preset_exists(self, name: str) -> bool:
        """
        Check if a preset exists.

        Args:
            name: Preset name to check

        Returns:
            True if preset exists, False otherwise
        """
        return name in self._BUILTIN_PRESETS or name in self._custom_presets
