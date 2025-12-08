"""
Unit tests for PresetManager component.

Tests preset loading, listing, and validation functionality.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from llm_judge_auditor.components.preset_manager import PresetInfo, PresetManager
from llm_judge_auditor.config import AggregationStrategy, ToolkitConfig


class TestPresetManager:
    """Test suite for PresetManager."""

    def test_initialization_without_preset_dir(self):
        """Test PresetManager initializes correctly without custom preset directory."""
        manager = PresetManager()
        assert manager.preset_dir is None
        assert len(manager._custom_presets) == 0

    def test_initialization_with_nonexistent_dir(self):
        """Test PresetManager handles nonexistent preset directory gracefully."""
        manager = PresetManager(preset_dir=Path("/nonexistent/path"))
        assert manager.preset_dir == Path("/nonexistent/path")
        assert len(manager._custom_presets) == 0

    def test_load_fast_preset(self):
        """Test loading the 'fast' preset."""
        manager = PresetManager()
        config = manager.load_preset("fast")

        assert isinstance(config, ToolkitConfig)
        assert config.verifier_model == "MiniCheck/flan-t5-base-finetuned"
        assert config.judge_models == ["microsoft/Phi-3-mini-4k-instruct"]
        assert config.quantize is True
        assert config.enable_retrieval is False
        assert config.aggregation_strategy == AggregationStrategy.MEAN
        assert config.batch_size == 1
        assert config.max_length == 512

    def test_load_balanced_preset(self):
        """Test loading the 'balanced' preset."""
        manager = PresetManager()
        config = manager.load_preset("balanced")

        assert isinstance(config, ToolkitConfig)
        assert config.verifier_model == "MiniCheck/flan-t5-large-finetuned"
        assert config.judge_models == ["meta-llama/Llama-3-8B", "mistralai/Mistral-7B-v0.1"]
        assert config.quantize is True
        assert config.enable_retrieval is True
        assert config.aggregation_strategy == AggregationStrategy.MEAN
        assert config.batch_size == 1
        assert config.max_length == 512

    def test_load_nonexistent_preset(self):
        """Test loading a nonexistent preset raises ValueError."""
        manager = PresetManager()

        with pytest.raises(ValueError) as exc_info:
            manager.load_preset("nonexistent")

        assert "Unknown preset 'nonexistent'" in str(exc_info.value)
        assert "fast" in str(exc_info.value)
        assert "balanced" in str(exc_info.value)

    def test_list_presets(self):
        """Test listing all available presets."""
        manager = PresetManager()
        presets = manager.list_presets()

        assert len(presets) >= 2  # At least fast and balanced
        assert all(isinstance(p, PresetInfo) for p in presets)

        preset_names = [p.name for p in presets]
        assert "fast" in preset_names
        assert "balanced" in preset_names

        # Check that each preset has a description and config
        for preset in presets:
            assert preset.description
            assert isinstance(preset.config, ToolkitConfig)

    def test_get_preset_names(self):
        """Test getting list of preset names."""
        manager = PresetManager()
        names = manager.get_preset_names()

        assert isinstance(names, list)
        assert "fast" in names
        assert "balanced" in names

    def test_preset_exists(self):
        """Test checking if presets exist."""
        manager = PresetManager()

        assert manager.preset_exists("fast") is True
        assert manager.preset_exists("balanced") is True
        assert manager.preset_exists("nonexistent") is False

    def test_load_custom_preset_from_yaml(self):
        """Test loading custom presets from YAML files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preset_dir = Path(tmpdir)

            # Create a custom preset YAML file
            custom_preset = {
                "description": "Custom test preset",
                "verifier_model": "custom/verifier",
                "judge_models": ["custom/judge1"],
                "quantize": False,
                "enable_retrieval": True,
                "aggregation_strategy": "median",
                "batch_size": 2,
                "max_length": 256,
            }

            custom_yaml = preset_dir / "custom.yaml"
            with open(custom_yaml, "w") as f:
                yaml.dump(custom_preset, f)

            # Load presets with custom directory
            manager = PresetManager(preset_dir=preset_dir)

            # Check custom preset is available
            assert manager.preset_exists("custom")
            assert "custom" in manager.get_preset_names()

            # Load and verify custom preset
            config = manager.load_preset("custom")
            assert config.verifier_model == "custom/verifier"
            assert config.judge_models == ["custom/judge1"]
            assert config.quantize is False
            assert config.enable_retrieval is True
            assert config.aggregation_strategy == AggregationStrategy.MEDIAN
            assert config.batch_size == 2
            assert config.max_length == 256

    def test_custom_preset_does_not_override_builtin(self):
        """Test that custom presets cannot override built-in presets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preset_dir = Path(tmpdir)

            # Try to create a custom "fast" preset (should be ignored)
            custom_preset = {
                "description": "Malicious override",
                "verifier_model": "malicious/model",
                "judge_models": ["malicious/judge"],
            }

            fast_yaml = preset_dir / "fast.yaml"
            with open(fast_yaml, "w") as f:
                yaml.dump(custom_preset, f)

            manager = PresetManager(preset_dir=preset_dir)

            # Load "fast" preset - should get built-in, not custom
            config = manager.load_preset("fast")
            assert config.verifier_model == "MiniCheck/flan-t5-base-finetuned"
            assert config.verifier_model != "malicious/model"

    def test_invalid_yaml_file_is_skipped(self):
        """Test that invalid YAML files are skipped gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preset_dir = Path(tmpdir)

            # Create an invalid YAML file
            invalid_yaml = preset_dir / "invalid.yaml"
            with open(invalid_yaml, "w") as f:
                f.write("{ invalid yaml content [[[")

            # Should not raise exception, just skip the invalid file
            manager = PresetManager(preset_dir=preset_dir)
            assert "invalid" not in manager.get_preset_names()

    def test_preset_info_repr(self):
        """Test PresetInfo string representation."""
        config = ToolkitConfig()
        info = PresetInfo("test", "Test preset", config)

        repr_str = repr(info)
        assert "PresetInfo" in repr_str
        assert "test" in repr_str
        assert "Test preset" in repr_str

    def test_multiple_custom_presets(self):
        """Test loading multiple custom presets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preset_dir = Path(tmpdir)

            # Create multiple custom presets
            for i in range(3):
                preset = {
                    "description": f"Custom preset {i}",
                    "verifier_model": f"custom/verifier{i}",
                    "judge_models": [f"custom/judge{i}"],
                }
                yaml_file = preset_dir / f"custom{i}.yaml"
                with open(yaml_file, "w") as f:
                    yaml.dump(preset, f)

            manager = PresetManager(preset_dir=preset_dir)

            # All custom presets should be available
            names = manager.get_preset_names()
            assert "custom0" in names
            assert "custom1" in names
            assert "custom2" in names

            # Verify each can be loaded
            for i in range(3):
                config = manager.load_preset(f"custom{i}")
                assert config.verifier_model == f"custom/verifier{i}"
