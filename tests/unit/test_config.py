"""
Unit tests for configuration module.

Tests the ToolkitConfig class and preset configurations.
"""

import pytest
from pathlib import Path
from llm_judge_auditor.config import (
    ToolkitConfig,
    AggregationStrategy,
    DeviceType,
    PresetConfig,
    APIConfig,
)


class TestToolkitConfig:
    """Test suite for ToolkitConfig."""

    def test_default_config_creation(self):
        """Test creating a config with default values."""
        config = ToolkitConfig()
        # Verifier is now optional when using API judges
        assert config.verifier_model is not None or config.api_config is not None
        # Judge models can be empty if using API judges
        assert config.judge_models is not None
        assert config.quantize is True
        assert config.device == DeviceType.AUTO
        # API config should be present
        assert config.api_config is not None
        assert config.api_config.enable_api_judges is True

    def test_custom_config_creation(self):
        """Test creating a config with custom values."""
        config = ToolkitConfig(
            verifier_model="custom-verifier",
            judge_models=["judge-1", "judge-2", "judge-3"],
            quantize=False,
            device=DeviceType.CPU,
            batch_size=4,
        )
        assert config.verifier_model == "custom-verifier"
        assert len(config.judge_models) == 3
        assert config.quantize is False
        assert config.device == DeviceType.CPU
        assert config.batch_size == 4

    def test_aggregation_strategy_enum(self):
        """Test aggregation strategy enum values."""
        config = ToolkitConfig(aggregation_strategy=AggregationStrategy.MEDIAN)
        assert config.aggregation_strategy == AggregationStrategy.MEDIAN

    def test_judge_weights_validation_valid(self):
        """Test that valid judge weights are accepted."""
        weights = {"judge-1": 0.5, "judge-2": 0.5}
        config = ToolkitConfig(
            judge_models=["judge-1", "judge-2"],
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
            judge_weights=weights,
        )
        assert config.judge_weights == weights

    def test_judge_weights_validation_invalid(self):
        """Test that invalid judge weights are rejected."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ToolkitConfig(
                judge_models=["judge-1", "judge-2"],
                judge_weights={"judge-1": 0.3, "judge-2": 0.5},  # Sum = 0.8
            )

    def test_retrieval_config(self):
        """Test retrieval configuration."""
        config = ToolkitConfig(
            enable_retrieval=True,
            knowledge_base_path=Path("/path/to/kb"),
            retrieval_top_k=5,
        )
        assert config.enable_retrieval is True
        assert config.knowledge_base_path == Path("/path/to/kb")
        assert config.retrieval_top_k == 5

    def test_cache_dir_default(self):
        """Test that cache_dir has a sensible default."""
        config = ToolkitConfig()
        assert config.cache_dir.name == "llm-judge-auditor"
        assert ".cache" in str(config.cache_dir)
    
    def test_api_config_integration(self):
        """Test that APIConfig is properly integrated into ToolkitConfig."""
        api_config = APIConfig(
            groq_api_key="test_groq_key",
            gemini_api_key="test_gemini_key",
            parallel_execution=False
        )
        config = ToolkitConfig(api_config=api_config)
        assert config.api_config.groq_api_key == "test_groq_key"
        assert config.api_config.gemini_api_key == "test_gemini_key"
        assert config.api_config.parallel_execution is False


class TestAPIConfig:
    """Test suite for APIConfig."""
    
    def test_default_api_config(self):
        """Test creating APIConfig with default values."""
        config = APIConfig()
        assert config.groq_api_key is None
        assert config.gemini_api_key is None
        assert config.groq_model == "llama-3.3-70b-versatile"
        assert config.gemini_model == "gemini-2.0-flash-exp"
        assert config.timeout == 30
        assert config.max_retries == 2
        assert config.parallel_execution is True
        assert config.enable_api_judges is True
    
    def test_custom_api_config(self):
        """Test creating APIConfig with custom values."""
        config = APIConfig(
            groq_api_key="test_key",
            groq_model="custom-model",
            timeout=60,
            max_retries=5,
            parallel_execution=False
        )
        assert config.groq_api_key == "test_key"
        assert config.groq_model == "custom-model"
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.parallel_execution is False
    
    def test_has_any_keys(self):
        """Test has_any_keys method."""
        # No keys
        config = APIConfig()
        assert config.has_any_keys() is False
        
        # Only Groq key
        config = APIConfig(groq_api_key="test_key")
        assert config.has_any_keys() is True
        
        # Only Gemini key
        config = APIConfig(gemini_api_key="test_key")
        assert config.has_any_keys() is True
        
        # Both keys
        config = APIConfig(groq_api_key="key1", gemini_api_key="key2")
        assert config.has_any_keys() is True


class TestPresetConfig:
    """Test suite for preset configurations."""

    def test_fast_preset(self):
        """Test fast preset configuration."""
        config = PresetConfig.fast()
        assert config.enable_retrieval is False
        # Fast preset now uses API judges by default
        assert config.api_config is not None
        assert config.api_config.enable_api_judges is True
        assert config.quantize is True

    def test_balanced_preset(self):
        """Test balanced preset configuration."""
        config = PresetConfig.balanced()
        assert config.enable_retrieval is True
        # Balanced preset now uses API judges by default
        assert config.api_config is not None
        assert config.api_config.enable_api_judges is True
        assert config.quantize is True

    def test_strict_preset(self):
        """Test strict preset configuration."""
        config = PresetConfig.strict()
        assert config.enable_retrieval is True
        # Strict preset now uses API judges by default
        assert config.api_config is not None
        assert config.api_config.enable_api_judges is True
        assert config.aggregation_strategy == AggregationStrategy.MEAN
        assert config.disagreement_threshold == 15.0

    def test_research_preset(self):
        """Test research preset configuration."""
        config = PresetConfig.research()
        assert config.enable_retrieval is True
        # Research preset now uses API judges by default
        assert config.api_config is not None
        assert config.api_config.enable_api_judges is True
        assert config.num_iterations == 200

    def test_from_preset_method(self):
        """Test loading preset via from_preset method."""
        config = ToolkitConfig.from_preset("balanced")
        assert config.enable_retrieval is True
        # Balanced preset now uses API judges by default
        assert config.api_config is not None
        assert config.api_config.enable_api_judges is True

    def test_from_preset_invalid_name(self):
        """Test that invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown preset"):
            ToolkitConfig.from_preset("invalid_preset")


class TestConfigValidation:
    """Test suite for configuration validation."""

    def test_batch_size_validation(self):
        """Test batch size must be positive."""
        with pytest.raises(ValueError):
            ToolkitConfig(batch_size=0)

    def test_retrieval_top_k_validation(self):
        """Test retrieval_top_k bounds."""
        with pytest.raises(ValueError):
            ToolkitConfig(retrieval_top_k=0)

        with pytest.raises(ValueError):
            ToolkitConfig(retrieval_top_k=20)

    def test_disagreement_threshold_validation(self):
        """Test disagreement_threshold bounds."""
        with pytest.raises(ValueError):
            ToolkitConfig(disagreement_threshold=-1.0)

        with pytest.raises(ValueError):
            ToolkitConfig(disagreement_threshold=150.0)
