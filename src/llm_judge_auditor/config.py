"""
Configuration schema for the LLM Judge Auditor toolkit.

This module defines the configuration structure using Pydantic for validation
and type safety. It includes the main ToolkitConfig class and related enums.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AggregationStrategy(str, Enum):
    """Strategy for aggregating scores from multiple judge models."""

    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"


class DeviceType(str, Enum):
    """Available device types for model execution."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"  # Apple Silicon
    AUTO = "auto"  # Auto-detect best available


class APIConfig(BaseModel):
    """
    Configuration for API-based judge services.
    
    This configuration controls API judge settings including API keys,
    model selection, retry behavior, and parallel execution.
    """
    
    # API Keys (loaded from environment by default)
    groq_api_key: Optional[str] = Field(
        default=None,
        description="Groq API key (defaults to GROQ_API_KEY environment variable)"
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Gemini API key (defaults to GEMINI_API_KEY environment variable)"
    )
    
    # Model selection
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model to use for evaluation"
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Gemini model to use for evaluation"
    )
    
    # API behavior
    timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Timeout for API calls in seconds"
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of retries for failed API calls"
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Base delay for exponential backoff in seconds"
    )
    
    # Execution settings
    parallel_execution: bool = Field(
        default=True,
        description="Execute API judges in parallel for faster evaluation"
    )
    
    # Feature flags
    enable_api_judges: bool = Field(
        default=True,
        description="Enable API-based judges (Groq, Gemini)"
    )
    
    def load_keys_from_env(self) -> None:
        """
        Load API keys from environment variables if not already set.
        
        Checks for:
        - GROQ_API_KEY
        - GEMINI_API_KEY
        """
        if self.groq_api_key is None:
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
        
        if self.gemini_api_key is None:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    def has_any_keys(self) -> bool:
        """
        Check if any API keys are configured.
        
        Returns:
            True if at least one API key is available
        """
        return bool(self.groq_api_key or self.gemini_api_key)


class ToolkitConfig(BaseModel):
    """
    Main configuration for the Evaluation Toolkit.

    This configuration controls all aspects of the evaluation pipeline including
    model selection, retrieval settings, aggregation strategy, and performance parameters.
    """

    # Model configuration
    verifier_model: Optional[str] = Field(
        default="MiniCheck/flan-t5-large-finetuned",
        description="HuggingFace model identifier for the specialized verifier (optional if using API judges)",
    )
    judge_models: List[str] = Field(
        default=[],
        description="List of HuggingFace model identifiers for judge ensemble (optional if using API judges)",
    )
    quantize: bool = Field(
        default=True, description="Enable 8-bit quantization to reduce memory usage"
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO, description="Device to run models on (cpu, cuda, mps, auto)"
    )
    
    # API configuration
    api_config: APIConfig = Field(
        default_factory=APIConfig,
        description="Configuration for API-based judge services"
    )

    # Retrieval configuration
    knowledge_base_path: Optional[Path] = Field(
        default=None, description="Path to knowledge base for retrieval-augmented verification"
    )
    retrieval_top_k: int = Field(
        default=3, ge=1, le=10, description="Number of passages to retrieve per claim"
    )
    enable_retrieval: bool = Field(
        default=False, description="Enable retrieval-augmented verification"
    )

    # Aggregation configuration
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.MEAN, description="Strategy for combining judge scores"
    )
    judge_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Weights for each judge model (only used with weighted_average strategy)",
    )
    disagreement_threshold: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Variance threshold for flagging low-confidence evaluations",
    )

    # Prompt configuration
    prompt_template_path: Optional[Path] = Field(
        default=None, description="Path to custom prompt templates"
    )
    custom_criteria: Optional[List[str]] = Field(
        default=None, description="Custom evaluation criteria beyond factual accuracy"
    )

    # Performance configuration
    batch_size: int = Field(default=1, ge=1, le=32, description="Batch size for model inference")
    max_length: int = Field(
        default=512, ge=128, le=2048, description="Maximum sequence length for models"
    )
    num_iterations: int = Field(
        default=100, ge=1, description="Number of iterations for property-based testing"
    )

    # Cache configuration
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "llm-judge-auditor",
        description="Directory for caching models and data",
    )

    @field_validator("judge_weights")
    @classmethod
    def validate_judge_weights(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate that judge weights sum to 1.0 if provided."""
        if v is not None:
            total = sum(v.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Judge weights must sum to 1.0, got {total}")
        return v

    @field_validator("cache_dir", mode="before")
    @classmethod
    def ensure_path(cls, v) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @classmethod
    def from_preset(cls, preset_name: str) -> "ToolkitConfig":
        """
        Load a pre-configured preset.

        Args:
            preset_name: Name of the preset ("fast", "balanced", "strict", "research")

        Returns:
            ToolkitConfig instance with preset values

        Raises:
            ValueError: If preset_name is not recognized
        """
        presets = {
            "fast": PresetConfig.fast(),
            "balanced": PresetConfig.balanced(),
            "strict": PresetConfig.strict(),
            "research": PresetConfig.research(),
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}"
            )

        return presets[preset_name]

    def model_dump_yaml(self) -> str:
        """Export configuration as YAML string."""
        import yaml

        return yaml.dump(self.model_dump(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ToolkitConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class PresetConfig:
    """Factory class for creating preset configurations."""

    @staticmethod
    def fast() -> ToolkitConfig:
        """
        Fast preset: Minimal processing for quick evaluations.

        - No retrieval
        - API judges preferred (faster than local models)
        - Quantization enabled for local models
        - Small batch size
        """
        api_config = APIConfig()
        api_config.load_keys_from_env()
        
        return ToolkitConfig(
            verifier_model=None,  # Optional when using API judges
            judge_models=[],  # Use API judges if available
            quantize=True,
            enable_retrieval=False,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=512,
            api_config=api_config,
        )

    @staticmethod
    def balanced() -> ToolkitConfig:
        """
        Balanced preset: Good accuracy with reasonable resource usage.

        - Retrieval enabled
        - API judges preferred (no model downloads required)
        - Verifier optional
        - Standard settings
        """
        api_config = APIConfig()
        api_config.load_keys_from_env()
        
        return ToolkitConfig(
            verifier_model=None,  # Optional when using API judges
            judge_models=[],  # Use API judges if available
            quantize=True,
            enable_retrieval=True,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=512,
            api_config=api_config,
        )

    @staticmethod
    def strict() -> ToolkitConfig:
        """
        Strict preset: Maximum accuracy with full pipeline.

        - Retrieval enabled
        - API judges for ensemble
        - Mean aggregation
        - Lower disagreement threshold
        """
        api_config = APIConfig()
        api_config.load_keys_from_env()
        
        return ToolkitConfig(
            verifier_model=None,  # Optional when using API judges
            judge_models=[],  # Use API judges if available
            quantize=True,
            enable_retrieval=True,
            aggregation_strategy=AggregationStrategy.MEAN,
            disagreement_threshold=15.0,
            batch_size=1,
            max_length=1024,
            api_config=api_config,
        )

    @staticmethod
    def research() -> ToolkitConfig:
        """
        Research preset: Maximum transparency and metrics.

        - All features enabled
        - API judges for evaluation
        - High iteration count for property testing
        - Detailed reporting
        """
        api_config = APIConfig()
        api_config.load_keys_from_env()
        
        return ToolkitConfig(
            verifier_model=None,  # Optional when using API judges
            judge_models=[],  # Use API judges if available
            quantize=True,
            enable_retrieval=True,
            aggregation_strategy=AggregationStrategy.MEAN,
            batch_size=1,
            max_length=1024,
            num_iterations=200,  # More iterations for thorough testing
            api_config=api_config,
        )
