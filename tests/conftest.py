"""
Pytest configuration and shared fixtures.

This module provides common fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
from llm_judge_auditor.config import ToolkitConfig, AggregationStrategy


@pytest.fixture
def sample_config():
    """Provide a basic test configuration."""
    return ToolkitConfig(
        verifier_model="test-verifier",
        judge_models=["test-judge-1", "test-judge-2"],
        quantize=False,
        enable_retrieval=False,
        aggregation_strategy=AggregationStrategy.MEAN,
        batch_size=1,
        max_length=128,
        num_iterations=10,  # Fewer iterations for faster tests
    )


@pytest.fixture
def sample_source_text():
    """Provide sample source text for testing."""
    return "The Eiffel Tower is located in Paris, France. It was completed in 1889."


@pytest.fixture
def sample_candidate_output():
    """Provide sample candidate output for testing."""
    return "The Eiffel Tower is in Paris and was built in 1889."


@pytest.fixture
def sample_hallucinated_output():
    """Provide sample output with hallucinations."""
    return "The Eiffel Tower is in London and was built in 1920."


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir
