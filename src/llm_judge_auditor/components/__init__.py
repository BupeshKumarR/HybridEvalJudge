"""
Core components for the evaluation pipeline.

This package contains the main components:
- DeviceManager: Hardware detection and optimization
- ModelManager: Model loading and initialization
- RetrievalComponent: Claim extraction and passage retrieval
- SpecializedVerifier: Statement-level fact-checking
- JudgeEnsemble: Multi-model evaluation
- PromptManager: Prompt template management
- AggregationEngine: Score aggregation
- ReportGenerator: Report generation
"""

from llm_judge_auditor.components.aggregation_engine import (
    AggregationEngine,
    AggregationStrategy,
)
from llm_judge_auditor.components.device_manager import Device, DeviceManager
from llm_judge_auditor.components.judge_ensemble import JudgeEnsemble, PairwiseResult
from llm_judge_auditor.components.model_downloader import ModelDownloader
from llm_judge_auditor.components.model_manager import ModelInfo, ModelManager
from llm_judge_auditor.components.preset_manager import PresetInfo, PresetManager
from llm_judge_auditor.components.prompt_manager import PromptManager
from llm_judge_auditor.components.report_generator import ReportGenerator
from llm_judge_auditor.components.retrieval_component import RetrievalComponent
from llm_judge_auditor.components.specialized_verifier import SpecializedVerifier

__all__ = [
    "AggregationEngine",
    "AggregationStrategy",
    "Device",
    "DeviceManager",
    "JudgeEnsemble",
    "ModelDownloader",
    "ModelInfo",
    "ModelManager",
    "PairwiseResult",
    "PresetInfo",
    "PresetManager",
    "PromptManager",
    "ReportGenerator",
    "RetrievalComponent",
    "SpecializedVerifier",
]
