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
- StreamingEvaluator: Streaming evaluation for large documents
- PluginRegistry: Plugin system for custom components
- AdversarialTester: Adversarial robustness testing
- ReliabilityValidator: Reliability validation and metrics
- VerifierTrainer: Fine-tuning specialized verifiers
- APIKeyManager: API key management for external judge services
- BaseJudgeClient: Base interface for API-based judge clients
"""

from llm_judge_auditor.components.adversarial_tester import (
    AdversarialTester,
    PerturbationResult,
    RobustnessReport,
    SymmetryReport,
)
from llm_judge_auditor.components.api_key_manager import (
    APIKeyManager,
    APIKeyStatus,
)
from llm_judge_auditor.components.base_judge_client import (
    BaseJudgeClient,
    JudgeVerdict,
)
from llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqAPIError,
    GroqRateLimitError,
    GroqAuthenticationError,
    GroqNetworkError,
)
from llm_judge_auditor.components.gemini_judge_client import (
    GeminiJudgeClient,
    GeminiAPIError,
    GeminiRateLimitError,
    GeminiAuthenticationError,
    GeminiNetworkError,
)
from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from llm_judge_auditor.components.aggregation_engine import (
    AggregationEngine,
    AggregationStrategy,
)
from llm_judge_auditor.components.claim_router import ClaimRouter
from llm_judge_auditor.components.device_manager import Device, DeviceManager
from llm_judge_auditor.components.judge_ensemble import JudgeEnsemble, PairwiseResult
from llm_judge_auditor.components.model_downloader import ModelDownloader
from llm_judge_auditor.components.model_manager import ModelInfo, ModelManager
from llm_judge_auditor.components.performance_tracker import (
    ComponentMetrics,
    Disagreement,
    PerformanceTracker,
)
from llm_judge_auditor.components.plugin_registry import (
    PluginMetadata,
    PluginRegistry,
)
from llm_judge_auditor.components.preset_manager import PresetInfo, PresetManager
from llm_judge_auditor.components.prompt_manager import PromptManager
from llm_judge_auditor.components.reliability_validator import (
    AgreementReport,
    ConsistencyReport,
    RankingCorrelationReport,
    ReliabilityValidator,
)
from llm_judge_auditor.components.report_generator import ReportGenerator
from llm_judge_auditor.components.retrieval_component import RetrievalComponent
from llm_judge_auditor.components.specialized_verifier import SpecializedVerifier
from llm_judge_auditor.components.streaming_evaluator import (
    PartialResult,
    StreamingEvaluator,
)
from llm_judge_auditor.components.verifier_trainer import (
    TrainingExample,
    VerifierDataset,
    VerifierTrainer,
)
from llm_judge_auditor.components.hallucination_metrics import (
    AbstentionResult,
    ClaimConsensus,
    ClaimVerificationMatrix,
    ClaimVerificationMatrixBuilder,
    ConsensusF1Result,
    FalseAcceptanceCalculator,
    FalseAcceptanceRateResult,
    HallucinationMetricsCalculator,
    HallucinationMetricsConfig,
    HallucinationProfile,
    KappaResult,
    MaHRResult,
    MiHRResult,
    ReliabilityLevel,
    UncertaintyResult,
)

__all__ = [
    "AbstentionResult",
    "AdversarialTester",
    "AggregationEngine",
    "AggregationStrategy",
    "AgreementReport",
    "APIJudgeEnsemble",
    "APIKeyManager",
    "APIKeyStatus",
    "BaseJudgeClient",
    "ClaimConsensus",
    "ClaimRouter",
    "ClaimVerificationMatrix",
    "ClaimVerificationMatrixBuilder",
    "ComponentMetrics",
    "ConsensusF1Result",
    "ConsistencyReport",
    "Device",
    "DeviceManager",
    "Disagreement",
    "FalseAcceptanceCalculator",
    "FalseAcceptanceRateResult",
    "GeminiJudgeClient",
    "GeminiAPIError",
    "GeminiRateLimitError",
    "GeminiAuthenticationError",
    "GeminiNetworkError",
    "GroqJudgeClient",
    "GroqAPIError",
    "GroqRateLimitError",
    "GroqAuthenticationError",
    "GroqNetworkError",
    "HallucinationMetricsCalculator",
    "HallucinationMetricsConfig",
    "HallucinationProfile",
    "JudgeEnsemble",
    "JudgeVerdict",
    "KappaResult",
    "MaHRResult",
    "MiHRResult",
    "ModelDownloader",
    "ModelInfo",
    "ModelManager",
    "PairwiseResult",
    "PartialResult",
    "PerformanceTracker",
    "PerturbationResult",
    "PluginMetadata",
    "PluginRegistry",
    "PresetInfo",
    "PresetManager",
    "PromptManager",
    "RankingCorrelationReport",
    "ReliabilityLevel",
    "ReliabilityValidator",
    "ReportGenerator",
    "RetrievalComponent",
    "RobustnessReport",
    "SpecializedVerifier",
    "StreamingEvaluator",
    "SymmetryReport",
    "TrainingExample",
    "UncertaintyResult",
    "VerifierDataset",
    "VerifierTrainer",
]
