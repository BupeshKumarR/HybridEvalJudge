"""
LLM Judge Auditor - Hybrid LLM Evaluation Toolkit

A comprehensive evaluation system combining specialized fact-checking models
with judge LLM ensembles for assessing factual accuracy, hallucinations, and bias.
"""

from llm_judge_auditor.components.streaming_evaluator import (
    PartialResult,
    StreamingEvaluator,
)
from llm_judge_auditor.components.verifier_trainer import (
    TrainingExample,
    VerifierTrainer,
)
from llm_judge_auditor.config import (
    AggregationStrategy,
    DeviceType,
    PresetConfig,
    ToolkitConfig,
)
from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit
from llm_judge_auditor.models import (
    BatchResult,
    Claim,
    ClaimType,
    EvaluationRequest,
    EvaluationResult,
    Issue,
    IssueSeverity,
    IssueType,
    JudgeResult,
    Passage,
    Verdict,
    VerdictLabel,
)

__version__ = "0.1.0"

__all__ = [
    "EvaluationToolkit",
    "StreamingEvaluator",
    "PartialResult",
    "ToolkitConfig",
    "AggregationStrategy",
    "PresetConfig",
    "DeviceType",
    "BatchResult",
    "Claim",
    "Passage",
    "Issue",
    "Verdict",
    "EvaluationRequest",
    "EvaluationResult",
    "JudgeResult",
    "VerdictLabel",
    "IssueType",
    "IssueSeverity",
    "ClaimType",
    "TrainingExample",
    "VerifierTrainer",
]
