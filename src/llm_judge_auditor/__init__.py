"""
LLM Judge Auditor - Hybrid LLM Evaluation Toolkit

A comprehensive evaluation system combining specialized fact-checking models
with judge LLM ensembles for assessing factual accuracy, hallucinations, and bias.
"""

from llm_judge_auditor.config import ToolkitConfig, AggregationStrategy, PresetConfig, DeviceType
from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit
from llm_judge_auditor.models import (
    BatchResult,
    Claim,
    Passage,
    Issue,
    Verdict,
    EvaluationRequest,
    EvaluationResult,
    JudgeResult,
    VerdictLabel,
    IssueType,
    IssueSeverity,
    ClaimType,
)

__version__ = "0.1.0"

__all__ = [
    "EvaluationToolkit",
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
]
