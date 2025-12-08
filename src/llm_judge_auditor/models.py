"""
Core data models for the LLM Judge Auditor toolkit.

This module defines the data structures used throughout the evaluation pipeline,
including claims, passages, verdicts, and evaluation results.
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class VerdictLabel(str, Enum):
    """Classification labels for statement verification."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"


class IssueType(str, Enum):
    """Types of issues that can be detected in candidate outputs."""

    HALLUCINATION = "hallucination"
    BIAS = "bias"
    INCONSISTENCY = "inconsistency"
    FACTUAL_ERROR = "factual_error"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    NUMERICAL_ERROR = "numerical_error"


class IssueSeverity(str, Enum):
    """Severity levels for detected issues."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ClaimType(str, Enum):
    """Types of claims for specialized routing."""

    FACTUAL = "factual"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    LOGICAL = "logical"
    COMMONSENSE = "commonsense"


@dataclass
class Claim:
    """
    A single claim extracted from candidate output.

    Attributes:
        text: The claim text
        source_span: Character offsets (start, end) in original text
        claim_type: Type of claim for specialized routing
    """

    text: str
    source_span: Tuple[int, int]
    claim_type: ClaimType = ClaimType.FACTUAL


@dataclass
class Passage:
    """
    A passage retrieved from a knowledge base.

    Attributes:
        text: The passage text
        source: Knowledge base identifier (e.g., "Wikipedia:Article_Name")
        relevance_score: Similarity score to the query claim
    """

    text: str
    source: str
    relevance_score: float


@dataclass
class Issue:
    """
    An issue detected during evaluation.

    Attributes:
        type: Type of issue (hallucination, bias, etc.)
        severity: Severity level (low, medium, high)
        description: Human-readable description
        evidence: Supporting evidence for the issue
    """

    type: IssueType
    severity: IssueSeverity
    description: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class Verdict:
    """
    Result from specialized verifier for a single statement.

    Attributes:
        label: Classification (SUPPORTED, REFUTED, NOT_ENOUGH_INFO)
        confidence: Confidence score (0.0 to 1.0)
        evidence: Evidence passages used for verification
        reasoning: Explanation of the verdict
    """

    label: VerdictLabel
    confidence: float
    evidence: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class JudgeResult:
    """
    Result from a single judge model evaluation.

    Attributes:
        model_name: Name of the judge model
        score: Factual accuracy score (0-100)
        reasoning: Chain-of-thought explanation
        flagged_issues: Issues detected by this judge
        confidence: Judge's confidence in the evaluation
    """

    model_name: str
    score: float
    reasoning: str
    flagged_issues: List[Issue] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class EvaluationRequest:
    """
    Request for evaluating a candidate output.

    Attributes:
        source_text: Reference document or context
        candidate_output: Text to be evaluated
        task: Evaluation task type
        criteria: Evaluation criteria to apply
        use_retrieval: Whether to use retrieval-augmented verification
    """

    source_text: str
    candidate_output: str
    task: str = "factual_accuracy"
    criteria: List[str] = field(default_factory=lambda: ["correctness"])
    use_retrieval: bool = False


@dataclass
class AggregationMetadata:
    """
    Metadata about the aggregation process.

    Attributes:
        strategy: Aggregation strategy used
        individual_scores: Scores from each judge
        variance: Variance in judge scores
        is_low_confidence: Whether disagreement exceeded threshold
        weights: Weights applied (if weighted average)
    """

    strategy: str
    individual_scores: Dict[str, float]
    variance: float
    is_low_confidence: bool
    weights: Optional[Dict[str, float]] = None


@dataclass
class Report:
    """
    Comprehensive evaluation report.

    Attributes:
        metadata: Evaluation metadata (timestamp, models, parameters)
        consensus_score: Final aggregated score
        individual_scores: Per-judge scores
        verifier_verdicts: Statement-level verdicts from verifier
        retrieval_provenance: Retrieved passages used
        reasoning: Per-judge reasoning explanations
        confidence: Overall confidence in evaluation
        disagreement_level: Level of disagreement among judges
        flagged_issues: All detected issues
        hallucination_categories: Count of hallucinations by type
    """

    metadata: Dict
    consensus_score: float
    individual_scores: Dict[str, float]
    verifier_verdicts: List[Verdict]
    retrieval_provenance: List[Passage]
    reasoning: Dict[str, str]
    confidence: float
    disagreement_level: float
    flagged_issues: List[Issue]
    hallucination_categories: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Report to a dictionary with proper enum handling.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return _dataclass_to_dict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Convert Report to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class EvaluationResult:
    """
    Complete result from an evaluation.

    Attributes:
        request: Original evaluation request
        consensus_score: Final aggregated score
        verifier_verdicts: Statement-level verdicts
        judge_results: Individual judge results
        aggregation_metadata: Metadata about aggregation
        report: Full evaluation report
    """

    request: EvaluationRequest
    consensus_score: float
    verifier_verdicts: List[Verdict]
    judge_results: List[JudgeResult]
    aggregation_metadata: AggregationMetadata
    report: Report
    flagged_issues: List[Issue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert EvaluationResult to a dictionary with proper enum handling.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return _dataclass_to_dict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Convert EvaluationResult to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """
        Create EvaluationResult from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            EvaluationResult instance
        """
        # Convert nested structures
        request = EvaluationRequest(**data["request"])

        verifier_verdicts = [
            Verdict(
                label=VerdictLabel(v["label"]),
                confidence=v["confidence"],
                evidence=v.get("evidence", []),
                reasoning=v.get("reasoning", ""),
            )
            for v in data["verifier_verdicts"]
        ]

        judge_results = [
            JudgeResult(
                model_name=j["model_name"],
                score=j["score"],
                reasoning=j["reasoning"],
                flagged_issues=[
                    Issue(
                        type=IssueType(i["type"]),
                        severity=IssueSeverity(i["severity"]),
                        description=i["description"],
                        evidence=i.get("evidence", []),
                    )
                    for i in j.get("flagged_issues", [])
                ],
                confidence=j.get("confidence", 1.0),
            )
            for j in data["judge_results"]
        ]

        aggregation_metadata = AggregationMetadata(**data["aggregation_metadata"])

        report_data = data["report"]
        report = Report(
            metadata=report_data["metadata"],
            consensus_score=report_data["consensus_score"],
            individual_scores=report_data["individual_scores"],
            verifier_verdicts=[
                Verdict(
                    label=VerdictLabel(v["label"]),
                    confidence=v["confidence"],
                    evidence=v.get("evidence", []),
                    reasoning=v.get("reasoning", ""),
                )
                for v in report_data["verifier_verdicts"]
            ],
            retrieval_provenance=[Passage(**p) for p in report_data["retrieval_provenance"]],
            reasoning=report_data["reasoning"],
            confidence=report_data["confidence"],
            disagreement_level=report_data["disagreement_level"],
            flagged_issues=[
                Issue(
                    type=IssueType(i["type"]),
                    severity=IssueSeverity(i["severity"]),
                    description=i["description"],
                    evidence=i.get("evidence", []),
                )
                for i in report_data["flagged_issues"]
            ],
            hallucination_categories=report_data["hallucination_categories"],
        )

        flagged_issues = [
            Issue(
                type=IssueType(i["type"]),
                severity=IssueSeverity(i["severity"]),
                description=i["description"],
                evidence=i.get("evidence", []),
            )
            for i in data.get("flagged_issues", [])
        ]

        return cls(
            request=request,
            consensus_score=data["consensus_score"],
            verifier_verdicts=verifier_verdicts,
            judge_results=judge_results,
            aggregation_metadata=aggregation_metadata,
            report=report,
            flagged_issues=flagged_issues,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationResult":
        """
        Create EvaluationResult from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            EvaluationResult instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class BatchResult:
    """
    Result from batch evaluation of multiple requests.

    Attributes:
        results: List of successful evaluation results
        errors: List of errors that occurred during batch processing
        statistics: Summary statistics (mean, median, etc.)
        metadata: Batch processing metadata (timestamp, total requests, etc.)
    """

    results: List[EvaluationResult]
    errors: List[Dict[str, Any]]
    statistics: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert BatchResult to a dictionary with proper enum handling.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return _dataclass_to_dict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Convert BatchResult to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    def save_to_file(self, filepath: str) -> None:
        """
        Save batch results to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())


def _dataclass_to_dict(obj: Any) -> Any:
    """
    Convert a dataclass to a dictionary, handling enums and nested structures.

    Args:
        obj: Object to convert (dataclass, enum, list, dict, or primitive)

    Returns:
        Dictionary representation suitable for JSON serialization
    """
    if isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, "__dataclass_fields__"):
        # It's a dataclass
        result = {}
        for field_name, field_value in asdict(obj).items():
            result[field_name] = _dataclass_to_dict(field_value)
        return result
    elif isinstance(obj, dict):
        return {key: _dataclass_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    else:
        return obj
