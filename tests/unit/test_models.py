"""
Unit tests for core data models.

Tests the data structures and JSON serialization/deserialization functionality.
"""

import json
from datetime import datetime

import pytest

from llm_judge_auditor.models import (
    AggregationMetadata,
    Claim,
    ClaimType,
    EvaluationRequest,
    EvaluationResult,
    Issue,
    IssueType,
    IssueSeverity,
    JudgeResult,
    Passage,
    Report,
    Verdict,
    VerdictLabel,
)


class TestClaim:
    """Tests for Claim dataclass."""

    def test_claim_creation(self):
        """Test basic claim creation."""
        claim = Claim(
            text="The Earth orbits the Sun.",
            source_span=(0, 27),
            claim_type=ClaimType.FACTUAL,
        )
        assert claim.text == "The Earth orbits the Sun."
        assert claim.source_span == (0, 27)
        assert claim.claim_type == ClaimType.FACTUAL

    def test_claim_default_type(self):
        """Test claim with default type."""
        claim = Claim(text="Test claim", source_span=(0, 10))
        assert claim.claim_type == ClaimType.FACTUAL


class TestPassage:
    """Tests for Passage dataclass."""

    def test_passage_creation(self):
        """Test basic passage creation."""
        passage = Passage(
            text="The Earth is the third planet from the Sun.",
            source="Wikipedia:Earth",
            relevance_score=0.95,
        )
        assert passage.text == "The Earth is the third planet from the Sun."
        assert passage.source == "Wikipedia:Earth"
        assert passage.relevance_score == 0.95


class TestIssue:
    """Tests for Issue dataclass."""

    def test_issue_creation(self):
        """Test basic issue creation."""
        issue = Issue(
            type=IssueType.HALLUCINATION,
            severity=IssueSeverity.HIGH,
            description="Unsupported claim about Mars",
            evidence=["Source text does not mention Mars"],
        )
        assert issue.type == IssueType.HALLUCINATION
        assert issue.severity == IssueSeverity.HIGH
        assert issue.description == "Unsupported claim about Mars"
        assert len(issue.evidence) == 1

    def test_issue_default_evidence(self):
        """Test issue with default empty evidence list."""
        issue = Issue(
            type=IssueType.BIAS,
            severity=IssueSeverity.MEDIUM,
            description="Biased language detected",
        )
        assert issue.evidence == []


class TestVerdict:
    """Tests for Verdict dataclass."""

    def test_verdict_creation(self):
        """Test basic verdict creation."""
        verdict = Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.92,
            evidence=["Earth orbits Sun - confirmed"],
            reasoning="The claim is supported by astronomical facts.",
        )
        assert verdict.label == VerdictLabel.SUPPORTED
        assert verdict.confidence == 0.92
        assert len(verdict.evidence) == 1
        assert "astronomical facts" in verdict.reasoning

    def test_verdict_defaults(self):
        """Test verdict with default values."""
        verdict = Verdict(label=VerdictLabel.NOT_ENOUGH_INFO, confidence=0.5)
        assert verdict.evidence == []
        assert verdict.reasoning == ""


class TestJudgeResult:
    """Tests for JudgeResult dataclass."""

    def test_judge_result_creation(self):
        """Test basic judge result creation."""
        issue = Issue(
            type=IssueType.FACTUAL_ERROR,
            severity=IssueSeverity.LOW,
            description="Minor inaccuracy",
        )
        result = JudgeResult(
            model_name="llama-3-8b",
            score=85.5,
            reasoning="The output is mostly accurate with minor issues.",
            flagged_issues=[issue],
            confidence=0.88,
        )
        assert result.model_name == "llama-3-8b"
        assert result.score == 85.5
        assert len(result.flagged_issues) == 1
        assert result.confidence == 0.88

    def test_judge_result_defaults(self):
        """Test judge result with default values."""
        result = JudgeResult(
            model_name="mistral-7b", score=90.0, reasoning="Good output"
        )
        assert result.flagged_issues == []
        assert result.confidence == 1.0


class TestEvaluationRequest:
    """Tests for EvaluationRequest dataclass."""

    def test_evaluation_request_creation(self):
        """Test basic evaluation request creation."""
        request = EvaluationRequest(
            source_text="The sky is blue.",
            candidate_output="The sky appears blue due to Rayleigh scattering.",
            task="factual_accuracy",
            criteria=["correctness", "completeness"],
            use_retrieval=True,
        )
        assert request.source_text == "The sky is blue."
        assert "Rayleigh scattering" in request.candidate_output
        assert request.task == "factual_accuracy"
        assert len(request.criteria) == 2
        assert request.use_retrieval is True

    def test_evaluation_request_defaults(self):
        """Test evaluation request with default values."""
        request = EvaluationRequest(
            source_text="Source", candidate_output="Output"
        )
        assert request.task == "factual_accuracy"
        assert request.criteria == ["correctness"]
        assert request.use_retrieval is False


class TestAggregationMetadata:
    """Tests for AggregationMetadata dataclass."""

    def test_aggregation_metadata_creation(self):
        """Test basic aggregation metadata creation."""
        metadata = AggregationMetadata(
            strategy="mean",
            individual_scores={"judge1": 85.0, "judge2": 90.0},
            variance=12.5,
            is_low_confidence=False,
            weights={"judge1": 0.5, "judge2": 0.5},
        )
        assert metadata.strategy == "mean"
        assert len(metadata.individual_scores) == 2
        assert metadata.variance == 12.5
        assert metadata.is_low_confidence is False
        assert metadata.weights is not None

    def test_aggregation_metadata_no_weights(self):
        """Test aggregation metadata without weights."""
        metadata = AggregationMetadata(
            strategy="median",
            individual_scores={"judge1": 80.0},
            variance=0.0,
            is_low_confidence=False,
        )
        assert metadata.weights is None


class TestReport:
    """Tests for Report dataclass."""

    def test_report_creation(self):
        """Test basic report creation."""
        verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9)
        passage = Passage(text="Test passage", source="Test", relevance_score=0.8)
        issue = Issue(
            type=IssueType.HALLUCINATION,
            severity=IssueSeverity.LOW,
            description="Test issue",
        )

        report = Report(
            metadata={"timestamp": "2024-01-01T00:00:00", "version": "1.0"},
            consensus_score=87.5,
            individual_scores={"judge1": 85.0, "judge2": 90.0},
            verifier_verdicts=[verdict],
            retrieval_provenance=[passage],
            reasoning={"judge1": "Good", "judge2": "Excellent"},
            confidence=0.85,
            disagreement_level=5.0,
            flagged_issues=[issue],
            hallucination_categories={"factual_error": 1},
        )

        assert report.consensus_score == 87.5
        assert len(report.verifier_verdicts) == 1
        assert len(report.retrieval_provenance) == 1
        assert len(report.flagged_issues) == 1
        assert report.hallucination_categories["factual_error"] == 1

    def test_report_to_dict(self):
        """Test report conversion to dictionary."""
        verdict = Verdict(label=VerdictLabel.SUPPORTED, confidence=0.9)
        report = Report(
            metadata={"test": "data"},
            consensus_score=85.0,
            individual_scores={"judge1": 85.0},
            verifier_verdicts=[verdict],
            retrieval_provenance=[],
            reasoning={"judge1": "Good"},
            confidence=0.9,
            disagreement_level=0.0,
            flagged_issues=[],
            hallucination_categories={},
        )

        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert report_dict["consensus_score"] == 85.0
        assert report_dict["verifier_verdicts"][0]["label"] == "SUPPORTED"

    def test_report_to_json(self):
        """Test report conversion to JSON."""
        verdict = Verdict(label=VerdictLabel.REFUTED, confidence=0.8)
        report = Report(
            metadata={"test": "data"},
            consensus_score=45.0,
            individual_scores={"judge1": 45.0},
            verifier_verdicts=[verdict],
            retrieval_provenance=[],
            reasoning={"judge1": "Poor"},
            confidence=0.8,
            disagreement_level=0.0,
            flagged_issues=[],
            hallucination_categories={},
        )

        json_str = report.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["consensus_score"] == 45.0
        assert parsed["verifier_verdicts"][0]["label"] == "REFUTED"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass and JSON serialization."""

    def create_sample_result(self) -> EvaluationResult:
        """Create a sample evaluation result for testing."""
        request = EvaluationRequest(
            source_text="The Earth is round.",
            candidate_output="The Earth is spherical.",
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        )

        verdict = Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.95,
            evidence=["Earth's shape is well-documented"],
            reasoning="The claim is scientifically accurate.",
        )

        issue = Issue(
            type=IssueType.FACTUAL_ERROR,
            severity=IssueSeverity.LOW,
            description="Minor terminology difference",
            evidence=["Round vs spherical"],
        )

        judge_result = JudgeResult(
            model_name="llama-3-8b",
            score=92.0,
            reasoning="Accurate statement with appropriate terminology.",
            flagged_issues=[issue],
            confidence=0.9,
        )

        aggregation_metadata = AggregationMetadata(
            strategy="mean",
            individual_scores={"llama-3-8b": 92.0},
            variance=0.0,
            is_low_confidence=False,
        )

        passage = Passage(
            text="The Earth is an oblate spheroid.",
            source="Wikipedia:Earth",
            relevance_score=0.88,
        )

        report = Report(
            metadata={
                "timestamp": datetime.now().isoformat(),
                "models": ["llama-3-8b"],
                "version": "1.0",
            },
            consensus_score=92.0,
            individual_scores={"llama-3-8b": 92.0},
            verifier_verdicts=[verdict],
            retrieval_provenance=[passage],
            reasoning={"llama-3-8b": "Accurate statement with appropriate terminology."},
            confidence=0.9,
            disagreement_level=0.0,
            flagged_issues=[issue],
            hallucination_categories={"factual_error": 1},
        )

        return EvaluationResult(
            request=request,
            consensus_score=92.0,
            verifier_verdicts=[verdict],
            judge_results=[judge_result],
            aggregation_metadata=aggregation_metadata,
            report=report,
            flagged_issues=[issue],
        )

    def test_evaluation_result_creation(self):
        """Test basic evaluation result creation."""
        result = self.create_sample_result()

        assert result.consensus_score == 92.0
        assert len(result.verifier_verdicts) == 1
        assert len(result.judge_results) == 1
        assert result.aggregation_metadata.strategy == "mean"
        assert len(result.flagged_issues) == 1

    def test_evaluation_result_to_dict(self):
        """Test evaluation result conversion to dictionary."""
        result = self.create_sample_result()
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["consensus_score"] == 92.0
        assert result_dict["request"]["source_text"] == "The Earth is round."
        assert result_dict["verifier_verdicts"][0]["label"] == "SUPPORTED"
        assert result_dict["judge_results"][0]["model_name"] == "llama-3-8b"
        assert result_dict["aggregation_metadata"]["strategy"] == "mean"

    def test_evaluation_result_to_json(self):
        """Test evaluation result conversion to JSON."""
        result = self.create_sample_result()
        json_str = result.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["consensus_score"] == 92.0
        assert parsed["request"]["candidate_output"] == "The Earth is spherical."

    def test_evaluation_result_json_round_trip(self):
        """Test JSON serialization and deserialization round trip."""
        original = self.create_sample_result()

        # Serialize to JSON
        json_str = original.to_json()

        # Deserialize from JSON
        restored = EvaluationResult.from_json(json_str)

        # Verify all fields match
        assert restored.consensus_score == original.consensus_score
        assert restored.request.source_text == original.request.source_text
        assert restored.request.candidate_output == original.request.candidate_output
        assert len(restored.verifier_verdicts) == len(original.verifier_verdicts)
        assert restored.verifier_verdicts[0].label == original.verifier_verdicts[0].label
        assert len(restored.judge_results) == len(original.judge_results)
        assert restored.judge_results[0].score == original.judge_results[0].score
        assert (
            restored.aggregation_metadata.strategy
            == original.aggregation_metadata.strategy
        )

    def test_evaluation_result_from_dict(self):
        """Test evaluation result creation from dictionary."""
        original = self.create_sample_result()
        result_dict = original.to_dict()

        restored = EvaluationResult.from_dict(result_dict)

        assert restored.consensus_score == original.consensus_score
        assert restored.request.source_text == original.request.source_text
        assert len(restored.verifier_verdicts) == len(original.verifier_verdicts)

    def test_evaluation_result_json_compact(self):
        """Test compact JSON serialization (no indentation)."""
        result = self.create_sample_result()
        json_str = result.to_json(indent=None)

        assert isinstance(json_str, str)
        assert "\n" not in json_str  # Compact format has no newlines
        parsed = json.loads(json_str)
        assert parsed["consensus_score"] == 92.0

    def test_evaluation_result_with_multiple_judges(self):
        """Test evaluation result with multiple judge results."""
        request = EvaluationRequest(
            source_text="Test source", candidate_output="Test output"
        )

        judge1 = JudgeResult(
            model_name="judge1", score=85.0, reasoning="Good", confidence=0.9
        )
        judge2 = JudgeResult(
            model_name="judge2", score=90.0, reasoning="Excellent", confidence=0.95
        )

        aggregation_metadata = AggregationMetadata(
            strategy="mean",
            individual_scores={"judge1": 85.0, "judge2": 90.0},
            variance=12.5,
            is_low_confidence=False,
        )

        report = Report(
            metadata={},
            consensus_score=87.5,
            individual_scores={"judge1": 85.0, "judge2": 90.0},
            verifier_verdicts=[],
            retrieval_provenance=[],
            reasoning={"judge1": "Good", "judge2": "Excellent"},
            confidence=0.925,
            disagreement_level=5.0,
            flagged_issues=[],
            hallucination_categories={},
        )

        result = EvaluationResult(
            request=request,
            consensus_score=87.5,
            verifier_verdicts=[],
            judge_results=[judge1, judge2],
            aggregation_metadata=aggregation_metadata,
            report=report,
        )

        # Test JSON round trip
        json_str = result.to_json()
        restored = EvaluationResult.from_json(json_str)

        assert len(restored.judge_results) == 2
        assert restored.judge_results[0].model_name == "judge1"
        assert restored.judge_results[1].model_name == "judge2"
        assert restored.consensus_score == 87.5


class TestEnums:
    """Tests for enum types."""

    def test_verdict_label_values(self):
        """Test VerdictLabel enum values."""
        assert VerdictLabel.SUPPORTED.value == "SUPPORTED"
        assert VerdictLabel.REFUTED.value == "REFUTED"
        assert VerdictLabel.NOT_ENOUGH_INFO.value == "NOT_ENOUGH_INFO"

    def test_issue_type_values(self):
        """Test IssueType enum values."""
        assert IssueType.HALLUCINATION.value == "hallucination"
        assert IssueType.BIAS.value == "bias"
        assert IssueType.FACTUAL_ERROR.value == "factual_error"

    def test_issue_severity_values(self):
        """Test IssueSeverity enum values."""
        assert IssueSeverity.LOW.value == "low"
        assert IssueSeverity.MEDIUM.value == "medium"
        assert IssueSeverity.HIGH.value == "high"

    def test_claim_type_values(self):
        """Test ClaimType enum values."""
        assert ClaimType.FACTUAL.value == "factual"
        assert ClaimType.TEMPORAL.value == "temporal"
        assert ClaimType.NUMERICAL.value == "numerical"
