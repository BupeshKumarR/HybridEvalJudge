"""
Unit tests for the ReportGenerator component.
"""

import json
import tempfile
from pathlib import Path

import pytest

from llm_judge_auditor.components.report_generator import ReportGenerator
from llm_judge_auditor.models import (
    AggregationMetadata,
    EvaluationRequest,
    EvaluationResult,
    Issue,
    IssueSeverity,
    IssueType,
    JudgeResult,
    Passage,
    Report,
    Verdict,
    VerdictLabel,
)


@pytest.fixture
def sample_report():
    """Create a sample report for testing."""
    metadata = {
        "timestamp": "2024-01-01T12:00:00",
        "task": "factual_accuracy",
        "criteria": ["correctness"],
        "retrieval_enabled": True,
        "verifier_model": "minicheck-flan-t5-large",
        "judge_models": ["llama-3-8b", "mistral-7b"],
        "aggregation_strategy": "mean",
        "num_retrieved_passages": 2,
        "num_verifier_verdicts": 2,
        "num_judge_results": 2,
    }

    individual_scores = {
        "llama-3-8b": 75.0,
        "mistral-7b": 80.0,
    }

    verifier_verdicts = [
        Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.9,
            evidence=["Evidence 1"],
            reasoning="The claim is supported by the source.",
        ),
        Verdict(
            label=VerdictLabel.REFUTED,
            confidence=0.85,
            evidence=["Evidence 2"],
            reasoning="The claim contradicts the source.",
        ),
    ]

    retrieval_provenance = [
        Passage(
            text="Paris is the capital of France.",
            source="Wikipedia:Paris",
            relevance_score=0.95,
        ),
        Passage(
            text="France is a country in Europe.",
            source="Wikipedia:France",
            relevance_score=0.88,
        ),
    ]

    reasoning = {
        "llama-3-8b": "The candidate output contains mostly accurate information with one error.",
        "mistral-7b": "The output is generally accurate but has minor issues.",
    }

    flagged_issues = [
        Issue(
            type=IssueType.HALLUCINATION,
            severity=IssueSeverity.HIGH,
            description="Refuted claim: The claim contradicts the source.",
            evidence=["Evidence 2"],
        ),
    ]

    hallucination_categories = {
        "factual_error": 1,
        "unsupported_claim": 0,
        "temporal_inconsistency": 0,
        "numerical_error": 0,
        "bias": 0,
        "inconsistency": 0,
        "other": 0,
    }

    report = Report(
        metadata=metadata,
        consensus_score=77.5,
        individual_scores=individual_scores,
        verifier_verdicts=verifier_verdicts,
        retrieval_provenance=retrieval_provenance,
        reasoning=reasoning,
        confidence=0.85,
        disagreement_level=2.5,
        flagged_issues=flagged_issues,
        hallucination_categories=hallucination_categories,
    )

    return report


@pytest.fixture
def sample_evaluation_result(sample_report):
    """Create a sample evaluation result for testing."""
    request = EvaluationRequest(
        source_text="Paris is the capital of France.",
        candidate_output="Paris is the capital of Germany.",
        task="factual_accuracy",
        criteria=["correctness"],
        use_retrieval=True,
    )

    judge_results = [
        JudgeResult(
            model_name="llama-3-8b",
            score=75.0,
            reasoning="The candidate output contains mostly accurate information with one error.",
            flagged_issues=[],
            confidence=0.9,
        ),
        JudgeResult(
            model_name="mistral-7b",
            score=80.0,
            reasoning="The output is generally accurate but has minor issues.",
            flagged_issues=[],
            confidence=0.8,
        ),
    ]

    aggregation_metadata = AggregationMetadata(
        strategy="mean",
        individual_scores={"llama-3-8b": 75.0, "mistral-7b": 80.0},
        variance=2.5,
        is_low_confidence=False,
        weights=None,
    )

    result = EvaluationResult(
        request=request,
        consensus_score=77.5,
        verifier_verdicts=sample_report.verifier_verdicts,
        judge_results=judge_results,
        aggregation_metadata=aggregation_metadata,
        report=sample_report,
        flagged_issues=sample_report.flagged_issues,
    )

    return result


class TestReportGenerator:
    """Test suite for ReportGenerator."""

    def test_initialization(self):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator()
        assert generator is not None

    def test_generate_report(self, sample_evaluation_result):
        """Test report generation from evaluation result."""
        generator = ReportGenerator()
        report = generator.generate_report(sample_evaluation_result)

        assert report is not None
        assert report.consensus_score == 77.5
        assert report.confidence == 0.85
        assert len(report.verifier_verdicts) == 2
        assert len(report.individual_scores) == 2

    def test_export_json(self, sample_report):
        """Test JSON export functionality."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            generator.export_json(sample_report, str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify content is valid JSON
            with open(output_path, "r") as f:
                data = json.load(f)

            assert data["consensus_score"] == 77.5
            assert data["confidence"] == 0.85
            assert len(data["verifier_verdicts"]) == 2
            assert len(data["individual_scores"]) == 2

    def test_export_json_with_nested_directories(self, sample_report):
        """Test JSON export with nested directory creation."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "report.json"
            generator.export_json(sample_report, str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify content
            with open(output_path, "r") as f:
                data = json.load(f)

            assert data["consensus_score"] == 77.5

    def test_export_markdown(self, sample_report):
        """Test Markdown export functionality."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify content contains expected sections
            with open(output_path, "r") as f:
                content = f.read()

            assert "# LLM Evaluation Report" in content
            assert "## Metadata" in content
            assert "## Evaluation Scores" in content
            assert "### Consensus Score:" in content
            assert "77.5" in content
            assert "## Chain-of-Thought Reasoning" in content
            assert "## Specialized Verifier Verdicts" in content
            assert "## Retrieval Provenance" in content
            assert "## Flagged Issues" in content
            assert "## Hallucination Categories" in content

    def test_export_markdown_with_nested_directories(self, sample_report):
        """Test Markdown export with nested directory creation."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify content
            with open(output_path, "r") as f:
                content = f.read()

            assert "# LLM Evaluation Report" in content

    def test_export_text(self, sample_report):
        """Test plain text export functionality."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"
            generator.export_text(sample_report, str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify content contains expected sections
            with open(output_path, "r") as f:
                content = f.read()

            assert "LLM EVALUATION REPORT" in content
            assert "METADATA" in content
            assert "EVALUATION SCORES" in content
            assert "Consensus Score: 77.5" in content
            assert "CHAIN-OF-THOUGHT REASONING" in content
            assert "SPECIALIZED VERIFIER VERDICTS" in content
            assert "FLAGGED ISSUES" in content
            assert "HALLUCINATION CATEGORIES" in content

    def test_markdown_includes_metadata(self, sample_report):
        """Test that Markdown export includes all metadata fields."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Check metadata fields
            assert "2024-01-01T12:00:00" in content
            assert "factual_accuracy" in content
            assert "minicheck-flan-t5-large" in content
            assert "llama-3-8b" in content
            assert "mistral-7b" in content
            assert "mean" in content

    def test_markdown_includes_individual_scores(self, sample_report):
        """Test that Markdown export includes individual judge scores."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Check individual scores
            assert "llama-3-8b" in content
            assert "75.00" in content
            assert "mistral-7b" in content
            assert "80.00" in content

    def test_markdown_includes_reasoning(self, sample_report):
        """Test that Markdown export includes chain-of-thought reasoning."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Check reasoning
            assert "The candidate output contains mostly accurate information with one error." in content
            assert "The output is generally accurate but has minor issues." in content

    def test_markdown_includes_verifier_verdicts(self, sample_report):
        """Test that Markdown export includes verifier verdicts."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Check verdicts
            assert "SUPPORTED" in content
            assert "REFUTED" in content
            assert "The claim is supported by the source." in content
            assert "The claim contradicts the source." in content

    def test_markdown_includes_retrieval_provenance(self, sample_report):
        """Test that Markdown export includes retrieval provenance."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Check retrieval provenance
            assert "Wikipedia:Paris" in content
            assert "Wikipedia:France" in content
            assert "Paris is the capital of France." in content

    def test_markdown_includes_flagged_issues(self, sample_report):
        """Test that Markdown export includes flagged issues."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Check flagged issues
            assert "hallucination" in content
            assert "high" in content
            assert "Refuted claim: The claim contradicts the source." in content

    def test_markdown_includes_confidence_and_disagreement(self, sample_report):
        """Test that Markdown export includes confidence and disagreement metrics."""
        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(sample_report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Check confidence and disagreement
            assert "0.85" in content  # confidence
            assert "2.50" in content  # disagreement level

    def test_report_with_no_retrieval(self):
        """Test report generation when retrieval is disabled."""
        metadata = {
            "timestamp": "2024-01-01T12:00:00",
            "task": "factual_accuracy",
            "criteria": ["correctness"],
            "retrieval_enabled": False,
            "verifier_model": "minicheck-flan-t5-large",
            "judge_models": ["llama-3-8b"],
            "aggregation_strategy": "mean",
        }

        report = Report(
            metadata=metadata,
            consensus_score=80.0,
            individual_scores={"llama-3-8b": 80.0},
            verifier_verdicts=[],
            retrieval_provenance=[],
            reasoning={"llama-3-8b": "Good output."},
            confidence=0.9,
            disagreement_level=0.0,
            flagged_issues=[],
            hallucination_categories={
                "factual_error": 0,
                "unsupported_claim": 0,
                "temporal_inconsistency": 0,
                "numerical_error": 0,
                "bias": 0,
                "inconsistency": 0,
                "other": 0,
            },
        )

        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Should indicate no retrieval
            assert "No retrieval performed or no passages retrieved" in content

    def test_report_with_no_issues(self):
        """Test report generation when no issues are flagged."""
        metadata = {
            "timestamp": "2024-01-01T12:00:00",
            "task": "factual_accuracy",
            "criteria": ["correctness"],
            "retrieval_enabled": False,
            "verifier_model": "minicheck-flan-t5-large",
            "judge_models": ["llama-3-8b"],
            "aggregation_strategy": "mean",
        }

        report = Report(
            metadata=metadata,
            consensus_score=95.0,
            individual_scores={"llama-3-8b": 95.0},
            verifier_verdicts=[],
            retrieval_provenance=[],
            reasoning={"llama-3-8b": "Excellent output."},
            confidence=0.95,
            disagreement_level=0.0,
            flagged_issues=[],
            hallucination_categories={
                "factual_error": 0,
                "unsupported_claim": 0,
                "temporal_inconsistency": 0,
                "numerical_error": 0,
                "bias": 0,
                "inconsistency": 0,
                "other": 0,
            },
        )

        generator = ReportGenerator()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            generator.export_markdown(report, str(output_path))

            with open(output_path, "r") as f:
                content = f.read()

            # Should indicate no issues
            assert "No issues flagged" in content
            assert "No hallucinations detected" in content
