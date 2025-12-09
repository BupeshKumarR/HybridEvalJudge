"""
Unit tests for the StreamingEvaluator component.

Tests cover:
- Initialization and configuration
- Text chunking with overlap
- Stream reading
- Chunk evaluation
- Result aggregation
- Error handling
"""

import io
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_judge_auditor.components.streaming_evaluator import (
    PartialResult,
    StreamingEvaluator,
)
from llm_judge_auditor.models import (
    AggregationMetadata,
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


class TestStreamingEvaluatorInitialization:
    """Test StreamingEvaluator initialization."""

    def test_init_with_valid_params(self):
        """Test initialization with valid parameters."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit, chunk_size=512, overlap=50)

        assert evaluator.toolkit == toolkit
        assert evaluator.chunk_size == 512
        assert evaluator.overlap == 50

    def test_init_with_default_params(self):
        """Test initialization with default parameters."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        assert evaluator.chunk_size == 512
        assert evaluator.overlap == 50

    def test_init_with_invalid_chunk_size(self):
        """Test initialization fails with invalid chunk_size."""
        toolkit = Mock()

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            StreamingEvaluator(toolkit, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            StreamingEvaluator(toolkit, chunk_size=-10)

    def test_init_with_invalid_overlap(self):
        """Test initialization fails with invalid overlap."""
        toolkit = Mock()

        with pytest.raises(ValueError, match="overlap cannot be negative"):
            StreamingEvaluator(toolkit, chunk_size=512, overlap=-5)

    def test_init_with_overlap_greater_than_chunk_size(self):
        """Test initialization fails when overlap >= chunk_size."""
        toolkit = Mock()

        with pytest.raises(ValueError, match="chunk_size must be greater than overlap"):
            StreamingEvaluator(toolkit, chunk_size=100, overlap=100)

        with pytest.raises(ValueError, match="chunk_size must be greater than overlap"):
            StreamingEvaluator(toolkit, chunk_size=100, overlap=150)


class TestTextChunking:
    """Test text chunking functionality."""

    def test_chunk_text_smaller_than_chunk_size(self):
        """Test chunking text that fits in a single chunk."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit, chunk_size=100, overlap=10)

        text = "This is a short text."
        chunks = evaluator._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_with_overlap(self):
        """Test chunking creates overlapping chunks."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit, chunk_size=20, overlap=5)

        text = "A" * 50  # 50 characters
        chunks = evaluator._chunk_text(text)

        # Should create multiple chunks
        assert len(chunks) > 1

        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            # Last 5 chars of chunk i should match first 5 chars of chunk i+1
            # (accounting for the way overlap works)
            assert len(chunks[i]) <= 20

    def test_chunk_text_sentence_boundary(self):
        """Test chunking prefers sentence boundaries."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit, chunk_size=50, overlap=10)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = evaluator._chunk_text(text)

        # Should break at sentence boundaries when possible
        assert len(chunks) >= 1
        # First chunk should end with a period if possible
        if len(chunks) > 1:
            assert chunks[0].rstrip().endswith(".")

    def test_chunk_text_exact_chunk_size(self):
        """Test chunking text that is exactly chunk_size."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit, chunk_size=20, overlap=5)

        text = "A" * 20
        chunks = evaluator._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty_string(self):
        """Test chunking empty string."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit, chunk_size=100, overlap=10)

        text = ""
        chunks = evaluator._chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == ""


class TestStreamReading:
    """Test stream reading functionality."""

    def test_read_stream_from_string_io(self):
        """Test reading from a StringIO stream."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        text = "This is test content from a stream."
        stream = io.StringIO(text)

        result = evaluator._read_stream(stream)
        assert result == text

    def test_read_stream_empty(self):
        """Test reading from an empty stream."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        stream = io.StringIO("")
        result = evaluator._read_stream(stream)

        assert result == ""

    def test_read_stream_multiline(self):
        """Test reading multiline content from stream."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        text = "Line 1\nLine 2\nLine 3"
        stream = io.StringIO(text)

        result = evaluator._read_stream(stream)
        assert result == text


class TestChunkEvaluation:
    """Test chunk evaluation functionality."""

    def test_evaluate_chunk_success(self):
        """Test successful chunk evaluation."""
        # Create mock toolkit with evaluate method
        toolkit = Mock()
        mock_result = Mock(spec=EvaluationResult)
        mock_result.consensus_score = 85.0
        mock_result.verifier_verdicts = []
        mock_result.judge_results = []
        mock_result.flagged_issues = []
        mock_result.report = Mock(retrieval_provenance=[])

        toolkit.evaluate.return_value = mock_result

        evaluator = StreamingEvaluator(toolkit)

        partial = evaluator._evaluate_chunk(
            source_text="Source text",
            chunk_text="Chunk text",
            chunk_index=0,
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        )

        assert isinstance(partial, PartialResult)
        assert partial.chunk_index == 0
        assert partial.chunk_text == "Chunk text"
        assert partial.consensus_score == 85.0

        # Verify toolkit.evaluate was called
        toolkit.evaluate.assert_called_once()

    def test_evaluate_chunk_failure(self):
        """Test chunk evaluation handles errors."""
        toolkit = Mock()
        toolkit.evaluate.side_effect = RuntimeError("Evaluation failed")

        evaluator = StreamingEvaluator(toolkit)

        with pytest.raises(RuntimeError, match="Failed to evaluate chunk 0"):
            evaluator._evaluate_chunk(
                source_text="Source text",
                chunk_text="Chunk text",
                chunk_index=0,
                task="factual_accuracy",
                criteria=["correctness"],
                use_retrieval=False,
            )


class TestResultAggregation:
    """Test result aggregation functionality."""

    def test_aggregate_stream_results_single_chunk(self):
        """Test aggregating results from a single chunk."""
        toolkit = Mock()
        toolkit.config = Mock()
        toolkit.config.enable_retrieval = False
        toolkit.config.verifier_model = "test-verifier"
        toolkit.config.judge_models = ["judge1"]
        toolkit.config.disagreement_threshold = 20.0

        evaluator = StreamingEvaluator(toolkit, chunk_size=100, overlap=10)

        # Create a partial result
        partial = PartialResult(
            chunk_index=0,
            chunk_text="Test chunk",
            consensus_score=80.0,
            verifier_verdicts=[],
            judge_results=[
                JudgeResult(
                    model_name="judge1",
                    score=80.0,
                    reasoning="Good",
                    flagged_issues=[],
                    confidence=0.9,
                )
            ],
            retrieved_passages=[],
            flagged_issues=[],
        )

        result = evaluator._aggregate_stream_results(
            partial_results=[partial],
            source_text="Source",
            candidate_text="Test chunk",
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        )

        assert isinstance(result, EvaluationResult)
        assert result.consensus_score == 80.0
        assert len(result.judge_results) == 1

    def test_aggregate_stream_results_multiple_chunks(self):
        """Test aggregating results from multiple chunks."""
        toolkit = Mock()
        toolkit.config = Mock()
        toolkit.config.enable_retrieval = False
        toolkit.config.verifier_model = "test-verifier"
        toolkit.config.judge_models = ["judge1"]
        toolkit.config.disagreement_threshold = 20.0

        evaluator = StreamingEvaluator(toolkit, chunk_size=10, overlap=2)

        # Create multiple partial results
        partials = [
            PartialResult(
                chunk_index=0,
                chunk_text="Chunk 1",  # 7 chars
                consensus_score=80.0,
                verifier_verdicts=[],
                judge_results=[
                    JudgeResult(
                        model_name="judge1",
                        score=80.0,
                        reasoning="Good",
                        flagged_issues=[],
                        confidence=0.9,
                    )
                ],
                retrieved_passages=[],
                flagged_issues=[],
            ),
            PartialResult(
                chunk_index=1,
                chunk_text="Chunk 2",  # 7 chars
                consensus_score=90.0,
                verifier_verdicts=[],
                judge_results=[
                    JudgeResult(
                        model_name="judge1",
                        score=90.0,
                        reasoning="Excellent",
                        flagged_issues=[],
                        confidence=0.95,
                    )
                ],
                retrieved_passages=[],
                flagged_issues=[],
            ),
        ]

        result = evaluator._aggregate_stream_results(
            partial_results=partials,
            source_text="Source",
            candidate_text="Chunk 1Chunk 2",
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        )

        assert isinstance(result, EvaluationResult)
        # Weighted average: (80*7 + 90*7) / 14 = 85.0
        assert result.consensus_score == 85.0
        assert len(result.judge_results) == 1
        assert result.judge_results[0].score == 85.0

    def test_aggregate_stream_results_empty_list(self):
        """Test aggregation fails with empty partial results."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        with pytest.raises(ValueError, match="Cannot aggregate empty partial_results"):
            evaluator._aggregate_stream_results(
                partial_results=[],
                source_text="Source",
                candidate_text="Candidate",
                task="factual_accuracy",
                criteria=["correctness"],
                use_retrieval=False,
            )

    def test_aggregate_deduplicates_verdicts(self):
        """Test aggregation deduplicates verdicts by reasoning."""
        toolkit = Mock()
        toolkit.config = Mock()
        toolkit.config.enable_retrieval = False
        toolkit.config.verifier_model = "test-verifier"
        toolkit.config.judge_models = ["judge1"]
        toolkit.config.disagreement_threshold = 20.0

        evaluator = StreamingEvaluator(toolkit)

        # Create partials with duplicate verdicts
        verdict1 = Verdict(
            label=VerdictLabel.SUPPORTED, confidence=0.9, reasoning="Same reasoning"
        )
        verdict2 = Verdict(
            label=VerdictLabel.SUPPORTED, confidence=0.8, reasoning="Same reasoning"
        )
        verdict3 = Verdict(
            label=VerdictLabel.REFUTED, confidence=0.7, reasoning="Different reasoning"
        )

        partials = [
            PartialResult(
                chunk_index=0,
                chunk_text="Chunk 1",
                consensus_score=80.0,
                verifier_verdicts=[verdict1, verdict3],
                judge_results=[
                    JudgeResult(
                        model_name="judge1",
                        score=80.0,
                        reasoning="Good",
                        confidence=0.9,
                    )
                ],
                retrieved_passages=[],
                flagged_issues=[],
            ),
            PartialResult(
                chunk_index=1,
                chunk_text="Chunk 2",
                consensus_score=85.0,
                verifier_verdicts=[verdict2],  # Duplicate reasoning
                judge_results=[
                    JudgeResult(
                        model_name="judge1",
                        score=85.0,
                        reasoning="Good",
                        confidence=0.9,
                    )
                ],
                retrieved_passages=[],
                flagged_issues=[],
            ),
        ]

        result = evaluator._aggregate_stream_results(
            partial_results=partials,
            source_text="Source",
            candidate_text="Chunk 1Chunk 2",
            task="factual_accuracy",
            criteria=["correctness"],
            use_retrieval=False,
        )

        # Should only have 2 unique verdicts (deduplicated by reasoning)
        assert len(result.verifier_verdicts) == 2


class TestStreamEvaluation:
    """Test full stream evaluation."""

    def test_evaluate_stream_success(self):
        """Test successful stream evaluation."""
        # Create mock toolkit
        toolkit = Mock()
        toolkit.config = Mock()
        toolkit.config.enable_retrieval = False
        toolkit.config.verifier_model = "test-verifier"
        toolkit.config.judge_models = ["judge1"]
        toolkit.config.disagreement_threshold = 20.0

        # Mock evaluate method
        mock_result = Mock(spec=EvaluationResult)
        mock_result.consensus_score = 85.0
        mock_result.verifier_verdicts = []
        mock_result.judge_results = [
            JudgeResult(
                model_name="judge1",
                score=85.0,
                reasoning="Good",
                confidence=0.9,
            )
        ]
        mock_result.flagged_issues = []
        mock_result.report = Mock(retrieval_provenance=[])

        toolkit.evaluate.return_value = mock_result

        evaluator = StreamingEvaluator(toolkit, chunk_size=50, overlap=10)

        # Create streams
        source_stream = io.StringIO("This is the source text.")
        candidate_stream = io.StringIO("This is the candidate text.")

        result = evaluator.evaluate_stream(source_stream, candidate_stream)

        assert isinstance(result, EvaluationResult)
        assert result.consensus_score == 85.0

    def test_evaluate_stream_empty_source(self):
        """Test stream evaluation fails with empty source."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        source_stream = io.StringIO("")
        candidate_stream = io.StringIO("Candidate text")

        with pytest.raises(ValueError, match="source_stream produced empty text"):
            evaluator.evaluate_stream(source_stream, candidate_stream)

    def test_evaluate_stream_empty_candidate(self):
        """Test stream evaluation fails with empty candidate."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        source_stream = io.StringIO("Source text")
        candidate_stream = io.StringIO("")

        with pytest.raises(ValueError, match="candidate_stream produced empty text"):
            evaluator.evaluate_stream(source_stream, candidate_stream)


class TestIssueCategorization:
    """Test issue categorization."""

    def test_categorize_issues_empty(self):
        """Test categorizing empty issue list."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        categories = evaluator._categorize_issues([])

        assert categories["factual_error"] == 0
        assert categories["unsupported_claim"] == 0
        assert categories["bias"] == 0

    def test_categorize_issues_various_types(self):
        """Test categorizing various issue types."""
        toolkit = Mock()
        evaluator = StreamingEvaluator(toolkit)

        issues = [
            Issue(
                type=IssueType.HALLUCINATION,
                severity=IssueSeverity.HIGH,
                description="Hallucination",
            ),
            Issue(
                type=IssueType.UNSUPPORTED_CLAIM,
                severity=IssueSeverity.MEDIUM,
                description="Unsupported",
            ),
            Issue(
                type=IssueType.BIAS, severity=IssueSeverity.LOW, description="Bias"
            ),
            Issue(
                type=IssueType.NUMERICAL_ERROR,
                severity=IssueSeverity.HIGH,
                description="Number error",
            ),
        ]

        categories = evaluator._categorize_issues(issues)

        assert categories["factual_error"] == 1
        assert categories["unsupported_claim"] == 1
        assert categories["bias"] == 1
        assert categories["numerical_error"] == 1
