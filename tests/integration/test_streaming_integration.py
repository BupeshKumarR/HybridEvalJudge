"""
Integration tests for StreamingEvaluator with the full evaluation pipeline.

These tests verify that the StreamingEvaluator correctly integrates with
the EvaluationToolkit and processes documents through the complete pipeline.

Note: These tests require actual models to be available and will be skipped
if the EvaluationToolkit cannot be initialized.
"""

import io

import pytest

from llm_judge_auditor import EvaluationToolkit, StreamingEvaluator
from llm_judge_auditor.models import EvaluationResult


# Skip all tests in this module if models are not available
pytestmark = pytest.mark.skip(
    reason="Integration tests require actual models to be loaded. "
    "Run manually when models are available."
)


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for StreamingEvaluator."""

    def test_streaming_with_fast_preset(self):
        """Test streaming evaluation with fast preset."""
        # Initialize toolkit
        toolkit = EvaluationToolkit.from_preset("fast")

        # Create streaming evaluator
        streaming = StreamingEvaluator(toolkit, chunk_size=100, overlap=20)

        # Create sample documents
        source_text = "The Eiffel Tower was completed in 1889 in Paris, France."
        candidate_text = "The Eiffel Tower was built in 1889 in Paris."

        # Create streams
        source_stream = io.StringIO(source_text)
        candidate_stream = io.StringIO(candidate_text)

        # Evaluate
        result = streaming.evaluate_stream(
            source_stream=source_stream,
            candidate_stream=candidate_stream,
            use_retrieval=False,
        )

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.consensus_score <= 100
        assert result.report is not None
        assert "num_chunks" in result.report.metadata
        assert result.report.metadata["num_chunks"] >= 1

    def test_streaming_with_multiple_chunks(self):
        """Test streaming evaluation creates and processes multiple chunks."""
        toolkit = EvaluationToolkit.from_preset("fast")
        streaming = StreamingEvaluator(toolkit, chunk_size=50, overlap=10)

        # Create longer text that will be chunked
        source_text = "The Eiffel Tower is in Paris. " * 10
        candidate_text = "The Eiffel Tower is located in Paris, France. " * 10

        source_stream = io.StringIO(source_text)
        candidate_stream = io.StringIO(candidate_text)

        result = streaming.evaluate_stream(
            source_stream=source_stream,
            candidate_stream=candidate_stream,
            use_retrieval=False,
        )

        # Should have created multiple chunks
        assert result.report.metadata["num_chunks"] > 1
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.consensus_score <= 100

    def test_streaming_preserves_evaluation_quality(self):
        """Test that streaming produces similar results to non-streaming."""
        toolkit = EvaluationToolkit.from_preset("fast")

        # Short text that fits in one chunk
        source_text = "Paris is the capital of France."
        candidate_text = "Paris is the capital of France."

        # Non-streaming evaluation
        result_normal = toolkit.evaluate(
            source_text=source_text,
            candidate_output=candidate_text,
            use_retrieval=False,
        )

        # Streaming evaluation with large chunk size (should be single chunk)
        streaming = StreamingEvaluator(toolkit, chunk_size=1000, overlap=0)
        source_stream = io.StringIO(source_text)
        candidate_stream = io.StringIO(candidate_text)

        result_streaming = streaming.evaluate_stream(
            source_stream=source_stream,
            candidate_stream=candidate_stream,
            use_retrieval=False,
        )

        # Scores should be similar (within 10 points)
        assert abs(result_normal.consensus_score - result_streaming.consensus_score) < 10

    def test_streaming_aggregates_issues_correctly(self):
        """Test that streaming correctly aggregates issues from multiple chunks."""
        toolkit = EvaluationToolkit.from_preset("fast")
        streaming = StreamingEvaluator(toolkit, chunk_size=50, overlap=10)

        # Text with potential issues
        source_text = "The Eiffel Tower was completed in 1889."
        candidate_text = (
            "The Eiffel Tower was completed in 1889. "
            "It is 500 meters tall. "  # Incorrect height
            "It was built in 1850. "  # Incorrect date
        )

        source_stream = io.StringIO(source_text)
        candidate_stream = io.StringIO(candidate_text)

        result = streaming.evaluate_stream(
            source_stream=source_stream,
            candidate_stream=candidate_stream,
            use_retrieval=False,
        )

        # Should have detected some issues
        assert isinstance(result.flagged_issues, list)
        # Score should reflect the inaccuracies
        assert result.consensus_score < 90

    def test_streaming_with_empty_overlap(self):
        """Test streaming works with zero overlap."""
        toolkit = EvaluationToolkit.from_preset("fast")
        streaming = StreamingEvaluator(toolkit, chunk_size=100, overlap=0)

        source_text = "The Eiffel Tower is in Paris. " * 5
        candidate_text = "The Eiffel Tower is in Paris. " * 5

        source_stream = io.StringIO(source_text)
        candidate_stream = io.StringIO(candidate_text)

        result = streaming.evaluate_stream(
            source_stream=source_stream,
            candidate_stream=candidate_stream,
            use_retrieval=False,
        )

        assert isinstance(result, EvaluationResult)
        assert result.report.metadata["overlap"] == 0

    def test_streaming_metadata_completeness(self):
        """Test that streaming results include complete metadata."""
        toolkit = EvaluationToolkit.from_preset("fast")
        streaming = StreamingEvaluator(toolkit, chunk_size=100, overlap=20)

        source_text = "Test source text."
        candidate_text = "Test candidate text."

        source_stream = io.StringIO(source_text)
        candidate_stream = io.StringIO(candidate_text)

        result = streaming.evaluate_stream(
            source_stream=source_stream,
            candidate_stream=candidate_stream,
            use_retrieval=False,
        )

        # Check metadata completeness
        metadata = result.report.metadata
        assert "timestamp" in metadata
        assert "num_chunks" in metadata
        assert "total_characters" in metadata
        assert "chunk_size" in metadata
        assert "overlap" in metadata
        assert "aggregation_strategy" in metadata
        assert metadata["aggregation_strategy"] == "weighted_average_streaming"
