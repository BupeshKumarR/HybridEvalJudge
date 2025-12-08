"""
Unit tests for error handling utilities.

Tests cover:
- Malformed output parsing with partial results
- Timeout handling
- Error logging with context

Requirements: 9.1, 9.2
"""

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from llm_judge_auditor.utils.error_handling import (
    MalformedOutputError,
    PartialResult,
    TimeoutError,
    TimeoutResult,
    handle_inference_error,
    log_error_with_context,
    parse_score_with_fallback,
    parse_structured_output,
    safe_parse_judge_output,
    timeout_wrapper,
    with_timeout,
)


class TestParseScoreWithFallback:
    """Test score parsing with fallback for malformed outputs."""

    def test_parse_valid_score(self):
        """Test parsing a valid score."""
        text = "SCORE: 85"
        result = parse_score_with_fallback(text)

        assert result.success is True
        assert result.data == 85.0
        assert result.error_message is None

    def test_parse_score_with_decimal(self):
        """Test parsing a score with decimal."""
        text = "SCORE: 85.5"
        result = parse_score_with_fallback(text)

        assert result.success is True
        assert result.data == 85.5

    def test_parse_score_case_insensitive(self):
        """Test parsing with different case."""
        text = "score: 75"
        result = parse_score_with_fallback(text)

        assert result.success is True
        assert result.data == 75.0

    def test_parse_score_with_percentage(self):
        """Test parsing percentage format."""
        text = "The score is 90%"
        result = parse_score_with_fallback(text)

        assert result.success is True
        assert result.data == 90.0

    def test_parse_score_with_fraction(self):
        """Test parsing fraction format."""
        text = "Score: 85/100"
        result = parse_score_with_fallback(text)

        assert result.success is True
        assert result.data == 85.0

    def test_parse_score_clamping(self):
        """Test that scores are clamped to valid range."""
        text = "SCORE: 150"
        result = parse_score_with_fallback(text)

        assert result.success is True
        assert result.data == 100.0  # Clamped to max

        text = "SCORE: -50"
        result = parse_score_with_fallback(text)

        assert result.success is True
        assert result.data == 0.0  # Clamped to min

    def test_parse_score_fallback(self):
        """Test fallback to default when parsing fails."""
        text = "No score here"
        result = parse_score_with_fallback(text, default_score=50.0)

        assert result.success is False
        assert result.data == 50.0
        assert result.error_message is not None
        assert "default" in result.error_message.lower()

    def test_parse_score_custom_default(self):
        """Test custom default score."""
        text = "Invalid"
        result = parse_score_with_fallback(text, default_score=75.0)

        assert result.success is False
        assert result.data == 75.0


class TestParseStructuredOutput:
    """Test structured output parsing with partial results."""

    def test_parse_complete_output(self):
        """Test parsing complete structured output."""
        text = "SCORE: 85\nREASONING: Good answer\nFLAGGED_ISSUES: None"
        result = parse_structured_output(
            text,
            required_fields={"score": 0.0, "reasoning": "", "flagged_issues": ""},
        )

        assert result.success is True
        assert result.data["score"] == 85.0
        assert "Good answer" in result.data["reasoning"]
        assert result.error_message is None

    def test_parse_partial_output(self):
        """Test parsing partial output with missing fields."""
        text = "SCORE: 85\nREASONING: Good answer"
        result = parse_structured_output(
            text,
            required_fields={"score": 0.0, "reasoning": "", "flagged_issues": ""},
        )

        assert result.success is False
        assert result.data["score"] == 85.0
        assert "Good answer" in result.data["reasoning"]
        assert result.data["flagged_issues"] == ""  # Default value
        assert "flagged_issues" in result.error_message

    def test_parse_with_optional_fields(self):
        """Test parsing with optional fields."""
        text = "SCORE: 85\nREASONING: Good"
        result = parse_structured_output(
            text,
            required_fields={"score": 0.0, "reasoning": ""},
            optional_fields={"confidence": 0.5},
        )

        assert result.success is True
        assert result.data["score"] == 85.0
        assert result.data["confidence"] == 0.5  # Default for optional

    def test_parse_case_insensitive(self):
        """Test case-insensitive field parsing."""
        text = "score: 85\nreasoning: Good"
        result = parse_structured_output(
            text, required_fields={"score": 0.0, "reasoning": ""}
        )

        assert result.success is True
        assert result.data["score"] == 85.0


class TestSafeParseJudgeOutput:
    """Test safe parsing of judge model output."""

    def test_parse_valid_judge_output(self):
        """Test parsing valid judge output."""
        response = "SCORE: 85\nREASONING: Good answer\nFLAGGED_ISSUES: None"
        result = safe_parse_judge_output(response, "test-model")

        assert result["score"] == 85.0
        assert "Good answer" in result["reasoning"]
        assert result["model_name"] == "test-model"
        assert result["parse_success"] is True
        assert len(result["parse_errors"]) == 0

    def test_parse_malformed_judge_output(self):
        """Test parsing malformed judge output with fallback."""
        response = "This is completely malformed output"
        result = safe_parse_judge_output(response, "test-model", default_score=50.0)

        assert result["score"] == 50.0  # Default
        assert result["reasoning"] is not None
        assert result["parse_success"] is False
        assert len(result["parse_errors"]) > 0

    def test_parse_partial_judge_output(self):
        """Test parsing partial judge output."""
        response = "SCORE: 75"  # Missing reasoning
        result = safe_parse_judge_output(response, "test-model")

        assert result["score"] == 75.0
        assert result["reasoning"] is not None  # Should have default
        assert result["parse_success"] is False  # Partial parse


class TestTimeoutHandling:
    """Test timeout handling utilities."""

    def test_timeout_wrapper_success(self):
        """Test timeout wrapper with successful function."""

        def fast_function(x):
            return x * 2

        result = timeout_wrapper(fast_function, 1.0, 5)

        assert result.success is True
        assert result.result == 10
        assert result.timed_out is False
        assert result.elapsed_time < 1.0

    def test_timeout_wrapper_slow_function(self):
        """Test timeout wrapper with slow function."""

        def slow_function(x):
            time.sleep(0.1)
            return x * 2

        result = timeout_wrapper(slow_function, 1.0, 5)

        assert result.success is True
        assert result.result == 10
        assert result.timed_out is False

    def test_with_timeout_decorator(self):
        """Test timeout decorator."""

        @with_timeout(1.0)
        def fast_function(x):
            return x * 2

        result = fast_function(5)

        assert result.success is True
        assert result.result == 10
        assert result.timed_out is False


class TestErrorLogging:
    """Test error logging with context."""

    def test_log_error_with_context(self, caplog):
        """Test logging error with context."""
        error = ValueError("Test error")
        context = {
            "operation": "test_operation",
            "model": "test-model",
            "input_length": 100,
        }

        with caplog.at_level(logging.ERROR):
            log_error_with_context(error, context)

        # Check that error was logged
        assert len(caplog.records) > 0
        assert "ValueError" in caplog.text
        assert "Test error" in caplog.text
        assert "test_operation" in caplog.text

    def test_log_error_custom_logger(self):
        """Test logging with custom logger."""
        mock_logger = MagicMock()
        error = RuntimeError("Custom error")
        context = {"key": "value"}

        log_error_with_context(error, context, logger_instance=mock_logger)

        # Verify logger was called
        assert mock_logger.error.called
        assert mock_logger.debug.called


class TestHandleInferenceError:
    """Test inference error handling."""

    def test_handle_oom_error(self):
        """Test handling out of memory error."""
        error = RuntimeError("CUDA out of memory")
        context = {"model": "test-model", "batch_size": 32}

        result = handle_inference_error(error, "inference", context)

        assert result["error_type"] == "RuntimeError"
        assert "out of memory" in result["error_message"].lower()
        assert result["action"] == "reduce_batch_size"
        assert result["operation"] == "inference"

    def test_handle_timeout_error(self):
        """Test handling timeout error."""
        error = TimeoutError("Operation timed out")
        context = {"model": "test-model"}

        result = handle_inference_error(error, "generation", context)

        assert result["error_type"] == "TimeoutError"
        assert result["action"] == "increase_timeout"

    def test_handle_cuda_error(self):
        """Test handling CUDA error."""
        error = RuntimeError("CUDA error: device-side assert triggered")
        context = {"model": "test-model"}

        result = handle_inference_error(error, "inference", context)

        assert result["action"] == "fallback_to_cpu"

    def test_handle_generic_error(self):
        """Test handling generic error."""
        error = ValueError("Some error")
        context = {"model": "test-model"}

        result = handle_inference_error(error, "operation", context)

        assert result["error_type"] == "ValueError"
        assert result["action"] == "retry"  # Default action


class TestPartialResult:
    """Test PartialResult dataclass."""

    def test_partial_result_success(self):
        """Test successful partial result."""
        result = PartialResult(success=True, data={"score": 85.0})

        assert result.success is True
        assert result.data["score"] == 85.0
        assert result.error_message is None

    def test_partial_result_failure(self):
        """Test failed partial result."""
        result = PartialResult(
            success=False,
            data={"score": 50.0},
            error_message="Parse failed",
            raw_output="Invalid output",
        )

        assert result.success is False
        assert result.data["score"] == 50.0
        assert result.error_message == "Parse failed"
        assert result.raw_output == "Invalid output"


class TestTimeoutResult:
    """Test TimeoutResult dataclass."""

    def test_timeout_result_success(self):
        """Test successful timeout result."""
        result = TimeoutResult(
            success=True, result=42, timed_out=False, elapsed_time=0.5
        )

        assert result.success is True
        assert result.result == 42
        assert result.timed_out is False
        assert result.elapsed_time == 0.5

    def test_timeout_result_timeout(self):
        """Test timed out result."""
        result = TimeoutResult(
            success=False,
            result=None,
            timed_out=True,
            elapsed_time=5.0,
            partial_output="Partial...",
        )

        assert result.success is False
        assert result.result is None
        assert result.timed_out is True
        assert result.elapsed_time == 5.0
        assert result.partial_output == "Partial..."
