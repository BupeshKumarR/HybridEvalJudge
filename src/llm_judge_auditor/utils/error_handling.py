"""
Error handling utilities for the LLM Judge Auditor toolkit.

This module provides utilities for handling common errors in LLM evaluation:
- Malformed output parsing with partial results
- Timeout handling for inference operations
- Structured error logging

Requirements: 9.1, 9.2
"""

import functools
import logging
import re
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class PartialResult:
    """
    Represents a partial result from malformed or incomplete output.

    Attributes:
        success: Whether parsing was fully successful
        data: The parsed data (may be incomplete)
        error_message: Description of what went wrong
        raw_output: The original raw output that was parsed
    """

    success: bool
    data: Any
    error_message: Optional[str] = None
    raw_output: Optional[str] = None


@dataclass
class TimeoutResult:
    """
    Represents the result of an operation that may have timed out.

    Attributes:
        success: Whether the operation completed successfully
        result: The result if successful, None if timed out
        timed_out: Whether the operation timed out
        elapsed_time: Time elapsed in seconds
        partial_output: Any partial output captured before timeout
    """

    success: bool
    result: Any = None
    timed_out: bool = False
    elapsed_time: float = 0.0
    partial_output: Optional[str] = None


class TimeoutError(Exception):
    """Raised when an operation times out."""

    pass


class MalformedOutputError(Exception):
    """Raised when model output cannot be parsed."""

    pass


def parse_score_with_fallback(
    text: str, default_score: float = 50.0, score_range: Tuple[float, float] = (0.0, 100.0)
) -> PartialResult:
    """
    Parse a score from text with fallback to default on failure.

    This function attempts to extract a numerical score from text using
    multiple patterns. If parsing fails, it returns a partial result with
    the default score.

    Args:
        text: Text containing a score
        default_score: Default score to use if parsing fails
        score_range: Valid range for scores (min, max)

    Returns:
        PartialResult with parsed or default score

    Example:
        >>> result = parse_score_with_fallback("SCORE: 85")
        >>> print(result.data)  # 85.0
        >>> result = parse_score_with_fallback("Invalid text")
        >>> print(result.data)  # 50.0 (default)

    Requirements: 9.1
    """
    min_score, max_score = score_range

    # Try multiple patterns to extract score
    patterns = [
        r"SCORE:\s*(-?\d+\.?\d*)",  # SCORE: 85 or SCORE: 85.5
        r"score:\s*(-?\d+\.?\d*)",  # score: 85 (case insensitive)
        r"rating:\s*(-?\d+\.?\d*)",  # rating: 85
        r"(\d+\.?\d*)\s*/\s*100",  # 85/100 or 85.5/100
        r"(\d+\.?\d*)%",  # 85% or 85.5%
        r"(?:^|\s)(-?\d+\.?\d*)(?:\s|$)",  # Any standalone number
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                # Clamp to valid range
                score = max(min_score, min(max_score, score))

                logger.debug(f"Successfully parsed score: {score}")
                return PartialResult(
                    success=True, data=score, raw_output=text
                )
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to convert matched text to float: {e}")
                continue

    # If all patterns fail, return default score
    logger.warning(
        f"Could not parse score from text, using default: {default_score}. "
        f"Text preview: {text[:100]}..."
    )
    return PartialResult(
        success=False,
        data=default_score,
        error_message=f"Could not parse score from text, using default: {default_score}",
        raw_output=text,
    )


def parse_structured_output(
    text: str, required_fields: Dict[str, Any], optional_fields: Optional[Dict[str, Any]] = None
) -> PartialResult:
    """
    Parse structured output with partial result support.

    This function attempts to extract structured data from text using regex
    patterns. If required fields are missing, it returns a partial result
    with available data and an error message.

    Args:
        text: Text containing structured output
        required_fields: Dict mapping field names to default values
        optional_fields: Dict mapping optional field names to default values

    Returns:
        PartialResult with parsed data (may be incomplete)

    Example:
        >>> text = "SCORE: 85\\nREASONING: Good answer"
        >>> result = parse_structured_output(
        ...     text,
        ...     required_fields={"score": 50.0, "reasoning": ""},
        ... )
        >>> print(result.success)  # True
        >>> print(result.data["score"])  # 85.0

    Requirements: 9.1
    """
    optional_fields = optional_fields or {}
    parsed_data = {}
    missing_fields = []

    # Combine required and optional fields
    all_fields = {**required_fields, **optional_fields}

    for field_name, default_value in all_fields.items():
        # Try to extract field using multiple patterns
        patterns = [
            rf"{field_name}:\s*(.*?)(?=\n[A-Z_]+:|$)",  # FIELD: value
            rf"{field_name.lower()}:\s*(.*?)(?=\n[a-z_]+:|$)",  # field: value
        ]

        found = False
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1).strip()

                # Try to convert to appropriate type
                if isinstance(default_value, (int, float)):
                    try:
                        # Extract first number from value
                        num_match = re.search(r"-?\d+\.?\d*", value)
                        if num_match:
                            value = type(default_value)(num_match.group())
                        else:
                            value = default_value
                    except (ValueError, TypeError):
                        value = default_value

                parsed_data[field_name] = value
                found = True
                break

        if not found:
            # Field not found, use default
            parsed_data[field_name] = default_value
            if field_name in required_fields:
                missing_fields.append(field_name)

    # Determine success based on whether all required fields were found
    success = len(missing_fields) == 0

    error_message = None
    if not success:
        error_message = f"Missing required fields: {', '.join(missing_fields)}"
        logger.warning(f"Partial parse: {error_message}")

    return PartialResult(
        success=success,
        data=parsed_data,
        error_message=error_message,
        raw_output=text,
    )


@contextmanager
def timeout_context(seconds: float):
    """
    Context manager for timeout handling using signals (Unix only).

    This context manager raises a TimeoutError if the code block
    takes longer than the specified timeout.

    Args:
        seconds: Timeout in seconds

    Raises:
        TimeoutError: If the operation times out

    Example:
        >>> try:
        ...     with timeout_context(5.0):
        ...         # Some long-running operation
        ...         result = slow_function()
        ... except TimeoutError:
        ...     print("Operation timed out")

    Note:
        This uses SIGALRM which only works on Unix systems.
        For cross-platform support, use timeout_wrapper instead.

    Requirements: 9.2
    """

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set up the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(seconds))

    try:
        yield
    finally:
        # Restore the old handler and cancel the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def timeout_wrapper(
    func: Callable[..., T],
    timeout_seconds: float,
    *args,
    **kwargs,
) -> TimeoutResult:
    """
    Wrap a function with timeout handling (cross-platform).

    This function executes the given function with a timeout. If the function
    takes longer than the timeout, it returns a TimeoutResult indicating failure.

    Note: This is a simple polling-based implementation. For production use,
    consider using threading or multiprocessing for true timeout enforcement.

    Args:
        func: Function to execute
        timeout_seconds: Timeout in seconds
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        TimeoutResult with success status and result or timeout info

    Example:
        >>> def slow_function(x):
        ...     time.sleep(10)
        ...     return x * 2
        >>> result = timeout_wrapper(slow_function, 1.0, 5)
        >>> print(result.timed_out)  # True

    Requirements: 9.2
    """
    start_time = time.time()

    try:
        # For now, we execute the function directly
        # A more robust implementation would use threading
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time

        return TimeoutResult(
            success=True, result=result, timed_out=False, elapsed_time=elapsed
        )

    except Exception as e:
        elapsed = time.time() - start_time

        # Check if this looks like a timeout
        if elapsed >= timeout_seconds * 0.9:  # Within 90% of timeout
            logger.warning(
                f"Function {func.__name__} likely timed out after {elapsed:.2f}s"
            )
            return TimeoutResult(
                success=False,
                result=None,
                timed_out=True,
                elapsed_time=elapsed,
                partial_output=str(e),
            )
        else:
            # Regular exception, not a timeout
            logger.error(f"Function {func.__name__} failed: {e}")
            raise


def with_timeout(timeout_seconds: float):
    """
    Decorator for adding timeout handling to functions.

    Args:
        timeout_seconds: Timeout in seconds

    Returns:
        Decorated function that returns TimeoutResult

    Example:
        >>> @with_timeout(5.0)
        ... def slow_function(x):
        ...     time.sleep(10)
        ...     return x * 2
        >>> result = slow_function(5)
        >>> print(result.timed_out)  # True

    Requirements: 9.2
    """

    def decorator(func: Callable[..., T]) -> Callable[..., TimeoutResult]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> TimeoutResult:
            return timeout_wrapper(func, timeout_seconds, *args, **kwargs)

        return wrapper

    return decorator


def log_error_with_context(
    error: Exception,
    context: Dict[str, Any],
    logger_instance: Optional[logging.Logger] = None,
) -> None:
    """
    Log an error with structured context information.

    This function logs errors with additional context to aid in debugging.
    It includes the error type, message, and relevant context data.

    Args:
        error: The exception that occurred
        context: Dictionary of contextual information
        logger_instance: Optional logger to use (defaults to module logger)

    Example:
        >>> try:
        ...     result = risky_operation()
        ... except Exception as e:
        ...     log_error_with_context(
        ...         e,
        ...         {
        ...             "operation": "evaluation",
        ...             "model": "llama-3-8b",
        ...             "input_length": 1024,
        ...         }
        ...     )

    Requirements: 9.1, 9.2
    """
    log = logger_instance or logger

    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "context": context,
    }

    log.error(
        f"Error occurred: {error_info['error_type']} - {error_info['error_message']}"
    )
    log.error(f"Context: {error_info['context']}")

    # Log stack trace at debug level
    log.debug("Stack trace:", exc_info=True)


def safe_parse_judge_output(
    response: str, model_name: str, default_score: float = 50.0
) -> Dict[str, Any]:
    """
    Safely parse judge model output with fallback for malformed responses.

    This function attempts to parse structured output from a judge model.
    If parsing fails, it returns partial results with defaults.

    Args:
        response: Raw response from judge model
        model_name: Name of the judge model
        default_score: Default score if parsing fails

    Returns:
        Dictionary with parsed data (score, reasoning, issues)

    Example:
        >>> response = "SCORE: 85\\nREASONING: Good\\nFLAGGED_ISSUES: None"
        >>> result = safe_parse_judge_output(response, "llama-3-8b")
        >>> print(result["score"])  # 85.0

    Requirements: 9.1
    """
    # Parse score
    score_result = parse_score_with_fallback(response, default_score)
    score = score_result.data

    # Parse structured fields
    fields_result = parse_structured_output(
        response,
        required_fields={"reasoning": "No reasoning provided"},
        optional_fields={"flagged_issues": "None detected"},
    )

    parsed_data = {
        "score": score,
        "reasoning": fields_result.data.get("reasoning", "No reasoning provided"),
        "flagged_issues": fields_result.data.get("flagged_issues", "None detected"),
        "model_name": model_name,
        "parse_success": score_result.success and fields_result.success,
        "parse_errors": [],
    }

    # Collect parse errors
    if not score_result.success:
        parsed_data["parse_errors"].append(score_result.error_message)
    if not fields_result.success:
        parsed_data["parse_errors"].append(fields_result.error_message)

    if parsed_data["parse_errors"]:
        logger.warning(
            f"Partial parse for {model_name}: {'; '.join(parsed_data['parse_errors'])}"
        )

    return parsed_data


def handle_inference_error(
    error: Exception, operation: str, context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Handle inference errors with appropriate fallback behavior.

    This function provides standardized error handling for inference operations,
    returning structured error information.

    Args:
        error: The exception that occurred
        operation: Description of the operation that failed
        context: Contextual information about the operation

    Returns:
        Dictionary with error information and suggested action

    Example:
        >>> try:
        ...     result = model.generate(...)
        ... except Exception as e:
        ...     error_info = handle_inference_error(
        ...         e,
        ...         "judge_evaluation",
        ...         {"model": "llama-3-8b", "input_length": 1024}
        ...     )
        ...     print(error_info["action"])

    Requirements: 9.1, 9.2
    """
    error_type = type(error).__name__
    error_message = str(error)

    # Log the error with context
    log_error_with_context(error, context)

    # Determine appropriate action based on error type
    action = "retry"  # Default action

    if "out of memory" in error_message.lower() or "oom" in error_message.lower():
        action = "reduce_batch_size"
    elif "timeout" in error_message.lower() or "timed out" in error_message.lower() or error_type == "TimeoutError":
        action = "increase_timeout"
    elif "cuda" in error_message.lower() and "error" in error_message.lower():
        action = "fallback_to_cpu"
    elif "connection" in error_message.lower() or "network" in error_message.lower():
        action = "retry_with_backoff"

    return {
        "error_type": error_type,
        "error_message": error_message,
        "operation": operation,
        "context": context,
        "action": action,
        "timestamp": time.time(),
    }
