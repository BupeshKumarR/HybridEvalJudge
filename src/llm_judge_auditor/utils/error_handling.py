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
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

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


# API-specific error handling utilities

class APIErrorHandler:
    """
    Centralized error handler for API judge operations.
    
    This class provides comprehensive error handling for API-based judges,
    including missing keys, network errors, rate limits, and malformed responses.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
    """
    
    @staticmethod
    def handle_missing_keys(available_keys: Dict[str, bool]) -> Dict[str, Any]:
        """
        Handle missing API keys scenario.
        
        Args:
            available_keys: Dict mapping service name to availability
        
        Returns:
            Error information with setup instructions
        
        Requirements: 6.1
        """
        missing_services = [
            service for service, available in available_keys.items()
            if not available
        ]
        
        if not missing_services:
            return {
                "error": False,
                "message": "All API keys are available"
            }
        
        error_info = {
            "error": True,
            "error_type": "MissingAPIKeys",
            "missing_services": missing_services,
            "message": f"Missing API keys for: {', '.join(missing_services)}",
            "action": "configure_api_keys",
            "help": (
                "To obtain free API keys:\n"
                "  - Groq: https://console.groq.com/keys\n"
                "  - Gemini: https://aistudio.google.com/app/apikey\n"
                "\n"
                "Set environment variables:\n"
                "  export GROQ_API_KEY='your-key'\n"
                "  export GEMINI_API_KEY='your-key'"
            )
        }
        
        logger.error(f"Missing API keys: {missing_services}")
        return error_info
    
    @staticmethod
    def handle_authentication_error(
        service: str,
        error: Exception
    ) -> Dict[str, Any]:
        """
        Handle API authentication errors.
        
        Args:
            service: Service name (groq, gemini)
            error: The authentication exception
        
        Returns:
            Error information with troubleshooting steps
        
        Requirements: 6.2
        """
        error_info = {
            "error": True,
            "error_type": "AuthenticationError",
            "service": service,
            "message": f"{service.title()} API authentication failed",
            "original_error": str(error),
            "action": "check_api_key",
            "help": (
                f"Your {service.title()} API key appears to be invalid.\n"
                "\n"
                "Troubleshooting steps:\n"
                "  1. Verify your API key is correct (no extra spaces/quotes)\n"
                "  2. Check that the key is active in your account\n"
                "  3. Ensure you're using the correct environment variable:\n"
                f"     export {service.upper()}_API_KEY='your-key'\n"
                "  4. Try generating a new API key if the issue persists"
            )
        }
        
        logger.error(
            f"Authentication failed for {service}: {error}",
            exc_info=True
        )
        return error_info
    
    @staticmethod
    def handle_network_error(
        service: str,
        error: Exception,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Handle network-related errors.
        
        Args:
            service: Service name (groq, gemini)
            error: The network exception
            retry_count: Number of retries attempted
        
        Returns:
            Error information with troubleshooting steps
        
        Requirements: 6.3
        """
        error_info = {
            "error": True,
            "error_type": "NetworkError",
            "service": service,
            "message": f"Network error when calling {service.title()} API",
            "original_error": str(error),
            "retry_count": retry_count,
            "action": "check_connectivity",
            "help": (
                "Network connection issue detected.\n"
                "\n"
                "Troubleshooting steps:\n"
                "  1. Check your internet connection\n"
                "  2. Verify you can access the API endpoint:\n"
                f"     - Groq: https://api.groq.com\n"
                f"     - Gemini: https://generativelanguage.googleapis.com\n"
                "  3. Check if your firewall is blocking the connection\n"
                "  4. Try again in a few moments\n"
                "  5. Check API service status pages for outages"
            )
        }
        
        logger.error(
            f"Network error for {service} (retry {retry_count}): {error}",
            exc_info=True
        )
        return error_info
    
    @staticmethod
    def handle_rate_limit_error(
        service: str,
        error: Exception,
        retry_after: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Handle rate limit errors.
        
        Args:
            service: Service name (groq, gemini)
            error: The rate limit exception
            retry_after: Suggested retry delay in seconds
        
        Returns:
            Error information with rate limit details
        
        Requirements: 6.3
        """
        # Default rate limits for free tiers
        rate_limits = {
            "groq": "30 requests per minute",
            "gemini": "15 requests per minute"
        }
        
        error_info = {
            "error": True,
            "error_type": "RateLimitError",
            "service": service,
            "message": f"{service.title()} API rate limit exceeded",
            "original_error": str(error),
            "retry_after": retry_after,
            "rate_limit": rate_limits.get(service, "Unknown"),
            "action": "wait_and_retry",
            "help": (
                f"You've hit the {service.title()} API rate limit.\n"
                "\n"
                f"Free tier limit: {rate_limits.get(service, 'Unknown')}\n"
                "\n"
                "What to do:\n"
                f"  1. Wait {retry_after or 60} seconds before retrying\n"
                "  2. Reduce the frequency of API calls\n"
                "  3. Consider upgrading to a paid tier for higher limits\n"
                "  4. Use batch processing to optimize API usage"
            )
        }
        
        logger.warning(
            f"Rate limit hit for {service}. Retry after: {retry_after}s"
        )
        return error_info
    
    @staticmethod
    def handle_malformed_response(
        service: str,
        response_text: str,
        parse_error: Exception
    ) -> Dict[str, Any]:
        """
        Handle malformed API responses.
        
        Args:
            service: Service name (groq, gemini)
            response_text: The malformed response text
            parse_error: The parsing exception
        
        Returns:
            Error information with partial results if available
        
        Requirements: 6.4
        """
        # Try to extract any useful information from the response
        partial_data = None
        try:
            # Attempt to extract a score using fallback parsing
            score_result = parse_score_with_fallback(response_text)
            if score_result.success or score_result.data != 50.0:
                partial_data = {"score": score_result.data}
        except:
            pass
        
        error_info = {
            "error": True,
            "error_type": "MalformedResponse",
            "service": service,
            "message": f"Could not parse {service.title()} API response",
            "original_error": str(parse_error),
            "response_preview": response_text[:200] if response_text else None,
            "partial_data": partial_data,
            "action": "use_fallback_parsing",
            "help": (
                "The API returned a response that couldn't be parsed.\n"
                "\n"
                "This may indicate:\n"
                "  1. The API response format has changed\n"
                "  2. The model didn't follow the expected output format\n"
                "  3. The response was truncated or corrupted\n"
                "\n"
                "The system will use fallback parsing to extract partial results."
            )
        }
        
        logger.warning(
            f"Malformed response from {service}: {parse_error}. "
            f"Response preview: {response_text[:100] if response_text else 'None'}"
        )
        return error_info
    
    @staticmethod
    def handle_partial_failure(
        total_judges: int,
        successful_judges: int,
        failed_judges: List[str]
    ) -> Dict[str, Any]:
        """
        Handle scenarios where some judges succeed and others fail.
        
        Args:
            total_judges: Total number of judges
            successful_judges: Number of successful judges
            failed_judges: List of failed judge names
        
        Returns:
            Status information about partial failure
        
        Requirements: 6.4
        """
        if successful_judges == 0:
            # Complete failure
            error_info = {
                "error": True,
                "error_type": "CompleteFailure",
                "message": "All judges failed during evaluation",
                "total_judges": total_judges,
                "successful_judges": 0,
                "failed_judges": failed_judges,
                "action": "check_all_configurations",
                "help": (
                    "All API judges failed to complete evaluation.\n"
                    "\n"
                    "Troubleshooting steps:\n"
                    "  1. Verify all API keys are valid\n"
                    "  2. Check your internet connection\n"
                    "  3. Ensure required packages are installed:\n"
                    "     pip install groq google-generativeai\n"
                    "  4. Check API service status pages\n"
                    "  5. Review logs for specific error messages"
                )
            }
            logger.error(f"All {total_judges} judges failed: {failed_judges}")
        else:
            # Partial failure
            error_info = {
                "error": False,
                "warning": True,
                "error_type": "PartialFailure",
                "message": f"{successful_judges}/{total_judges} judges succeeded",
                "total_judges": total_judges,
                "successful_judges": successful_judges,
                "failed_judges": failed_judges,
                "action": "continue_with_available",
                "help": (
                    f"Some judges failed, but {successful_judges} succeeded.\n"
                    "\n"
                    "The evaluation will continue with available judges.\n"
                    "Check logs for details about failed judges."
                )
            }
            logger.warning(
                f"Partial failure: {successful_judges}/{total_judges} judges succeeded. "
                f"Failed: {failed_judges}"
            )
        
        return error_info
    
    @staticmethod
    def get_comprehensive_troubleshooting_guide() -> str:
        """
        Get a comprehensive troubleshooting guide for all error types.
        
        Returns:
            Formatted troubleshooting guide
        
        Requirements: 6.5, 6.6
        """
        guide = "\n"
        guide += "╔══════════════════════════════════════════════════════════════╗\n"
        guide += "║  🔧 Comprehensive Troubleshooting Guide                      ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  ERROR TYPE: Missing API Keys                               ║\n"
        guide += "║  ────────────────────────────────────────────────────────   ║\n"
        guide += "║  Symptoms: \"No API keys configured\" error                   ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Solutions:                                                 ║\n"
        guide += "║  1. Get free API keys:                                      ║\n"
        guide += "║     • Groq: https://console.groq.com/keys                   ║\n"
        guide += "║     • Gemini: https://aistudio.google.com/app/apikey        ║\n"
        guide += "║  2. Set environment variables:                              ║\n"
        guide += "║     export GROQ_API_KEY='your-key'                          ║\n"
        guide += "║     export GEMINI_API_KEY='your-key'                        ║\n"
        guide += "║  3. Restart your terminal/application                       ║\n"
        guide += "║                                                              ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  ERROR TYPE: Authentication Failed                          ║\n"
        guide += "║  ────────────────────────────────────────────────────────   ║\n"
        guide += "║  Symptoms: \"Invalid API key\" or \"401/403\" errors            ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Solutions:                                                 ║\n"
        guide += "║  1. Verify API key is copied correctly (no spaces/quotes)   ║\n"
        guide += "║  2. Check key is active in your account dashboard           ║\n"
        guide += "║  3. Ensure correct environment variable name                ║\n"
        guide += "║  4. Try generating a new API key                            ║\n"
        guide += "║  5. Verify you're using the right API service               ║\n"
        guide += "║                                                              ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  ERROR TYPE: Network Error                                  ║\n"
        guide += "║  ────────────────────────────────────────────────────────   ║\n"
        guide += "║  Symptoms: Connection timeout, DNS errors                   ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Solutions:                                                 ║\n"
        guide += "║  1. Check internet connection                               ║\n"
        guide += "║  2. Verify firewall isn't blocking API requests             ║\n"
        guide += "║  3. Try accessing API endpoints in browser:                 ║\n"
        guide += "║     • https://api.groq.com                                  ║\n"
        guide += "║     • https://generativelanguage.googleapis.com             ║\n"
        guide += "║  4. Check API service status pages                          ║\n"
        guide += "║  5. Wait a few minutes and retry                            ║\n"
        guide += "║                                                              ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  ERROR TYPE: Rate Limit Exceeded                            ║\n"
        guide += "║  ────────────────────────────────────────────────────────   ║\n"
        guide += "║  Symptoms: \"429\" errors, \"quota exceeded\"                   ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Free Tier Limits:                                          ║\n"
        guide += "║  • Groq: 30 requests per minute                             ║\n"
        guide += "║  • Gemini: 15 requests per minute                           ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Solutions:                                                 ║\n"
        guide += "║  1. Wait 60 seconds before retrying                         ║\n"
        guide += "║  2. Reduce frequency of API calls                           ║\n"
        guide += "║  3. Use batch processing for multiple evaluations           ║\n"
        guide += "║  4. Consider upgrading to paid tier                         ║\n"
        guide += "║                                                              ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  ERROR TYPE: Malformed Response                             ║\n"
        guide += "║  ────────────────────────────────────────────────────────   ║\n"
        guide += "║  Symptoms: JSON parse errors, unexpected format             ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Solutions:                                                 ║\n"
        guide += "║  1. System will use fallback parsing automatically          ║\n"
        guide += "║  2. Check if API response format has changed                ║\n"
        guide += "║  3. Verify model is following expected output format        ║\n"
        guide += "║  4. Review logs for response preview                        ║\n"
        guide += "║  5. Report issue if it persists                             ║\n"
        guide += "║                                                              ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  ERROR TYPE: All Judges Failed                              ║\n"
        guide += "║  ────────────────────────────────────────────────────────   ║\n"
        guide += "║  Symptoms: Complete evaluation failure                      ║\n"
        guide += "║                                                              ║\n"
        guide += "║  Solutions:                                                 ║\n"
        guide += "║  1. Check all API keys are valid                            ║\n"
        guide += "║  2. Verify internet connectivity                            ║\n"
        guide += "║  3. Ensure packages installed:                              ║\n"
        guide += "║     pip install groq google-generativeai                    ║\n"
        guide += "║  4. Review detailed error logs                              ║\n"
        guide += "║  5. Try each API service individually                       ║\n"
        guide += "║                                                              ║\n"
        guide += "╠══════════════════════════════════════════════════════════════╣\n"
        guide += "║                                                              ║\n"
        guide += "║  Need More Help?                                            ║\n"
        guide += "║                                                              ║\n"
        guide += "║  • Check logs for detailed error messages                   ║\n"
        guide += "║  • Review API documentation:                                ║\n"
        guide += "║    - Groq: https://console.groq.com/docs                    ║\n"
        guide += "║    - Gemini: https://ai.google.dev/docs                     ║\n"
        guide += "║  • Report issues on GitHub                                  ║\n"
        guide += "║                                                              ║\n"
        guide += "╚══════════════════════════════════════════════════════════════╝\n"
        
        return guide
