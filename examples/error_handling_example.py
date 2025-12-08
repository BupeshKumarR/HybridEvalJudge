"""
Example demonstrating error handling utilities in the LLM Judge Auditor toolkit.

This example shows:
1. Malformed output parsing with partial results
2. Timeout handling for inference operations
3. Structured error logging

Requirements: 9.1, 9.2
"""

import logging
import time

from llm_judge_auditor.utils.error_handling import (
    TimeoutError,
    handle_inference_error,
    log_error_with_context,
    parse_score_with_fallback,
    parse_structured_output,
    safe_parse_judge_output,
    timeout_wrapper,
    with_timeout,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_malformed_output_parsing():
    """Demonstrate parsing malformed model outputs with fallback."""
    print("\n" + "=" * 80)
    print("Example 1: Malformed Output Parsing")
    print("=" * 80)

    # Example 1: Valid output
    print("\n1. Parsing valid output:")
    valid_output = "SCORE: 85\nREASONING: Good answer\nFLAGGED_ISSUES: None"
    result = safe_parse_judge_output(valid_output, "test-model")
    print(f"  Score: {result['score']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Parse success: {result['parse_success']}")

    # Example 2: Malformed output (missing score)
    print("\n2. Parsing malformed output (missing score):")
    malformed_output = "This is a completely malformed response without proper structure"
    result = safe_parse_judge_output(malformed_output, "test-model", default_score=50.0)
    print(f"  Score: {result['score']} (using default)")
    print(f"  Reasoning: {result['reasoning'][:50]}...")
    print(f"  Parse success: {result['parse_success']}")
    print(f"  Parse errors: {result['parse_errors']}")

    # Example 3: Partial output (score only)
    print("\n3. Parsing partial output (score only):")
    partial_output = "SCORE: 75"
    result = safe_parse_judge_output(partial_output, "test-model")
    print(f"  Score: {result['score']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Parse success: {result['parse_success']}")

    # Example 4: Score parsing with various formats
    print("\n4. Score parsing with various formats:")
    test_cases = [
        "SCORE: 85",
        "score: 90.5",
        "The rating is 75%",
        "Score: 80/100",
        "Invalid text with no score",
    ]
    for text in test_cases:
        result = parse_score_with_fallback(text, default_score=50.0)
        print(f"  '{text[:30]}...' -> {result.data} (success: {result.success})")


def example_timeout_handling():
    """Demonstrate timeout handling for long-running operations."""
    print("\n" + "=" * 80)
    print("Example 2: Timeout Handling")
    print("=" * 80)

    # Example 1: Fast function (completes within timeout)
    print("\n1. Fast function (completes within timeout):")

    def fast_function(x):
        return x * 2

    result = timeout_wrapper(fast_function, 1.0, 5)
    print(f"  Result: {result.result}")
    print(f"  Success: {result.success}")
    print(f"  Timed out: {result.timed_out}")
    print(f"  Elapsed time: {result.elapsed_time:.4f}s")

    # Example 2: Slow function (takes time but completes)
    print("\n2. Slow function (takes time but completes):")

    def slow_function(x):
        time.sleep(0.1)
        return x * 2

    result = timeout_wrapper(slow_function, 1.0, 5)
    print(f"  Result: {result.result}")
    print(f"  Success: {result.success}")
    print(f"  Timed out: {result.timed_out}")
    print(f"  Elapsed time: {result.elapsed_time:.4f}s")

    # Example 3: Using timeout decorator
    print("\n3. Using timeout decorator:")

    @with_timeout(1.0)
    def decorated_function(x):
        return x * 3

    result = decorated_function(7)
    print(f"  Result: {result.result}")
    print(f"  Success: {result.success}")
    print(f"  Elapsed time: {result.elapsed_time:.4f}s")


def example_error_logging():
    """Demonstrate structured error logging with context."""
    print("\n" + "=" * 80)
    print("Example 3: Structured Error Logging")
    print("=" * 80)

    # Example 1: Logging a ValueError with context
    print("\n1. Logging ValueError with context:")
    try:
        # Simulate an error
        raise ValueError("Invalid input parameter")
    except Exception as e:
        context = {
            "operation": "model_inference",
            "model_name": "llama-3-8b",
            "input_length": 1024,
            "batch_size": 8,
        }
        log_error_with_context(e, context)
        print("  Error logged with context (check logs above)")

    # Example 2: Handling inference errors with suggested actions
    print("\n2. Handling inference errors with suggested actions:")

    # Out of memory error
    oom_error = RuntimeError("CUDA out of memory")
    error_info = handle_inference_error(
        oom_error, "inference", {"model": "llama-3-8b", "batch_size": 32}
    )
    print(f"  Error type: {error_info['error_type']}")
    print(f"  Suggested action: {error_info['action']}")

    # Timeout error
    timeout_error = TimeoutError("Operation timed out after 60s")
    error_info = handle_inference_error(
        timeout_error, "generation", {"model": "mistral-7b"}
    )
    print(f"  Error type: {error_info['error_type']}")
    print(f"  Suggested action: {error_info['action']}")

    # CUDA error
    cuda_error = RuntimeError("CUDA error: device-side assert triggered")
    error_info = handle_inference_error(cuda_error, "inference", {"model": "phi-3"})
    print(f"  Error type: {error_info['error_type']}")
    print(f"  Suggested action: {error_info['action']}")


def example_structured_output_parsing():
    """Demonstrate parsing structured output with required and optional fields."""
    print("\n" + "=" * 80)
    print("Example 4: Structured Output Parsing")
    print("=" * 80)

    # Example 1: Complete output
    print("\n1. Parsing complete structured output:")
    complete_output = """
SCORE: 85
REASONING: The answer is accurate and well-supported.
CONFIDENCE: 0.9
FLAGGED_ISSUES: None detected
"""
    result = parse_structured_output(
        complete_output,
        required_fields={"score": 0.0, "reasoning": ""},
        optional_fields={"confidence": 0.5, "flagged_issues": ""},
    )
    print(f"  Success: {result.success}")
    print(f"  Score: {result.data['score']}")
    print(f"  Reasoning: {result.data['reasoning'][:50]}...")
    print(f"  Confidence: {result.data['confidence']}")

    # Example 2: Partial output (missing optional field)
    print("\n2. Parsing partial output (missing optional field):")
    partial_output = """
SCORE: 75
REASONING: Mostly correct but lacks detail.
"""
    result = parse_structured_output(
        partial_output,
        required_fields={"score": 0.0, "reasoning": ""},
        optional_fields={"confidence": 0.5},
    )
    print(f"  Success: {result.success}")
    print(f"  Score: {result.data['score']}")
    print(f"  Confidence: {result.data['confidence']} (default)")

    # Example 3: Missing required field
    print("\n3. Parsing output with missing required field:")
    incomplete_output = "REASONING: Some reasoning text"
    result = parse_structured_output(
        incomplete_output, required_fields={"score": 0.0, "reasoning": ""}
    )
    print(f"  Success: {result.success}")
    print(f"  Error: {result.error_message}")
    print(f"  Score: {result.data['score']} (default)")


def main():
    """Run all error handling examples."""
    print("\n" + "=" * 80)
    print("LLM Judge Auditor - Error Handling Examples")
    print("=" * 80)

    # Run examples
    example_malformed_output_parsing()
    example_timeout_handling()
    example_error_logging()
    example_structured_output_parsing()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
