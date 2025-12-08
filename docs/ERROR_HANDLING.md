# Error Handling in LLM Judge Auditor

This document describes the error handling capabilities implemented in the LLM Judge Auditor toolkit.

## Overview

The toolkit includes comprehensive error handling utilities to ensure robust operation when dealing with:
- Malformed model outputs
- Inference timeouts
- Various runtime errors

**Requirements Addressed:** 9.1, 9.2

## Features

### 1. Malformed Output Parsing (Requirement 9.1)

The toolkit can gracefully handle malformed or incomplete model outputs by parsing partial results and using sensible defaults.

**Key Functions:**
- `parse_score_with_fallback()` - Parse scores from text with multiple format support
- `parse_structured_output()` - Parse structured fields with required/optional support
- `safe_parse_judge_output()` - Safely parse judge model outputs with fallback

**Example:**
```python
from llm_judge_auditor.utils.error_handling import safe_parse_judge_output

# Even if the model output is malformed, we get usable results
malformed_output = "This is completely malformed"
result = safe_parse_judge_output(malformed_output, "model-name", default_score=50.0)

print(result["score"])  # 50.0 (default)
print(result["parse_success"])  # False
print(result["parse_errors"])  # List of what went wrong
```

**Features:**
- Multiple score format detection (SCORE: 85, 85%, 85/100, etc.)
- Case-insensitive parsing
- Score clamping to valid ranges (0-100)
- Partial result extraction when some fields are missing
- Detailed error reporting

### 2. Timeout Handling (Requirement 9.2)

The toolkit provides utilities for handling long-running operations with timeouts.

**Key Functions:**
- `timeout_wrapper()` - Wrap any function with timeout tracking
- `with_timeout()` - Decorator for adding timeout handling
- `timeout_context()` - Context manager for timeout handling (Unix only)

**Example:**
```python
from llm_judge_auditor.utils.error_handling import timeout_wrapper

def slow_inference(prompt):
    # Some long-running model inference
    return model.generate(prompt)

# Wrap with timeout
result = timeout_wrapper(slow_inference, timeout_seconds=60.0, prompt="...")

if result.timed_out:
    print(f"Operation timed out after {result.elapsed_time}s")
else:
    print(f"Result: {result.result}")
```

**Features:**
- Cross-platform timeout tracking
- Elapsed time measurement
- Partial output capture on timeout
- Decorator and wrapper interfaces

### 3. Structured Error Logging (Requirements 9.1, 9.2)

The toolkit provides structured error logging with contextual information for debugging.

**Key Functions:**
- `log_error_with_context()` - Log errors with structured context
- `handle_inference_error()` - Handle inference errors with suggested actions

**Example:**
```python
from llm_judge_auditor.utils.error_handling import (
    log_error_with_context,
    handle_inference_error
)

try:
    result = model.generate(prompt)
except Exception as e:
    # Log with context
    log_error_with_context(e, {
        "operation": "inference",
        "model": "llama-3-8b",
        "input_length": 1024
    })
    
    # Get suggested action
    error_info = handle_inference_error(e, "inference", context)
    print(f"Suggested action: {error_info['action']}")
```

**Features:**
- Structured error context
- Automatic action suggestions based on error type:
  - Out of memory → reduce_batch_size
  - Timeout → increase_timeout
  - CUDA error → fallback_to_cpu
  - Network error → retry_with_backoff
- Stack trace logging at debug level

## Integration with Components

### Judge Ensemble

The `JudgeEnsemble` component has been enhanced with error handling:

1. **Malformed Output Parsing**: The `_parse_factual_accuracy_response()` method now uses `safe_parse_judge_output()` to handle malformed model outputs gracefully.

2. **Timeout Handling**: The `_generate_response()` method includes timeout tracking and logging for long-running inference operations.

3. **Error Context Logging**: All error paths include structured logging with context for debugging.

**Example Integration:**
```python
# In judge_ensemble.py
def _parse_factual_accuracy_response(self, response: str, model_name: str) -> JudgeResult:
    try:
        # Use safe parsing utility
        parsed = safe_parse_judge_output(response, model_name, default_score=50.0)
        
        # Log if parsing was not fully successful
        if not parsed["parse_success"]:
            logger.warning(f"Partial parse for {model_name}: {parsed['parse_errors']}")
        
        # Continue with partial results...
    except Exception as e:
        # Log error with context
        log_error_with_context(e, {
            "operation": "parse_factual_accuracy_response",
            "model_name": model_name,
            "response_length": len(response)
        })
        
        # Return default result
        return JudgeResult(...)
```

## Data Classes

### PartialResult

Represents a partial result from malformed or incomplete output.

```python
@dataclass
class PartialResult:
    success: bool
    data: Any
    error_message: Optional[str] = None
    raw_output: Optional[str] = None
```

### TimeoutResult

Represents the result of an operation that may have timed out.

```python
@dataclass
class TimeoutResult:
    success: bool
    result: Any = None
    timed_out: bool = False
    elapsed_time: float = 0.0
    partial_output: Optional[str] = None
```

## Testing

Comprehensive unit tests are provided in `tests/unit/test_error_handling.py`:

- 28 test cases covering all error handling utilities
- Tests for malformed output parsing
- Tests for timeout handling
- Tests for error logging
- Tests for structured output parsing

Run tests with:
```bash
pytest tests/unit/test_error_handling.py -v
```

## Examples

A complete example demonstrating all error handling features is available in `examples/error_handling_example.py`.

Run the example with:
```bash
python examples/error_handling_example.py
```

## Best Practices

1. **Always use safe parsing for model outputs**: Model outputs can be unpredictable. Use `safe_parse_judge_output()` instead of manual parsing.

2. **Set appropriate timeouts**: Different operations have different expected durations. Set timeouts based on your use case.

3. **Log errors with context**: Always include relevant context when logging errors to aid debugging.

4. **Handle partial results gracefully**: When parsing fails partially, use the available data and log warnings rather than failing completely.

5. **Provide sensible defaults**: When parsing fails, use sensible defaults (e.g., score=50.0 for neutral evaluation).

## Future Enhancements

Potential improvements for error handling:

1. **Retry logic**: Automatic retry with exponential backoff for transient errors
2. **Circuit breaker**: Prevent cascading failures by temporarily disabling failing components
3. **Metrics collection**: Track error rates and types for monitoring
4. **Advanced timeout handling**: True timeout enforcement using threading/multiprocessing
5. **Error recovery strategies**: Automatic fallback to simpler models or methods

## References

- Requirements: 9.1 (malformed output parsing), 9.2 (timeout handling)
- Design Document: Error Handling section
- Implementation: `src/llm_judge_auditor/utils/error_handling.py`
- Tests: `tests/unit/test_error_handling.py`
- Examples: `examples/error_handling_example.py`
