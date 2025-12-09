# Comprehensive Error Handling Implementation Summary

## Overview

Task 8 from the Ollama Judge Integration spec has been completed. This implementation provides comprehensive error handling for all API judge operations, covering all requirements from 6.1 through 6.6.

## What Was Implemented

### 1. APIErrorHandler Class

A centralized error handler in `src/llm_judge_auditor/utils/error_handling.py` that provides:

#### Missing API Keys Handling (Requirement 6.1)
- Detects missing API keys on initialization
- Provides formatted setup guide with links to obtain free keys
- Shows which specific keys are missing
- Includes environment variable setup instructions

#### Authentication Error Handling (Requirement 6.2)
- Catches invalid API key errors
- Provides service-specific troubleshooting steps
- Suggests key validation and regeneration
- No retry on authentication failures (fail fast)

#### Network Error Handling (Requirement 6.3)
- Automatic retry with exponential backoff (1s, 2s, 4s)
- Maximum 2 retries per request
- Detailed logging of retry attempts
- Graceful failure after max retries
- Troubleshooting guide for connectivity issues

#### Rate Limit Error Handling (Requirement 6.3)
- Automatic retry with longer delays
- Respects `Retry-After` header when provided
- Documents free tier limits (Groq: 30/min, Gemini: 15/min)
- Provides guidance on rate limit management
- Suggests batch processing and paid tier upgrades

#### Malformed Response Handling (Requirement 6.4)
- Fallback parsing for invalid JSON
- Multiple pattern matching for score extraction
- Default values for missing fields
- Low confidence scores for fallback results
- Detailed logging of parse failures

#### Partial Failure Handling (Requirement 6.4)
- Continues with successful judges when some fail
- Logs which judges failed
- Provides aggregated results from available judges
- Warning messages with failure details
- Complete failure detection and handling

#### Comprehensive Troubleshooting Guide (Requirements 6.5, 6.6)
- Formatted guide covering all error types
- Specific symptoms and solutions for each error
- Links to API documentation
- Step-by-step troubleshooting instructions
- Common issues and resolutions

### 2. Error Handling Methods

The `APIErrorHandler` class provides these static methods:

```python
# Handle specific error types
handle_missing_keys(available_keys) -> Dict[str, Any]
handle_authentication_error(service, error) -> Dict[str, Any]
handle_network_error(service, error, retry_count) -> Dict[str, Any]
handle_rate_limit_error(service, error, retry_after) -> Dict[str, Any]
handle_malformed_response(service, response_text, parse_error) -> Dict[str, Any]
handle_partial_failure(total_judges, successful_judges, failed_judges) -> Dict[str, Any]

# Get comprehensive guide
get_comprehensive_troubleshooting_guide() -> str
```

Each method returns a structured dictionary with:
- `error`: Boolean indicating if this is an error
- `error_type`: Classification of the error
- `message`: Human-readable error message
- `action`: Suggested action to take
- `help`: Detailed troubleshooting steps

### 3. Integration with Existing Components

The error handling is integrated throughout the system:

#### Judge Clients (Groq & Gemini)
- Custom exception classes for each error type
- Automatic retry logic in `_call_api_with_retry()`
- Fallback parsing in `_parse_response()`
- Detailed error logging with context

#### API Judge Ensemble
- Parallel execution with individual failure handling
- Partial failure detection and reporting
- Graceful degradation when judges fail
- Comprehensive logging of ensemble operations

#### API Key Manager
- Key validation with test calls
- Formatted setup guides with validation status
- Error message collection and display
- Troubleshooting guide generation

### 4. Documentation

Created comprehensive documentation:

#### `docs/API_ERROR_HANDLING.md`
- Complete guide to all error types
- Code examples for each scenario
- Best practices for error handling
- Error handling flow diagram
- API reference for error classes

#### `examples/comprehensive_error_handling_example.py`
- Demonstrates all 8 error scenarios
- Shows real-world usage patterns
- Includes validation and troubleshooting
- Can be run to test error handling

### 5. Error Types Covered

#### Custom Exceptions

**Groq:**
- `GroqAPIError` - Base exception
- `GroqAuthenticationError` - Invalid API key (401/403)
- `GroqNetworkError` - Connection/timeout issues
- `GroqRateLimitError` - Rate limit exceeded (429)

**Gemini:**
- `GeminiAPIError` - Base exception
- `GeminiAuthenticationError` - Invalid API key (401/403)
- `GeminiNetworkError` - Connection/timeout issues
- `GeminiRateLimitError` - Rate limit exceeded (429)

#### Error Scenarios

1. **Missing Keys** - No API keys configured
2. **Invalid Keys** - Authentication failures
3. **Network Issues** - Timeouts, DNS failures
4. **Rate Limits** - Quota exceeded
5. **Malformed Responses** - JSON parse errors
6. **Partial Failures** - Some judges fail
7. **Complete Failures** - All judges fail

## Testing

The implementation was tested with:

1. **Comprehensive Example** - `examples/comprehensive_error_handling_example.py`
   - Demonstrates all 8 error scenarios
   - Shows error handling in action
   - Validates error messages and troubleshooting

2. **Real-World Scenario** - Included in the example
   - Full evaluation flow with error handling
   - Key validation and ensemble initialization
   - Partial and complete failure handling

## Requirements Coverage

✅ **6.1** - Missing API keys with setup instructions
- Detects missing keys
- Provides formatted setup guide
- Shows links to obtain free keys
- Includes environment variable instructions

✅ **6.2** - Authentication error handling
- Catches invalid key errors
- Provides service-specific troubleshooting
- Suggests key validation steps
- No retry on auth failures

✅ **6.3** - Network and rate limit handling
- Automatic retry with exponential backoff
- Respects rate limit headers
- Documents free tier limits
- Provides connectivity troubleshooting

✅ **6.4** - Malformed response and partial failure handling
- Fallback parsing for invalid JSON
- Continues with successful judges
- Logs failure details
- Provides partial results

✅ **6.5** - Comprehensive troubleshooting guide
- Covers all error types
- Specific symptoms and solutions
- Links to documentation
- Step-by-step instructions

✅ **6.6** - Complete error handling coverage
- All scenarios handled
- Graceful degradation
- Detailed logging
- User-friendly error messages

## Key Features

### 1. Automatic Retry Logic
- Exponential backoff for network errors
- Respects API rate limit headers
- Maximum 2 retries per request
- Detailed logging of retry attempts

### 2. Fallback Parsing
- Multiple pattern matching for scores
- Default values for missing fields
- Partial result extraction
- Low confidence indicators

### 3. Graceful Degradation
- Continues with available judges
- Partial failure warnings
- Complete failure detection
- Comprehensive error reporting

### 4. User-Friendly Messages
- Formatted setup guides
- Clear error descriptions
- Actionable troubleshooting steps
- Links to documentation

### 5. Comprehensive Logging
- Error context capture
- Retry attempt tracking
- Parse failure details
- Judge failure reporting

## Usage Examples

### Basic Error Handling

```python
from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.utils.error_handling import APIErrorHandler

# Load and validate keys
api_key_manager = APIKeyManager()
available_keys = api_key_manager.load_keys()

if not api_key_manager.has_any_keys():
    error_info = APIErrorHandler.handle_missing_keys(available_keys)
    print(error_info['help'])
    print(api_key_manager.get_setup_instructions())
```

### Handling Evaluation Errors

```python
from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble

try:
    verdicts = ensemble.evaluate(source, candidate)
    
    # Check for partial failures
    if len(verdicts) < ensemble.get_judge_count():
        failed_judges = [
            name for name in ensemble.get_judge_names()
            if name not in [v.judge_name for v in verdicts]
        ]
        
        error_info = APIErrorHandler.handle_partial_failure(
            total_judges=ensemble.get_judge_count(),
            successful_judges=len(verdicts),
            failed_judges=failed_judges
        )
        
        print(f"Warning: {error_info['message']}")
        
except RuntimeError as e:
    print(f"Error: All judges failed - {e}")
    print(APIErrorHandler.get_comprehensive_troubleshooting_guide())
```

### Displaying Troubleshooting Guide

```python
from llm_judge_auditor.utils.error_handling import APIErrorHandler

# Show comprehensive troubleshooting guide
guide = APIErrorHandler.get_comprehensive_troubleshooting_guide()
print(guide)
```

## Files Modified/Created

### Modified Files
1. `src/llm_judge_auditor/utils/error_handling.py`
   - Added `APIErrorHandler` class
   - Added error handling methods for all scenarios
   - Added comprehensive troubleshooting guide

### Created Files
1. `examples/comprehensive_error_handling_example.py`
   - Demonstrates all error scenarios
   - Shows real-world usage
   - Validates error handling

2. `docs/API_ERROR_HANDLING.md`
   - Complete error handling guide
   - Code examples
   - Best practices
   - API reference

3. `COMPREHENSIVE_ERROR_HANDLING_SUMMARY.md` (this file)
   - Implementation summary
   - Requirements coverage
   - Usage examples

## Benefits

1. **Improved User Experience**
   - Clear error messages
   - Actionable troubleshooting steps
   - Formatted setup guides

2. **Increased Reliability**
   - Automatic retry logic
   - Graceful degradation
   - Partial failure handling

3. **Better Debugging**
   - Comprehensive logging
   - Error context capture
   - Detailed error information

4. **Easier Onboarding**
   - Setup instructions
   - Validation feedback
   - Troubleshooting guides

## Next Steps

The comprehensive error handling is now complete. The system can:
- Handle all error scenarios gracefully
- Provide clear troubleshooting guidance
- Continue operation with partial failures
- Give users actionable error messages

All requirements (6.1 through 6.6) have been satisfied.
