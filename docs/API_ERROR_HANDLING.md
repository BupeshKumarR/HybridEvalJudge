# API Error Handling Guide

This guide covers comprehensive error handling for API-based judge integration in the LLM Judge Auditor toolkit.

## Overview

The system provides robust error handling for all common scenarios when using API-based judges (Groq and Gemini):

- **Missing API Keys** - Clear setup instructions when keys are not configured
- **Authentication Errors** - Detailed troubleshooting for invalid keys
- **Network Errors** - Automatic retry with exponential backoff
- **Rate Limit Errors** - Intelligent handling of API rate limits
- **Malformed Responses** - Fallback parsing for unexpected formats
- **Partial Failures** - Graceful degradation when some judges fail

## Error Types and Handling

### 1. Missing API Keys

**Symptoms:**
- "No API keys configured" error
- Empty judge ensemble

**Automatic Handling:**
- System detects missing keys on initialization
- Displays formatted setup guide with links
- Shows which keys are missing

**Example:**
```python
from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.utils.error_handling import APIErrorHandler

api_key_manager = APIKeyManager()
available_keys = api_key_manager.load_keys()

if not api_key_manager.has_any_keys():
    error_info = APIErrorHandler.handle_missing_keys(available_keys)
    print(error_info['help'])
    print(api_key_manager.get_setup_instructions())
```

**Resolution:**
1. Get free API keys:
   - Groq: https://console.groq.com/keys
   - Gemini: https://aistudio.google.com/app/apikey
2. Set environment variables:
   ```bash
   export GROQ_API_KEY='your-key'
   export GEMINI_API_KEY='your-key'
   ```
3. Restart your application

**Requirements:** 6.1

---

### 2. Authentication Errors

**Symptoms:**
- "Invalid API key" errors
- 401/403 HTTP status codes
- "Authentication failed" messages

**Automatic Handling:**
- Caught by judge clients (no retry)
- Detailed error message with service name
- Troubleshooting steps provided

**Example:**
```python
from llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqAuthenticationError
)
from llm_judge_auditor.utils.error_handling import APIErrorHandler

try:
    client = GroqJudgeClient(api_key="invalid_key")
    verdict = client.evaluate(source, candidate)
except GroqAuthenticationError as e:
    error_info = APIErrorHandler.handle_authentication_error("groq", e)
    print(error_info['help'])
```

**Resolution:**
1. Verify API key is copied correctly (no spaces/quotes)
2. Check key is active in your account dashboard
3. Ensure correct environment variable name
4. Try generating a new API key
5. Verify you're using the right API service

**Requirements:** 6.2

---

### 3. Network Errors

**Symptoms:**
- Connection timeouts
- DNS resolution failures
- "Network error" messages

**Automatic Handling:**
- Automatic retry with exponential backoff (1s, 2s, 4s)
- Maximum 2 retries per request
- Detailed logging of retry attempts
- Graceful failure after max retries

**Example:**
```python
# Network errors are handled automatically by judge clients
from llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqNetworkError
)

try:
    client = GroqJudgeClient(api_key=api_key)
    verdict = client.evaluate(source, candidate)
    # Automatic retry on network errors
except GroqNetworkError as e:
    # Only raised after all retries exhausted
    error_info = APIErrorHandler.handle_network_error("groq", e, retry_count=2)
    print(error_info['help'])
```

**Resolution:**
1. Check your internet connection
2. Verify firewall isn't blocking API requests
3. Try accessing API endpoints in browser
4. Check API service status pages
5. Wait a few minutes and retry

**Requirements:** 6.3

---

### 4. Rate Limit Errors

**Symptoms:**
- "Rate limit exceeded" errors
- 429 HTTP status codes
- "Quota exceeded" messages

**Free Tier Limits:**
- **Groq:** 30 requests per minute
- **Gemini:** 15 requests per minute

**Automatic Handling:**
- Automatic retry with longer delay
- Respects `Retry-After` header if provided
- Exponential backoff if no header
- Maximum 2 retries

**Example:**
```python
from llm_judge_auditor.components.groq_judge_client import (
    GroqJudgeClient,
    GroqRateLimitError
)

try:
    client = GroqJudgeClient(api_key=api_key)
    verdict = client.evaluate(source, candidate)
    # Automatic retry on rate limits
except GroqRateLimitError as e:
    # Only raised after all retries exhausted
    retry_after = e.retry_after
    error_info = APIErrorHandler.handle_rate_limit_error(
        "groq", e, retry_after
    )
    print(f"Wait {retry_after} seconds before retrying")
```

**Resolution:**
1. Wait the suggested time before retrying
2. Reduce frequency of API calls
3. Use batch processing for multiple evaluations
4. Consider upgrading to paid tier for higher limits

**Requirements:** 6.3

---

### 5. Malformed Responses

**Symptoms:**
- JSON parse errors
- Missing required fields
- Unexpected response format

**Automatic Handling:**
- Fallback parsing attempts to extract score
- Uses default values for missing fields
- Low confidence score for fallback results
- Detailed logging of parse failures

**Example:**
```python
# Malformed responses are handled automatically by judge clients
from llm_judge_auditor.components.groq_judge_client import GroqJudgeClient

client = GroqJudgeClient(api_key=api_key)
verdict = client.evaluate(source, candidate)

# Check if fallback parsing was used
if verdict.metadata.get('fallback_parsing'):
    print(f"Warning: Fallback parsing used (confidence: {verdict.confidence})")
    print(f"Score: {verdict.score}")
```

**Fallback Parsing:**
The system attempts to extract information using multiple patterns:
- Score patterns: `SCORE: 85`, `85/100`, `85%`
- Field patterns: `FIELD: value`
- Default values for missing data

**Resolution:**
1. System handles automatically with fallback parsing
2. Check if API response format has changed
3. Verify model is following expected output format
4. Review logs for response preview
5. Report issue if it persists

**Requirements:** 6.4

---

### 6. Partial Failures

**Symptoms:**
- Some judges succeed, others fail
- "Partial failure" warnings
- Reduced number of verdicts

**Automatic Handling:**
- Ensemble continues with successful judges
- Logs which judges failed
- Provides aggregated results from available judges
- Warning message with failure details

**Example:**
```python
from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from llm_judge_auditor.utils.error_handling import APIErrorHandler

ensemble = APIJudgeEnsemble(config, api_key_manager)

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
        print(f"Failed judges: {failed_judges}")
        
except RuntimeError as e:
    # All judges failed
    print(f"Error: All judges failed - {e}")
```

**Resolution:**
- **Partial Success:** Continue with available judges
- **Complete Failure:** Check all configurations
  1. Verify all API keys are valid
  2. Check internet connection
  3. Ensure required packages installed
  4. Review logs for specific errors

**Requirements:** 6.4

---

## Comprehensive Troubleshooting

### Quick Diagnostics

```python
from llm_judge_auditor.components.api_key_manager import APIKeyManager

# Load and validate keys
api_key_manager = APIKeyManager()
api_key_manager.load_keys()

# Validate all keys
validation_results = api_key_manager.validate_all_keys(verbose=True)

# Display validation summary
print(api_key_manager.get_validation_summary())

# Show setup guide if needed
if not all(validation_results.values()):
    print(api_key_manager.get_setup_instructions(show_validation=True))
```

### Troubleshooting Guide

```python
from llm_judge_auditor.utils.error_handling import APIErrorHandler

# Display comprehensive troubleshooting guide
guide = APIErrorHandler.get_comprehensive_troubleshooting_guide()
print(guide)
```

The comprehensive guide covers:
- Missing API keys
- Authentication failures
- Network errors
- Rate limit issues
- Malformed responses
- Complete failures

**Requirements:** 6.5, 6.6

---

## Best Practices

### 1. Always Validate Keys on Startup

```python
api_key_manager = APIKeyManager()
api_key_manager.load_keys()

if api_key_manager.has_any_keys():
    # Validate with lightweight test calls
    validation_results = api_key_manager.validate_all_keys()
    
    if not any(validation_results.values()):
        print("No valid API keys found!")
        print(api_key_manager.get_setup_instructions(show_validation=True))
        sys.exit(1)
```

### 2. Handle Partial Failures Gracefully

```python
try:
    verdicts = ensemble.evaluate(source, candidate)
    
    if len(verdicts) < ensemble.get_judge_count():
        logger.warning("Some judges failed, continuing with available judges")
    
    # Use available verdicts
    consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts)
    
except RuntimeError as e:
    logger.error(f"All judges failed: {e}")
    # Fallback behavior or error reporting
```

### 3. Log Errors with Context

```python
from llm_judge_auditor.utils.error_handling import log_error_with_context

try:
    verdict = client.evaluate(source, candidate)
except Exception as e:
    log_error_with_context(
        error=e,
        context={
            "service": "groq",
            "source_length": len(source),
            "candidate_length": len(candidate),
            "task": "factual_accuracy"
        }
    )
```

### 4. Use Parallel Execution with Error Handling

```python
# Parallel execution handles individual failures automatically
ensemble = APIJudgeEnsemble(
    config=config,
    api_key_manager=api_key_manager,
    parallel_execution=True  # Recommended
)

# Each judge failure is logged but doesn't stop others
verdicts = ensemble.evaluate(source, candidate)
```

### 5. Monitor Rate Limits

```python
import time

# Track request timing
start_time = time.time()

try:
    verdict = client.evaluate(source, candidate)
except GroqRateLimitError as e:
    # Wait before retrying
    if e.retry_after:
        time.sleep(e.retry_after)
    else:
        time.sleep(60)  # Default wait
```

---

## Error Handling Flow

```
┌─────────────────────────────────────────┐
│  API Judge Evaluation Request          │
└──────────────┬──────────────────────────┘
               │
               ├─→ Check API Keys
               │   ├─ Missing? → Show Setup Guide
               │   └─ Invalid? → Authentication Error
               │
               ├─→ Make API Call
               │   ├─ Network Error? → Retry (2x)
               │   ├─ Rate Limit? → Wait & Retry
               │   └─ Success? → Continue
               │
               ├─→ Parse Response
               │   ├─ Valid JSON? → Extract Data
               │   ├─ Malformed? → Fallback Parsing
               │   └─ Empty? → Use Defaults
               │
               └─→ Return Verdict
                   ├─ Success → JudgeVerdict
                   ├─ Partial → Warning + Verdict
                   └─ Failure → Error + Troubleshooting
```

---

## Testing Error Handling

See `examples/comprehensive_error_handling_example.py` for complete demonstrations of all error scenarios.

Run the example:
```bash
python examples/comprehensive_error_handling_example.py
```

This demonstrates:
1. Missing API keys handling
2. Authentication error handling
3. Network error handling
4. Rate limit handling
5. Malformed response handling
6. Partial failure handling
7. Comprehensive troubleshooting
8. Real-world scenario

---

## API Reference

### APIErrorHandler

Centralized error handler for API judge operations.

**Methods:**

- `handle_missing_keys(available_keys)` - Handle missing API keys
- `handle_authentication_error(service, error)` - Handle auth failures
- `handle_network_error(service, error, retry_count)` - Handle network issues
- `handle_rate_limit_error(service, error, retry_after)` - Handle rate limits
- `handle_malformed_response(service, response_text, parse_error)` - Handle parse errors
- `handle_partial_failure(total_judges, successful_judges, failed_judges)` - Handle partial failures
- `get_comprehensive_troubleshooting_guide()` - Get full troubleshooting guide

### Judge Client Exceptions

**Groq:**
- `GroqAPIError` - Base exception
- `GroqAuthenticationError` - Invalid API key
- `GroqNetworkError` - Network issues
- `GroqRateLimitError` - Rate limit exceeded

**Gemini:**
- `GeminiAPIError` - Base exception
- `GeminiAuthenticationError` - Invalid API key
- `GeminiNetworkError` - Network issues
- `GeminiRateLimitError` - Rate limit exceeded

---

## Related Documentation

- [API Judge Ensemble Guide](API_JUDGE_ENSEMBLE.md)
- [API Key Setup Guide](../demo/FREE_SETUP_GUIDE.md)
- [Error Handling Utilities](ERROR_HANDLING.md)
- [Usage Guide](USAGE_GUIDE.md)

---

## Requirements Coverage

This error handling implementation satisfies:

- **6.1** - Missing API keys with setup instructions
- **6.2** - Authentication error handling with troubleshooting
- **6.3** - Network and rate limit error handling with retry
- **6.4** - Malformed response and partial failure handling
- **6.5** - Comprehensive troubleshooting guide
- **6.6** - Complete error handling for all scenarios
