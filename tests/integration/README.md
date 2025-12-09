# Integration Tests

This directory contains integration tests for the LLM Judge Auditor toolkit.

## Test Coverage

### API Judge Integration (`test_api_judges.py`)

Tests real API integration with Groq and Gemini services:

1. **TestAPIKeyManager**: API key management tests
   - Loading keys from environment variables
   - Key validation
   - Setup instructions generation

2. **TestGroqJudgeIntegration**: Groq API integration tests
   - Basic evaluation with Groq API
   - Hallucination detection
   - Bias detection
   - Invalid API key handling
   - Response time verification

3. **TestGeminiJudgeIntegration**: Gemini API integration tests
   - Basic evaluation with Gemini API
   - Hallucination detection
   - Bias detection
   - Invalid API key handling
   - Response time verification

4. **TestAPIJudgeEnsemble**: Ensemble coordination tests
   - Ensemble initialization with available keys
   - Parallel vs sequential execution
   - Verdict aggregation
   - Disagreement detection
   - Partial failure handling

5. **TestEndToEndEvaluation**: Complete evaluation flow tests
   - Full evaluation pipeline from keys to verdict
   - Multiple sequential evaluations

6. **TestDemoIntegration**: Demo functionality tests
   - Demo component initialization
   - Demo execution readiness

**Requirements for API Judge Tests:**
- Set `GROQ_API_KEY` environment variable for Groq tests
- Set `GEMINI_API_KEY` environment variable for Gemini tests
- Tests will skip if API keys are not available

**Running API Judge Tests:**
```bash
# Run all API judge tests (requires API keys)
pytest tests/integration/test_api_judges.py -v

# Run only tests that don't require API keys
pytest tests/integration/test_api_judges.py -v -m "not requires_api_keys"

# Check API key availability
pytest tests/integration/test_api_judges.py::test_api_availability_summary -v -s
```

### Full Pipeline Integration (`test_full_pipeline.py`)

Tests the complete evaluation pipeline with all components working together:

1. **TestFullPipelineIntegration**: End-to-end pipeline tests
   - Basic evaluation pipeline
   - Evaluation with retrieval enabled
   - Batch evaluation
   - Hallucination detection
   - Multiple judge models
   - Pipeline stage ordering

2. **TestPresetIntegration**: Preset configuration tests
   - Fast preset initialization and evaluation
   - Balanced preset initialization and evaluation
   - Preset comparison

3. **TestErrorHandlingIntegration**: Error handling tests
   - Empty input validation
   - Batch evaluation error resilience
   - Batch evaluation fail-fast mode
   - Invalid preset names
   - Model loading failures

4. **TestComponentIntegration**: Component interaction tests
   - Verifier to aggregation flow
   - Retrieval to verifier flow
   - Judge ensemble to aggregation flow
   - Statistics calculation

### Streaming Integration (`test_streaming_integration.py`)

Tests streaming evaluation functionality:

1. **TestStreamingIntegration**: Streaming evaluator tests
   - Streaming with fast preset
   - Multiple chunk processing
   - Evaluation quality preservation
   - Issue aggregation
   - Metadata completeness

## Running Integration Tests

### Run all integration tests:
```bash
python -m pytest tests/integration/ -v
```

### Run specific test file:
```bash
python -m pytest tests/integration/test_api_judges.py -v
python -m pytest tests/integration/test_full_pipeline.py -v
python -m pytest tests/integration/test_streaming_integration.py -v
```

### Run specific test class:
```bash
python -m pytest tests/integration/test_full_pipeline.py::TestFullPipelineIntegration -v
python -m pytest tests/integration/test_api_judges.py::TestGroqJudgeIntegration -v
```

### Run with coverage:
```bash
python -m pytest tests/integration/ --cov=llm_judge_auditor --cov-report=html
```

## Test Behavior

### API Judge Tests
- **Require API keys**: Tests will skip if `GROQ_API_KEY` or `GEMINI_API_KEY` are not set
- **Make real API calls**: Tests interact with actual Groq and Gemini APIs
- **Respect rate limits**: Tests include appropriate delays and retry logic
- **Validate responses**: Tests verify actual API responses and behavior

### Pipeline Tests
- **Skip without models**: Most tests will skip in environments without real models loaded
- **Expected behavior**: Allows test suite to run in CI/CD without large model downloads
- **Error handling tests**: Run normally without requiring models

## Requirements

- pytest
- hypothesis (for property-based tests)
- groq (for Groq API tests)
- google-generativeai (for Gemini API tests)
- All llm_judge_auditor dependencies

## Setting Up API Keys

To run API judge integration tests, you need free API keys:

1. **Get Groq API Key** (free):
   - Sign up: https://console.groq.com
   - Get key: https://console.groq.com/keys
   - Set: `export GROQ_API_KEY="your-key"`

2. **Get Gemini API Key** (free):
   - Sign up: https://aistudio.google.com
   - Get key: https://aistudio.google.com/app/apikey
   - Set: `export GEMINI_API_KEY="your-key"`

3. **Run tests**:
   ```bash
   pytest tests/integration/test_api_judges.py -v
   ```

## Notes

- Integration tests verify that components work together correctly
- Tests use realistic configurations (fast, balanced presets)
- Error handling is tested to ensure graceful degradation
- All tests follow the multi-stage pipeline architecture
- API judge tests validate real API integration and behavior
- Tests are designed to be run with actual API keys for comprehensive validation
