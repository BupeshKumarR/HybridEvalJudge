# Design Document: Free API Judge Integration

## Overview

This design implements a judge ensemble using free API-based models (Groq Llama 3.1 70B and Google Gemini Flash) to replace the current HuggingFace model dependencies. The system will call these APIs to evaluate LLM outputs, aggregate their scores, and provide comprehensive evaluation reports.

## Architecture

### High-Level Flow

```
User Input (LLM Response to Evaluate)
    ↓
Evaluation Toolkit
    ↓
API Judge Ensemble
    ├─→ Groq Llama 3.1 70B Judge
    │   ├─ Format prompt
    │   ├─ Call Groq API
    │   └─ Parse response → Score
    │
    └─→ Google Gemini Flash Judge
        ├─ Format prompt
        ├─ Call Gemini API
        └─ Parse response → Score
    ↓
Aggregation Engine
    ├─ Combine scores
    ├─ Calculate consensus
    └─ Identify disagreements
    ↓
Report Generator
    └─ Final evaluation report with scores
```

### Component Integration

```
┌─────────────────────────────────────────┐
│      Evaluation Toolkit                 │
│  (Main Orchestrator)                    │
└──────────────┬──────────────────────────┘
               │
               ├─→ API Key Manager
               │   └─ Validates & stores API keys
               │
               ├─→ API Judge Ensemble
               │   ├─ Groq Judge Client
               │   │  └─ groq Python SDK
               │   │
               │   └─ Gemini Judge Client
               │      └─ google-generativeai SDK
               │
               ├─→ Prompt Manager
               │   └─ Formats prompts for each API
               │
               ├─→ Aggregation Engine
               │   └─ Combines judge scores
               │
               └─→ Report Generator
                   └─ Creates final report
```

## Components and Interfaces

### 1. API Key Manager

**Purpose:** Manage and validate API keys for Groq and Gemini

**Interface:**
```python
class APIKeyManager:
    def __init__(self):
        self.groq_key: Optional[str] = None
        self.gemini_key: Optional[str] = None
    
    def load_keys(self) -> Dict[str, bool]:
        """Load API keys from environment variables or config"""
        # Check GROQ_API_KEY environment variable
        # Check GEMINI_API_KEY environment variable
        # Return dict of which keys are available
    
    def validate_key(self, service: str, key: str) -> bool:
        """Validate an API key by making a test call"""
    
    def get_setup_instructions(self) -> str:
        """Return instructions for obtaining free API keys"""
```

**Implementation Details:**
- Check environment variables: `GROQ_API_KEY`, `GEMINI_API_KEY`
- Fall back to config file if env vars not set
- Validate keys on first use with lightweight test call
- Cache validation results to avoid repeated checks

### 2. Groq Judge Client

**Purpose:** Interface with Groq API for LLM evaluation

**Interface:**
```python
class GroqJudgeClient:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def evaluate(
        self, 
        source_text: str, 
        candidate_output: str,
        task: str
    ) -> JudgeVerdict:
        """
        Evaluate candidate output against source text
        
        Returns:
            JudgeVerdict with score, reasoning, and issues
        """
    
    def _format_prompt(self, source_text: str, candidate_output: str, task: str) -> str:
        """Format evaluation prompt for Groq"""
    
    def _parse_response(self, response: str) -> JudgeVerdict:
        """Parse Groq response into structured verdict"""
```

**API Details:**
- **Endpoint:** Groq Chat Completions API
- **Model:** `llama-3.1-70b-versatile` (free tier: 30 requests/minute)
- **SDK:** `groq` Python package
- **Response Format:** JSON with score, reasoning, issues

### 3. Gemini Judge Client

**Purpose:** Interface with Google Gemini API for LLM evaluation

**Interface:**
```python
class GeminiJudgeClient:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.client = genai.GenerativeModel(model)
        genai.configure(api_key=api_key)
        self.model = model
    
    def evaluate(
        self, 
        source_text: str, 
        candidate_output: str,
        task: str
    ) -> JudgeVerdict:
        """
        Evaluate candidate output against source text
        
        Returns:
            JudgeVerdict with score, reasoning, and issues
        """
    
    def _format_prompt(self, source_text: str, candidate_output: str, task: str) -> str:
        """Format evaluation prompt for Gemini"""
    
    def _parse_response(self, response: str) -> JudgeVerdict:
        """Parse Gemini response into structured verdict"""
```

**API Details:**
- **Endpoint:** Google Generative AI API
- **Model:** `gemini-1.5-flash` (free tier: 15 requests/minute)
- **SDK:** `google-generativeai` Python package
- **Response Format:** JSON with score, reasoning, issues

### 4. API Judge Ensemble

**Purpose:** Coordinate multiple API judges and aggregate results

**Interface:**
```python
class APIJudgeEnsemble:
    def __init__(self, config: ToolkitConfig, api_key_manager: APIKeyManager):
        self.judges: List[BaseJudgeClient] = []
        self._initialize_judges(api_key_manager)
    
    def _initialize_judges(self, api_key_manager: APIKeyManager):
        """Initialize available judges based on API keys"""
        if api_key_manager.groq_key:
            self.judges.append(GroqJudgeClient(api_key_manager.groq_key))
        if api_key_manager.gemini_key:
            self.judges.append(GeminiJudgeClient(api_key_manager.gemini_key))
    
    def evaluate(
        self,
        source_text: str,
        candidate_output: str,
        task: str
    ) -> List[JudgeVerdict]:
        """
        Evaluate with all available judges in parallel
        
        Returns:
            List of verdicts from each judge
        """
    
    def get_judge_count(self) -> int:
        """Return number of active judges"""
```

**Implementation Details:**
- Initialize judges based on available API keys
- Call judges in parallel using `asyncio` or `concurrent.futures`
- Handle individual judge failures gracefully
- Return all successful verdicts for aggregation

## Data Models

### JudgeVerdict

```python
@dataclass
class JudgeVerdict:
    """Verdict from a single judge"""
    judge_name: str  # "groq-llama-3.1-70b" or "gemini-flash"
    score: float  # 0-100
    confidence: float  # 0-1
    reasoning: str  # Judge's explanation
    issues: List[Issue]  # Identified problems
    metadata: Dict[str, Any]  # API response time, tokens used, etc.
```

### APIConfig

```python
@dataclass
class APIConfig:
    """Configuration for API judges"""
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_model: str = "llama-3.1-70b-versatile"
    gemini_model: str = "gemini-1.5-flash"
    timeout: int = 30  # seconds
    max_retries: int = 2
    parallel_calls: bool = True
```

## Prompt Templates

### Evaluation Prompt Structure

```python
EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator assessing the quality of an AI-generated response.

**Task:** {task}

**Source/Reference Text:**
{source_text}

**AI-Generated Response to Evaluate:**
{candidate_output}

**Your Task:**
1. Compare the AI response against the source text
2. Identify any inaccuracies, hallucinations, or missing information
3. Assign a score from 0-100 (100 = perfect, 0 = completely wrong)
4. Provide reasoning for your score
5. List specific issues found

**Response Format (JSON):**
{{
    "score": <0-100>,
    "confidence": <0.0-1.0>,
    "reasoning": "<your explanation>",
    "issues": [
        {{
            "severity": "<low|medium|high|critical>",
            "description": "<what's wrong>",
            "location": "<where in the response>"
        }}
    ]
}}
"""
```

### Task-Specific Prompts

Different prompts for different evaluation tasks:
- **factual_accuracy**: Focus on correctness vs source
- **hallucination_detection**: Focus on unsupported claims
- **bias_detection**: Focus on unfair or discriminatory content
- **completeness**: Focus on missing information

## Error Handling

### API Error Scenarios

1. **Missing API Keys**
   - Detection: Check environment variables on init
   - Response: Display setup guide with links
   - Fallback: Continue with available judges

2. **Invalid API Keys**
   - Detection: 401/403 errors from API
   - Response: Clear error message indicating which key is invalid
   - Fallback: Continue with valid judges

3. **Rate Limiting**
   - Detection: 429 errors from API
   - Response: Wait according to `Retry-After` header
   - Fallback: Exponential backoff (1s, 2s, 4s)

4. **Network Errors**
   - Detection: Connection timeouts, DNS failures
   - Response: Retry once after 2 seconds
   - Fallback: Continue with other judges

5. **Malformed Responses**
   - Detection: JSON parse errors
   - Response: Log warning, attempt to extract score from text
   - Fallback: Assign neutral score with low confidence

### Retry Strategy

```python
def call_api_with_retry(
    api_call: Callable,
    max_retries: int = 2,
    base_delay: float = 1.0
) -> Any:
    """
    Call API with exponential backoff retry
    
    Retry on:
    - Network errors
    - 5xx server errors
    - Rate limit errors (with longer delay)
    
    Don't retry on:
    - 4xx client errors (except 429)
    - Invalid API keys
    """
    for attempt in range(max_retries + 1):
        try:
            return api_call()
        except RateLimitError as e:
            if attempt < max_retries:
                delay = e.retry_after or (base_delay * (2 ** attempt))
                time.sleep(delay)
            else:
                raise
        except (NetworkError, ServerError) as e:
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise
        except ClientError:
            # Don't retry client errors
            raise
```

## Testing Strategy

### Unit Tests

1. **API Key Manager Tests**
   - Test loading keys from environment
   - Test validation logic
   - Test setup instructions generation

2. **Judge Client Tests**
   - Test prompt formatting
   - Test response parsing
   - Test error handling
   - Mock API responses

3. **Ensemble Tests**
   - Test parallel execution
   - Test partial failure handling
   - Test aggregation logic

### Integration Tests

1. **Real API Tests** (with test keys)
   - Test Groq API integration
   - Test Gemini API integration
   - Test end-to-end evaluation flow

2. **Demo Integration Tests**
   - Test demo with API judges
   - Test setup guide display
   - Test evaluation output

### Property-Based Tests

*Property 1: Score consistency*
**For any** valid source text and candidate output, when evaluated by multiple judges, the consensus score should be within a reasonable range (not wildly different)
**Validates: Requirements 1.5**

*Property 2: API key validation*
**For any** API key string, validation should either succeed or fail with a clear error message
**Validates: Requirements 3.6**

*Property 3: Parallel execution*
**For any** set of judges, parallel execution should produce the same results as sequential execution
**Validates: Requirements 5.7**

## Performance Considerations

### Latency Optimization

1. **Parallel API Calls**
   - Call both judges simultaneously
   - Use `asyncio` for non-blocking I/O
   - Expected latency: ~2-5 seconds (vs 4-10 sequential)

2. **Response Caching**
   - Cache API responses for identical inputs
   - Use content hash as cache key
   - Configurable TTL (default: 1 hour)

3. **Timeout Management**
   - Set reasonable timeouts (30s default)
   - Fail fast on slow APIs
   - Continue with faster judges

### Cost Management

Both APIs are free but have rate limits:

- **Groq**: 30 requests/minute (free tier)
- **Gemini**: 15 requests/minute (free tier)

**Rate Limit Handling:**
- Track request counts per minute
- Implement token bucket algorithm
- Queue requests when approaching limits
- Display warning when limits are hit

## Integration with Existing System

### Changes to EvaluationToolkit

```python
class EvaluationToolkit:
    def __init__(self, config: ToolkitConfig, enable_profiling: bool = False):
        # ... existing code ...
        
        # NEW: Initialize API key manager
        self.api_key_manager = APIKeyManager()
        self.api_key_manager.load_keys()
        
        # NEW: Initialize API judge ensemble
        if self.api_key_manager.has_any_keys():
            self.judge_ensemble = APIJudgeEnsemble(
                config=config,
                api_key_manager=self.api_key_manager
            )
        else:
            # Show setup guide
            print(self.api_key_manager.get_setup_instructions())
            raise RuntimeError("No API keys configured")
        
        # MODIFIED: Make verifier optional
        try:
            self.verifier = SpecializedVerifier(...)
        except Exception as e:
            logger.warning(f"Verifier unavailable: {e}")
            self.verifier = None
```

### Backward Compatibility

- Existing HuggingFace model support remains
- API judges are preferred when keys are available
- Config option to force HuggingFace models: `use_api_judges=False`

## Setup Guide for Users

### Quick Start

```bash
# 1. Get free API keys
# Groq: https://console.groq.com/keys
# Gemini: https://aistudio.google.com/app/apikey

# 2. Set environment variables
export GROQ_API_KEY="your-groq-key-here"
export GEMINI_API_KEY="your-gemini-key-here"

# 3. Install required packages
pip install groq google-generativeai

# 4. Run demo
python demo/demo.py
```

### Detailed Setup

The system will display this guide when API keys are missing:

```
╔══════════════════════════════════════════════════════════════╗
║  🔑 API Key Setup Required                                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  This demo uses FREE API-based judges for evaluation:       ║
║                                                              ║
║  1️⃣  Groq Llama 3.1 70B (FREE)                              ║
║     • Sign up: https://console.groq.com                     ║
║     • Get API key: https://console.groq.com/keys            ║
║     • Set: export GROQ_API_KEY="your-key"                   ║
║                                                              ║
║  2️⃣  Google Gemini Flash (FREE)                             ║
║     • Sign up: https://aistudio.google.com                  ║
║     • Get API key: https://aistudio.google.com/app/apikey   ║
║     • Set: export GEMINI_API_KEY="your-key"                 ║
║                                                              ║
║  💡 Both APIs are completely FREE!                          ║
║  💡 You need at least ONE key to run the demo               ║
║  💡 Using BOTH keys gives better evaluation accuracy        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
```

## Dependencies

### New Python Packages

```python
# requirements.txt additions
groq>=0.4.0  # Groq API client
google-generativeai>=0.3.0  # Gemini API client
```

### Optional Dependencies

```python
# For async parallel calls
aiohttp>=3.9.0
```

## Migration Path

### Phase 1: Add API Judge Support (This Spec)
- Implement API judge clients
- Add to ensemble alongside existing judges
- Make verifier optional

### Phase 2: Update Demo
- Add API key setup guide
- Update demo to use API judges
- Add troubleshooting documentation

### Phase 3: Deprecate HuggingFace Dependency
- Make HuggingFace models fully optional
- Update all examples to use API judges
- Document migration for existing users

## Success Metrics

1. **Functionality**
   - ✅ Demo runs successfully with API keys
   - ✅ Evaluation produces accurate scores
   - ✅ Both judges contribute to ensemble

2. **Performance**
   - ✅ Evaluation completes in <10 seconds
   - ✅ Parallel calls reduce latency by 50%
   - ✅ Rate limits are respected

3. **User Experience**
   - ✅ Setup guide is clear and helpful
   - ✅ Error messages are actionable
   - ✅ API key validation works correctly

4. **Reliability**
   - ✅ System handles API failures gracefully
   - ✅ Retries work as expected
   - ✅ Partial failures don't crash the system
