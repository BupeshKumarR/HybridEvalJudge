# API Judge Ensemble

The API Judge Ensemble component coordinates multiple API-based judge clients (Groq, Gemini) to evaluate LLM outputs comprehensively.

## Overview

The `APIJudgeEnsemble` class manages multiple API judge clients and provides:
- Automatic initialization based on available API keys
- Parallel or sequential execution of judges
- Graceful handling of partial failures
- Score aggregation and disagreement detection

## Features

### 1. Flexible Initialization

The ensemble automatically initializes judges based on available API keys:

```python
from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.config import ToolkitConfig

# Load API keys
api_key_manager = APIKeyManager()
api_key_manager.load_keys()

# Initialize ensemble
config = ToolkitConfig()
ensemble = APIJudgeEnsemble(
    config=config,
    api_key_manager=api_key_manager,
    parallel_execution=True  # Enable parallel execution
)

print(f"Initialized {ensemble.get_judge_count()} judges")
```

### 2. Parallel Execution

Judges are called in parallel by default for faster evaluation:

```python
verdicts = ensemble.evaluate(
    source_text="The Eiffel Tower is in Paris.",
    candidate_output="The Eiffel Tower is in London.",
    task="factual_accuracy"
)

# Evaluation completes in ~2-3 seconds instead of 4-6 seconds
```

### 3. Graceful Failure Handling

If one judge fails, the ensemble continues with remaining judges:

```python
# Even if Gemini fails, Groq results are still returned
verdicts = ensemble.evaluate(source_text, candidate_output)

# Returns verdicts from successful judges only
print(f"Received {len(verdicts)} verdicts")
```

### 4. Score Aggregation

Aggregate scores from multiple judges:

```python
consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts)

print(f"Consensus Score: {consensus:.1f}/100")
print(f"Individual Scores: {individual}")
print(f"Disagreement Level: {disagreement:.2f}")
```

### 5. Disagreement Detection

Identify when judges significantly disagree:

```python
disagreement_analysis = ensemble.identify_disagreements(verdicts, threshold=20.0)

if disagreement_analysis["has_disagreement"]:
    print("⚠️  Judges disagree!")
    print(f"Score range: {disagreement_analysis['score_range']}")
    print(f"Outliers: {disagreement_analysis['outliers']}")
```

## API Reference

### APIJudgeEnsemble

#### Constructor

```python
APIJudgeEnsemble(
    config: ToolkitConfig,
    api_key_manager: APIKeyManager,
    parallel_execution: bool = True
)
```

**Parameters:**
- `config`: Toolkit configuration
- `api_key_manager`: Manager for API keys
- `parallel_execution`: Whether to execute judges in parallel (default: True)

#### Methods

##### evaluate()

```python
evaluate(
    source_text: str,
    candidate_output: str,
    task: str = "factual_accuracy"
) -> List[JudgeVerdict]
```

Evaluate candidate output using all available judges.

**Parameters:**
- `source_text`: Reference document or context
- `candidate_output`: Text to be evaluated
- `task`: Evaluation task type (e.g., "factual_accuracy", "bias_detection")

**Returns:**
- List of `JudgeVerdict` objects from successful judges

**Raises:**
- `RuntimeError`: If no judges are available or all judges fail

##### aggregate_verdicts()

```python
aggregate_verdicts(
    verdicts: List[JudgeVerdict]
) -> Tuple[float, Dict[str, float], float]
```

Aggregate verdicts from multiple judges.

**Parameters:**
- `verdicts`: List of JudgeVerdict objects

**Returns:**
- Tuple of (consensus_score, individual_scores_dict, disagreement_level)

##### identify_disagreements()

```python
identify_disagreements(
    verdicts: List[JudgeVerdict],
    threshold: float = 20.0
) -> Dict[str, any]
```

Identify and analyze disagreements between judges.

**Parameters:**
- `verdicts`: List of JudgeVerdict objects
- `threshold`: Variance threshold for flagging disagreement (default: 20.0)

**Returns:**
- Dictionary with disagreement analysis:
  - `has_disagreement`: bool
  - `variance`: float
  - `score_range`: tuple (min, max)
  - `outliers`: list of judge names with outlier scores
  - `reasoning_summary`: dict of judge_name -> reasoning

##### get_judge_count()

```python
get_judge_count() -> int
```

Get the number of active judges in the ensemble.

**Returns:**
- Number of initialized judges

##### get_judge_names()

```python
get_judge_names() -> List[str]
```

Get the names of all active judges.

**Returns:**
- List of judge names

## Usage Examples

### Basic Usage

```python
from llm_judge_auditor.components.api_judge_ensemble import APIJudgeEnsemble
from llm_judge_auditor.components.api_key_manager import APIKeyManager
from llm_judge_auditor.config import ToolkitConfig

# Setup
api_key_manager = APIKeyManager()
api_key_manager.load_keys()

config = ToolkitConfig()
ensemble = APIJudgeEnsemble(config, api_key_manager)

# Evaluate
verdicts = ensemble.evaluate(
    source_text="The capital of France is Paris.",
    candidate_output="The capital of France is London.",
    task="factual_accuracy"
)

# Aggregate
consensus, individual, disagreement = ensemble.aggregate_verdicts(verdicts)
print(f"Consensus: {consensus:.1f}/100")
```

### Handling Disagreements

```python
# Evaluate
verdicts = ensemble.evaluate(source_text, candidate_output)

# Check for disagreements
disagreement = ensemble.identify_disagreements(verdicts, threshold=20.0)

if disagreement["has_disagreement"]:
    print("⚠️  Judges disagree significantly!")
    print(f"Variance: {disagreement['variance']:.2f}")
    print(f"Score range: {disagreement['score_range']}")
    
    # Show reasoning from each judge
    for judge_name, reasoning in disagreement["reasoning_summary"].items():
        print(f"\n{judge_name}:")
        print(reasoning)
```

### Sequential Execution

```python
# Use sequential execution for debugging
ensemble = APIJudgeEnsemble(
    config=config,
    api_key_manager=api_key_manager,
    parallel_execution=False
)

verdicts = ensemble.evaluate(source_text, candidate_output)
```

## Error Handling

The ensemble handles various error scenarios gracefully:

### No API Keys Available

```python
# If no API keys are configured
try:
    verdicts = ensemble.evaluate(source_text, candidate_output)
except RuntimeError as e:
    print(f"Error: {e}")
    # "No judges available for evaluation. Please configure at least one API key."
```

### Partial Judge Failure

```python
# If one judge fails, others continue
verdicts = ensemble.evaluate(source_text, candidate_output)

# Returns verdicts from successful judges only
if len(verdicts) < ensemble.get_judge_count():
    print(f"⚠️  Only {len(verdicts)}/{ensemble.get_judge_count()} judges succeeded")
```

### All Judges Fail

```python
# If all judges fail
try:
    verdicts = ensemble.evaluate(source_text, candidate_output)
except RuntimeError as e:
    print(f"Error: {e}")
    # "All judges failed during evaluation. Please check API keys and network connectivity."
```

## Performance Considerations

### Parallel vs Sequential Execution

- **Parallel** (default): ~2-3 seconds for 2 judges
- **Sequential**: ~4-6 seconds for 2 judges

Parallel execution is recommended for production use.

### Rate Limits

Both APIs have free tier rate limits:
- **Groq**: 30 requests/minute
- **Gemini**: 15 requests/minute

The ensemble respects these limits through the individual judge clients' retry logic.

## Integration with Existing Components

The API Judge Ensemble integrates with:

- **APIKeyManager**: Loads and validates API keys
- **GroqJudgeClient**: Handles Groq API calls
- **GeminiJudgeClient**: Handles Gemini API calls
- **AggregationEngine**: Can be used for more advanced aggregation strategies

## Testing

Comprehensive unit tests are available in `tests/unit/test_api_judge_ensemble.py`:

```bash
pytest tests/unit/test_api_judge_ensemble.py -v
```

Tests cover:
- Initialization with different API key configurations
- Parallel and sequential evaluation
- Partial failure handling
- Score aggregation
- Disagreement detection

## See Also

- [API Key Manager](API_KEY_MANAGER.md)
- [Groq Judge Client](GROQ_JUDGE_CLIENT.md)
- [Gemini Judge Client](GEMINI_JUDGE_CLIENT.md)
- [Aggregation Engine](AGGREGATION_ENGINE.md)
