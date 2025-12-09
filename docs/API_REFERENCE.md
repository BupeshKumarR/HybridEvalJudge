# API Reference

Complete API reference for the LLM Judge Auditor toolkit.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Models](#data-models)
3. [Configuration](#configuration)
4. [Components](#components)
5. [Utilities](#utilities)

## Core Classes

### EvaluationToolkit

Main orchestrator class for evaluating LLM outputs.

```python
from llm_judge_auditor import EvaluationToolkit
```

#### Constructor

```python
EvaluationToolkit(config: ToolkitConfig)
```

**Parameters**:
- `config` (ToolkitConfig): Configuration object

**Example**:
```python
config = ToolkitConfig.from_preset("balanced")
toolkit = EvaluationToolkit(config)
```

#### Class Methods

##### `from_preset(preset_name: str) -> EvaluationToolkit`

Create toolkit from a preset configuration.

**Parameters**:
- `preset_name` (str): Name of preset ("fast", "balanced", "strict", "research")

**Returns**: EvaluationToolkit instance

**Example**:
```python
toolkit = EvaluationToolkit.from_preset("balanced")
```

#### Instance Methods

##### `evaluate(source_text: str, candidate_output: str) -> EvaluationResult`

Evaluate a single candidate output.

**Parameters**:
- `source_text` (str): Reference text containing facts
- `candidate_output` (str): Text to evaluate

**Returns**: EvaluationResult object

**Example**:
```python
result = toolkit.evaluate(
    source_text="Paris is the capital of France.",
    candidate_output="Paris is the capital of France."
)
```

##### `evaluate_request(request: EvaluationRequest) -> EvaluationResult`

Evaluate using a structured request object.

**Parameters**:
- `request` (EvaluationRequest): Evaluation request with all parameters

**Returns**: EvaluationResult object

**Example**:
```python
request = EvaluationRequest(
    source_text="...",
    candidate_output="...",
    task="factual_accuracy",
    criteria=["correctness", "completeness"]
)
result = toolkit.evaluate_request(request)
```

##### `batch_evaluate(requests: List[EvaluationRequest], continue_on_error: bool = True) -> BatchResult`

Evaluate multiple requests in batch.

**Parameters**:
- `requests` (List[EvaluationRequest]): List of evaluation requests
- `continue_on_error` (bool): Continue processing if a request fails

**Returns**: BatchResult object

**Example**:
```python
batch_result = toolkit.batch_evaluate(
    requests=[request1, request2, request3],
    continue_on_error=True
)
```

##### `get_stats() -> Dict[str, Any]`

Get toolkit statistics and configuration info.

**Returns**: Dictionary with statistics

**Example**:
```python
stats = toolkit.get_stats()
print(f"Number of judges: {stats['num_judges']}")
```

## Data Models

### EvaluationRequest

Request object for evaluation.

```python
from llm_judge_auditor import EvaluationRequest
```

**Attributes**:
- `source_text` (str): Reference text
- `candidate_output` (str): Text to evaluate
- `task` (str): Task type ("factual_accuracy", "bias_detection", "pairwise_ranking")
- `criteria` (List[str]): Evaluation criteria
- `use_retrieval` (bool): Enable retrieval augmentation

**Example**:
```python
request = EvaluationRequest(
    source_text="The Earth orbits the Sun.",
    candidate_output="The Sun orbits the Earth.",
    task="factual_accuracy",
    criteria=["correctness"],
    use_retrieval=False
)
```

### EvaluationResult

Result object from evaluation.

**Attributes**:
- `request` (EvaluationRequest): Original request
- `consensus_score` (float): Aggregated score (0-100)
- `verifier_verdicts` (List[Verdict]): Statement-level verdicts
- `judge_results` (List[JudgeResult]): Individual judge results
- `flagged_issues` (List[Issue]): Detected issues
- `report` (Report): Detailed report

**Methods**:
- `to_json(indent: int = None) -> str`: Export to JSON
- `save_to_file(path: str) -> None`: Save to file

**Example**:
```python
result = toolkit.evaluate(source_text, candidate_output)
print(f"Score: {result.consensus_score:.2f}")
result.save_to_file("result.json")
```

### BatchResult

Result object from batch evaluation.

**Attributes**:
- `results` (List[EvaluationResult]): Individual results
- `statistics` (Dict[str, float]): Batch statistics (mean, median, std, etc.)
- `metadata` (Dict[str, Any]): Batch metadata
- `errors` (List[Dict]): Errors that occurred

**Methods**:
- `to_json(indent: int = None) -> str`: Export to JSON
- `to_csv() -> str`: Export to CSV
- `save_to_file(path: str) -> None`: Save to file

**Example**:
```python
batch_result = toolkit.batch_evaluate(requests)
print(f"Mean score: {batch_result.statistics['mean']:.2f}")
batch_result.save_to_file("batch_results.json")
```

### Verdict

Statement-level verification result.

**Attributes**:
- `claim` (str): The claim being verified
- `label` (VerdictLabel): SUPPORTED, REFUTED, or NOT_ENOUGH_INFO
- `confidence` (float): Confidence score (0-1)
- `evidence` (List[str]): Supporting evidence
- `reasoning` (str): Explanation

**Example**:
```python
for verdict in result.verifier_verdicts:
    print(f"{verdict.label.value}: {verdict.claim}")
    print(f"Confidence: {verdict.confidence:.2f}")
```

### JudgeResult

Individual judge evaluation result.

**Attributes**:
- `model_name` (str): Judge model name
- `score` (float): Score (0-100)
- `reasoning` (str): Chain-of-thought explanation
- `flagged_issues` (List[Issue]): Issues detected by this judge
- `confidence` (float): Confidence in evaluation

**Example**:
```python
for judge_result in result.judge_results:
    print(f"{judge_result.model_name}: {judge_result.score:.2f}")
    print(f"Reasoning: {judge_result.reasoning}")
```

### Issue

Detected problem in candidate output.

**Attributes**:
- `type` (IssueType): hallucination, bias, inconsistency
- `severity` (IssueSeverity): low, medium, high
- `description` (str): Description of the issue
- `evidence` (List[str]): Supporting evidence

**Example**:
```python
for issue in result.flagged_issues:
    print(f"[{issue.severity.value}] {issue.type.value}")
    print(f"Description: {issue.description}")
```

### Claim

Extracted claim from text.

**Attributes**:
- `text` (str): Claim text
- `source_span` (Tuple[int, int]): Character offsets in original text
- `claim_type` (ClaimType): FACTUAL, TEMPORAL, NUMERICAL, etc.

**Example**:
```python
claim = Claim(
    text="Paris is the capital of France.",
    source_span=(0, 31),
    claim_type=ClaimType.FACTUAL
)
```

### Passage

Retrieved passage from knowledge base.

**Attributes**:
- `text` (str): Passage text
- `source` (str): Source identifier
- `relevance_score` (float): Relevance score (0-1)

**Example**:
```python
for passage in result.report.retrieval_provenance:
    print(f"Source: {passage.source}")
    print(f"Relevance: {passage.relevance_score:.2f}")
```

## Configuration

### ToolkitConfig

Main configuration class.

```python
from llm_judge_auditor import ToolkitConfig
```

**Attributes**:
- `verifier_model` (str): Verifier model name
- `judge_models` (List[str]): Judge model names
- `quantize` (bool): Enable 8-bit quantization
- `device` (str): Device ("cpu", "cuda", "mps", "auto")
- `enable_retrieval` (bool): Enable retrieval augmentation
- `knowledge_base_path` (Optional[str]): Path to knowledge base
- `retrieval_top_k` (int): Number of passages to retrieve
- `aggregation_strategy` (AggregationStrategy): Aggregation method
- `judge_weights` (Optional[Dict[str, float]]): Judge weights
- `disagreement_threshold` (float): Disagreement threshold
- `batch_size` (int): Batch size for inference
- `max_length` (int): Maximum sequence length

**Class Methods**:

##### `from_preset(name: str) -> ToolkitConfig`

Load a preset configuration.

**Parameters**:
- `name` (str): Preset name ("fast", "balanced", "strict", "research")

**Example**:
```python
config = ToolkitConfig.from_preset("balanced")
```

##### `from_yaml(path: str) -> ToolkitConfig`

Load configuration from YAML file.

**Parameters**:
- `path` (str): Path to YAML file

**Example**:
```python
config = ToolkitConfig.from_yaml("config/my_config.yaml")
```

**Instance Methods**:

##### `to_yaml(path: str) -> None`

Save configuration to YAML file.

**Example**:
```python
config.to_yaml("config/saved_config.yaml")
```

### AggregationStrategy

Enum for aggregation strategies.

**Values**:
- `MEAN`: Simple average
- `MEDIAN`: Median score
- `WEIGHTED_AVERAGE`: Weighted average using judge_weights
- `MAJORITY_VOTE`: Majority voting

**Example**:
```python
from llm_judge_auditor import AggregationStrategy

config.aggregation_strategy = AggregationStrategy.WEIGHTED_AVERAGE
config.judge_weights = {"judge1": 0.6, "judge2": 0.4}
```

### DeviceType

Enum for device types.

**Values**:
- `CPU`: CPU execution
- `CUDA`: NVIDIA GPU
- `MPS`: Apple Silicon GPU
- `AUTO`: Auto-detect best device

**Example**:
```python
from llm_judge_auditor import DeviceType

config.device = DeviceType.AUTO
```

## Components

### DeviceManager

Hardware detection and optimization.

```python
from llm_judge_auditor.components import DeviceManager
```

**Methods**:
- `detect_devices() -> List[Device]`: Detect available devices
- `select_optimal_device(model_size: int) -> Device`: Select best device
- `auto_configure(config: ToolkitConfig) -> ToolkitConfig`: Auto-configure

### ModelManager

Model loading and management.

```python
from llm_judge_auditor.components import ModelManager
```

**Methods**:
- `load_verifier(model_name: str, quantize: bool) -> VerifierModel`: Load verifier
- `load_judge_ensemble(model_names: List[str], quantize: bool) -> List[JudgeModel]`: Load judges
- `verify_models_ready() -> bool`: Check if models are ready

### RetrievalComponent

Retrieval-augmented verification.

```python
from llm_judge_auditor.components import RetrievalComponent
```

**Methods**:
- `initialize_knowledge_base(kb_path: str) -> None`: Initialize KB
- `extract_claims(text: str) -> List[Claim]`: Extract claims
- `retrieve_passages(claim: Claim, top_k: int) -> List[Passage]`: Retrieve passages

### SpecializedVerifier

Statement-level fact-checking.

```python
from llm_judge_auditor.components import SpecializedVerifier
```

**Methods**:
- `verify_statement(statement: str, context: str, passages: List[Passage]) -> Verdict`: Verify statement
- `batch_verify(statements: List[str], contexts: List[str]) -> List[Verdict]`: Batch verify

### JudgeEnsemble

Judge model ensemble.

```python
from llm_judge_auditor.components import JudgeEnsemble
```

**Methods**:
- `evaluate_single(judge: JudgeModel, prompt: str) -> JudgeResult`: Single judge evaluation
- `evaluate_all(prompt: str) -> List[JudgeResult]`: All judges evaluation
- `pairwise_compare(candidate_a: str, candidate_b: str, source: str) -> PairwiseResult`: Pairwise comparison

### AggregationEngine

Result aggregation.

```python
from llm_judge_auditor.components import AggregationEngine
```

**Methods**:
- `set_strategy(strategy: AggregationStrategy) -> None`: Set strategy
- `aggregate_scores(verifier_result: Verdict, judge_results: List[JudgeResult]) -> AggregatedScore`: Aggregate
- `detect_disagreement(results: List[JudgeResult]) -> DisagreementReport`: Detect disagreement

### ReportGenerator

Report generation.

```python
from llm_judge_auditor.components import ReportGenerator
```

**Methods**:
- `generate_report(evaluation: EvaluationResult) -> Report`: Generate report
- `export_json(report: Report, path: str, indent: Optional[int] = 2) -> None`: Export to JSON
- `export_csv(report: Report, path: str) -> None`: Export to CSV with detailed sections
- `export_markdown(report: Report, path: str) -> None`: Export to Markdown
- `export_text(report: Report, path: str) -> None`: Export to plain text
- `get_retrieval_provenance_summary(report: Report) -> Dict[str, Any]`: Get detailed retrieval provenance summary
- `get_hallucination_summary(report: Report) -> Dict[str, Any]`: Get detailed hallucination categorization summary

## Utilities

### Error Handling

```python
from llm_judge_auditor.utils import (
    EvaluationError,
    ModelLoadError,
    InferenceError,
    RetrievalError
)
```

**Exception Classes**:
- `EvaluationError`: Base exception
- `ModelLoadError`: Model loading failures
- `InferenceError`: Inference failures
- `RetrievalError`: Retrieval failures

**Example**:
```python
try:
    result = toolkit.evaluate(source_text, candidate_output)
except ModelLoadError as e:
    logger.error(f"Failed to load model: {e}")
except InferenceError as e:
    logger.error(f"Inference failed: {e}")
```

## Enums

### VerdictLabel

```python
from llm_judge_auditor import VerdictLabel

VerdictLabel.SUPPORTED
VerdictLabel.REFUTED
VerdictLabel.NOT_ENOUGH_INFO
```

### IssueType

```python
from llm_judge_auditor import IssueType

IssueType.HALLUCINATION
IssueType.BIAS
IssueType.INCONSISTENCY
```

### IssueSeverity

```python
from llm_judge_auditor import IssueSeverity

IssueSeverity.LOW
IssueSeverity.MEDIUM
IssueSeverity.HIGH
```

### ClaimType

```python
from llm_judge_auditor import ClaimType

ClaimType.FACTUAL
ClaimType.TEMPORAL
ClaimType.NUMERICAL
ClaimType.LOGICAL
ClaimType.COMMONSENSE
```

## Type Hints

The toolkit uses comprehensive type hints for better IDE support:

```python
from typing import List, Dict, Optional, Tuple
from llm_judge_auditor import (
    EvaluationToolkit,
    EvaluationRequest,
    EvaluationResult,
    ToolkitConfig
)

def evaluate_text(
    toolkit: EvaluationToolkit,
    source: str,
    candidate: str
) -> EvaluationResult:
    return toolkit.evaluate(source, candidate)
```

## Additional Resources

- [Usage Guide](USAGE_GUIDE.md)
- [CLI Reference](CLI_USAGE.md)
- [Configuration Guide](../config/README.md)
- [Examples](../examples/)


## PluginRegistry

Registry for managing custom plugins (verifiers, judges, aggregators).

```python
from llm_judge_auditor.components import PluginRegistry
```

### Constructor

```python
PluginRegistry(plugins_dir: Optional[str] = None)
```

**Parameters**:
- `plugins_dir` (Optional[str]): Path to plugins directory for auto-discovery

**Example**:
```python
# Create registry with auto-discovery
registry = PluginRegistry(plugins_dir="plugins")

# Or create empty registry
registry = PluginRegistry()
```

### Methods

#### `register_verifier(name, loader, version="1.0.0", description="", author="", compatible_versions=None)`

Register a custom verifier plugin.

**Parameters**:
- `name` (str): Unique name for the verifier
- `loader` (Callable): Function that returns a verifier instance
- `version` (str): Version string (default: "1.0.0")
- `description` (str): Optional description
- `author` (str): Optional author information
- `compatible_versions` (Optional[List[str]]): Compatible toolkit versions

**Example**:
```python
def load_my_verifier():
    return MyCustomVerifier()

registry.register_verifier("my_verifier", load_my_verifier, version="1.0.0")
```

#### `register_judge(name, loader, version="1.0.0", description="", author="", compatible_versions=None)`

Register a custom judge plugin.

**Parameters**:
- `name` (str): Unique name for the judge
- `loader` (Callable): Function that returns a judge instance
- `version` (str): Version string (default: "1.0.0")
- `description` (str): Optional description
- `author` (str): Optional author information
- `compatible_versions` (Optional[List[str]]): Compatible toolkit versions

**Example**:
```python
def load_my_judge():
    return MyCustomJudge()

registry.register_judge("my_judge", load_my_judge, version="1.0.0")
```

#### `register_aggregator(name, aggregator, version="1.0.0", description="", author="", compatible_versions=None)`

Register a custom aggregation strategy.

**Parameters**:
- `name` (str): Unique name for the aggregator
- `aggregator` (Callable[[List[float]], float]): Function that aggregates scores
- `version` (str): Version string (default: "1.0.0")
- `description` (str): Optional description
- `author` (str): Optional author information
- `compatible_versions` (Optional[List[str]]): Compatible toolkit versions

**Example**:
```python
def harmonic_mean(scores):
    return len(scores) / sum(1/s for s in scores if s > 0)

registry.register_aggregator("harmonic_mean", harmonic_mean)
```

#### `get_verifier(name: str) -> Any`

Get a verifier plugin instance.

**Parameters**:
- `name` (str): Name of the verifier

**Returns**: Verifier instance

**Raises**: KeyError if verifier not registered

#### `get_judge(name: str) -> Any`

Get a judge plugin instance.

**Parameters**:
- `name` (str): Name of the judge

**Returns**: Judge instance

**Raises**: KeyError if judge not registered

#### `get_aggregator(name: str) -> Callable[[List[float]], float]`

Get an aggregator plugin function.

**Parameters**:
- `name` (str): Name of the aggregator

**Returns**: Aggregator function

**Raises**: KeyError if aggregator not registered

#### `list_plugins() -> Dict[str, List[str]]`

List all registered plugins by type.

**Returns**: Dictionary mapping plugin types to lists of plugin names

**Example**:
```python
plugins = registry.list_plugins()
print(plugins)
# {'verifiers': ['custom_verifier'], 'judges': ['custom_judge'], 'aggregators': ['harmonic_mean']}
```

#### `get_plugin_info(name: str) -> Optional[PluginMetadata]`

Get metadata for a specific plugin.

**Parameters**:
- `name` (str): Name of the plugin

**Returns**: PluginMetadata if found, None otherwise

#### `discover_plugins(plugins_dir: str) -> Dict[str, int]`

Discover and load plugins from a directory.

**Parameters**:
- `plugins_dir` (str): Path to the plugins directory

**Returns**: Dictionary with counts of discovered plugins by type

**Example**:
```python
discovered = registry.discover_plugins("plugins")
print(f"Discovered {discovered['verifiers']} verifiers")
```

#### `check_compatibility(plugin_name: str, toolkit_version: str) -> bool`

Check if a plugin is compatible with the current toolkit version.

**Parameters**:
- `plugin_name` (str): Name of the plugin
- `toolkit_version` (str): Current toolkit version string

**Returns**: True if compatible, False otherwise

#### `unregister_verifier(name: str)`

Unregister a verifier plugin.

#### `unregister_judge(name: str)`

Unregister a judge plugin.

#### `unregister_aggregator(name: str)`

Unregister an aggregator plugin.

#### `clear_all()`

Clear all registered plugins.

### Plugin Interfaces

#### Verifier Protocol

Custom verifiers should implement:

```python
class CustomVerifier:
    def verify_statement(
        self, 
        statement: str, 
        context: str, 
        passages: Optional[List[Any]] = None
    ) -> Verdict:
        """Verify a single statement."""
        pass
    
    def batch_verify(
        self,
        statements: List[str],
        contexts: List[str],
        passages_list: Optional[List[List[Any]]] = None
    ) -> List[Verdict]:
        """Verify multiple statements."""
        pass
```

#### Judge Protocol

Custom judges should implement:

```python
class CustomJudge:
    def evaluate(
        self, 
        source_text: str, 
        candidate_output: str, 
        retrieved_context: str = ""
    ) -> JudgeResult:
        """Evaluate a candidate output."""
        pass
```

#### Aggregator Protocol

Custom aggregators should be callable functions:

```python
def custom_aggregator(scores: List[float]) -> float:
    """Aggregate multiple scores into a single consensus score."""
    pass
```

### Plugin Discovery

Plugins are automatically discovered from Python modules in the plugins directory. Each plugin module should define a `register_plugin(registry)` function:

```python
# plugins/my_plugin.py

def register_plugin(registry):
    """Register custom components."""
    
    def load_my_verifier():
        return MyVerifier()
    
    registry.register_verifier("my_verifier", load_my_verifier)
```

See `plugins/README.md` and `examples/plugin_system_example.py` for more details.

