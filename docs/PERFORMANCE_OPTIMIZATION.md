## Performance Optimization

This document describes the performance optimization features available in the LLM Judge Auditor toolkit.

### Overview

The toolkit includes several performance optimization features to improve evaluation speed and efficiency:

1. **Profiling** - Identify bottlenecks in the evaluation pipeline
2. **Model Loading Optimization** - Efficient model caching and lazy loading
3. **Parallel Judge Evaluation** - Concurrent evaluation with multiple judges
4. **Batch Processing Optimization** - Efficient processing of multiple requests

### Profiling

Profiling helps identify performance bottlenecks in your evaluation pipeline.

#### Enabling Profiling

```python
from llm_judge_auditor.config import ToolkitConfig
from llm_judge_auditor.evaluation_toolkit import EvaluationToolkit

# Create toolkit with profiling enabled
config = ToolkitConfig.from_preset("balanced")
toolkit = EvaluationToolkit(config, enable_profiling=True)

# Run evaluations
result = toolkit.evaluate(
    source_text="Paris is the capital of France.",
    candidate_output="Paris is in France."
)

# Get profiling summary
print(toolkit.get_profiling_summary())
```

#### Profiling Output

The profiling summary shows timing information for each stage of the pipeline:

```
Profiling Summary
================================================================================

retrieval_stage:
  Calls: 5
  Total: 2.450s
  Average: 0.490s
  Min: 0.450s
  Max: 0.550s

verification_stage:
  Calls: 5
  Total: 3.200s
  Average: 0.640s
  Min: 0.600s
  Max: 0.700s

judge_ensemble_stage:
  Calls: 5
  Total: 8.500s
  Average: 1.700s
  Min: 1.650s
  Max: 1.750s

aggregation_stage:
  Calls: 5
  Total: 0.050s
  Average: 0.010s
  Min: 0.008s
  Max: 0.012s
```

#### Identifying Bottlenecks

```python
# Get top 5 bottlenecks by total time
bottlenecks = toolkit.get_profiling_bottlenecks(5)
for name, total_time in bottlenecks:
    print(f"{name}: {total_time:.3f}s")
```

Output:
```
judge_ensemble_stage: 8.500s
verification_stage: 3.200s
retrieval_stage: 2.450s
aggregation_stage: 0.050s
```

#### Resetting Profiling Data

```python
# Reset profiling data to start fresh
toolkit.reset_profiling()
```

### Model Loading Optimization

The toolkit optimizes model loading through caching and lazy loading.

#### Model Path Caching

Model paths are cached in memory to avoid redundant lookups:

```python
# First load: downloads and caches model path
toolkit1 = EvaluationToolkit(config)

# Second load: uses cached path (faster)
toolkit2 = EvaluationToolkit(config)
```

#### Lazy Loading

Models are loaded on first use rather than at initialization:

```python
# Models are not loaded until first evaluation
toolkit = EvaluationToolkit(config)

# Models loaded here on first evaluation
result = toolkit.evaluate(source, candidate)

# Subsequent evaluations use cached models
result2 = toolkit.evaluate(source2, candidate2)
```

### Parallel Judge Evaluation

When using multiple judge models, parallel evaluation can significantly speed up processing.

#### Sequential vs Parallel

```python
# Sequential evaluation (default)
# Judges run one after another
result = toolkit.evaluate(
    source_text=source,
    candidate_output=candidate,
    parallel_judges=False  # Default
)

# Parallel evaluation
# Judges run concurrently
result = toolkit.evaluate(
    source_text=source,
    candidate_output=candidate,
    parallel_judges=True
)
```

#### Performance Comparison

With 3 judge models, parallel evaluation typically provides 2-3x speedup:

```
Sequential: 6.5s (2.2s per judge × 3 judges)
Parallel:   2.5s (judges run concurrently)
Speedup:    2.6x faster
```

#### When to Use Parallel Evaluation

**Use parallel evaluation when:**
- You have multiple judge models (2+)
- You have multiple CPU cores or GPUs
- Latency is more important than resource usage

**Use sequential evaluation when:**
- You have limited memory
- You want to minimize resource usage
- You're running on a single-core system

#### Controlling Parallelism

```python
# Control maximum number of parallel workers
from llm_judge_auditor.components.judge_ensemble import JudgeEnsemble

ensemble = JudgeEnsemble(model_manager, prompt_manager)

# Limit to 2 concurrent judges
results = ensemble.evaluate_all(
    source_text=source,
    candidate_output=candidate,
    parallel=True,
    max_workers=2
)
```

### Batch Processing Optimization

Batch processing can be optimized by combining with parallel judge evaluation.

#### Basic Batch Processing

```python
from llm_judge_auditor.models import EvaluationRequest

# Create batch requests
requests = [
    EvaluationRequest(
        source_text="Paris is the capital of France.",
        candidate_output="Paris is in France."
    ),
    EvaluationRequest(
        source_text="The Earth orbits the Sun.",
        candidate_output="The Sun orbits the Earth."
    ),
    # ... more requests
]

# Process batch
batch_result = toolkit.batch_evaluate(requests)
```

#### Optimized Batch Processing

```python
# Use parallel judges for each request in the batch
batch_result = toolkit.batch_evaluate(
    requests,
    parallel_judges=True  # Faster processing
)

print(f"Processed: {len(batch_result.results)}/{len(requests)}")
print(f"Mean score: {batch_result.statistics['mean']:.2f}")
```

#### Performance Impact

For a batch of 10 requests with 3 judges each:

```
Sequential judges:  65s (6.5s per request)
Parallel judges:    25s (2.5s per request)
Speedup:            2.6x faster
```

### Best Practices

#### 1. Profile First

Always profile your evaluation pipeline to identify actual bottlenecks:

```python
toolkit = EvaluationToolkit(config, enable_profiling=True)

# Run representative evaluations
for request in sample_requests:
    toolkit.evaluate(request.source_text, request.candidate_output)

# Analyze bottlenecks
bottlenecks = toolkit.get_profiling_bottlenecks(5)
print(toolkit.get_profiling_summary())
```

#### 2. Use Parallel Evaluation Appropriately

Enable parallel evaluation when you have multiple judges and sufficient resources:

```python
# Good: Multiple judges, sufficient resources
config = ToolkitConfig(
    judge_models=["llama-3-8b", "mistral-7b", "phi-3-mini"],
    # ... other config
)
toolkit = EvaluationToolkit(config)
result = toolkit.evaluate(source, candidate, parallel_judges=True)

# Not beneficial: Single judge
config = ToolkitConfig(
    judge_models=["llama-3-8b"],  # Only one judge
    # ... other config
)
toolkit = EvaluationToolkit(config)
result = toolkit.evaluate(source, candidate, parallel_judges=False)
```

#### 3. Optimize Model Loading

Reuse toolkit instances to benefit from model caching:

```python
# Good: Reuse toolkit
toolkit = EvaluationToolkit(config)
for request in requests:
    result = toolkit.evaluate(request.source_text, request.candidate_output)

# Inefficient: Create new toolkit each time
for request in requests:
    toolkit = EvaluationToolkit(config)  # Reloads models!
    result = toolkit.evaluate(request.source_text, request.candidate_output)
```

#### 4. Monitor Performance

Use performance tracking to monitor component performance:

```python
# Run evaluations
for request in requests:
    toolkit.evaluate(request.source_text, request.candidate_output)

# Get performance report
report = toolkit.get_performance_report()

print(f"Verifier avg latency: {report['verifier_metrics']['average_latency']:.3f}s")
print(f"Judge avg latency: {report['judge_metrics']['average_latency']:.3f}s")
print(f"Disagreements: {report['disagreements']['total_count']}")
```

#### 5. Balance Speed and Resource Usage

Consider your resource constraints when optimizing:

```python
# High-speed configuration (more resources)
config_fast = ToolkitConfig(
    judge_models=["llama-3-8b", "mistral-7b"],
    quantize=True,  # Reduce memory
    enable_retrieval=False,  # Skip retrieval for speed
)
toolkit_fast = EvaluationToolkit(config_fast)
result = toolkit_fast.evaluate(source, candidate, parallel_judges=True)

# Resource-constrained configuration
config_light = ToolkitConfig(
    judge_models=["phi-3-mini"],  # Smaller model
    quantize=True,
    enable_retrieval=False,
)
toolkit_light = EvaluationToolkit(config_light)
result = toolkit_light.evaluate(source, candidate, parallel_judges=False)
```

### Performance Metrics

#### Typical Performance Characteristics

| Configuration | Time per Evaluation | Memory Usage |
|--------------|---------------------|--------------|
| 1 judge, sequential | 2-3s | 4-8 GB |
| 3 judges, sequential | 6-9s | 12-24 GB |
| 3 judges, parallel | 2-4s | 12-24 GB |
| With retrieval | +1-2s | +2-4 GB |

*Note: Actual performance depends on hardware, model sizes, and input length.*

#### Optimization Impact

| Optimization | Typical Speedup | Memory Impact |
|-------------|-----------------|---------------|
| Parallel judges (3 models) | 2-3x | None |
| Model caching | 10-20x (initialization) | None |
| Skip retrieval | 1.3-1.5x | -2-4 GB |
| Quantization | 1.1-1.2x | -50% |

### Troubleshooting

#### High Memory Usage

If you encounter memory issues:

1. Enable quantization:
   ```python
   config = ToolkitConfig(quantize=True, ...)
   ```

2. Reduce number of judges:
   ```python
   config = ToolkitConfig(judge_models=["phi-3-mini"], ...)
   ```

3. Use sequential evaluation:
   ```python
   result = toolkit.evaluate(..., parallel_judges=False)
   ```

#### Slow Evaluation

If evaluation is slower than expected:

1. Profile to identify bottlenecks:
   ```python
   toolkit = EvaluationToolkit(config, enable_profiling=True)
   # ... run evaluations ...
   print(toolkit.get_profiling_summary())
   ```

2. Enable parallel judges:
   ```python
   result = toolkit.evaluate(..., parallel_judges=True)
   ```

3. Disable retrieval if not needed:
   ```python
   config = ToolkitConfig(enable_retrieval=False, ...)
   ```

4. Use smaller models:
   ```python
   config = ToolkitConfig(
       verifier_model="minicheck-flan-t5-base",  # Smaller
       judge_models=["phi-3-mini"],  # Smaller
       ...
   )
   ```

### Examples

See `examples/performance_optimization_example.py` for complete working examples of:
- Profiling and bottleneck identification
- Sequential vs parallel evaluation comparison
- Optimized batch processing
- Model loading optimization
- Performance tracking and monitoring

### Requirements

These optimizations address the following requirements:
- **1.1**: Model initialization and loading optimization
- **1.2**: Efficient model management and caching
- **5.1**: Batch processing optimization

### API Reference

#### EvaluationToolkit

```python
class EvaluationToolkit:
    def __init__(self, config: ToolkitConfig, enable_profiling: bool = False):
        """Initialize toolkit with optional profiling."""
        
    def evaluate(
        self,
        source_text: str,
        candidate_output: str,
        parallel_judges: bool = False,
        ...
    ) -> EvaluationResult:
        """Evaluate with optional parallel judge processing."""
        
    def batch_evaluate(
        self,
        requests: List[EvaluationRequest],
        parallel_judges: bool = False,
        ...
    ) -> BatchResult:
        """Batch evaluate with optional parallel processing."""
        
    def get_profiling_summary(self) -> str:
        """Get profiling summary."""
        
    def get_profiling_bottlenecks(self, top_n: int = 5) -> List[tuple[str, float]]:
        """Get top bottlenecks."""
        
    def reset_profiling(self):
        """Reset profiling data."""
```

#### JudgeEnsemble

```python
class JudgeEnsemble:
    def evaluate_all(
        self,
        source_text: str,
        candidate_output: str,
        parallel: bool = False,
        max_workers: Optional[int] = None,
        ...
    ) -> List[JudgeResult]:
        """Evaluate with all judges, optionally in parallel."""
```

#### Profiler

```python
class Profiler:
    def profile(self, name: str, detailed: bool = False):
        """Context manager for profiling a code block."""
        
    def get_summary(self) -> str:
        """Get profiling summary."""
        
    def get_bottlenecks(self, top_n: int = 5) -> List[tuple[str, float]]:
        """Get top bottlenecks."""
        
    def reset(self):
        """Reset profiling data."""
```
