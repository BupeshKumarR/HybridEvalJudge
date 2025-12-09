# Component Performance Tracking

The LLM Judge Auditor toolkit includes comprehensive performance tracking capabilities that allow you to monitor and compare the performance of the specialized verifier and judge ensemble components.

## Overview

The `PerformanceTracker` component provides:

- **Separate metrics tracking** for verifier and judge ensemble
- **Latency monitoring** for each component
- **Confidence score tracking** across evaluations
- **Accuracy measurement** when ground truth is available
- **Disagreement logging** between verifier and judges
- **Comparative analysis** showing which component performs better on different claim types

## Requirements Satisfied

This implementation satisfies the following requirements:

- **13.1**: Track separate metrics for the Specialized Verifier and Judge Model Ensemble
- **13.2**: Include accuracy, latency, and confidence scores for each component in performance reports
- **13.3**: Log disagreements between verifier and judges with both verdicts and reasoning
- **13.4**: Identify which component performs better on different types of claims

## Usage

### Basic Usage

The performance tracker is automatically initialized when you create an `EvaluationToolkit`:

```python
from llm_judge_auditor import EvaluationToolkit

# Initialize toolkit (performance tracker is created automatically)
toolkit = EvaluationToolkit.from_preset("balanced")

# Run some evaluations
result1 = toolkit.evaluate(source1, candidate1)
result2 = toolkit.evaluate(source2, candidate2)
result3 = toolkit.evaluate(source3, candidate3)

# Get performance report
report = toolkit.get_performance_report()

print(f"Verifier avg latency: {report['verifier_metrics']['average_latency']:.3f}s")
print(f"Judge avg latency: {report['judge_metrics']['average_latency']:.3f}s")
print(f"Disagreements: {report['disagreements']['total_count']}")
```

### Getting a Human-Readable Summary

```python
# Get a formatted summary
summary = toolkit.get_performance_summary()
print(summary)
```

Output:
```
=== Performance Tracker Summary ===

Verifier Metrics:
  Total Evaluations: 10
  Average Latency: 0.045s
  Average Confidence: 0.87
  Accuracy: 95.00%

Judge Ensemble Metrics:
  Total Evaluations: 10
  Average Latency: 0.112s
  Average Confidence: 0.82
  Accuracy: 88.00%

Disagreements:
  Total: 2
  Rate: 20.00%
```

### Resetting Performance Tracking

If you want to start fresh tracking for a new batch of evaluations:

```python
# Reset all metrics
toolkit.reset_performance_tracking()

# Start tracking new evaluations
result = toolkit.evaluate(source, candidate)
```

## Performance Report Structure

The performance report returned by `get_performance_report()` contains:

### Verifier Metrics

```python
{
    "verifier_metrics": {
        "component_name": "specialized_verifier",
        "total_evaluations": 10,
        "average_latency": 0.045,
        "average_confidence": 0.87,
        "accuracy": 0.95,  # Only if ground truth available
        "claim_type_performance": {
            "factual": {"total": 5, "correct": 5, "incorrect": 0},
            "temporal": {"total": 3, "correct": 2, "incorrect": 1},
            "numerical": {"total": 2, "correct": 2, "incorrect": 0}
        }
    }
}
```

### Judge Metrics

```python
{
    "judge_metrics": {
        "component_name": "judge_ensemble",
        "total_evaluations": 10,
        "average_latency": 0.112,
        "average_confidence": 0.82,
        "accuracy": 0.88,  # Only if ground truth available
        "claim_type_performance": {
            "factual": {"total": 5, "correct": 4, "incorrect": 1},
            "temporal": {"total": 3, "correct": 3, "incorrect": 0},
            "numerical": {"total": 2, "correct": 1, "incorrect": 1}
        }
    }
}
```

### Disagreement Statistics

```python
{
    "disagreements": {
        "total_count": 2,
        "disagreement_rate": 0.20,
        "disagreements_by_claim_type": {
            "temporal": 1,
            "numerical": 1
        },
        "recent_disagreements": [
            {
                "statement": "The moon landing was in 1968.",
                "verifier_verdict": "REFUTED",
                "verifier_confidence": 0.85,
                "judge_consensus": 77.5,
                "judge_confidence": 0.72,
                "judge_results": [
                    {
                        "model_name": "llama-3-8b",
                        "score": 75.0,
                        "reasoning": "The date seems plausible..."
                    }
                ],
                "claim_type": "temporal"
            }
        ]
    }
}
```

### Comparative Analysis

```python
{
    "comparative_analysis": {
        "latency_comparison": {
            "verifier_avg": 0.045,
            "judge_avg": 0.112,
            "faster_component": "verifier"
        },
        "confidence_comparison": {
            "verifier_avg": 0.87,
            "judge_avg": 0.82,
            "more_confident_component": "verifier"
        },
        "accuracy_comparison": {
            "verifier_accuracy": 0.95,
            "judge_accuracy": 0.88,
            "more_accurate_component": "verifier"
        },
        "claim_type_performance": {
            "factual": {
                "verifier": {"accuracy": 1.0, "total_evaluations": 5},
                "judge": {"accuracy": 0.8, "total_evaluations": 5},
                "better_component": "verifier"
            },
            "temporal": {
                "verifier": {"accuracy": 0.67, "total_evaluations": 3},
                "judge": {"accuracy": 1.0, "total_evaluations": 3},
                "better_component": "judge_ensemble"
            },
            "numerical": {
                "verifier": {"accuracy": 1.0, "total_evaluations": 2},
                "judge": {"accuracy": 0.5, "total_evaluations": 2},
                "better_component": "verifier"
            }
        }
    }
}
```

## Disagreement Detection

A disagreement is logged when:

1. **Verifier says REFUTED but judges give high score (>70)**
   - Example: Verifier detects a factual error, but judges rate it as accurate

2. **Verifier says SUPPORTED but judges give low score (<30)**
   - Example: Verifier confirms accuracy, but judges find issues

3. **Verifier says NOT_ENOUGH_INFO but judges are very confident (score <20 or >80)**
   - Example: Verifier is uncertain, but judges have strong opinions

## Advanced Usage

### Direct Use of PerformanceTracker

You can also use the `PerformanceTracker` directly for custom tracking:

```python
from llm_judge_auditor.components import PerformanceTracker
from llm_judge_auditor.models import Verdict, VerdictLabel, JudgeResult, ClaimType

tracker = PerformanceTracker()

# Track verifier evaluation
tracker.start_verifier_timing()
# ... run verifier ...
latency = tracker.end_verifier_timing()

verdict = Verdict(
    label=VerdictLabel.SUPPORTED,
    confidence=0.9,
    evidence=["Evidence"],
    reasoning="Reasoning"
)

tracker.record_verifier_result(
    verdict=verdict,
    latency=latency,
    claim_type=ClaimType.FACTUAL,
    correct=True  # If ground truth is available
)

# Track judge evaluation
tracker.start_judge_timing()
# ... run judges ...
latency = tracker.end_judge_timing()

judge_results = [
    JudgeResult(
        model_name="llama-3-8b",
        score=85.0,
        reasoning="Looks good",
        confidence=0.85
    )
]

tracker.record_judge_results(
    judge_results=judge_results,
    latency=latency,
    claim_type=ClaimType.FACTUAL,
    correct=True
)

# Log disagreements
tracker.log_disagreement(
    statement="Test statement",
    verifier_verdict=verdict,
    judge_results=judge_results,
    judge_consensus_score=85.0,
    claim_type=ClaimType.FACTUAL
)

# Generate report
report = tracker.generate_report()
```

## Integration with Evaluation Pipeline

The performance tracker is automatically integrated into the evaluation pipeline:

1. **Verifier Stage**: Latency and results are tracked automatically
2. **Judge Stage**: Ensemble latency and individual judge results are tracked
3. **Aggregation Stage**: Disagreements are detected and logged
4. **Reporting**: Performance metrics are available via `get_performance_report()`

## Performance Insights

The performance tracker helps you:

1. **Identify bottlenecks**: See which component is slower
2. **Compare accuracy**: Understand which component is more reliable
3. **Detect disagreements**: Find cases where components disagree
4. **Optimize by claim type**: Route claims to the component that performs better on that type
5. **Monitor confidence**: Track how confident each component is in its evaluations

## Example Output

See `examples/performance_tracker_example.py` for a complete working example that demonstrates:

- Recording metrics for both components
- Logging disagreements
- Generating comprehensive reports
- Comparative analysis
- Resetting tracking for new batches

## API Reference

### EvaluationToolkit Methods

- `get_performance_report() -> Dict`: Get comprehensive performance report
- `get_performance_summary() -> str`: Get human-readable summary
- `reset_performance_tracking()`: Reset all performance metrics

### PerformanceTracker Methods

- `start_verifier_timing()`: Start timing verifier evaluation
- `end_verifier_timing() -> float`: End timing and return elapsed time
- `start_judge_timing()`: Start timing judge evaluation
- `end_judge_timing() -> float`: End timing and return elapsed time
- `record_verifier_result(verdict, latency, claim_type, correct)`: Record verifier result
- `record_judge_results(judge_results, latency, claim_type, correct)`: Record judge results
- `log_disagreement(statement, verifier_verdict, judge_results, judge_consensus_score, claim_type)`: Log disagreement
- `generate_report() -> Dict`: Generate performance report
- `get_summary() -> str`: Get human-readable summary
- `reset()`: Reset all metrics

## See Also

- [API Reference](API_REFERENCE.md)
- [Usage Guide](USAGE_GUIDE.md)
- [Reliability Validation](RELIABILITY_VALIDATION.md)
