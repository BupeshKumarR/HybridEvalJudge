# Benchmark Validation

This document describes the benchmark validation infrastructure for the LLM Judge Auditor toolkit.

## Overview

The toolkit is validated against standard benchmarks to ensure accuracy and reliability:

- **FEVER** (Fact Extraction and VERification): Tests fact-checking accuracy
- **TruthfulQA**: Tests detection of common misconceptions and hallucinations

## Quick Start

### 1. Download Benchmark Datasets

```bash
# Download all benchmarks
python scripts/download_benchmarks.py --all

# Download specific benchmark
python scripts/download_benchmarks.py --dataset fever
python scripts/download_benchmarks.py --dataset truthfulqa
```

### 2. Run Benchmark Validation

```bash
# Run all benchmarks with balanced preset
python scripts/run_benchmarks.py --all --preset balanced

# Run specific benchmark
python scripts/run_benchmarks.py --dataset fever --preset fast

# Limit number of samples for quick testing
python scripts/run_benchmarks.py --all --max-samples 50
```

### 3. View Results

Results are saved to `benchmarks/results/benchmark_results.json` by default.

## Benchmark Datasets

### FEVER (Fact Extraction and VERification)

**Purpose**: Evaluate fact-checking and claim verification accuracy

**Dataset Structure**:
- Claims with supporting/refuting evidence
- Labels: SUPPORTS, REFUTES, NOT ENOUGH INFO
- ~185,000 claims in full dataset

**Metrics**:
- Accuracy: Percentage of correctly classified claims
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)
- F1 Score: Harmonic mean of precision and recall

**Baseline Performance**:
- MiniCheck baseline: ~85% accuracy, ~82% F1 score

**Citation**:
```
@inproceedings{thorne2018fever,
  title={FEVER: a large-scale dataset for Fact Extraction and VERification},
  author={Thorne, James and Vlachos, Andreas and Christodoulopoulos, Christos and Mittal, Arpit},
  booktitle={NAACL-HLT},
  year={2018}
}
```

### TruthfulQA

**Purpose**: Evaluate detection of common misconceptions and false beliefs

**Dataset Structure**:
- Questions with correct and incorrect answers
- Categories: Health, Science, Geography, History, etc.
- ~800 questions in full dataset

**Metrics**:
- Accuracy: Percentage of questions answered truthfully
- Precision: Correct answers / Total answers given
- Recall: Correct answers / Total correct answers available
- F1 Score: Harmonic mean of precision and recall

**Baseline Performance**:
- GPT-3.5 baseline: ~75% accuracy, ~70% F1 score

**Citation**:
```
@article{lin2021truthfulqa,
  title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
  author={Lin, Stephanie and Hilton, Jacob and Evans, Owain},
  journal={arXiv preprint arXiv:2109.07958},
  year={2021}
}
```

## Usage Examples

### Basic Validation

```bash
# Run all benchmarks with default settings
python scripts/run_benchmarks.py --all
```

### Custom Configuration

```bash
# Use fast preset for quick validation
python scripts/run_benchmarks.py --all --preset fast

# Use strict preset for maximum accuracy
python scripts/run_benchmarks.py --all --preset strict

# Specify custom output location
python scripts/run_benchmarks.py --all --output results/my_results.json
```

### Development Testing

```bash
# Quick test with limited samples
python scripts/run_benchmarks.py --all --max-samples 10

# Test specific dataset
python scripts/run_benchmarks.py --dataset fever --max-samples 50
```

## Understanding Results

### Output Format

Results are saved as JSON with the following structure:

```json
[
  {
    "dataset": "FEVER",
    "total_samples": 100,
    "correct": 85,
    "incorrect": 15,
    "accuracy": 0.85,
    "precision": 0.87,
    "recall": 0.83,
    "f1_score": 0.85,
    "avg_confidence": 0.89,
    "avg_latency_ms": 245.3,
    "errors": 0
  },
  {
    "dataset": "TruthfulQA",
    "total_samples": 50,
    "correct": 38,
    "incorrect": 12,
    "accuracy": 0.76,
    "precision": 0.79,
    "recall": 0.74,
    "f1_score": 0.76,
    "avg_confidence": 0.82,
    "avg_latency_ms": 312.7,
    "errors": 0
  }
]
```

### Key Metrics

- **Accuracy**: Overall correctness rate
- **Precision**: How many predicted positives are actually correct
- **Recall**: How many actual positives are correctly identified
- **F1 Score**: Balance between precision and recall
- **Avg Confidence**: Average confidence score from the toolkit
- **Avg Latency**: Average processing time per sample

### Baseline Comparison

The script automatically compares results to published baselines:

```
Comparison to Baselines
============================================================

FEVER:
  Baseline (MiniCheck baseline):
    Accuracy: 85.00%
    F1 Score: 82.00%
  Our Results:
    Accuracy: 85.00% (+0.00%)
    F1 Score: 85.00% (+3.00%)

TruthfulQA:
  Baseline (GPT-3.5 baseline):
    Accuracy: 75.00%
    F1 Score: 70.00%
  Our Results:
    Accuracy: 76.00% (+1.00%)
    F1 Score: 76.00% (+6.00%)
```

## Integration with Testing

### Running as Part of Test Suite

You can integrate benchmark validation into your test suite:

```python
# tests/integration/test_benchmarks.py
import pytest
from scripts.run_benchmarks import BenchmarkEvaluator

def test_fever_accuracy():
    """Test that FEVER accuracy meets minimum threshold."""
    evaluator = BenchmarkEvaluator(preset="balanced")
    result = evaluator.evaluate_fever(max_samples=100)
    
    # Should be within 10% of baseline
    assert result.accuracy >= 0.75, f"FEVER accuracy too low: {result.accuracy}"

def test_truthfulqa_accuracy():
    """Test that TruthfulQA accuracy meets minimum threshold."""
    evaluator = BenchmarkEvaluator(preset="balanced")
    result = evaluator.evaluate_truthfulqa(max_samples=50)
    
    # Should be within 10% of baseline
    assert result.accuracy >= 0.65, f"TruthfulQA accuracy too low: {result.accuracy}"
```

### Continuous Integration

Add to your CI pipeline:

```yaml
# .github/workflows/benchmarks.yml
name: Benchmark Validation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Download benchmarks
        run: python scripts/download_benchmarks.py --all
      - name: Run benchmarks
        run: python scripts/run_benchmarks.py --all --max-samples 50
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/results/
```

## Troubleshooting

### Dataset Not Found

If you see "Dataset not found" errors:

```bash
# Download the datasets first
python scripts/download_benchmarks.py --all
```

### Memory Issues

If you encounter out-of-memory errors:

```bash
# Use fast preset with quantization
python scripts/run_benchmarks.py --all --preset fast

# Limit number of samples
python scripts/run_benchmarks.py --all --max-samples 50
```

### Slow Performance

To speed up validation:

```bash
# Use fast preset
python scripts/run_benchmarks.py --all --preset fast

# Reduce sample size
python scripts/run_benchmarks.py --all --max-samples 100
```

## Full Dataset Access

### FEVER

The sample dataset includes only a few examples. To use the full FEVER dataset:

1. Register at https://fever.ai/
2. Download the official dataset
3. Place files in `benchmarks/fever/`:
   - `train.jsonl`
   - `dev.jsonl`
   - `test.jsonl`

### TruthfulQA

To use the full TruthfulQA dataset:

1. Visit https://github.com/sylinrl/TruthfulQA
2. Download `TruthfulQA.csv`
3. Convert to JSON format (or modify the script to read CSV)
4. Place in `benchmarks/truthfulqa/truthfulqa.json`

## Advanced Usage

### Custom Evaluation Logic

You can extend the benchmark evaluator for custom evaluation logic:

```python
from scripts.run_benchmarks import BenchmarkEvaluator

class CustomEvaluator(BenchmarkEvaluator):
    def _evaluate_claim(self, claim, evidence, true_label):
        # Custom evaluation logic
        result = self.toolkit.evaluate(
            source_text=evidence,
            candidate_output=claim,
            custom_criteria=["factual_accuracy", "completeness"]
        )
        
        # Map toolkit output to benchmark format
        predicted_label = self._map_score_to_label(result.consensus_score)
        
        return {
            "correct": predicted_label == true_label,
            "confidence": result.confidence,
            "predicted_label": predicted_label
        }
```

### Batch Processing

For faster evaluation on large datasets:

```python
evaluator = BenchmarkEvaluator(preset="balanced")

# Load all samples
samples = evaluator._load_fever_samples()

# Batch evaluate
results = evaluator.toolkit.batch_evaluate(
    sources=[s["evidence"] for s in samples],
    candidates=[s["claim"] for s in samples],
    batch_size=16
)
```

## Performance Targets

Based on the design document (Requirement 10.3), the toolkit should achieve:

- **FEVER**: Within 10% of MiniCheck baseline (~85% accuracy)
- **TruthfulQA**: Competitive with GPT-3.5 baseline (~75% accuracy)
- **Latency**: < 500ms per evaluation on average
- **Consistency**: Variance < 5 points across multiple runs

## See Also

- [Reliability Validation](RELIABILITY_VALIDATION.md) - Inter-model agreement metrics
- [Adversarial Testing](ADVERSARIAL_TESTING.md) - Robustness testing
- [Performance Tracking](PERFORMANCE_TRACKING.md) - Component-level metrics
