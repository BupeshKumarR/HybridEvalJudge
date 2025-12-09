# Scripts

This directory contains utility scripts for the LLM Judge Auditor toolkit.

## Available Scripts

### Benchmark Validation

#### `download_benchmarks.py`

Download benchmark datasets for validation.

**Usage:**
```bash
# Download all benchmarks
python scripts/download_benchmarks.py

# Download specific benchmark
python scripts/download_benchmarks.py --dataset fever
python scripts/download_benchmarks.py --dataset truthfulqa

# Specify output directory
python scripts/download_benchmarks.py --output-dir /path/to/benchmarks
```

**Supported Datasets:**
- **FEVER**: Fact Extraction and VERification dataset
- **TruthfulQA**: Dataset for evaluating truthfulness

**Output:**
Creates a `benchmarks/` directory with:
- `benchmarks/fever/` - FEVER dataset files
- `benchmarks/truthfulqa/` - TruthfulQA dataset files

---

#### `run_benchmarks.py`

Run benchmark validation on downloaded datasets.

**Usage:**
```bash
# Run all benchmarks
python scripts/run_benchmarks.py

# Run specific benchmark
python scripts/run_benchmarks.py --dataset fever
python scripts/run_benchmarks.py --dataset truthfulqa

# Use different preset
python scripts/run_benchmarks.py --preset fast
python scripts/run_benchmarks.py --preset balanced
python scripts/run_benchmarks.py --preset strict

# Limit number of samples
python scripts/run_benchmarks.py --max-samples 50

# Specify output file
python scripts/run_benchmarks.py --output results/my_results.json
```

**Options:**
- `--dataset`: Which dataset to evaluate (fever, truthfulqa, or omit for all)
- `--preset`: Toolkit preset to use (default: balanced)
- `--benchmarks-dir`: Directory containing benchmark datasets (default: benchmarks)
- `--output`: Output file for results (default: benchmarks/results/benchmark_results.json)
- `--max-samples`: Maximum number of samples to evaluate per dataset

**Output:**
- JSON file with detailed results
- Console output with metrics and baseline comparison
- Metrics include: accuracy, precision, recall, F1 score, confidence, latency

---

## Quick Start

```bash
# 1. Download benchmarks
python scripts/download_benchmarks.py

# 2. Run validation
python scripts/run_benchmarks.py --max-samples 50

# 3. View results
cat benchmarks/results/benchmark_results.json
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Benchmark Validation

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

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
        run: pip install -r requirements.txt
      - name: Download benchmarks
        run: python scripts/download_benchmarks.py
      - name: Run benchmarks
        run: python scripts/run_benchmarks.py --max-samples 100
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/results/
```

## Development

### Adding New Benchmarks

To add a new benchmark dataset:

1. Add download logic to `download_benchmarks.py`:
```python
def download_new_benchmark(self) -> None:
    """Download new benchmark dataset."""
    # Implementation here
```

2. Add evaluation logic to `run_benchmarks.py`:
```python
def evaluate_new_benchmark(self, max_samples: int = None) -> BenchmarkResult:
    """Evaluate on new benchmark dataset."""
    # Implementation here
```

3. Update documentation in `docs/BENCHMARK_VALIDATION.md`

### Testing Scripts

Scripts have integration tests in `tests/integration/test_benchmarks.py`:

```bash
pytest tests/integration/test_benchmarks.py -v
```

## See Also

- [Benchmark Validation Documentation](../docs/BENCHMARK_VALIDATION.md)
- [Benchmark Validation Example](../examples/benchmark_validation_example.py)
- [Integration Tests](../tests/integration/test_benchmarks.py)
