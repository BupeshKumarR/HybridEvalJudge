# Examples Index

Quick reference guide to all example scripts in the LLM Judge Auditor toolkit.

## Quick Navigation

| Example | Difficulty | Purpose | Run Time |
|---------|-----------|---------|----------|
| [simple_evaluation.py](#simple-evaluation) | ⭐ Beginner | Basic evaluation workflow | < 1 min |
| [basic_usage.py](#basic-usage) | ⭐ Beginner | Data models and config | < 1 min |
| [batch_processing_example.py](#batch-processing) | ⭐⭐ Intermediate | Multiple evaluations | 1-2 min |
| [evaluation_toolkit_example.py](#evaluation-toolkit) | ⭐⭐ Intermediate | Advanced features | 2-5 min |
| [benchmark_validation_example.py](#benchmark-validation) | ⭐⭐ Intermediate | Benchmark validation | 2-5 min |
| [verifier_trainer_example.py](#verifier-trainer) | ⭐⭐⭐ Advanced | Fine-tuning verifiers | 5-10 min |
| [Component Examples](#component-examples) | ⭐⭐⭐ Advanced | Individual components | Varies |

## Beginner Examples

### Simple Evaluation

**File**: `examples/simple_evaluation.py`

**Purpose**: The simplest possible evaluation workflow - perfect for getting started.

**What you'll learn**:
- How to initialize the toolkit
- How to evaluate a single output
- How to view and interpret results
- How to export results

**Run it**:
```bash
python examples/simple_evaluation.py
```

**Key code snippet**:
```python
toolkit = EvaluationToolkit.from_preset("fast")
result = toolkit.evaluate(source_text, candidate_output)
print(f"Score: {result.consensus_score:.2f}/100")
```

---

### Basic Usage

**File**: `examples/basic_usage.py`

**Purpose**: Introduction to the toolkit's data models and configuration system.

**What you'll learn**:
- How to load preset configurations
- How to create custom configurations
- How to work with data models (Claim, EvaluationRequest)
- Understanding the toolkit's structure

**Run it**:
```bash
python examples/basic_usage.py
```

**Key code snippet**:
```python
config = ToolkitConfig.from_preset("balanced")
request = EvaluationRequest(
    source_text="...",
    candidate_output="...",
    task="factual_accuracy"
)
```

---

## Intermediate Examples

### Batch Processing

**File**: `examples/batch_processing_example.py`

**Purpose**: Efficiently evaluate multiple outputs with error handling.

**What you'll learn**:
- How to create multiple evaluation requests
- How to process batches with error resilience
- How to view batch statistics
- How to save batch results

**Run it**:
```bash
python examples/batch_processing_example.py
```

**Key code snippet**:
```python
batch_result = toolkit.batch_evaluate(
    requests=requests,
    continue_on_error=True
)
print(f"Mean score: {batch_result.statistics['mean']:.2f}")
```

---

### Evaluation Toolkit

**File**: `examples/evaluation_toolkit_example.py`

**Purpose**: Comprehensive demonstration of advanced features.

**What you'll learn**:
- Custom configuration options
- Hallucination detection
- Different export formats
- Advanced evaluation scenarios

**Run it**:
```bash
python examples/evaluation_toolkit_example.py
```

**Note**: This example includes multiple sub-examples. Uncomment the ones you want to run in the `main()` function.

---

### Benchmark Validation

**File**: `examples/benchmark_validation_example.py`

**Purpose**: Demonstrate benchmark validation on FEVER and TruthfulQA datasets.

**What you'll learn**:
- How to download benchmark datasets
- Running FEVER and TruthfulQA evaluations
- Comparing different presets on benchmarks
- Saving and loading benchmark results
- Comparing results to published baselines
- Implementing custom evaluation logic

**Run it**:
```bash
python examples/benchmark_validation_example.py
```

**Prerequisites**:
```bash
# Download benchmarks first (done automatically by example)
python scripts/download_benchmarks.py
```

**Key code snippet**:
```python
from run_benchmarks import BenchmarkEvaluator

evaluator = BenchmarkEvaluator(preset="balanced")
fever_result = evaluator.evaluate_fever(max_samples=50)
print(f"FEVER Accuracy: {fever_result.accuracy:.2%}")
```

**Note**: This example uses mock evaluation for demonstration. In production, it would use actual models.

---

## Component Examples

### Data Models

**File**: `examples/data_models_example.py`

**Purpose**: Working with core data structures.

**Components covered**: Claim, Passage, Issue, Verdict, EvaluationRequest, EvaluationResult

```bash
python examples/data_models_example.py
```

---

### Device Detection

**File**: `examples/device_detection_example.py`

**Purpose**: Hardware detection and optimization.

**Components covered**: DeviceManager, device selection, auto-configuration

```bash
python examples/device_detection_example.py
```

---

### Aggregation Engine

**File**: `examples/aggregation_engine_example.py`

**Purpose**: Combining results from multiple judges.

**Components covered**: AggregationEngine, aggregation strategies, disagreement detection

```bash
python examples/aggregation_engine_example.py
```

---

### Judge Ensemble

**File**: `examples/judge_ensemble_example.py`

**Purpose**: Working with multiple judge models.

**Components covered**: JudgeEnsemble, individual judges, pairwise comparison

```bash
python examples/judge_ensemble_example.py
```

---

### Specialized Verifier

**File**: `examples/specialized_verifier_example.py`

**Purpose**: Statement-level fact-checking.

**Components covered**: SpecializedVerifier, verdicts, confidence scores

```bash
python examples/specialized_verifier_example.py
```

---

### Retrieval Component

**File**: `examples/retrieval_component_example.py`

**Purpose**: Retrieval-augmented verification.

**Components covered**: RetrievalComponent, knowledge bases, claim extraction

```bash
python examples/retrieval_component_example.py
```

---

### Prompt Manager

**File**: `examples/prompt_manager_example.py`

**Purpose**: Managing evaluation prompts.

**Components covered**: PromptManager, templates, customization

```bash
python examples/prompt_manager_example.py
```

---

### Preset Manager

**File**: `examples/preset_manager_example.py`

**Purpose**: Working with preset configurations.

**Components covered**: PresetManager, loading presets, customization

```bash
python examples/preset_manager_example.py
```

---

### Report Generator

**File**: `examples/report_generator_example.py`

**Purpose**: Generating evaluation reports.

**Components covered**: ReportGenerator, export formats, report structure

```bash
python examples/report_generator_example.py
```

---

### Streaming Evaluator

**File**: `examples/streaming_evaluator_example.py`

**Purpose**: Processing large documents incrementally.

**Components covered**: StreamingEvaluator, chunking, memory-efficient processing

```bash
python examples/streaming_evaluator_example.py
```

---

### Verifier Trainer

**File**: `examples/verifier_trainer_example.py`

**Purpose**: Fine-tuning specialized verifier models for fact-checking.

**Components covered**: VerifierTrainer, FEVER dataset, custom training data, model evaluation

```bash
python examples/verifier_trainer_example.py
```

**What you'll learn**:
- How to load training data in FEVER format
- How to fine-tune small models (< 1B parameters) for fact verification
- How to evaluate trained models with accuracy, precision, recall, and F1 metrics
- How to save and load trained verifiers
- How to create custom training datasets programmatically
- How to analyze label distribution in training data

**Key code snippet**:
```python
trainer = VerifierTrainer(
    base_model="google/flan-t5-base",
    output_dir="models/my_verifier"
)
train_data = trainer.load_fever_dataset("data/fever_train.jsonl")
metrics = trainer.train(train_data=train_data, num_epochs=3)
trainer.save_model("models/my_verifier/final")
```

---

### Plugin System

**File**: `examples/plugin_system_example.py`

**Purpose**: Extending the toolkit with custom components.

**Components covered**: PluginRegistry, custom verifiers/judges/aggregators, plugin discovery

```bash
python examples/plugin_system_example.py
```

---

### Adversarial Testing

**File**: `examples/adversarial_tester_example.py`

**Purpose**: Testing robustness against adversarial perturbations.

**Components covered**: AdversarialTester, perturbation generation, robustness testing, symmetry testing

```bash
python examples/adversarial_tester_example.py
```

**What you'll learn**:
- How to generate adversarial perturbations (date shifts, location swaps, number changes)
- How to test detection rates
- How to verify pairwise ranking symmetry
- How to analyze robustness by perturbation type

---

### Error Handling

**File**: `examples/error_handling_example.py`

**Purpose**: Handling errors gracefully.

**Components covered**: Error types, recovery strategies, logging

```bash
python examples/error_handling_example.py
```

---

### Performance Optimization

**File**: `examples/performance_optimization_example.py`

**Purpose**: Demonstrating performance optimization features.

**Components covered**: Profiling, parallel judge evaluation, model caching, batch optimization

```bash
python examples/performance_optimization_example.py
```

**What you'll learn**:
- How to enable profiling to identify bottlenecks
- How to use parallel judge evaluation for faster processing
- How to optimize batch processing
- How to analyze performance metrics
- How to compare sequential vs parallel evaluation

**Key code snippet**:
```python
# Enable profiling
toolkit = EvaluationToolkit(config, enable_profiling=True)

# Use parallel judges for faster evaluation
result = toolkit.evaluate(
    source_text=source,
    candidate_output=candidate,
    parallel_judges=True  # 2-3x speedup with multiple judges
)

# View profiling results
print(toolkit.get_profiling_summary())
bottlenecks = toolkit.get_profiling_bottlenecks(5)
```

---

### CLI Usage

**File**: `examples/cli_example.py`

**Purpose**: Using the command-line interface.

**Components covered**: CLI commands, arguments, programmatic usage

```bash
python examples/cli_example.py
```

---

## Learning Path

### Path 1: Quick Start (15 minutes)

For users who want to get started quickly:

1. `simple_evaluation.py` - Basic workflow
2. `batch_processing_example.py` - Multiple evaluations
3. Start using the toolkit!

### Path 2: Comprehensive (1 hour)

For users who want to understand everything:

1. `basic_usage.py` - Data models and config
2. `simple_evaluation.py` - Basic evaluation
3. `batch_processing_example.py` - Batch processing
4. `evaluation_toolkit_example.py` - Advanced features
5. Component examples as needed

### Path 3: Component-Focused (2 hours)

For users who want deep understanding:

1. `basic_usage.py` - Foundation
2. `device_detection_example.py` - Hardware
3. `specialized_verifier_example.py` - Verification
4. `judge_ensemble_example.py` - Judges
5. `aggregation_engine_example.py` - Aggregation
6. `report_generator_example.py` - Reporting
7. `evaluation_toolkit_example.py` - Integration

## Running Examples

### Run a Single Example

```bash
python examples/simple_evaluation.py
```

### Run All Examples

```bash
for example in examples/*.py; do
    echo "Running $example..."
    python "$example"
    echo "---"
done
```

### Run with Custom Settings

Edit the example file to customize:

```python
# Change preset
toolkit = EvaluationToolkit.from_preset("balanced")  # or "fast", "strict", "research"

# Customize configuration
config = ToolkitConfig.from_preset("fast")
config.batch_size = 4
config.enable_retrieval = True
toolkit = EvaluationToolkit(config)
```

## Example Output Formats

### Console Output

All examples print formatted output to the console with:
- Clear section headers
- Progress indicators
- Results and statistics
- Next steps

### File Output

Many examples save results to files:
- JSON: `result.json`, `batch_results.json`
- CSV: `batch_results.csv`
- Text: `report.txt`

## Troubleshooting

### Common Issues

**Import Error**:
```bash
# Solution: Activate virtual environment
source .venv/bin/activate
pip install -e .
```

**Model Not Found**:
```python
# Solution: Examples use mock models by default
# For real evaluation, configure actual models
config = ToolkitConfig(
    verifier_model="MiniCheck/flan-t5-base-finetuned",
    judge_models=["microsoft/Phi-3-mini-4k-instruct"]
)
```

**Out of Memory**:
```python
# Solution: Use fast preset or enable quantization
config = ToolkitConfig.from_preset("fast")
config.quantize = True
```

## Next Steps

After exploring examples:

1. **Read Documentation**:
   - [Usage Guide](USAGE_GUIDE.md)
   - [API Reference](API_REFERENCE.md)
   - [CLI Usage](CLI_USAGE.md)

2. **Try Your Own Data**:
   - Modify examples with your own text
   - Experiment with different configurations
   - Test different presets

3. **Build Your Application**:
   - Use examples as templates
   - Integrate into your workflow
   - Customize for your use case

## Additional Resources

- [Main README](../README.md)
- [Quick Start Guide](../QUICKSTART.md)
- [Examples README](../examples/README.md)
- [Contributing Guide](../CONTRIBUTING.md)
