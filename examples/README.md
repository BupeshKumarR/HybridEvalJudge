# Examples

This directory contains example scripts demonstrating various features of the LLM Judge Auditor toolkit.

## Getting Started

Before running examples, ensure you have:

1. **Activated the virtual environment**:
   ```bash
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate.bat  # Windows
   ```

2. **Verified installation**:
   ```bash
   pytest  # All tests should pass
   ```

## Example Scripts

### Beginner Examples

#### 1. Simple Evaluation (`simple_evaluation.py`)

**Start here!** The most basic example showing how to evaluate a single output.

```bash
python examples/simple_evaluation.py
```

**What it demonstrates**:
- Initializing toolkit with a preset
- Evaluating a candidate output
- Viewing results and scores
- Exporting results to JSON

**Recommended for**: First-time users, quick testing

---

#### 2. Basic Usage (`basic_usage.py`)

Introduction to data models and configuration.

```bash
python examples/basic_usage.py
```

**What it demonstrates**:
- Loading preset configurations
- Creating custom configurations
- Working with data models (Claim, EvaluationRequest)
- Understanding the toolkit structure

**Recommended for**: Understanding the toolkit's building blocks

---

### Intermediate Examples

#### 3. Batch Processing (`batch_processing_example.py`)

Process multiple evaluations efficiently.

```bash
python examples/batch_processing_example.py
```

**What it demonstrates**:
- Creating multiple evaluation requests
- Batch processing with error resilience
- Viewing batch statistics
- Handling errors in batch mode
- Saving batch results

**Recommended for**: Production use, evaluating multiple outputs

---

#### 4. Evaluation Toolkit (`evaluation_toolkit_example.py`)

Advanced features and customization.

```bash
python examples/evaluation_toolkit_example.py
```

**What it demonstrates**:
- Custom configurations
- Hallucination detection
- Exporting results in different formats
- Advanced evaluation scenarios

**Recommended for**: Advanced users, custom workflows

---

### Component Examples

These examples demonstrate individual components of the toolkit:

#### Data Models (`data_models_example.py`)

```bash
python examples/data_models_example.py
```

Working with core data structures: Claim, Passage, Issue, Verdict, etc.

---

#### Device Detection (`device_detection_example.py`)

```bash
python examples/device_detection_example.py
```

Hardware detection and optimization for CPU, CUDA, and MPS.

---

#### Aggregation Engine (`aggregation_engine_example.py`)

```bash
python examples/aggregation_engine_example.py
```

Combining results from multiple judges using different strategies.

---

#### Judge Ensemble (`judge_ensemble_example.py`)

```bash
python examples/judge_ensemble_example.py
```

Working with multiple judge models for evaluation.

---

#### Specialized Verifier (`specialized_verifier_example.py`)

```bash
python examples/specialized_verifier_example.py
```

Statement-level fact-checking with specialized models.

---

#### Retrieval Component (`retrieval_component_example.py`)

```bash
python examples/retrieval_component_example.py
```

Retrieval-augmented verification with knowledge bases.

---

#### Prompt Manager (`prompt_manager_example.py`)

```bash
python examples/prompt_manager_example.py
```

Managing and customizing evaluation prompts.

---

#### Preset Manager (`preset_manager_example.py`)

```bash
python examples/preset_manager_example.py
```

Working with preset configurations.

---

#### Report Generator (`report_generator_example.py`)

```bash
python examples/report_generator_example.py
```

Generating comprehensive evaluation reports.

---

#### Error Handling (`error_handling_example.py`)

```bash
python examples/error_handling_example.py
```

Handling errors gracefully in evaluation workflows.

---

#### CLI Usage (`cli_example.py`)

```bash
python examples/cli_example.py
```

Using the command-line interface programmatically.

---

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
done
```

### Run with Custom Configuration

Most examples can be modified to use custom configurations:

```python
# Edit the example file
config = ToolkitConfig.from_preset("balanced")  # Change preset
config.batch_size = 4  # Customize settings
toolkit = EvaluationToolkit(config)
```

## Example Output

### Simple Evaluation Output

```
================================================================================
Simple Evaluation Example
================================================================================

1. Initializing toolkit with 'fast' preset...
   (This uses minimal resources for quick evaluation)
   ✓ Toolkit initialized

2. Setting up evaluation...
   Source text: Facts about the Eiffel Tower
   Candidate output: Summary of those facts

3. Running evaluation...
   ✓ Evaluation complete

4. Results:
--------------------------------------------------------------------------------
   Consensus Score:    85.50/100
   Confidence:         0.92
   Disagreement:       5.20
   Issues Found:       0

   Individual Judge Scores:
     • Phi-3-mini: 87.00
     • Mistral-7B: 84.00

   Verifier Verdicts: 3 claims checked
     1. SUPPORTED (confidence: 0.95)
     2. SUPPORTED (confidence: 0.89)
     3. SUPPORTED (confidence: 0.91)

   ✓ No issues detected - output appears factually accurate!

5. Exporting results...
   ✓ Results saved to: simple_evaluation_result.json

================================================================================
Evaluation complete!
================================================================================
```

### Batch Processing Output

```
================================================================================
Batch Processing Example
================================================================================

1. Initializing toolkit with 'fast' preset...

2. Creating batch of evaluation requests...
   Created 5 evaluation requests

3. Processing batch (with error resilience enabled)...

4. Batch Processing Results:
--------------------------------------------------------------------------------
Total requests:        5
Successful:            5
Failed:                0
Success rate:          100.0%

5. Batch Statistics:
--------------------------------------------------------------------------------
Mean score:            78.40
Median score:          82.00
Std deviation:         12.35
Min score:             58.00
Max score:             92.00
25th percentile:       71.00
75th percentile:       87.00

6. Individual Results:
--------------------------------------------------------------------------------
[Results for each request...]

8. Saving results to JSON...
   Results saved to: batch_results.json

================================================================================
Batch processing example complete!
================================================================================
```

## Troubleshooting

### Issue: Import Errors

**Solution**: Ensure virtual environment is activated and package is installed

```bash
source .venv/bin/activate
pip install -e .
```

### Issue: Model Not Found

**Solution**: Examples use mock models by default. For real evaluation, configure actual models:

```python
config = ToolkitConfig(
    verifier_model="MiniCheck/flan-t5-base-finetuned",
    judge_models=["microsoft/Phi-3-mini-4k-instruct"]
)
```

### Issue: Out of Memory

**Solution**: Use the "fast" preset or enable quantization:

```python
config = ToolkitConfig.from_preset("fast")
config.quantize = True
```

## Next Steps

After exploring the examples:

1. **Read the Usage Guide**: [docs/USAGE_GUIDE.md](../docs/USAGE_GUIDE.md)
2. **Check API Reference**: [docs/API_REFERENCE.md](../docs/API_REFERENCE.md)
3. **Try CLI**: [docs/CLI_USAGE.md](../docs/CLI_USAGE.md)
4. **Build Your Own**: Use examples as templates for your use case

## Contributing Examples

Have a useful example? Contributions are welcome!

1. Create your example script
2. Add documentation in this README
3. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Additional Resources

- [Main README](../README.md)
- [Quick Start Guide](../QUICKSTART.md)
- [Configuration Guide](../config/README.md)
- [Test Suite](../tests/)
