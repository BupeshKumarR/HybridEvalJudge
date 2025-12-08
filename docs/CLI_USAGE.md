# CLI Usage Guide

The LLM Judge Auditor toolkit provides a command-line interface for evaluating LLM outputs without writing code.

## Installation

After installing the package, the `llm-judge` command will be available:

```bash
pip install -e .
```

## Quick Start

### Single Evaluation

Evaluate a single candidate output against a source text:

```bash
llm-judge evaluate \
  --source "Paris is the capital of France." \
  --candidate "Paris is the capital of Germany." \
  --preset fast
```

### Using Files

For longer texts, use files instead:

```bash
llm-judge evaluate \
  --source-file source.txt \
  --candidate-file candidate.txt \
  --preset balanced \
  --output report.json
```

### Batch Evaluation

Evaluate multiple outputs at once:

```bash
llm-judge batch-evaluate \
  --input batch_requests.json \
  --output batch_results.json \
  --preset fast
```

## Commands

### `evaluate`

Evaluate a single candidate output against a source text.

**Required Arguments:**
- `--source TEXT` or `--source-file PATH`: Source text (reference document)
- `--candidate TEXT` or `--candidate-file PATH`: Candidate output to evaluate

**Optional Arguments:**
- `--config PATH`: Path to YAML configuration file
- `--preset {fast,balanced,strict,research}`: Preset configuration (default: balanced)
- `--output PATH`: Path to save evaluation report
- `--output-format {json,markdown,text}`: Output format (default: json)
- `--no-retrieval`: Disable retrieval-augmented verification

**Example:**

```bash
llm-judge evaluate \
  --source "The Eiffel Tower was completed in 1889." \
  --candidate "The Eiffel Tower was built in 1889." \
  --preset balanced \
  --output report.json \
  --output-format markdown
```

### `batch-evaluate`

Evaluate multiple candidate outputs in batch.

**Required Arguments:**
- `--input PATH`: Path to JSON file containing batch requests
- `--output PATH`: Path to save batch results

**Optional Arguments:**
- `--config PATH`: Path to YAML configuration file
- `--preset {fast,balanced,strict,research}`: Preset configuration (default: balanced)
- `--continue-on-error`: Continue processing if an evaluation fails (default: True)

**Batch Input Format:**

```json
[
  {
    "source_text": "Paris is the capital of France.",
    "candidate_output": "Paris is in France.",
    "task": "factual_accuracy",
    "criteria": ["correctness"],
    "use_retrieval": false
  },
  {
    "source_text": "The Earth orbits the Sun.",
    "candidate_output": "The Sun orbits the Earth.",
    "task": "factual_accuracy",
    "criteria": ["correctness"],
    "use_retrieval": false
  }
]
```

**Example:**

```bash
llm-judge batch-evaluate \
  --input batch_input.json \
  --output batch_results.json \
  --preset fast \
  --continue-on-error
```

## Presets

The toolkit includes several pre-configured presets optimized for different use cases:

### `fast`
- **Use case**: Quick evaluations, development testing
- **Features**: No retrieval, single lightweight judge (Phi-3-mini)
- **Resource usage**: Low memory, fast execution
- **Accuracy**: Basic

### `balanced` (default)
- **Use case**: General-purpose evaluation
- **Features**: Retrieval enabled, 2 judges (LLaMA-3-8B, Mistral-7B)
- **Resource usage**: Moderate memory, reasonable speed
- **Accuracy**: Good

### `strict`
- **Use case**: High-accuracy evaluation, production use
- **Features**: Full pipeline, 3 judges, weighted aggregation
- **Resource usage**: Higher memory, slower execution
- **Accuracy**: High

### `research`
- **Use case**: Research, benchmarking, maximum transparency
- **Features**: All features enabled, detailed metrics
- **Resource usage**: Higher memory, slower execution
- **Accuracy**: High with full provenance

## Configuration Files

Instead of using presets, you can provide a custom YAML configuration file:

```yaml
# custom_config.yaml
verifier_model: "MiniCheck/flan-t5-large-finetuned"
judge_models:
  - "meta-llama/Llama-3-8B"
  - "mistralai/Mistral-7B-v0.1"
quantize: true
device: "auto"
enable_retrieval: true
aggregation_strategy: "mean"
batch_size: 1
max_length: 512
```

Use it with:

```bash
llm-judge evaluate \
  --source "Source text" \
  --candidate "Candidate text" \
  --config custom_config.yaml
```

## Output Formats

### JSON (default)

Structured output with all evaluation details:

```bash
llm-judge evaluate \
  --source "..." \
  --candidate "..." \
  --output report.json \
  --output-format json
```

### Markdown

Human-readable report with sections:

```bash
llm-judge evaluate \
  --source "..." \
  --candidate "..." \
  --output report.md \
  --output-format markdown
```

### Text

Simple plain text report:

```bash
llm-judge evaluate \
  --source "..." \
  --candidate "..." \
  --output report.txt \
  --output-format text
```

## Examples

### Example 1: Quick Factual Check

```bash
llm-judge evaluate \
  --source "Water boils at 100°C at sea level." \
  --candidate "Water boils at 100 degrees Celsius under standard atmospheric pressure." \
  --preset fast \
  --no-retrieval
```

### Example 2: Detailed Evaluation with Report

```bash
llm-judge evaluate \
  --source-file reference_document.txt \
  --candidate-file model_output.txt \
  --preset balanced \
  --output detailed_report.json \
  --output-format json
```

### Example 3: Batch Processing

Create `batch_input.json`:

```json
[
  {
    "source_text": "The capital of France is Paris.",
    "candidate_output": "Paris is the capital of France."
  },
  {
    "source_text": "The Earth is round.",
    "candidate_output": "The Earth is flat."
  }
]
```

Run batch evaluation:

```bash
llm-judge batch-evaluate \
  --input batch_input.json \
  --output batch_results.json \
  --preset balanced
```

### Example 4: Custom Configuration

Create `my_config.yaml`:

```yaml
verifier_model: "MiniCheck/flan-t5-base-finetuned"
judge_models:
  - "microsoft/Phi-3-mini-4k-instruct"
quantize: true
enable_retrieval: false
aggregation_strategy: "mean"
```

Run evaluation:

```bash
llm-judge evaluate \
  --source "Source text here" \
  --candidate "Candidate text here" \
  --config my_config.yaml \
  --output report.json
```

## Tips

1. **Start with `fast` preset**: Use the fast preset for initial testing and development
2. **Use files for long texts**: For texts longer than a few sentences, use `--source-file` and `--candidate-file`
3. **Disable retrieval for speed**: Add `--no-retrieval` to speed up evaluations when external knowledge isn't needed
4. **Save reports**: Always use `--output` to save evaluation reports for later analysis
5. **Batch processing**: Use batch evaluation for processing multiple outputs efficiently

## Troubleshooting

### Models not found

If you get a "model not found" error, the models need to be downloaded first. The toolkit will automatically download models on first use, but this requires an internet connection and may take some time.

### Out of memory

If you run out of memory:
- Use the `fast` preset
- Add `--no-retrieval` to disable retrieval
- Ensure quantization is enabled in your config (`quantize: true`)

### Slow evaluation

If evaluation is too slow:
- Use the `fast` preset
- Disable retrieval with `--no-retrieval`
- Use a smaller model in a custom config

## Getting Help

For more information on any command:

```bash
llm-judge --help
llm-judge evaluate --help
llm-judge batch-evaluate --help
```

## Python API

For more advanced use cases, consider using the Python API directly. See the examples in the `examples/` directory.
