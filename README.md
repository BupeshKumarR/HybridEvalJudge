# LLM Judge Auditor

Hybrid LLM Evaluation Toolkit combining specialized verifiers and judge ensembles for comprehensive evaluation of AI-generated text.

> **⚠️ Important:** This project uses a dedicated virtual environment. See [Installation](#installation) below.

## Features

- **Multi-stage Pipeline**: Combines specialized fact-checking models with judge LLM ensembles
- **Retrieval-Augmented Verification**: Optional integration with external knowledge bases
- **Local Execution**: Runs entirely locally without cloud APIs or paid services
- **Quantization Support**: 8-bit quantization for memory efficiency on consumer hardware
- **Comprehensive Reporting**: Transparent reports with provenance, reasoning, and metrics
- **Batch Processing**: Efficient evaluation of multiple outputs with error resilience
- **Property-Based Testing**: Rigorous correctness validation using Hypothesis

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Command-Line Interface](#command-line-interface)
- [Testing](#testing)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.9 or higher
- 8GB+ RAM recommended (4GB minimum with quantization)
- Optional: CUDA-capable GPU for faster inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/llm-judge-auditor.git
cd llm-judge-auditor
```

### Step 2: Set Up Virtual Environment

**On macOS/Linux:**
```bash
# Run the setup script
./setup_env.sh

# Activate the environment
source .venv/bin/activate
```

**On Windows:**
```bash
# Run the setup script
setup_env.bat

# Activate the environment
.venv\Scripts\activate.bat
```

**Manual Setup (Alternative):**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows)
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Step 3: Verify Installation

```bash
# Run tests to verify everything works
pytest

# Should see all tests passing
```

### Development Installation

For development with all optional dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Evaluation

```python
from llm_judge_auditor import EvaluationToolkit

# Initialize toolkit with a preset
toolkit = EvaluationToolkit.from_preset("fast")

# Evaluate a candidate output
result = toolkit.evaluate(
    source_text="The Eiffel Tower is located in Paris, France.",
    candidate_output="The Eiffel Tower is in London, England."
)

# View results
print(f"Score: {result.consensus_score:.2f}/100")
print(f"Issues: {len(result.flagged_issues)}")
```

### Run Example Scripts

```bash
# Simple evaluation example
python examples/simple_evaluation.py

# Batch processing example
python examples/batch_processing_example.py

# Advanced features
python examples/evaluation_toolkit_example.py
```

## Usage Examples

### Example 1: Simple Evaluation

```python
from llm_judge_auditor import EvaluationToolkit

# Initialize with fast preset (minimal resources)
toolkit = EvaluationToolkit.from_preset("fast")

# Define source and candidate
source_text = "Python was created by Guido van Rossum in 1991."
candidate_output = "Python was created by Guido van Rossum."

# Evaluate
result = toolkit.evaluate(source_text, candidate_output)

# Display results
print(f"Consensus Score: {result.consensus_score:.2f}")
print(f"Confidence: {result.report.confidence:.2f}")

# Check for issues
for issue in result.flagged_issues:
    print(f"[{issue.severity.value}] {issue.description}")
```

### Example 2: Batch Processing

```python
from llm_judge_auditor import EvaluationToolkit, EvaluationRequest

# Initialize toolkit
toolkit = EvaluationToolkit.from_preset("balanced")

# Create multiple requests
requests = [
    EvaluationRequest(
        source_text="Paris is the capital of France.",
        candidate_output="Paris is the capital of France.",
        task="factual_accuracy"
    ),
    EvaluationRequest(
        source_text="Water boils at 100°C at sea level.",
        candidate_output="Water boils at 212°F at sea level.",
        task="factual_accuracy"
    ),
]

# Process batch
batch_result = toolkit.batch_evaluate(requests, continue_on_error=True)

# View statistics
print(f"Mean score: {batch_result.statistics['mean']:.2f}")
print(f"Success rate: {batch_result.metadata['success_rate']:.1%}")

# Save results
batch_result.save_to_file("batch_results.json")
```

### Example 3: Custom Configuration

```python
from llm_judge_auditor import EvaluationToolkit, ToolkitConfig, AggregationStrategy

# Create custom configuration
config = ToolkitConfig(
    verifier_model="MiniCheck/flan-t5-base-finetuned",
    judge_models=["microsoft/Phi-3-mini-4k-instruct"],
    quantize=True,
    enable_retrieval=False,
    aggregation_strategy=AggregationStrategy.MEAN,
    batch_size=1,
    max_length=512,
)

# Initialize toolkit with custom config
toolkit = EvaluationToolkit(config)

# Use as normal
result = toolkit.evaluate(source_text, candidate_output)
```

### Example 4: Hallucination Detection

```python
from llm_judge_auditor import EvaluationToolkit

toolkit = EvaluationToolkit.from_preset("balanced")

source_text = "The Great Wall of China was built over many centuries."
candidate_output = "The Great Wall of China was completed in one year."

result = toolkit.evaluate(source_text, candidate_output)

# Check hallucination categories
print("Hallucination Categories:")
for category, count in result.report.hallucination_categories.items():
    if count > 0:
        print(f"  {category}: {count}")

# View flagged issues
for issue in result.flagged_issues:
    print(f"[{issue.severity.value}] {issue.description}")
```

## Configuration

### Presets

The toolkit includes four built-in presets:

- **fast**: Minimal resources, single lightweight judge, no retrieval
- **balanced**: Standard configuration with 2 judges and retrieval enabled
- **strict**: Full pipeline with all features and adversarial testing
- **research**: Maximum transparency with all metrics and benchmarks

```python
# Load a preset
config = ToolkitConfig.from_preset("balanced")

# Or customize a preset
config = ToolkitConfig.from_preset("fast")
config.enable_retrieval = True
config.batch_size = 4
```

### Configuration Options

```python
config = ToolkitConfig(
    # Model configuration
    verifier_model="MiniCheck/flan-t5-base-finetuned",
    judge_models=["microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-v0.1"],
    quantize=True,
    device="auto",  # "cpu", "cuda", "mps", or "auto"
    
    # Retrieval configuration
    enable_retrieval=True,
    knowledge_base_path="/path/to/kb",
    retrieval_top_k=3,
    
    # Aggregation configuration
    aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE,
    judge_weights={"Phi-3": 0.6, "Mistral": 0.4},
    disagreement_threshold=20.0,
    
    # Performance configuration
    batch_size=2,
    max_length=1024,
)
```

### Configuration Files

You can also load configuration from YAML files:

```yaml
# config/my_config.yaml
verifier_model: "MiniCheck/flan-t5-base-finetuned"
judge_models:
  - "microsoft/Phi-3-mini-4k-instruct"
quantize: true
enable_retrieval: false
aggregation_strategy: "mean"
batch_size: 1
max_length: 512
```

```python
# Load from file
config = ToolkitConfig.from_yaml("config/my_config.yaml")
toolkit = EvaluationToolkit(config)
```

## Command-Line Interface

The toolkit includes a CLI for common tasks:

### Evaluate a Single Output

```bash
llm-judge-auditor evaluate \
  --source "The capital of France is Paris." \
  --candidate "The capital of France is London." \
  --preset balanced \
  --output result.json
```

### Batch Evaluation

```bash
llm-judge-auditor batch-evaluate \
  --input requests.json \
  --preset fast \
  --output batch_results.json \
  --continue-on-error
```

### Using Custom Configuration

```bash
llm-judge-auditor evaluate \
  --source "..." \
  --candidate "..." \
  --config config/my_config.yaml \
  --output result.json
```

See [docs/CLI_USAGE.md](docs/CLI_USAGE.md) for complete CLI documentation.

## Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=llm_judge_auditor --cov-report=html
```

### Run Specific Test Types

```bash
# Unit tests only
pytest tests/unit/

# Property-based tests only
pytest tests/property/

# Integration tests only
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_evaluation_toolkit.py
```

### Property-Based Testing

The toolkit uses Hypothesis for property-based testing:

```bash
# Run property tests with more examples
pytest tests/property/ --hypothesis-show-statistics

# Run with specific seed for reproducibility
pytest tests/property/ --hypothesis-seed=12345
```

## Documentation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)**: Quick start guide for immediate usage
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)**: Comprehensive usage guide
- **[docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)**: Detailed setup instructions

### Reference
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)**: Complete API reference
- **[docs/CLI_USAGE.md](docs/CLI_USAGE.md)**: Command-line interface documentation
- **[docs/ERROR_HANDLING.md](docs/ERROR_HANDLING.md)**: Error handling guide
- **[config/README.md](config/README.md)**: Configuration guide

### Development
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Contributing guidelines
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)**: Current project status

### Example Scripts

- **[examples/simple_evaluation.py](examples/simple_evaluation.py)**: Basic evaluation (start here!)
- **[examples/batch_processing_example.py](examples/batch_processing_example.py)**: Batch processing
- **[examples/evaluation_toolkit_example.py](examples/evaluation_toolkit_example.py)**: Advanced features
- **[examples/basic_usage.py](examples/basic_usage.py)**: Data models and configuration
- **[examples/](examples/)**: Additional component examples

## Project Structure

```
llm-judge-auditor/
├── src/llm_judge_auditor/          # Main package
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Configuration system
│   ├── models.py                   # Data models
│   ├── evaluation_toolkit.py       # Main orchestrator
│   ├── cli.py                      # CLI interface
│   ├── components/                 # Core components
│   │   ├── device_manager.py       # Hardware detection
│   │   ├── model_manager.py        # Model loading
│   │   ├── retrieval_component.py  # Retrieval system
│   │   ├── specialized_verifier.py # Fact-checking
│   │   ├── judge_ensemble.py       # Judge models
│   │   ├── aggregation_engine.py   # Result aggregation
│   │   └── report_generator.py     # Report generation
│   └── utils/                      # Utilities
│       └── error_handling.py       # Error handling
├── tests/                          # Test suite
│   ├── unit/                       # Unit tests
│   ├── property/                   # Property-based tests
│   └── integration/                # Integration tests
├── config/                         # Configuration files
│   ├── default_config.yaml         # Default configuration
│   ├── presets/                    # Preset configurations
│   └── prompts/                    # Prompt templates
├── examples/                       # Usage examples
├── docs/                           # Documentation
└── [build files]                   # Setup and build files
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/llm-judge-auditor.git
cd llm-judge-auditor
./setup_env.sh
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
make format

# Check linting
make lint
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{llm_judge_auditor,
  title = {LLM Judge Auditor: Hybrid Evaluation Toolkit},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/llm-judge-auditor}
}
```

## Acknowledgments

This toolkit builds on research in:
- Fact-checking models (MiniCheck, HHEM)
- LLM-as-a-judge evaluation
- Retrieval-augmented verification

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/llm-judge-auditor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/llm-judge-auditor/discussions)
- **Documentation**: [docs/](docs/)

---

**Status**: Active Development | **Version**: 0.1.0 | **Python**: 3.9+
