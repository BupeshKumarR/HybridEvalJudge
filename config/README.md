# Configuration Files

This directory contains configuration files and templates for the LLM Judge Auditor toolkit.

## Directory Structure

```
config/
├── default_config.yaml      # Default configuration with all options
├── presets/                 # Pre-configured evaluation modes
│   ├── fast.yaml           # Fast mode: minimal processing
│   └── balanced.yaml       # Balanced mode: good accuracy
└── prompts/                # Prompt templates for evaluation tasks
    ├── factual_accuracy.txt
    ├── pairwise_ranking.txt
    └── bias_detection.txt
```

## Configuration Files

### default_config.yaml

The default configuration file contains all available options with sensible defaults. You can:

1. **Use as-is**: Load the default configuration
2. **Customize**: Copy and modify for your specific needs
3. **Override**: Load and override specific values programmatically

**Loading the default config:**

```python
from llm_judge_auditor.config import ToolkitConfig

# Load from YAML file
config = ToolkitConfig.from_yaml("config/default_config.yaml")
```

### Configuration Options

- **Model Configuration**
  - `verifier_model`: HuggingFace model for specialized verification
  - `judge_models`: List of judge models for ensemble evaluation
  - `quantize`: Enable 8-bit quantization (reduces memory usage)
  - `device`: Device to run on (cpu, cuda, mps, auto)

- **Retrieval Configuration**
  - `knowledge_base_path`: Path to knowledge base for retrieval
  - `retrieval_top_k`: Number of passages to retrieve per claim
  - `enable_retrieval`: Enable/disable retrieval-augmented verification

- **Aggregation Configuration**
  - `aggregation_strategy`: How to combine judge scores (mean, median, weighted_average, majority_vote)
  - `judge_weights`: Weights for weighted_average strategy
  - `disagreement_threshold`: Variance threshold for low-confidence flagging

- **Prompt Configuration**
  - `prompt_template_path`: Path to custom prompt templates
  - `custom_criteria`: Additional evaluation criteria

- **Performance Configuration**
  - `batch_size`: Batch size for inference
  - `max_length`: Maximum sequence length
  - `num_iterations`: Iterations for property-based testing

## Presets

Presets are pre-configured evaluation modes optimized for different use cases.

### Available Presets

#### fast
- **Use case**: Quick evaluations, limited resources
- **Features**: No retrieval, single lightweight judge (Phi-3-mini)
- **Resource usage**: Low memory, fast inference

```python
from llm_judge_auditor.config import ToolkitConfig

config = ToolkitConfig.from_preset("fast")
```

Or load from file:
```python
config = ToolkitConfig.from_yaml("config/presets/fast.yaml")
```

#### balanced
- **Use case**: Production evaluations, good accuracy
- **Features**: Retrieval enabled, 2 judges (LLaMA-3, Mistral)
- **Resource usage**: Moderate memory, reasonable speed

```python
config = ToolkitConfig.from_preset("balanced")
```

Or load from file:
```python
config = ToolkitConfig.from_yaml("config/presets/balanced.yaml")
```

### Creating Custom Presets

You can create your own preset by copying an existing preset file and modifying it:

```bash
cp config/presets/balanced.yaml config/presets/my_custom.yaml
# Edit my_custom.yaml with your preferred settings
```

Then load it:
```python
config = ToolkitConfig.from_yaml("config/presets/my_custom.yaml")
```

## Prompt Templates

Prompt templates define how evaluation tasks are presented to judge models. Templates use `{variable}` syntax for substitution.

### Available Templates

#### factual_accuracy.txt
Evaluates factual correctness of candidate outputs against source text.

**Required variables:**
- `source_text`: The ground truth reference text
- `candidate_output`: The text to evaluate
- `retrieved_context`: Additional context from knowledge base (can be empty)

#### pairwise_ranking.txt
Compares two candidate outputs to determine which is more accurate.

**Required variables:**
- `source_text`: The ground truth reference text
- `candidate_a`: First candidate to compare
- `candidate_b`: Second candidate to compare

#### bias_detection.txt
Detects bias, stereotypes, and harmful language in outputs.

**Required variables:**
- `candidate_output`: The text to analyze for bias

### Using Prompt Templates

Prompt templates are managed by the `PromptManager` class:

```python
from llm_judge_auditor.components.prompt_manager import PromptManager

pm = PromptManager()

# Get a prompt with variable substitution
prompt = pm.get_prompt(
    "factual_accuracy",
    source_text="Paris is the capital of France.",
    candidate_output="Paris is a city in France.",
    retrieved_context=""
)
```

### Customizing Prompts

You can customize prompts programmatically:

```python
custom_template = """
Your custom prompt here with {source_text} and {candidate_output}
"""

pm.customize_prompt("factual_accuracy", custom_template)
```

Or create a new template file and load it (future feature).

## Examples

### Example 1: Load default config and override specific values

```python
from llm_judge_auditor.config import ToolkitConfig

# Load default
config = ToolkitConfig.from_yaml("config/default_config.yaml")

# Override specific values
config.enable_retrieval = True
config.batch_size = 4
```

### Example 2: Use a preset with modifications

```python
# Start with fast preset
config = ToolkitConfig.from_preset("fast")

# Enable retrieval for this evaluation
config.enable_retrieval = True
config.knowledge_base_path = "/path/to/kb"
```

### Example 3: Create a completely custom config

```python
from llm_judge_auditor.config import ToolkitConfig, AggregationStrategy

config = ToolkitConfig(
    verifier_model="MiniCheck/flan-t5-large-finetuned",
    judge_models=["meta-llama/Llama-3-8B"],
    quantize=True,
    device="cuda",
    enable_retrieval=False,
    aggregation_strategy=AggregationStrategy.MEAN,
    batch_size=2,
    max_length=1024
)
```

### Example 4: Export config to YAML

```python
config = ToolkitConfig.from_preset("balanced")

# Export to YAML string
yaml_str = config.model_dump_yaml()
print(yaml_str)

# Save to file
with open("my_config.yaml", "w") as f:
    f.write(yaml_str)
```

## Best Practices

1. **Start with a preset**: Use `fast` or `balanced` as a starting point
2. **Test locally first**: Use `fast` preset to verify your setup works
3. **Enable retrieval carefully**: Retrieval requires a knowledge base and increases latency
4. **Monitor memory usage**: Adjust `quantize` and number of `judge_models` based on available RAM/VRAM
5. **Customize prompts for your domain**: The default prompts are general-purpose; domain-specific prompts may improve accuracy
6. **Version control your configs**: Keep custom configs in version control for reproducibility

## Troubleshooting

### Out of Memory Errors

- Enable quantization: `quantize: true`
- Reduce number of judge models
- Use smaller models (e.g., Phi-3-mini instead of LLaMA-3-8B)
- Reduce `max_length`

### Slow Inference

- Use `fast` preset
- Disable retrieval: `enable_retrieval: false`
- Use CPU if GPU is overloaded: `device: cpu`
- Reduce `retrieval_top_k`

### Model Loading Errors

- Check model names are correct HuggingFace identifiers
- Ensure you have internet connection for first download
- Check `cache_dir` has sufficient disk space
- Verify models are compatible with your hardware (e.g., some models require CUDA)
