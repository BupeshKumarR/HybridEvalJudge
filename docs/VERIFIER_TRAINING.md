# Verifier Training Guide

This guide explains how to fine-tune specialized verifier models for fact-checking using the VerifierTrainer class.

## Overview

The VerifierTrainer enables you to fine-tune small models (< 1B parameters) on fact-checking datasets for statement-level verification. The trained models perform three-way classification:

- **SUPPORTED**: The claim is supported by the evidence
- **REFUTED**: The claim contradicts the evidence
- **NOT_ENOUGH_INFO**: Insufficient information to verify the claim

## Quick Start

```python
from llm_judge_auditor.components.verifier_trainer import VerifierTrainer

# Initialize trainer
trainer = VerifierTrainer(
    base_model="google/flan-t5-base",
    output_dir="models/my_verifier"
)

# Load training data (FEVER format)
train_data = trainer.load_fever_dataset("data/fever_train.jsonl")
eval_data = trainer.load_fever_dataset("data/fever_dev.jsonl")

# Train the model
metrics = trainer.train(
    train_data=train_data,
    eval_data=eval_data,
    num_epochs=3,
    batch_size=16
)

# Evaluate on test set
test_data = trainer.load_fever_dataset("data/fever_test.jsonl")
test_metrics = trainer.evaluate(test_data)

print(f"Test Accuracy: {test_metrics['eval_accuracy']:.3f}")
print(f"Test F1: {test_metrics['eval_f1']:.3f}")

# Save the trained model
trainer.save_model("models/my_verifier/final")
```

## Data Formats

### FEVER Format

The FEVER format is a JSONL file where each line contains:

```json
{
  "claim": "The Eiffel Tower is in Paris.",
  "evidence": "The Eiffel Tower is located in Paris, France.",
  "label": "SUPPORTS"
}
```

**Supported labels:**
- `"SUPPORTS"` → SUPPORTED
- `"REFUTES"` → REFUTED
- `"NOT ENOUGH INFO"` → NOT_ENOUGH_INFO

### Custom JSON Format

You can also use a JSON array format:

```json
[
  {
    "claim": "Water boils at 100°C at sea level.",
    "evidence": "At standard atmospheric pressure, water boils at 100°C.",
    "label": "SUPPORTED"
  },
  {
    "claim": "Water freezes at 10°C.",
    "evidence": "Water freezes at 0°C at standard pressure.",
    "label": "REFUTED"
  }
]
```

Load custom data with:

```python
train_data = trainer.load_custom_dataset("data/custom.json", format="json")
```

### Programmatic Data Creation

Create training examples directly in code:

```python
from llm_judge_auditor.components.verifier_trainer import TrainingExample
from llm_judge_auditor.models import VerdictLabel

training_examples = [
    TrainingExample(
        claim="The Earth orbits the Sun.",
        evidence="The Earth revolves around the Sun in an elliptical orbit.",
        label=VerdictLabel.SUPPORTED
    ),
    TrainingExample(
        claim="The Sun orbits the Earth.",
        evidence="The Earth revolves around the Sun, not vice versa.",
        label=VerdictLabel.REFUTED
    )
]

trainer.train(train_data=training_examples, num_epochs=3)
```

## Training Configuration

### Basic Parameters

```python
trainer.train(
    train_data=train_data,           # Required: training examples
    eval_data=eval_data,              # Optional: evaluation examples
    num_epochs=3,                     # Number of training epochs
    batch_size=16,                    # Training batch size
    learning_rate=5e-5,               # Learning rate
    warmup_steps=500,                 # Warmup steps
    save_steps=1000,                  # Save checkpoint every N steps
    eval_steps=500,                   # Evaluate every N steps
    logging_steps=100                 # Log every N steps
)
```

### Recommended Settings

**For small datasets (< 1K examples):**
```python
trainer.train(
    train_data=train_data,
    num_epochs=5,
    batch_size=8,
    learning_rate=3e-5,
    warmup_steps=100
)
```

**For medium datasets (1K-10K examples):**
```python
trainer.train(
    train_data=train_data,
    num_epochs=3,
    batch_size=16,
    learning_rate=5e-5,
    warmup_steps=500
)
```

**For large datasets (> 10K examples):**
```python
trainer.train(
    train_data=train_data,
    num_epochs=2,
    batch_size=32,
    learning_rate=5e-5,
    warmup_steps=1000
)
```

## Model Selection

### Recommended Base Models

**Small models (< 500M parameters):**
- `google/flan-t5-small` (80M) - Fast, good for prototyping
- `google/flan-t5-base` (250M) - Balanced performance

**Medium models (500M-1B parameters):**
- `google/flan-t5-large` (770M) - High accuracy, similar to MiniCheck
- `facebook/bart-base` (140M) - Good for longer sequences

**Example:**
```python
# For production use
trainer = VerifierTrainer(base_model="google/flan-t5-large")

# For development/testing
trainer = VerifierTrainer(base_model="google/flan-t5-small")
```

## Evaluation Metrics

The trainer reports four key metrics:

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision across all classes (macro-averaged)
- **Recall**: Recall across all classes (macro-averaged)
- **F1 Score**: F1 score across all classes (macro-averaged)

```python
metrics = trainer.evaluate(test_data)

print(f"Accuracy:  {metrics['eval_accuracy']:.3f}")
print(f"Precision: {metrics['eval_precision']:.3f}")
print(f"Recall:    {metrics['eval_recall']:.3f}")
print(f"F1 Score:  {metrics['eval_f1']:.3f}")
```

## Saving and Loading Models

### Saving

```python
# Save after training
trainer.save_model("models/my_verifier/final")

# The model is also auto-saved during training to output_dir
```

### Loading

```python
# Load a trained model
trainer = VerifierTrainer()
trainer.load_model("models/my_verifier/final")

# Now you can evaluate or use it for inference
metrics = trainer.evaluate(test_data)
```

### Using with ModelManager

```python
from llm_judge_auditor.components.model_manager import ModelManager
from llm_judge_auditor.config import ToolkitConfig

# Configure to use your trained model
config = ToolkitConfig(
    verifier_model="models/my_verifier/final",
    judge_models=[],
    quantize=False
)

manager = ModelManager(config)
model, tokenizer = manager.load_verifier()
```

## Best Practices

### 1. Data Quality

- **Balance your dataset**: Aim for roughly equal numbers of each label
- **Diverse evidence**: Include varied evidence types and domains
- **Quality over quantity**: 1K high-quality examples > 10K noisy examples

### 2. Training Strategy

- **Start small**: Begin with a small model and dataset to validate your pipeline
- **Use validation data**: Always evaluate on held-out data during training
- **Monitor metrics**: Watch for overfitting (train accuracy >> eval accuracy)
- **Early stopping**: The trainer automatically saves the best model based on F1 score

### 3. Hyperparameter Tuning

- **Learning rate**: Start with 5e-5, decrease if training is unstable
- **Batch size**: Larger is better, but limited by GPU memory
- **Epochs**: 2-5 epochs is usually sufficient; more may cause overfitting

### 4. Domain Adaptation

For domain-specific fact-checking:

1. Start with a model pre-trained on FEVER
2. Fine-tune on your domain data
3. Use domain-specific evidence sources

```python
# Load pre-trained FEVER model
trainer = VerifierTrainer(base_model="path/to/fever_model")

# Fine-tune on domain data
domain_data = trainer.load_custom_dataset("medical_claims.json")
trainer.train(domain_data, num_epochs=2, learning_rate=3e-5)
```

## Troubleshooting

### Out of Memory Errors

**Solution 1: Reduce batch size**
```python
trainer.train(train_data=data, batch_size=4)  # Instead of 16
```

**Solution 2: Use a smaller model**
```python
trainer = VerifierTrainer(base_model="google/flan-t5-small")
```

**Solution 3: Reduce sequence length**
```python
dataset = VerifierDataset(examples, tokenizer, max_length=256)  # Instead of 512
```

### Poor Performance

**Check label distribution:**
```python
from collections import Counter
labels = [ex.label for ex in train_data]
print(Counter(labels))
```

**Increase training data:**
- Aim for at least 500 examples per class
- Use data augmentation if needed

**Adjust learning rate:**
```python
# Try lower learning rate
trainer.train(train_data=data, learning_rate=3e-5)
```

### Training Too Slow

**Use GPU:**
```python
trainer = VerifierTrainer(device="cuda")
```

**Increase batch size:**
```python
trainer.train(train_data=data, batch_size=32)
```

**Reduce evaluation frequency:**
```python
trainer.train(train_data=data, eval_steps=1000)  # Instead of 500
```

## Advanced Usage

### Custom Training Loop

For more control, you can access the underlying components:

```python
from llm_judge_auditor.components.verifier_trainer import VerifierDataset
from transformers import Trainer, TrainingArguments

# Create dataset
dataset = VerifierDataset(train_data, trainer.tokenizer)

# Custom training arguments
args = TrainingArguments(
    output_dir="custom_output",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    fp16=True,  # Mixed precision training
    # ... other arguments
)

# Create custom trainer
custom_trainer = Trainer(
    model=trainer.model,
    args=args,
    train_dataset=dataset,
    compute_metrics=trainer._compute_metrics
)

custom_trainer.train()
```

### Multi-GPU Training

```python
# The trainer automatically uses all available GPUs
# Just ensure your batch size is appropriate

trainer.train(
    train_data=train_data,
    batch_size=16,  # Per-device batch size
    # Effective batch size = 16 * num_gpus
)
```

## Integration with Evaluation Toolkit

After training, integrate your verifier with the main toolkit:

```python
from llm_judge_auditor import EvaluationToolkit
from llm_judge_auditor.config import ToolkitConfig

# Configure with your trained verifier
config = ToolkitConfig(
    verifier_model="models/my_verifier/final",
    judge_models=["meta-llama/Meta-Llama-3-8B-Instruct"],
    quantize=True
)

# Create toolkit
toolkit = EvaluationToolkit(config)

# Evaluate
result = toolkit.evaluate(
    source_text="The Eiffel Tower was completed in 1889.",
    candidate_output="The Eiffel Tower was built in 1889 in Paris."
)

print(f"Score: {result.consensus_score}")
```

## Example Datasets

### FEVER Dataset

Download from: https://fever.ai/dataset/fever.html

```bash
# Download FEVER
wget https://fever.ai/download/fever/train.jsonl
wget https://fever.ai/download/fever/dev.jsonl
wget https://fever.ai/download/fever/test.jsonl
```

### Creating Synthetic Data

For domain-specific applications, you can generate synthetic training data:

```python
import random

def generate_synthetic_examples(num_examples=1000):
    """Generate synthetic fact-checking examples."""
    examples = []
    
    # Define templates
    templates = [
        ("The {entity} is located in {location}.", "SUPPORTED"),
        ("The {entity} was built in {year}.", "REFUTED"),
        ("The {entity} has {attribute}.", "NOT_ENOUGH_INFO"),
    ]
    
    entities = ["Eiffel Tower", "Statue of Liberty", "Great Wall"]
    locations = ["Paris", "New York", "China"]
    years = ["1889", "1886", "220 BC"]
    
    for _ in range(num_examples):
        template, label = random.choice(templates)
        claim = template.format(
            entity=random.choice(entities),
            location=random.choice(locations),
            year=random.choice(years),
            attribute="interesting features"
        )
        evidence = f"Historical information about {random.choice(entities)}."
        
        examples.append(TrainingExample(
            claim=claim,
            evidence=evidence,
            label=VerdictLabel[label.replace(" ", "_")]
        ))
    
    return examples
```

## Performance Benchmarks

Expected performance on FEVER dev set:

| Model | Parameters | Accuracy | F1 Score | Training Time* |
|-------|-----------|----------|----------|----------------|
| flan-t5-small | 80M | ~0.65 | ~0.62 | ~30 min |
| flan-t5-base | 250M | ~0.72 | ~0.70 | ~1 hour |
| flan-t5-large | 770M | ~0.78 | ~0.76 | ~3 hours |

*Training time on single V100 GPU with 10K examples, 3 epochs

## References

- [FEVER Dataset Paper](https://arxiv.org/abs/1803.05355)
- [MiniCheck Paper](https://arxiv.org/abs/2404.10774)
- [T5 Model](https://arxiv.org/abs/1910.10683)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## Next Steps

1. **Prepare your data**: Convert to FEVER format or use custom format
2. **Train a model**: Start with a small model and dataset
3. **Evaluate**: Test on held-out data
4. **Integrate**: Use with the EvaluationToolkit
5. **Iterate**: Improve based on error analysis

For more examples, see `examples/verifier_trainer_example.py`.
