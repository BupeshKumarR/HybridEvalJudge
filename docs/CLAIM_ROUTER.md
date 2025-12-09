# Claim Router

The ClaimRouter component provides intelligent routing of claims to specialized judges based on claim type classification. This enables more accurate evaluations by matching claims to judges with expertise in specific domains.

## Overview

The ClaimRouter analyzes claims to determine their type (numerical, temporal, logical, commonsense, or factual) and routes them to judges that specialize in evaluating those types of claims. This improves evaluation quality by leveraging judge specializations.

## Features

- **Automatic Claim Classification**: Classifies claims into five types based on content analysis
- **Specialized Routing**: Routes claims to judges with relevant expertise
- **Batch Processing**: Efficiently routes multiple claims and groups them by assigned judge
- **Dynamic Configuration**: Update judge specializations at runtime
- **Fallback Handling**: Gracefully handles cases where no specialist is available

## Claim Types

The ClaimRouter classifies claims into the following types:

1. **NUMERICAL**: Claims containing numbers, quantities, measurements, percentages, or currency
   - Example: "The unemployment rate increased by 5.2% last quarter."

2. **TEMPORAL**: Claims involving dates, times, or temporal references
   - Example: "The Eiffel Tower was completed in 1889."

3. **LOGICAL**: Claims with logical reasoning, conditionals, or causal relationships
   - Example: "If the temperature rises, then the ice will melt."

4. **COMMONSENSE**: Claims requiring common sense reasoning about human behavior or everyday concepts
   - Example: "People generally feel happier when the weather is warm."

5. **FACTUAL**: General factual statements (default category)
   - Example: "Paris is the capital of France."

## Usage

### Basic Initialization

```python
from llm_judge_auditor.components import ClaimRouter
from llm_judge_auditor.models import Claim, ClaimType

# Initialize without specializations (general-purpose judges)
router = ClaimRouter()

# Initialize with judge specializations
specializations = {
    "llama-3-8b": ["factual", "logical"],
    "mistral-7b": ["numerical", "temporal"],
    "phi-3-mini": ["commonsense"],
}
router = ClaimRouter(specializations)
```

### Classifying Claims

```python
# Create a claim
claim = Claim(
    text="The temperature was 25 degrees Celsius.",
    source_span=(0, 40)
)

# Classify the claim
claim_type = router.classify_claim(claim)
print(f"Claim type: {claim_type}")  # Output: ClaimType.NUMERICAL
```

### Routing to Judges

```python
# Route a single claim to the best judge
available_judges = ["llama-3-8b", "mistral-7b", "phi-3-mini"]
selected_judge = router.route_to_judge(claim, available_judges)
print(f"Selected judge: {selected_judge}")  # Output: mistral-7b
```

### Batch Routing

```python
# Route multiple claims at once
claims = [
    Claim(text="The temperature was 25 degrees.", source_span=(0, 31)),
    Claim(text="Paris is the capital of France.", source_span=(0, 32)),
    Claim(text="The event happened in 1989.", source_span=(0, 27)),
]

# Get routing assignments
routing = router.route_claims_to_judges(claims, available_judges)

# Process claims by judge
for judge_name, judge_claims in routing.items():
    if judge_claims:
        print(f"{judge_name}: {len(judge_claims)} claims")
        # Evaluate claims with this judge...
```

### Managing Specializations

```python
# Get specializations for a judge
specs = router.get_judge_specializations("mistral-7b")
print(f"Mistral-7B specializes in: {specs}")

# Get judges specialized in a claim type
numerical_judges = router.get_specialized_judges(ClaimType.NUMERICAL)
print(f"Numerical specialists: {numerical_judges}")

# Update specializations dynamically
router.update_specializations("llama-3-8b", ["factual", "logical", "numerical"])
```

## Integration with Evaluation Pipeline

The ClaimRouter can be integrated into the evaluation pipeline to improve accuracy:

```python
from llm_judge_auditor import EvaluationToolkit
from llm_judge_auditor.components import ClaimRouter, RetrievalComponent

# Initialize components
toolkit = EvaluationToolkit.from_preset("balanced")
router = ClaimRouter({
    "llama-3-8b": ["factual", "logical"],
    "mistral-7b": ["numerical", "temporal"],
})

# Extract claims from candidate output
retrieval = RetrievalComponent()
claims = retrieval.extract_claims(candidate_output)

# Route claims to specialized judges
available_judges = list(toolkit.model_manager.get_all_judges().keys())
routing = router.route_claims_to_judges(claims, available_judges)

# Evaluate each claim with its assigned judge
for judge_name, judge_claims in routing.items():
    for claim in judge_claims:
        result = toolkit.judge_ensemble.evaluate_single(
            judge_name=judge_name,
            source_text=source_text,
            candidate_output=claim.text
        )
        # Process result...
```

## Configuration

Judge specializations can be configured in the toolkit configuration:

```yaml
models:
  judges:
    - name: "meta-llama/Meta-Llama-3-8B-Instruct"
      weight: 0.4
      specialization: ["factual", "logical"]
    - name: "mistralai/Mistral-7B-Instruct-v0.2"
      weight: 0.35
      specialization: ["numerical", "temporal"]
    - name: "microsoft/Phi-3-mini-4k-instruct"
      weight: 0.25
      specialization: ["commonsense"]

aggregation:
  use_claim_routing: true  # Enable claim routing
```

## Classification Algorithm

The ClaimRouter uses pattern matching to classify claims:

1. **Strong Logical Patterns**: Checks for if-then constructs and logical connectives first
2. **Specific Numerical Patterns**: Looks for percentages, measurements, currency
3. **Temporal Patterns**: Identifies dates, times, temporal references
4. **Commonsense Patterns**: Detects human behavior and everyday concepts
5. **Weak Logical Patterns**: Checks for causal relationships and quantifiers
6. **Generic Numbers**: Falls back to numerical for any remaining numbers
7. **Default**: Classifies as factual if no other patterns match

This hierarchical approach ensures accurate classification by checking more specific patterns before generic ones.

## Benefits

- **Improved Accuracy**: Specialized judges perform better on claims matching their expertise
- **Efficient Resource Use**: Distributes evaluation workload based on judge strengths
- **Flexible Configuration**: Easy to add new judges or update specializations
- **Graceful Degradation**: Works with general-purpose judges when specialists unavailable

## Requirements

Validates: Requirements 2.3 (specialized judge selection)

## See Also

- [Judge Ensemble Documentation](USAGE_GUIDE.md#judge-ensemble)
- [Evaluation Toolkit API](API_REFERENCE.md#evaluation-toolkit)
- [Configuration Guide](../config/README.md)
