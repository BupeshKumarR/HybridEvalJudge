# Adversarial Testing Guide

## Overview

The AdversarialTester component provides robustness testing capabilities for the LLM Judge Auditor toolkit. It generates adversarial variants of input texts with subtle factual perturbations and tests whether the evaluation system can detect these manipulations.

## Purpose

Adversarial testing helps evaluate:
- **Robustness**: Can the system detect subtle factual errors?
- **Consistency**: Does the system maintain consistent rankings?
- **Reliability**: How well does the system perform under adversarial conditions?

## Features

### Perturbation Types

The AdversarialTester supports four types of perturbations:

1. **Date Shift**: Modifies dates by small amounts (days, months, years)
   - Example: "1889" → "1890"
   - Useful for testing temporal reasoning

2. **Location Swap**: Replaces locations with similar but incorrect ones
   - Example: "Paris" → "London"
   - Tests geographic knowledge

3. **Number Change**: Alters numbers by small percentages
   - Example: "324 meters" → "340 meters"
   - Tests numerical accuracy

4. **Entity Replace**: Swaps named entities with similar ones
   - Example: "Einstein" → "Newton"
   - Tests entity recognition

### Testing Capabilities

1. **Robustness Testing**: Generate perturbations and measure detection rates
2. **Symmetry Testing**: Verify pairwise ranking consistency (A vs B = B vs A)
3. **Detailed Reporting**: Get comprehensive reports with detection rates by type

## Usage

### Basic Usage

```python
from llm_judge_auditor import EvaluationToolkit
from llm_judge_auditor.components import AdversarialTester

# Initialize toolkit and tester
toolkit = EvaluationToolkit.from_preset("balanced")
tester = AdversarialTester(toolkit, detection_threshold=10.0)

# Generate perturbations
text = "The Eiffel Tower was completed in 1889 in Paris."
perturbations = tester.generate_perturbations(
    text=text,
    perturbation_types=["date_shift", "location_swap"],
    num_variants=2
)

# Test robustness
source = "The Eiffel Tower was built for the 1889 World's Fair in Paris."
report = tester.test_robustness(
    source=source,
    original=text,
    perturbations=perturbations
)

print(f"Detection Rate: {report.detection_rate:.1f}%")
```

### Generating Perturbations

```python
# Generate date perturbations
date_perturbations = tester.generate_perturbations(
    text="The event happened in 1989.",
    perturbation_types=["date_shift"],
    num_variants=3
)

# Generate multiple types
mixed_perturbations = tester.generate_perturbations(
    text="Einstein developed relativity in 1905.",
    perturbation_types=["date_shift", "entity_replace"],
    num_variants=2
)

# Each perturbation is a tuple: (perturbed_text, type, changes)
for perturbed, pert_type, changes in perturbations:
    print(f"[{pert_type}] {perturbed}")
    print(f"Changes: {', '.join(changes)}")
```

### Testing Robustness

```python
# Test with custom threshold
report = tester.test_robustness(
    source=source_text,
    original=original_text,
    perturbations=perturbations,
    detection_threshold=15.0  # Higher threshold = stricter detection
)

# Access detailed results
print(f"Total Tests: {report.total_tests}")
print(f"Detected: {report.detected_count}")
print(f"Missed: {report.missed_count}")
print(f"Detection Rate: {report.detection_rate:.1f}%")

# View by-type statistics
for pert_type, stats in report.by_type.items():
    print(f"{pert_type}: {stats['detection_rate']:.1f}%")

# Examine individual results
for result in report.perturbation_results:
    print(f"Original Score: {result.original_score}")
    print(f"Perturbed Score: {result.perturbed_score}")
    print(f"Detected: {result.detected}")
```

### Testing Symmetry

```python
# Test pairwise ranking symmetry
candidate_a = "The tower was built in 1889."
candidate_b = "The tower was built in 1890."

symmetry_report = tester.test_symmetry(
    candidate_a=candidate_a,
    candidate_b=candidate_b,
    source=source_text
)

print(f"A vs B Winner: {symmetry_report.ab_winner}")
print(f"B vs A Winner: {symmetry_report.ba_winner}")
print(f"Is Symmetric: {symmetry_report.is_symmetric}")
```

## Configuration

### Detection Threshold

The detection threshold determines how much the score must drop to consider a perturbation "detected":

```python
# Strict detection (small score drops count)
tester = AdversarialTester(toolkit, detection_threshold=5.0)

# Moderate detection (default)
tester = AdversarialTester(toolkit, detection_threshold=10.0)

# Lenient detection (only large score drops count)
tester = AdversarialTester(toolkit, detection_threshold=20.0)
```

### Perturbation Variants

Control how many variants to generate per perturbation type:

```python
# Generate 1 variant per type (faster)
perturbations = tester.generate_perturbations(
    text=text,
    perturbation_types=["date_shift", "location_swap"],
    num_variants=1
)

# Generate 5 variants per type (more thorough)
perturbations = tester.generate_perturbations(
    text=text,
    perturbation_types=["date_shift", "location_swap"],
    num_variants=5
)
```

## Report Structure

### RobustnessReport

```python
@dataclass
class RobustnessReport:
    total_tests: int                    # Total perturbations tested
    detected_count: int                 # Number detected
    missed_count: int                   # Number missed
    detection_rate: float               # Percentage detected
    false_positive_rate: float          # False positive rate
    perturbation_results: List[...]     # Detailed results
    by_type: Dict[str, Dict]           # Stats by perturbation type
    metadata: Dict[str, Any]           # Test metadata
```

### PerturbationResult

```python
@dataclass
class PerturbationResult:
    original_text: str                  # Original text
    perturbed_text: str                 # Perturbed text
    perturbation_type: str              # Type of perturbation
    perturbations_applied: List[str]    # Specific changes made
    detected: bool                      # Was it detected?
    original_score: float               # Score for original
    perturbed_score: float              # Score for perturbed
    score_delta: float                  # Difference in scores
```

### SymmetryReport

```python
@dataclass
class SymmetryReport:
    candidate_a: str                    # First candidate
    candidate_b: str                    # Second candidate
    source: str                         # Source text
    ab_winner: str                      # Winner of A vs B
    ba_winner: str                      # Winner of B vs A
    is_symmetric: bool                  # Are rankings consistent?
    ab_reasoning: str                   # Reasoning for A vs B
    ba_reasoning: str                   # Reasoning for B vs A
```

## Best Practices

### 1. Choose Appropriate Perturbation Types

Select perturbation types relevant to your use case:

```python
# For historical texts
perturbation_types = ["date_shift", "entity_replace"]

# For geographic content
perturbation_types = ["location_swap"]

# For scientific/technical content
perturbation_types = ["number_change", "entity_replace"]

# Comprehensive testing
perturbation_types = ["date_shift", "location_swap", "number_change", "entity_replace"]
```

### 2. Set Appropriate Detection Thresholds

Consider your application's requirements:

```python
# High-stakes applications (medical, legal)
tester = AdversarialTester(toolkit, detection_threshold=5.0)

# General purpose
tester = AdversarialTester(toolkit, detection_threshold=10.0)

# Exploratory testing
tester = AdversarialTester(toolkit, detection_threshold=15.0)
```

### 3. Generate Sufficient Variants

More variants provide better coverage:

```python
# Quick smoke test
num_variants = 1

# Standard testing
num_variants = 3

# Thorough evaluation
num_variants = 5-10
```

### 4. Analyze Results by Type

Different perturbation types may have different detection rates:

```python
report = tester.test_robustness(source, original, perturbations)

for pert_type, stats in report.by_type.items():
    rate = stats['detection_rate']
    if rate < 70:
        print(f"Warning: Low detection rate for {pert_type}: {rate:.1f}%")
```

### 5. Test Symmetry Regularly

Ensure consistent pairwise rankings:

```python
# Test multiple pairs
pairs = [
    (candidate_a, candidate_b),
    (candidate_c, candidate_d),
    # ...
]

for a, b in pairs:
    report = tester.test_symmetry(a, b, source)
    if not report.is_symmetric:
        print(f"Warning: Asymmetric ranking detected!")
```

## Integration with Evaluation Pipeline

### During Development

```python
# Test robustness during development
def test_evaluation_robustness():
    toolkit = EvaluationToolkit.from_preset("balanced")
    tester = AdversarialTester(toolkit)
    
    # Generate test cases
    test_cases = [
        ("Source 1", "Candidate 1"),
        ("Source 2", "Candidate 2"),
        # ...
    ]
    
    for source, candidate in test_cases:
        perturbations = tester.generate_perturbations(
            text=candidate,
            perturbation_types=["date_shift", "location_swap"],
            num_variants=2
        )
        
        report = tester.test_robustness(source, candidate, perturbations)
        assert report.detection_rate >= 80.0, f"Low detection rate: {report.detection_rate}"
```

### In Production Monitoring

```python
# Periodic robustness checks
def monitor_robustness():
    toolkit = EvaluationToolkit.from_preset("balanced")
    tester = AdversarialTester(toolkit)
    
    # Sample from production data
    samples = get_production_samples(n=10)
    
    total_detection_rate = 0
    for source, candidate in samples:
        perturbations = tester.generate_perturbations(
            text=candidate,
            perturbation_types=["date_shift", "location_swap"],
            num_variants=1
        )
        
        report = tester.test_robustness(source, candidate, perturbations)
        total_detection_rate += report.detection_rate
    
    avg_rate = total_detection_rate / len(samples)
    log_metric("adversarial_detection_rate", avg_rate)
```

## Limitations

1. **Perturbation Coverage**: Only tests specific perturbation types
2. **Language Support**: Currently optimized for English text
3. **Context Dependency**: Some perturbations may not be detectable without broader context
4. **Randomness**: Results may vary due to random perturbation selection

## Examples

See `examples/adversarial_tester_example.py` for complete working examples.

## API Reference

### AdversarialTester

```python
class AdversarialTester:
    def __init__(
        self,
        evaluation_toolkit: EvaluationToolkit,
        detection_threshold: float = 10.0
    )
    
    def generate_perturbations(
        self,
        text: str,
        perturbation_types: List[str],
        num_variants: int = 1
    ) -> List[Tuple[str, str, List[str]]]
    
    def test_robustness(
        self,
        source: str,
        original: str,
        perturbations: List[Tuple[str, str, List[str]]],
        detection_threshold: Optional[float] = None
    ) -> RobustnessReport
    
    def test_symmetry(
        self,
        candidate_a: str,
        candidate_b: str,
        source: str
    ) -> SymmetryReport
```

## Related Documentation

- [Usage Guide](USAGE_GUIDE.md) - General toolkit usage
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Examples](../examples/README.md) - More examples
