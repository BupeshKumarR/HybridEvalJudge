# Reliability Validation

The Reliability Validator provides comprehensive metrics to ensure the evaluation system produces consistent, reliable results. It implements three key validation approaches:

1. **Evaluation Consistency Checking**: Ensures repeated evaluations have low variance
2. **Inter-Model Agreement**: Measures agreement between judge models using Cohen's kappa
3. **Ranking Correlation**: Validates pairwise rankings using Kendall's Tau and Spearman's rho

## Overview

Reliability validation is essential for building trust in AI evaluation systems. The `ReliabilityValidator` component helps you:

- Verify that your evaluation system produces consistent scores across multiple runs
- Measure how well your judge models agree with each other
- Validate that pairwise rankings correlate with ground truth
- Identify potential issues with model calibration or configuration

## Installation

The ReliabilityValidator is included in the core package:

```python
from llm_judge_auditor.components import ReliabilityValidator
```

## Quick Start

```python
from llm_judge_auditor.components import ReliabilityValidator

# Initialize validator
validator = ReliabilityValidator(consistency_threshold=5.0)

# Check consistency across multiple evaluations
scores = [85.0, 87.0, 84.5, 86.0, 85.5]
consistency_report = validator.check_consistency(scores)
print(f"Consistent: {consistency_report.is_consistent}")

# Check inter-model agreement
judge_scores = {
    "llama-3": [85.0, 45.0, 90.0, 30.0],
    "mistral": [80.0, 40.0, 88.0, 35.0],
    "phi-3": [82.0, 48.0, 92.0, 32.0],
}
agreement_report = validator.calculate_inter_model_agreement(judge_scores)
print(f"Cohen's kappa: {agreement_report.cohens_kappa:.3f}")

# Validate ranking correlation
predicted = [("A", "B"), ("C", "D"), ("E", "F")]
ground_truth = [("A", "B"), ("C", "D"), ("E", "F")]
ranking_report = validator.calculate_ranking_correlation(predicted, ground_truth)
print(f"Spearman's rho: {ranking_report.spearmans_rho:.3f}")
```

## Features

### 1. Evaluation Consistency Checking

Checks whether repeated evaluations of the same input produce consistent scores. Consistency is defined as variance below a threshold (default: 5 points).

**Requirements Validated**: 10.1

```python
validator = ReliabilityValidator(consistency_threshold=5.0)

# Evaluate the same input multiple times
scores = [85.0, 87.0, 84.5, 86.0, 85.5, 86.5]

report = validator.check_consistency(scores)

print(f"Mean Score: {report.mean_score:.2f}")
print(f"Variance: {report.variance:.2f}")
print(f"Std Deviation: {report.std_deviation:.2f}")
print(f"Is Consistent: {report.is_consistent}")
```

**ConsistencyReport Fields**:
- `mean_score`: Average score across all evaluations
- `variance`: Sample variance of scores
- `std_deviation`: Sample standard deviation
- `is_consistent`: True if variance < threshold
- `num_evaluations`: Number of evaluations
- `scores`: Original list of scores

**Interpretation**:
- Variance < 5: Consistent (good)
- Variance 5-10: Moderately consistent (acceptable)
- Variance > 10: Inconsistent (needs investigation)

### 2. Inter-Model Agreement (Cohen's Kappa)

Measures agreement between judge models by computing Cohen's kappa for each pair of judges. This helps identify whether judges are making similar decisions.

**Requirements Validated**: 10.4

```python
validator = ReliabilityValidator()

# Scores from three judges on the same test cases
judge_scores = {
    "llama-3-8b": [85.0, 45.0, 90.0, 30.0, 75.0],
    "mistral-7b": [80.0, 40.0, 88.0, 35.0, 70.0],
    "phi-3-mini": [82.0, 48.0, 92.0, 32.0, 73.0],
}

report = validator.calculate_inter_model_agreement(
    judge_scores,
    threshold=50.0  # Score threshold for binary classification
)

print(f"Cohen's Kappa: {report.cohens_kappa:.3f}")
print(f"Agreement Level: {report.agreement_level}")

# View pairwise agreements
for (judge_a, judge_b), kappa in report.pairwise_agreements.items():
    print(f"{judge_a} vs {judge_b}: κ = {kappa:.3f}")
```

**AgreementReport Fields**:
- `cohens_kappa`: Average Cohen's kappa across all judge pairs
- `agreement_level`: Interpretation of kappa value
- `num_models`: Number of judge models
- `pairwise_agreements`: Dictionary of kappa values for each pair

**Cohen's Kappa Interpretation**:
- < 0.00: Poor agreement
- 0.00-0.20: Slight agreement
- 0.21-0.40: Fair agreement
- 0.41-0.60: Moderate agreement
- 0.61-0.80: Substantial agreement
- 0.81-1.00: Almost perfect agreement

**Best Practices**:
- Aim for κ > 0.6 (substantial agreement)
- If κ < 0.4, consider recalibrating judges or adjusting prompts
- Use the same threshold across all evaluations for consistency

### 3. Ranking Correlation

Validates pairwise rankings by comparing predicted rankings against ground truth using Kendall's Tau and Spearman's rho.

**Requirements Validated**: 10.5

```python
validator = ReliabilityValidator()

# Predicted rankings from your system
predicted = [
    ("Model_A", "Model_B"),  # Model_A beats Model_B
    ("Model_C", "Model_D"),  # Model_C beats Model_D
    ("Model_E", "Model_F"),  # Model_E beats Model_F
]

# Ground truth rankings
ground_truth = [
    ("Model_A", "Model_B"),
    ("Model_C", "Model_D"),
    ("Model_E", "Model_F"),
]

report = validator.calculate_ranking_correlation(predicted, ground_truth)

print(f"Kendall's Tau: {report.kendalls_tau:.3f} (p={report.kendalls_tau_p_value:.4f})")
print(f"Spearman's Rho: {report.spearmans_rho:.3f} (p={report.spearmans_rho_p_value:.4f})")
print(f"Statistically Significant: {report.is_significant}")
```

**RankingCorrelationReport Fields**:
- `kendalls_tau`: Kendall's Tau correlation coefficient
- `kendalls_tau_p_value`: Statistical significance (p-value)
- `spearmans_rho`: Spearman's rho correlation coefficient
- `spearmans_rho_p_value`: Statistical significance (p-value)
- `num_pairs`: Number of ranking pairs
- `is_significant`: True if both p-values < 0.05

**Interpretation**:
- τ or ρ > 0.7: Strong correlation (good)
- τ or ρ 0.4-0.7: Moderate correlation (acceptable)
- τ or ρ < 0.4: Weak correlation (needs improvement)
- p-value < 0.05: Statistically significant

## Integration with Evaluation Toolkit

The ReliabilityValidator can be integrated into your evaluation workflow:

```python
from llm_judge_auditor import EvaluationToolkit
from llm_judge_auditor.components import ReliabilityValidator

# Initialize toolkit and validator
toolkit = EvaluationToolkit.from_preset("balanced")
validator = ReliabilityValidator()

# Run multiple evaluations
source = "Paris is the capital of France."
candidate = "Paris is the capital city of France."

scores = []
for _ in range(5):
    result = toolkit.evaluate(source, candidate)
    scores.append(result.consensus_score)

# Check consistency
consistency_report = validator.check_consistency(scores)
print(f"Evaluation consistency: {consistency_report.is_consistent}")

# Check inter-model agreement
judge_scores = {
    judge.model_name: [judge.score]
    for judge in result.judge_results
}
# Collect more scores...
agreement_report = validator.calculate_inter_model_agreement(judge_scores)
print(f"Judge agreement: {agreement_report.agreement_level}")
```

## Comprehensive Validation Example

```python
from llm_judge_auditor.components import ReliabilityValidator

def validate_evaluation_system():
    """Comprehensive reliability validation."""
    validator = ReliabilityValidator()
    
    # 1. Consistency Check
    print("1. Checking evaluation consistency...")
    scores = [84.5, 86.0, 85.5, 87.0, 85.0]
    consistency = validator.check_consistency(scores)
    
    if consistency.is_consistent:
        print(f"   ✓ PASS: Variance = {consistency.variance:.2f}")
    else:
        print(f"   ✗ FAIL: Variance = {consistency.variance:.2f}")
    
    # 2. Inter-Model Agreement
    print("\n2. Checking inter-model agreement...")
    judge_scores = {
        "judge_1": [85.0, 45.0, 90.0, 30.0, 75.0],
        "judge_2": [80.0, 40.0, 88.0, 35.0, 70.0],
        "judge_3": [82.0, 48.0, 92.0, 32.0, 73.0],
    }
    agreement = validator.calculate_inter_model_agreement(judge_scores)
    
    if agreement.cohens_kappa > 0.6:
        print(f"   ✓ PASS: κ = {agreement.cohens_kappa:.3f} ({agreement.agreement_level})")
    else:
        print(f"   ✗ FAIL: κ = {agreement.cohens_kappa:.3f} ({agreement.agreement_level})")
    
    # 3. Ranking Correlation
    print("\n3. Checking ranking correlation...")
    predicted = [("A", "B"), ("C", "D"), ("E", "F")]
    ground_truth = [("A", "B"), ("C", "D"), ("E", "F")]
    ranking = validator.calculate_ranking_correlation(predicted, ground_truth)
    
    if ranking.spearmans_rho > 0.7:
        print(f"   ✓ PASS: ρ = {ranking.spearmans_rho:.3f}")
    else:
        print(f"   ✗ FAIL: ρ = {ranking.spearmans_rho:.3f}")
    
    # Overall assessment
    print("\n" + "=" * 60)
    all_pass = (
        consistency.is_consistent and
        agreement.cohens_kappa > 0.6 and
        ranking.spearmans_rho > 0.7
    )
    
    if all_pass:
        print("✓ SYSTEM RELIABLE")
    else:
        print("✗ SYSTEM NEEDS IMPROVEMENT")
    
    return all_pass

if __name__ == "__main__":
    validate_evaluation_system()
```

## Troubleshooting

### Low Consistency (High Variance)

**Problem**: Repeated evaluations produce inconsistent scores.

**Possible Causes**:
- Temperature setting too high (increase randomness)
- Non-deterministic model behavior
- Insufficient context or ambiguous inputs

**Solutions**:
- Lower temperature (e.g., 0.1 or 0.0)
- Use deterministic sampling (top_k=1)
- Provide more context in prompts
- Run more evaluations and use median instead of mean

### Low Inter-Model Agreement

**Problem**: Judge models disagree frequently (κ < 0.4).

**Possible Causes**:
- Models have different calibration
- Prompts are ambiguous
- Models have different strengths/weaknesses

**Solutions**:
- Recalibrate models with consistent prompts
- Use more specific evaluation criteria
- Consider weighted aggregation based on model strengths
- Fine-tune models on the same training data

### Low Ranking Correlation

**Problem**: Rankings don't match ground truth (ρ < 0.4).

**Possible Causes**:
- System bias or miscalibration
- Ground truth may be incorrect
- Evaluation criteria mismatch

**Solutions**:
- Validate ground truth labels
- Adjust evaluation criteria to match ground truth
- Use adversarial testing to identify systematic errors
- Consider ensemble methods to improve ranking quality

## API Reference

### ReliabilityValidator

```python
class ReliabilityValidator:
    def __init__(self, consistency_threshold: float = 5.0)
    
    def check_consistency(
        self,
        scores: List[float]
    ) -> ConsistencyReport
    
    def calculate_inter_model_agreement(
        self,
        judge_scores: Dict[str, List[float]],
        threshold: float = 50.0
    ) -> AgreementReport
    
    def calculate_ranking_correlation(
        self,
        predicted_rankings: List[Tuple[str, str]],
        ground_truth_rankings: List[Tuple[str, str]]
    ) -> RankingCorrelationReport
```

### Data Classes

```python
@dataclass
class ConsistencyReport:
    mean_score: float
    variance: float
    std_deviation: float
    is_consistent: bool
    num_evaluations: int
    scores: List[float]

@dataclass
class AgreementReport:
    cohens_kappa: float
    agreement_level: str
    num_models: int
    pairwise_agreements: Dict[Tuple[str, str], float]

@dataclass
class RankingCorrelationReport:
    kendalls_tau: float
    kendalls_tau_p_value: float
    spearmans_rho: float
    spearmans_rho_p_value: float
    num_pairs: int
    is_significant: bool
```

## Best Practices

1. **Regular Validation**: Run reliability checks periodically, especially after:
   - Model updates or changes
   - Prompt modifications
   - Configuration changes

2. **Baseline Establishment**: Establish baseline metrics for your system:
   - Target variance < 5 for consistency
   - Target κ > 0.6 for agreement
   - Target ρ > 0.7 for ranking correlation

3. **Continuous Monitoring**: Track metrics over time to detect degradation

4. **Multiple Metrics**: Don't rely on a single metric; use all three validation approaches

5. **Statistical Significance**: Pay attention to p-values; ensure correlations are statistically significant

## Related Documentation

- [Evaluation Toolkit](USAGE_GUIDE.md)
- [Adversarial Testing](ADVERSARIAL_TESTING.md)
- [Error Handling](ERROR_HANDLING.md)

## References

- Cohen, J. (1960). "A coefficient of agreement for nominal scales"
- Kendall, M. G. (1938). "A new measure of rank correlation"
- Spearman, C. (1904). "The proof and measurement of association between two things"
