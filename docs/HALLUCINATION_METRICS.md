# Hallucination Quantification Metrics

This document describes the research-backed hallucination quantification metrics implemented in the LLM Judge Auditor toolkit.

## Overview

The toolkit provides comprehensive hallucination quantification through:

- **MiHR (Micro Hallucination Rate)**: Claim-level hallucination measurement
- **MaHR (Macro Hallucination Rate)**: Response-level hallucination measurement
- **FactScore**: Factual precision metric
- **Consensus F1**: Cross-model agreement metric
- **Fleiss' Kappa**: Inter-judge agreement statistic
- **Uncertainty Quantification**: Shannon entropy with epistemic/aleatoric decomposition
- **False Acceptance Rate**: Model abstention behavior measurement

## Quick Start

```python
from llm_judge_auditor.components.hallucination_metrics import (
    HallucinationMetricsCalculator,
    ClaimVerificationMatrixBuilder,
    FalseAcceptanceCalculator,
)
from llm_judge_auditor.models import Verdict, VerdictLabel

# Initialize calculator
calculator = HallucinationMetricsCalculator()

# Compute MiHR from verdicts
verdicts = [
    Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
    Verdict(VerdictLabel.REFUTED, 0.85, [], "Refuted"),
    Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.7, [], "NEI"),
]
mihr = calculator.compute_mihr(verdicts)
print(f"MiHR: {mihr.value:.2%}")  # 66.67% (2/3 unsupported)
```

## Metrics Reference

### MiHR (Micro Hallucination Rate)

**Formula**: `MiHR = unsupported_claims / total_claims`

MiHR measures the fraction of individual claims that are not supported by evidence.

```python
mihr_result = calculator.compute_mihr(verdicts)
print(f"MiHR: {mihr_result.value:.2%}")
print(f"Unsupported: {mihr_result.unsupported_claims}/{mihr_result.total_claims}")
```

**Interpretation**:
- 0% = All claims supported (no hallucinations)
- 100% = All claims unsupported (complete hallucination)
- Threshold: MiHR > 30% is considered high risk

**Edge Cases**:
- Zero claims: Returns `None` with `has_claims=False`

### MaHR (Macro Hallucination Rate)

**Formula**: `MaHR = responses_with_hallucinations / total_responses`

MaHR measures the fraction of responses that contain at least one hallucination.

```python
response_verdicts = [
    [Verdict(VerdictLabel.SUPPORTED, 0.9, [], "")],  # No hallucination
    [Verdict(VerdictLabel.REFUTED, 0.85, [], "")],   # Has hallucination
]
mahr_result = calculator.compute_mahr(response_verdicts)
print(f"MaHR: {mahr_result.value:.2%}")  # 50%
```

**Interpretation**:
- 0% = No responses contain hallucinations
- 100% = All responses contain hallucinations

### FactScore

**Formula**: `FactScore = verified_claims / total_claims`

FactScore measures factual precision - the fraction of claims that are verified as supported.

```python
factscore = calculator.compute_factscore(verdicts)
print(f"FactScore: {factscore:.2%}")
```

**Interpretation**:
- 100% = All claims verified (perfect factual accuracy)
- 0% = No claims verified

### Consensus F1

**Formula**: 
- `Precision = model_claims_supported_by_others / model_claims`
- `Recall = consensus_claims_included / total_consensus_claims`
- `F1 = 2 × (precision × recall) / (precision + recall)`

Consensus F1 measures cross-model agreement on claims.

```python
from llm_judge_auditor.models import Claim, ClaimType

# Build claim verification matrix
model_claims = {
    "model_a": [Claim("Paris is the capital", (0, 20), ClaimType.FACTUAL)],
    "model_b": [Claim("Paris is the capital", (0, 20), ClaimType.FACTUAL)],
}
matrix_builder = ClaimVerificationMatrixBuilder()
matrix = matrix_builder.build_matrix(model_claims)

# Compute F1 for a specific model
f1_result = calculator.compute_consensus_f1(matrix, "model_a")
print(f"F1: {f1_result.f1:.2%}")
```

### Fleiss' Kappa

**Formula**: `κ = (Po - Pe) / (1 - Pe)`

Where:
- Po = observed agreement among raters
- Pe = expected agreement by chance

```python
judge_verdicts = {
    "judge_1": [Verdict(VerdictLabel.SUPPORTED, 0.9, [], "")],
    "judge_2": [Verdict(VerdictLabel.SUPPORTED, 0.85, [], "")],
    "judge_3": [Verdict(VerdictLabel.REFUTED, 0.8, [], "")],
}
kappa_result = calculator.compute_fleiss_kappa_from_verdicts(judge_verdicts)
print(f"Kappa: {kappa_result.kappa:.4f}")
print(f"Interpretation: {kappa_result.interpretation}")
```

**Interpretation Scale**:
| Kappa Value | Interpretation |
|-------------|----------------|
| < 0.2 | Poor agreement |
| 0.2 - 0.4 | Fair agreement |
| 0.4 - 0.6 | Moderate agreement |
| 0.6 - 0.8 | Substantial agreement |
| > 0.8 | Almost perfect agreement |

**Edge Cases**:
- Fewer than 2 judges: Returns undefined with error message

### Uncertainty Quantification

**Shannon Entropy**: `H(p) = -Σ pᵢ log pᵢ`

**Epistemic Uncertainty**: `Var(E[p])` across inference samples (model uncertainty)

**Aleatoric Uncertainty**: `E[Var(p)]` within inference samples (data noise)

**Total Uncertainty**: `epistemic + aleatoric`

```python
# Basic entropy
probabilities = [0.7, 0.2, 0.1]
uncertainty = calculator.compute_uncertainty(probabilities)
print(f"Shannon entropy: {uncertainty.shannon_entropy:.4f}")

# With epistemic/aleatoric decomposition
inference_samples = [
    [0.7, 0.2, 0.1],
    [0.65, 0.25, 0.1],
    [0.72, 0.18, 0.1],
]
full_uncertainty = calculator.compute_uncertainty(probabilities, inference_samples)
print(f"Epistemic: {full_uncertainty.epistemic:.6f}")
print(f"Aleatoric: {full_uncertainty.aleatoric:.6f}")
print(f"Total: {full_uncertainty.total:.6f}")
print(f"High uncertainty: {full_uncertainty.is_high_uncertainty}")
```

**Interpretation**:
- High epistemic uncertainty = model is unsure (may hallucinate)
- High aleatoric uncertainty = inherent ambiguity in the data
- Threshold: Total uncertainty > 0.8 is flagged as high risk

### False Acceptance Rate (FAR)

**Formula**: `FAR = failed_abstentions / total_nonexistent_queries`

FAR measures how often a model fails to abstain when asked about non-existent entities.

```python
far_calculator = FalseAcceptanceCalculator()

# Evaluate a single query
result = far_calculator.evaluate_abstention(
    query="Who is Dr. Fake Person?",
    response="Dr. Fake Person was a scientist...",
    is_nonexistent=True
)
print(f"False acceptance: {result.is_false_acceptance}")

# Compute FAR across multiple queries
far_result = far_calculator.evaluate_and_compute_far(
    queries=["Who is X?", "Who is Y?"],
    responses=["X was...", "I don't know about Y."],
    is_nonexistent_flags=[True, True]
)
print(f"FAR: {far_result.value:.2%}")
```

**Interpretation**:
- 0% = Perfect abstention (model always refuses non-existent queries)
- 100% = Complete failure (model always generates content)

## Hallucination Profile

The `HallucinationProfile` combines all metrics into a comprehensive report:

```python
profile = calculator.generate_hallucination_profile(
    verdicts=verdicts,
    response_verdicts=response_verdicts,
    claim_matrix=matrix,
    judge_verdicts=judge_verdicts,
    probabilities=probabilities,
    inference_samples=inference_samples,
)

print(f"MiHR: {profile.mihr.value:.2%}")
print(f"MaHR: {profile.mahr.value:.2%}")
print(f"FactScore: {profile.factscore:.2%}")
print(f"Consensus F1: {profile.consensus_f1.f1:.2%}")
print(f"Fleiss' Kappa: {profile.fleiss_kappa.kappa:.4f}")
print(f"Reliability: {profile.reliability.value}")
print(f"High risk: {profile.is_high_risk}")
```

### Reliability Classification

Based on the metrics, profiles are classified as:

| Level | Criteria |
|-------|----------|
| HIGH | MiHR ≤ 0.15, Kappa ≥ 0.6, Uncertainty ≤ 0.5 |
| MEDIUM | Between HIGH and LOW thresholds |
| LOW | MiHR > 0.3 OR Kappa < 0.4 OR Uncertainty > 0.8 |

### High Risk Flagging

A profile is flagged as high risk if ANY of:
- MiHR > 0.3 (30% of claims unsupported)
- Kappa < 0.4 (poor inter-judge agreement)
- Uncertainty > 0.8 (high model uncertainty)

### JSON Serialization

Profiles can be serialized to JSON for storage and reporting:

```python
# Serialize to JSON
json_str = profile.to_json(indent=2)

# Deserialize from JSON
restored = HallucinationProfile.from_json(json_str)
```

## Configuration

Customize thresholds using `HallucinationMetricsConfig`:

```python
from llm_judge_auditor.components.hallucination_metrics import (
    HallucinationMetricsConfig,
    HallucinationMetricsCalculator,
)

config = HallucinationMetricsConfig(
    mihr_high_risk_threshold=0.2,      # Stricter (default: 0.3)
    kappa_low_threshold=0.5,           # Stricter (default: 0.4)
    uncertainty_high_threshold=0.6,    # Stricter (default: 0.8)
)
calculator = HallucinationMetricsCalculator(config=config)
```

## Examples

See the examples directory for complete working examples:

- `hallucination_metrics_example.py` - MiHR, MaHR, FactScore basics
- `consensus_analysis_example.py` - Cross-model consensus and Fleiss' Kappa
- `uncertainty_quantification_example.py` - Shannon entropy and decomposition
- `hallucination_profile_example.py` - Complete profile generation
- `false_acceptance_rate_example.py` - FAR computation

## References

The metrics implemented are based on research in hallucination detection:

- **MiHR/MaHR**: Micro and Macro hallucination rates for granular analysis
- **FactScore**: Factual precision metric from Min et al.
- **Fleiss' Kappa**: Standard inter-rater agreement statistic
- **Shannon Entropy**: Information-theoretic uncertainty measure
- **Epistemic/Aleatoric Decomposition**: Uncertainty decomposition from Bayesian deep learning

## Requirements Validation

These metrics validate the following requirements:

- **15.1-15.5**: MiHR and MaHR computation
- **16.1-16.5**: FactScore and Consensus F1
- **17.1-17.4**: Fleiss' Kappa
- **18.1-18.5**: Uncertainty quantification
- **19.1-19.5**: Hallucination profile generation
- **20.1-20.4**: False Acceptance Rate
