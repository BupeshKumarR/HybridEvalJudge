"""
Example demonstrating uncertainty quantification for hallucination detection.

This script shows how to use the HallucinationMetricsCalculator to quantify
model uncertainty using:
- Shannon entropy: H(p) = -Σ pᵢ log pᵢ
- Epistemic uncertainty: Var(E[p]) across inference samples
- Aleatoric uncertainty: E[Var(p)] within inference samples
- High uncertainty flagging for hallucination risk assessment

Validates: Requirements 18.1, 18.2, 18.3, 18.4, 18.5
"""

import math

from llm_judge_auditor.components.hallucination_metrics import (
    HallucinationMetricsCalculator,
    HallucinationMetricsConfig,
)


def main():
    """Demonstrate uncertainty quantification for hallucination detection."""
    print("=" * 70)
    print("Uncertainty Quantification Example")
    print("=" * 70)

    calculator = HallucinationMetricsCalculator()

    # =========================================================================
    # 1. Shannon Entropy Basics
    # =========================================================================
    print("\n1. Shannon Entropy Basics")
    print("-" * 70)
    print("H(p) = -Σ pᵢ log pᵢ (using natural log)")
    print("Higher entropy = more uncertainty")
    print()

    # Confident prediction (low entropy)
    confident_probs = [0.95, 0.03, 0.02]
    confident_entropy = calculator.compute_shannon_entropy(confident_probs)
    print(f"Confident prediction: {confident_probs}")
    print(f"Shannon entropy: {confident_entropy:.4f}")

    # Uncertain prediction (high entropy)
    uncertain_probs = [0.34, 0.33, 0.33]
    uncertain_entropy = calculator.compute_shannon_entropy(uncertain_probs)
    print(f"\nUncertain prediction: {uncertain_probs}")
    print(f"Shannon entropy: {uncertain_entropy:.4f}")

    # Maximum entropy for 3 classes
    max_entropy = math.log(3)
    print(f"\nMaximum entropy for 3 classes: {max_entropy:.4f}")
    print(f"Uncertain prediction is {uncertain_entropy/max_entropy:.1%} of maximum")

    # =========================================================================
    # 2. Binary Classification Entropy
    # =========================================================================
    print("\n2. Binary Classification Entropy")
    print("-" * 70)

    binary_cases = [
        [1.0, 0.0],    # Certain positive
        [0.9, 0.1],    # High confidence positive
        [0.7, 0.3],    # Moderate confidence
        [0.5, 0.5],    # Maximum uncertainty
    ]

    for probs in binary_cases:
        entropy = calculator.compute_shannon_entropy(probs)
        print(f"P(positive)={probs[0]:.1f}, P(negative)={probs[1]:.1f} → H={entropy:.4f}")

    # =========================================================================
    # 3. Epistemic Uncertainty (Model Uncertainty)
    # =========================================================================
    print("\n3. Epistemic Uncertainty (Model Uncertainty)")
    print("-" * 70)
    print("Epistemic = Var(E[p]) across inference samples")
    print("High epistemic uncertainty = model is unsure about its predictions")
    print()

    # Consistent model (low epistemic uncertainty)
    consistent_samples = [
        [0.8, 0.15, 0.05],
        [0.82, 0.13, 0.05],
        [0.79, 0.16, 0.05],
        [0.81, 0.14, 0.05],
        [0.80, 0.15, 0.05],
    ]
    consistent_epistemic = calculator.compute_epistemic_uncertainty(consistent_samples)
    print(f"Consistent model samples:")
    for i, s in enumerate(consistent_samples):
        print(f"  Sample {i+1}: {s}")
    print(f"Epistemic uncertainty: {consistent_epistemic:.6f}")

    # Inconsistent model (high epistemic uncertainty)
    inconsistent_samples = [
        [0.9, 0.05, 0.05],
        [0.3, 0.6, 0.1],
        [0.5, 0.3, 0.2],
        [0.7, 0.2, 0.1],
        [0.4, 0.4, 0.2],
    ]
    inconsistent_epistemic = calculator.compute_epistemic_uncertainty(inconsistent_samples)
    print(f"\nInconsistent model samples:")
    for i, s in enumerate(inconsistent_samples):
        print(f"  Sample {i+1}: {s}")
    print(f"Epistemic uncertainty: {inconsistent_epistemic:.6f}")

    # =========================================================================
    # 4. Aleatoric Uncertainty (Data Noise)
    # =========================================================================
    print("\n4. Aleatoric Uncertainty (Data Noise)")
    print("-" * 70)
    print("Aleatoric = E[Var(p)] within inference samples")
    print("High aleatoric uncertainty = inherent noise in the data")
    print()

    # Low aleatoric (peaked distributions)
    peaked_samples = [
        [0.95, 0.03, 0.02],
        [0.94, 0.04, 0.02],
        [0.96, 0.02, 0.02],
    ]
    peaked_aleatoric = calculator.compute_aleatoric_uncertainty(peaked_samples)
    print(f"Peaked distributions (low aleatoric):")
    for s in peaked_samples:
        print(f"  {s}")
    print(f"Aleatoric uncertainty: {peaked_aleatoric:.6f}")

    # High aleatoric (spread distributions)
    spread_samples = [
        [0.4, 0.35, 0.25],
        [0.38, 0.32, 0.30],
        [0.42, 0.33, 0.25],
    ]
    spread_aleatoric = calculator.compute_aleatoric_uncertainty(spread_samples)
    print(f"\nSpread distributions (high aleatoric):")
    for s in spread_samples:
        print(f"  {s}")
    print(f"Aleatoric uncertainty: {spread_aleatoric:.6f}")

    # =========================================================================
    # 5. Complete Uncertainty Decomposition
    # =========================================================================
    print("\n5. Complete Uncertainty Decomposition")
    print("-" * 70)
    print("Total uncertainty = Epistemic + Aleatoric")
    print()

    # Scenario 1: Low total uncertainty
    low_uncertainty_samples = [
        [0.9, 0.07, 0.03],
        [0.91, 0.06, 0.03],
        [0.89, 0.08, 0.03],
        [0.90, 0.07, 0.03],
    ]
    low_result = calculator.compute_uncertainty(
        probabilities=low_uncertainty_samples[0],
        inference_samples=low_uncertainty_samples
    )
    print("Scenario 1: Confident, consistent model")
    print(f"  Shannon entropy: {low_result.shannon_entropy:.4f}")
    print(f"  Epistemic: {low_result.epistemic:.6f}")
    print(f"  Aleatoric: {low_result.aleatoric:.6f}")
    print(f"  Total: {low_result.total:.6f}")
    print(f"  High uncertainty: {low_result.is_high_uncertainty}")

    # Scenario 2: High epistemic, low aleatoric
    high_epistemic_samples = [
        [0.9, 0.05, 0.05],
        [0.2, 0.7, 0.1],
        [0.5, 0.3, 0.2],
        [0.8, 0.1, 0.1],
    ]
    high_epistemic_result = calculator.compute_uncertainty(
        probabilities=high_epistemic_samples[0],
        inference_samples=high_epistemic_samples
    )
    print("\nScenario 2: High epistemic (model disagrees with itself)")
    print(f"  Shannon entropy: {high_epistemic_result.shannon_entropy:.4f}")
    print(f"  Epistemic: {high_epistemic_result.epistemic:.6f}")
    print(f"  Aleatoric: {high_epistemic_result.aleatoric:.6f}")
    print(f"  Total: {high_epistemic_result.total:.6f}")
    print(f"  High uncertainty: {high_epistemic_result.is_high_uncertainty}")

    # Scenario 3: Low epistemic, high aleatoric
    high_aleatoric_samples = [
        [0.35, 0.35, 0.30],
        [0.34, 0.36, 0.30],
        [0.36, 0.34, 0.30],
        [0.35, 0.35, 0.30],
    ]
    high_aleatoric_result = calculator.compute_uncertainty(
        probabilities=high_aleatoric_samples[0],
        inference_samples=high_aleatoric_samples
    )
    print("\nScenario 3: High aleatoric (inherently uncertain)")
    print(f"  Shannon entropy: {high_aleatoric_result.shannon_entropy:.4f}")
    print(f"  Epistemic: {high_aleatoric_result.epistemic:.6f}")
    print(f"  Aleatoric: {high_aleatoric_result.aleatoric:.6f}")
    print(f"  Total: {high_aleatoric_result.total:.6f}")
    print(f"  High uncertainty: {high_aleatoric_result.is_high_uncertainty}")

    # =========================================================================
    # 6. High Uncertainty Flagging
    # =========================================================================
    print("\n6. High Uncertainty Flagging")
    print("-" * 70)
    print("Default threshold: 0.8")
    print()

    # Test various uncertainty levels
    test_cases = [
        ("Low uncertainty", [0.95, 0.03, 0.02], None),
        ("Medium uncertainty", [0.6, 0.25, 0.15], None),
        ("High uncertainty", [0.35, 0.35, 0.30], None),
    ]

    for name, probs, samples in test_cases:
        result = calculator.compute_uncertainty(probs, samples)
        flag = calculator.flag_high_uncertainty(result)
        print(f"{name}: total={result.total:.4f}, flagged={flag}")

    # Custom threshold
    print("\nWith custom threshold (0.5):")
    for name, probs, samples in test_cases:
        result = calculator.compute_uncertainty(probs, samples)
        flag = calculator.flag_high_uncertainty(result, threshold=0.5)
        print(f"{name}: total={result.total:.4f}, flagged={flag}")

    # =========================================================================
    # 7. Practical Application: Hallucination Risk Assessment
    # =========================================================================
    print("\n7. Practical Application: Hallucination Risk Assessment")
    print("-" * 70)
    print("High uncertainty often correlates with hallucination risk")
    print()

    # Simulate model outputs for different claim types
    claim_scenarios = [
        {
            "claim": "Paris is the capital of France",
            "type": "Well-known fact",
            "samples": [
                [0.98, 0.01, 0.01],
                [0.97, 0.02, 0.01],
                [0.98, 0.01, 0.01],
            ],
        },
        {
            "claim": "The meeting was scheduled for 3pm",
            "type": "Specific detail (potential hallucination)",
            "samples": [
                [0.6, 0.25, 0.15],
                [0.4, 0.35, 0.25],
                [0.55, 0.30, 0.15],
            ],
        },
        {
            "claim": "Dr. Smith published 47 papers in 2023",
            "type": "Precise number (high hallucination risk)",
            "samples": [
                [0.3, 0.4, 0.3],
                [0.5, 0.2, 0.3],
                [0.25, 0.45, 0.3],
            ],
        },
    ]

    for scenario in claim_scenarios:
        result = calculator.compute_uncertainty(
            probabilities=scenario["samples"][0],
            inference_samples=scenario["samples"]
        )
        risk_level = "HIGH" if result.is_high_uncertainty else (
            "MEDIUM" if result.total > 0.3 else "LOW"
        )
        print(f"Claim: \"{scenario['claim']}\"")
        print(f"  Type: {scenario['type']}")
        print(f"  Total uncertainty: {result.total:.4f}")
        print(f"  Risk level: {risk_level}")
        print()

    # =========================================================================
    # 8. Custom Configuration
    # =========================================================================
    print("\n8. Custom Configuration")
    print("-" * 70)

    # Stricter uncertainty threshold for high-stakes applications
    strict_config = HallucinationMetricsConfig(
        uncertainty_high_threshold=0.3  # Much stricter
    )
    strict_calculator = HallucinationMetricsCalculator(config=strict_config)

    medium_probs = [0.6, 0.25, 0.15]
    default_result = calculator.compute_uncertainty(medium_probs)
    strict_result = strict_calculator.compute_uncertainty(medium_probs)

    print(f"Probabilities: {medium_probs}")
    print(f"Default threshold (0.8): flagged = {default_result.is_high_uncertainty}")
    print(f"Strict threshold (0.3): flagged = {strict_result.is_high_uncertainty}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
