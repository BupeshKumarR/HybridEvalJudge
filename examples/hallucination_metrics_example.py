"""
Example demonstrating hallucination quantification metrics.

This script shows how to use the HallucinationMetricsCalculator to compute
research-backed hallucination metrics including:
- MiHR (Micro Hallucination Rate): unsupported_claims / total_claims
- MaHR (Macro Hallucination Rate): responses_with_hallucinations / total_responses
- FactScore: verified_claims / total_claims
- Uncertainty quantification with Shannon entropy

Validates: Requirements 15.1, 15.2, 15.3, 15.4, 15.5, 16.1, 18.1-18.5
"""

from llm_judge_auditor.components.hallucination_metrics import (
    HallucinationMetricsCalculator,
    HallucinationMetricsConfig,
)
from llm_judge_auditor.models import (
    Verdict,
    VerdictLabel,
)


def main():
    """Demonstrate hallucination metrics computation."""
    print("=" * 70)
    print("Hallucination Metrics Example")
    print("=" * 70)

    # Initialize calculator with default thresholds
    calculator = HallucinationMetricsCalculator()

    # =========================================================================
    # 1. MiHR (Micro Hallucination Rate) Computation
    # =========================================================================
    print("\n1. MiHR (Micro Hallucination Rate)")
    print("-" * 70)
    print("MiHR = unsupported_claims / total_claims")
    print()

    # Create sample verdicts for a response with some hallucinations
    verdicts = [
        Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.95,
            evidence=["Source confirms this claim"],
            reasoning="Directly supported by source text",
        ),
        Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.88,
            evidence=["Evidence found in paragraph 2"],
            reasoning="Claim aligns with source information",
        ),
        Verdict(
            label=VerdictLabel.REFUTED,
            confidence=0.92,
            evidence=["Source contradicts this claim"],
            reasoning="Statement contradicts source information",
        ),
        Verdict(
            label=VerdictLabel.NOT_ENOUGH_INFO,
            confidence=0.70,
            evidence=[],
            reasoning="No evidence found to verify this claim",
        ),
        Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.85,
            evidence=["Partial evidence found"],
            reasoning="Claim is partially supported",
        ),
    ]

    mihr_result = calculator.compute_mihr(verdicts)
    print(f"Total claims: {mihr_result.total_claims}")
    print(f"Unsupported claims: {mihr_result.unsupported_claims}")
    print(f"MiHR value: {mihr_result.value:.2%}")
    print(f"Has claims: {mihr_result.has_claims}")

    # Edge case: zero claims
    print("\nEdge case - Zero claims:")
    empty_mihr = calculator.compute_mihr([])
    print(f"MiHR value: {empty_mihr.value}")
    print(f"Has claims: {empty_mihr.has_claims}")

    # =========================================================================
    # 2. MaHR (Macro Hallucination Rate) Computation
    # =========================================================================
    print("\n2. MaHR (Macro Hallucination Rate)")
    print("-" * 70)
    print("MaHR = responses_with_hallucinations / total_responses")
    print()

    # Create verdicts for multiple responses
    response_verdicts = [
        # Response 1: No hallucinations (all supported)
        [
            Verdict(VerdictLabel.SUPPORTED, 0.9, ["evidence"], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.85, ["evidence"], "Supported"),
        ],
        # Response 2: Has hallucination (one refuted)
        [
            Verdict(VerdictLabel.SUPPORTED, 0.9, ["evidence"], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.88, ["evidence"], "Refuted"),
        ],
        # Response 3: Has hallucination (not enough info)
        [
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.7, [], "No evidence"),
        ],
        # Response 4: No hallucinations
        [
            Verdict(VerdictLabel.SUPPORTED, 0.95, ["evidence"], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.92, ["evidence"], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.88, ["evidence"], "Supported"),
        ],
    ]

    mahr_result = calculator.compute_mahr(response_verdicts)
    print(f"Total responses: {mahr_result.total_responses}")
    print(f"Responses with hallucinations: {mahr_result.responses_with_hallucinations}")
    print(f"MaHR value: {mahr_result.value:.2%}")

    # =========================================================================
    # 3. FactScore Computation
    # =========================================================================
    print("\n3. FactScore")
    print("-" * 70)
    print("FactScore = verified_claims / total_claims")
    print()

    factscore = calculator.compute_factscore(verdicts)
    print(f"FactScore: {factscore:.2%}")
    print("(Higher is better - indicates factual precision)")

    # =========================================================================
    # 4. Uncertainty Quantification
    # =========================================================================
    print("\n4. Uncertainty Quantification")
    print("-" * 70)
    print("Shannon entropy: H(p) = -Σ pᵢ log pᵢ")
    print()

    # Example probability distribution (e.g., from model output)
    probabilities = [0.7, 0.2, 0.1]  # High confidence prediction
    
    uncertainty_result = calculator.compute_uncertainty(probabilities)
    print(f"Shannon entropy: {uncertainty_result.shannon_entropy:.4f}")
    print(f"Is high uncertainty: {uncertainty_result.is_high_uncertainty}")

    # Compare with uncertain prediction
    uncertain_probs = [0.35, 0.35, 0.30]  # Low confidence prediction
    uncertain_result = calculator.compute_uncertainty(uncertain_probs)
    print(f"\nUncertain prediction entropy: {uncertain_result.shannon_entropy:.4f}")
    print(f"Is high uncertainty: {uncertain_result.is_high_uncertainty}")

    # =========================================================================
    # 5. Epistemic and Aleatoric Uncertainty
    # =========================================================================
    print("\n5. Epistemic and Aleatoric Uncertainty Decomposition")
    print("-" * 70)
    print("Epistemic = Var(E[p]) across inference samples")
    print("Aleatoric = E[Var(p)] within inference samples")
    print()

    # Multiple inference samples (e.g., from Monte Carlo dropout)
    inference_samples = [
        [0.7, 0.2, 0.1],
        [0.65, 0.25, 0.1],
        [0.72, 0.18, 0.1],
        [0.68, 0.22, 0.1],
        [0.71, 0.19, 0.1],
    ]

    full_uncertainty = calculator.compute_uncertainty(
        probabilities=probabilities,
        inference_samples=inference_samples
    )
    print(f"Shannon entropy: {full_uncertainty.shannon_entropy:.4f}")
    print(f"Epistemic uncertainty: {full_uncertainty.epistemic:.6f}")
    print(f"Aleatoric uncertainty: {full_uncertainty.aleatoric:.6f}")
    print(f"Total uncertainty: {full_uncertainty.total:.6f}")
    print(f"Is high uncertainty: {full_uncertainty.is_high_uncertainty}")

    # =========================================================================
    # 6. High Risk Detection
    # =========================================================================
    print("\n6. High Risk Detection")
    print("-" * 70)
    print("High risk if: MiHR > 0.3 OR Kappa < 0.4 OR uncertainty > 0.8")
    print()

    # Check if current metrics indicate high risk
    is_high_risk = calculator.is_high_risk(
        mihr=mihr_result,
        uncertainty=full_uncertainty
    )
    print(f"Current MiHR: {mihr_result.value:.2%}")
    print(f"Current uncertainty: {full_uncertainty.total:.4f}")
    print(f"Is high risk: {is_high_risk}")

    # Create high-risk scenario
    high_risk_verdicts = [
        Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
        Verdict(VerdictLabel.REFUTED, 0.85, [], "Refuted"),
        Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.7, [], "No evidence"),
        Verdict(VerdictLabel.SUPPORTED, 0.8, [], "Supported"),
    ]
    high_risk_mihr = calculator.compute_mihr(high_risk_verdicts)
    is_high_risk_scenario = calculator.is_high_risk(mihr=high_risk_mihr)
    print(f"\nHigh-risk scenario MiHR: {high_risk_mihr.value:.2%}")
    print(f"Is high risk: {is_high_risk_scenario}")

    # =========================================================================
    # 7. Custom Thresholds
    # =========================================================================
    print("\n7. Custom Thresholds")
    print("-" * 70)

    custom_config = HallucinationMetricsConfig(
        mihr_high_risk_threshold=0.2,  # Stricter threshold
        kappa_low_threshold=0.5,
        uncertainty_high_threshold=0.6,
    )
    strict_calculator = HallucinationMetricsCalculator(config=custom_config)

    # Same MiHR but with stricter threshold
    is_high_risk_strict = strict_calculator.is_high_risk(mihr=mihr_result)
    print(f"MiHR value: {mihr_result.value:.2%}")
    print(f"Default threshold (0.3): High risk = {is_high_risk}")
    print(f"Strict threshold (0.2): High risk = {is_high_risk_strict}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
