"""
Example demonstrating comprehensive hallucination profile generation.

This script shows how to generate a complete HallucinationProfile that combines
all quantified metrics:
- MiHR and MaHR (hallucination rates)
- FactScore (factual precision)
- Consensus F1 (cross-model agreement)
- Fleiss' Kappa (inter-judge agreement)
- Uncertainty quantification
- Reliability classification and high-risk flagging

Validates: Requirements 19.1, 19.2, 19.3, 19.4, 19.5
"""

import json

from llm_judge_auditor.components.hallucination_metrics import (
    ClaimVerificationMatrixBuilder,
    HallucinationMetricsCalculator,
    HallucinationMetricsConfig,
    HallucinationProfile,
)
from llm_judge_auditor.models import (
    Claim,
    ClaimType,
    Verdict,
    VerdictLabel,
)


def main():
    """Demonstrate comprehensive hallucination profile generation."""
    print("=" * 70)
    print("Hallucination Profile Generation Example")
    print("=" * 70)

    # Initialize components
    calculator = HallucinationMetricsCalculator()
    matrix_builder = ClaimVerificationMatrixBuilder()

    # =========================================================================
    # 1. Prepare Sample Data
    # =========================================================================
    print("\n1. Preparing Sample Data")
    print("-" * 70)

    # Primary response verdicts (for MiHR and FactScore)
    primary_verdicts = [
        Verdict(VerdictLabel.SUPPORTED, 0.95, ["Source confirms"], "Supported"),
        Verdict(VerdictLabel.SUPPORTED, 0.88, ["Evidence found"], "Supported"),
        Verdict(VerdictLabel.REFUTED, 0.92, ["Source contradicts"], "Refuted"),
        Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.70, [], "No evidence"),
        Verdict(VerdictLabel.SUPPORTED, 0.85, ["Partial evidence"], "Supported"),
    ]
    print(f"Primary response: {len(primary_verdicts)} claims")
    print(f"  - Supported: {sum(1 for v in primary_verdicts if v.label == VerdictLabel.SUPPORTED)}")
    print(f"  - Refuted: {sum(1 for v in primary_verdicts if v.label == VerdictLabel.REFUTED)}")
    print(f"  - NEI: {sum(1 for v in primary_verdicts if v.label == VerdictLabel.NOT_ENOUGH_INFO)}")

    # Multiple response verdicts (for MaHR)
    response_verdicts = [
        primary_verdicts,
        [  # Response 2: No hallucinations
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.85, [], "Supported"),
        ],
        [  # Response 3: Has hallucination
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.88, [], "Refuted"),
        ],
    ]
    print(f"\nTotal responses for MaHR: {len(response_verdicts)}")

    # Model claims for consensus analysis
    model_claims = {
        "model_a": [
            Claim("Paris is the capital of France", (0, 30), ClaimType.FACTUAL),
            Claim("The Eiffel Tower is 330 meters tall", (31, 65), ClaimType.NUMERICAL),
            Claim("France has 67 million people", (66, 100), ClaimType.NUMERICAL),
        ],
        "model_b": [
            Claim("Paris is the capital of France", (0, 30), ClaimType.FACTUAL),
            Claim("The Eiffel Tower is 330 meters tall", (31, 65), ClaimType.NUMERICAL),
        ],
        "model_c": [
            Claim("Paris is the capital of France", (0, 30), ClaimType.FACTUAL),
            Claim("France has 67 million people", (31, 65), ClaimType.NUMERICAL),
        ],
    }
    claim_matrix = matrix_builder.build_matrix(model_claims)
    print(f"\nClaim matrix: {len(claim_matrix.claims)} unique claims, {len(claim_matrix.models)} models")

    # Judge verdicts for Fleiss' Kappa
    judge_verdicts = {
        "judge_1": [
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], ""),
            Verdict(VerdictLabel.SUPPORTED, 0.85, [], ""),
            Verdict(VerdictLabel.REFUTED, 0.8, [], ""),
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.7, [], ""),
            Verdict(VerdictLabel.SUPPORTED, 0.88, [], ""),
        ],
        "judge_2": [
            Verdict(VerdictLabel.SUPPORTED, 0.92, [], ""),
            Verdict(VerdictLabel.SUPPORTED, 0.87, [], ""),
            Verdict(VerdictLabel.REFUTED, 0.82, [], ""),
            Verdict(VerdictLabel.REFUTED, 0.75, [], ""),
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], ""),
        ],
        "judge_3": [
            Verdict(VerdictLabel.SUPPORTED, 0.88, [], ""),
            Verdict(VerdictLabel.SUPPORTED, 0.8, [], ""),
            Verdict(VerdictLabel.REFUTED, 0.85, [], ""),
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.65, [], ""),
            Verdict(VerdictLabel.SUPPORTED, 0.86, [], ""),
        ],
    }
    print(f"Judge verdicts: {len(judge_verdicts)} judges, {len(list(judge_verdicts.values())[0])} items each")

    # Probability distribution and inference samples for uncertainty
    probabilities = [0.7, 0.2, 0.1]
    inference_samples = [
        [0.7, 0.2, 0.1],
        [0.65, 0.25, 0.1],
        [0.72, 0.18, 0.1],
        [0.68, 0.22, 0.1],
    ]
    print(f"Uncertainty data: {len(inference_samples)} inference samples")

    # =========================================================================
    # 2. Generate Complete Hallucination Profile
    # =========================================================================
    print("\n2. Generating Hallucination Profile")
    print("-" * 70)

    profile = calculator.generate_hallucination_profile(
        verdicts=primary_verdicts,
        response_verdicts=response_verdicts,
        claim_matrix=claim_matrix,
        judge_verdicts=judge_verdicts,
        probabilities=probabilities,
        inference_samples=inference_samples,
        consensus_threshold=0.5,
    )

    # =========================================================================
    # 3. Display Profile Results
    # =========================================================================
    print("\n3. Profile Results")
    print("-" * 70)

    # MiHR
    print("\nMicro Hallucination Rate (MiHR):")
    print(f"  Value: {profile.mihr.value:.2%}")
    print(f"  Unsupported claims: {profile.mihr.unsupported_claims}/{profile.mihr.total_claims}")

    # MaHR
    print("\nMacro Hallucination Rate (MaHR):")
    print(f"  Value: {profile.mahr.value:.2%}")
    print(f"  Responses with hallucinations: {profile.mahr.responses_with_hallucinations}/{profile.mahr.total_responses}")

    # FactScore
    print("\nFactScore:")
    print(f"  Value: {profile.factscore:.2%}")

    # Consensus F1
    print("\nConsensus F1:")
    print(f"  Precision: {profile.consensus_f1.precision:.2%}")
    print(f"  Recall: {profile.consensus_f1.recall:.2%}")
    print(f"  F1: {profile.consensus_f1.f1:.2%}")

    # Fleiss' Kappa
    print("\nFleiss' Kappa:")
    print(f"  Value: {profile.fleiss_kappa.kappa:.4f}")
    print(f"  Interpretation: {profile.fleiss_kappa.interpretation}")
    print(f"  Observed agreement: {profile.fleiss_kappa.observed_agreement:.4f}")

    # Uncertainty
    print("\nUncertainty:")
    print(f"  Shannon entropy: {profile.uncertainty.shannon_entropy:.4f}")
    print(f"  Epistemic: {profile.uncertainty.epistemic:.6f}")
    print(f"  Aleatoric: {profile.uncertainty.aleatoric:.6f}")
    print(f"  Total: {profile.uncertainty.total:.6f}")
    print(f"  High uncertainty: {profile.uncertainty.is_high_uncertainty}")

    # Reliability and Risk
    print("\nReliability Assessment:")
    print(f"  Reliability level: {profile.reliability.value}")
    print(f"  Is high risk: {profile.is_high_risk}")

    # Claims analysis
    print("\nClaim Analysis:")
    print(f"  Consensus claims: {len(profile.consensus_claims)}")
    for claim in profile.consensus_claims:
        print(f"    - {claim.text}")
    print(f"  Disputed claims: {len(profile.disputed_claims)}")
    for claim in profile.disputed_claims:
        print(f"    - {claim.text}")

    # =========================================================================
    # 4. JSON Serialization
    # =========================================================================
    print("\n4. JSON Serialization")
    print("-" * 70)

    json_str = profile.to_json(indent=2)
    print("Profile serialized to JSON:")
    print(json_str[:500] + "..." if len(json_str) > 500 else json_str)

    # Round-trip test
    restored_profile = HallucinationProfile.from_json(json_str)
    print(f"\nRound-trip successful: {restored_profile.mihr.value == profile.mihr.value}")

    # =========================================================================
    # 5. High-Risk Scenario
    # =========================================================================
    print("\n5. High-Risk Scenario")
    print("-" * 70)

    # Create a high-risk profile
    high_risk_verdicts = [
        Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
        Verdict(VerdictLabel.REFUTED, 0.85, [], "Refuted"),
        Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.7, [], "NEI"),
        Verdict(VerdictLabel.SUPPORTED, 0.8, [], "Supported"),
    ]

    high_risk_profile = calculator.generate_hallucination_profile(
        verdicts=high_risk_verdicts,
    )

    print(f"MiHR: {high_risk_profile.mihr.value:.2%} (threshold: 30%)")
    print(f"Reliability: {high_risk_profile.reliability.value}")
    print(f"Is high risk: {high_risk_profile.is_high_risk}")

    # =========================================================================
    # 6. Custom Thresholds
    # =========================================================================
    print("\n6. Custom Thresholds")
    print("-" * 70)

    strict_config = HallucinationMetricsConfig(
        mihr_high_risk_threshold=0.2,  # Stricter
        kappa_low_threshold=0.5,
        uncertainty_high_threshold=0.5,
    )
    strict_calculator = HallucinationMetricsCalculator(config=strict_config)

    strict_profile = strict_calculator.generate_hallucination_profile(
        verdicts=primary_verdicts,
    )

    print(f"MiHR: {strict_profile.mihr.value:.2%}")
    print(f"Default threshold (30%): high_risk = {profile.is_high_risk}")
    print(f"Strict threshold (20%): high_risk = {strict_profile.is_high_risk}")

    # =========================================================================
    # 7. Minimal Profile (Only Required Data)
    # =========================================================================
    print("\n7. Minimal Profile (Only Required Data)")
    print("-" * 70)

    minimal_profile = calculator.generate_hallucination_profile(
        verdicts=primary_verdicts,
        # No optional data provided
    )

    print(f"MiHR: {minimal_profile.mihr.value:.2%}")
    print(f"FactScore: {minimal_profile.factscore:.2%}")
    print(f"MaHR: {minimal_profile.mahr}")
    print(f"Consensus F1: {minimal_profile.consensus_f1}")
    print(f"Fleiss' Kappa: {minimal_profile.fleiss_kappa}")
    print(f"Uncertainty: {minimal_profile.uncertainty}")
    print(f"Reliability: {minimal_profile.reliability.value}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
