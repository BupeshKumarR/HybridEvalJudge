"""
Example demonstrating cross-model consensus analysis.

This script shows how to use the ClaimVerificationMatrixBuilder and
HallucinationMetricsCalculator to analyze agreement across multiple models:
- Building claim verification matrices
- Computing Consensus F1 scores
- Computing Fleiss' Kappa for inter-judge agreement
- Identifying disputed and consensus claims

Validates: Requirements 16.2, 16.3, 16.4, 16.5, 17.1, 17.2, 17.3, 17.4
"""

from llm_judge_auditor.components.hallucination_metrics import (
    ClaimVerificationMatrixBuilder,
    HallucinationMetricsCalculator,
)
from llm_judge_auditor.models import (
    Claim,
    ClaimType,
    Verdict,
    VerdictLabel,
)


def main():
    """Demonstrate cross-model consensus analysis."""
    print("=" * 70)
    print("Cross-Model Consensus Analysis Example")
    print("=" * 70)

    # Initialize components
    matrix_builder = ClaimVerificationMatrixBuilder()
    calculator = HallucinationMetricsCalculator()

    # =========================================================================
    # 1. Building a Claim Verification Matrix
    # =========================================================================
    print("\n1. Building Claim Verification Matrix")
    print("-" * 70)
    print("Matrix tracks which claims are supported by which models")
    print()

    # Define claims extracted from different model responses
    # Simulating 3 models responding to the same query
    model_claims = {
        "gpt-4": [
            Claim("Paris is the capital of France", (0, 30), ClaimType.FACTUAL),
            Claim("The Eiffel Tower is 330 meters tall", (31, 65), ClaimType.NUMERICAL),
            Claim("France has a population of 67 million", (66, 100), ClaimType.NUMERICAL),
        ],
        "claude-3": [
            Claim("Paris is the capital of France", (0, 30), ClaimType.FACTUAL),
            Claim("The Eiffel Tower is 330 meters tall", (31, 65), ClaimType.NUMERICAL),
            Claim("France joined the EU in 1957", (66, 95), ClaimType.TEMPORAL),
        ],
        "gemini": [
            Claim("Paris is the capital of France", (0, 30), ClaimType.FACTUAL),
            Claim("France has a population of 67 million", (31, 65), ClaimType.NUMERICAL),
            Claim("The French Revolution began in 1789", (66, 100), ClaimType.TEMPORAL),
        ],
    }

    # Build the matrix
    matrix = matrix_builder.build_matrix(model_claims)

    print(f"Number of unique claims: {len(matrix.claims)}")
    print(f"Number of models: {len(matrix.models)}")
    print(f"Models: {matrix.models}")
    print("\nUnique claims:")
    for i, claim in enumerate(matrix.claims):
        support_count = matrix.get_claim_support_count(i)
        print(f"  {i+1}. '{claim.text}' - supported by {support_count}/{len(matrix.models)} models")

    # =========================================================================
    # 2. Identifying Consensus and Disputed Claims
    # =========================================================================
    print("\n2. Consensus and Disputed Claims")
    print("-" * 70)

    consensus_claims = matrix_builder.identify_consensus_claims(matrix, threshold=0.5)
    disputed_claims = matrix_builder.identify_disputed_claims(matrix, threshold=0.5)

    print(f"Consensus claims (>= 50% agreement):")
    for claim in consensus_claims:
        print(f"  - {claim.text}")

    print(f"\nDisputed claims (< 50% agreement):")
    for claim in disputed_claims:
        print(f"  - {claim.text}")

    # =========================================================================
    # 3. Computing Claim Consensus Details
    # =========================================================================
    print("\n3. Detailed Claim Consensus Analysis")
    print("-" * 70)

    consensus_details = matrix_builder.compute_claim_consensus(matrix)
    for detail in consensus_details:
        status = "✓ Consensus" if detail.is_consensus else "✗ Disputed"
        print(f"  [{status}] '{detail.claim.text}'")
        print(f"      Support: {detail.support_count}/{detail.total_models} ({detail.consensus_ratio:.0%})")

    # =========================================================================
    # 4. Computing Consensus F1
    # =========================================================================
    print("\n4. Consensus F1 Scores")
    print("-" * 70)
    print("Precision = claims supported by others / model's claims")
    print("Recall = consensus claims included / total consensus claims")
    print("F1 = 2 × (precision × recall) / (precision + recall)")
    print()

    # Compute F1 for each model
    for model_name in matrix.models:
        f1_result = calculator.compute_consensus_f1(matrix, model_name)
        print(f"{model_name}:")
        print(f"  Precision: {f1_result.precision:.2%}")
        print(f"  Recall: {f1_result.recall:.2%}")
        print(f"  F1: {f1_result.f1:.2%}")
        print()

    # Compute average F1 across all models
    avg_f1 = calculator.compute_average_consensus_f1(matrix)
    print(f"Average across all models:")
    print(f"  Precision: {avg_f1.precision:.2%}")
    print(f"  Recall: {avg_f1.recall:.2%}")
    print(f"  F1: {avg_f1.f1:.2%}")

    # =========================================================================
    # 5. Fleiss' Kappa for Inter-Judge Agreement
    # =========================================================================
    print("\n5. Fleiss' Kappa (Inter-Judge Agreement)")
    print("-" * 70)
    print("κ = (Po - Pe) / (1 - Pe)")
    print("Interpretation: poor (<0.2), fair (0.2-0.4), moderate (0.4-0.6),")
    print("               substantial (0.6-0.8), almost perfect (>0.8)")
    print()

    # Create judge verdicts for the same set of claims
    # Each judge evaluates the same 5 claims
    judge_verdicts = {
        "judge_1": [
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.85, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.8, [], "Refuted"),
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.7, [], "NEI"),
            Verdict(VerdictLabel.SUPPORTED, 0.88, [], "Supported"),
        ],
        "judge_2": [
            Verdict(VerdictLabel.SUPPORTED, 0.92, [], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.87, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.82, [], "Refuted"),
            Verdict(VerdictLabel.REFUTED, 0.75, [], "Refuted"),
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
        ],
        "judge_3": [
            Verdict(VerdictLabel.SUPPORTED, 0.88, [], "Supported"),
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.6, [], "NEI"),
            Verdict(VerdictLabel.REFUTED, 0.85, [], "Refuted"),
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.65, [], "NEI"),
            Verdict(VerdictLabel.SUPPORTED, 0.86, [], "Supported"),
        ],
    }

    kappa_result = calculator.compute_fleiss_kappa_from_verdicts(judge_verdicts)
    print(f"Fleiss' Kappa: {kappa_result.kappa:.4f}")
    print(f"Interpretation: {kappa_result.interpretation}")
    print(f"Observed agreement (Po): {kappa_result.observed_agreement:.4f}")
    print(f"Expected agreement (Pe): {kappa_result.expected_agreement:.4f}")

    # =========================================================================
    # 6. High Agreement Example
    # =========================================================================
    print("\n6. High Agreement Example")
    print("-" * 70)

    # Judges mostly agree
    high_agreement_verdicts = {
        "judge_1": [
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
        ],
        "judge_2": [
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
        ],
        "judge_3": [
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
        ],
    }

    high_kappa = calculator.compute_fleiss_kappa_from_verdicts(high_agreement_verdicts)
    print(f"Fleiss' Kappa: {high_kappa.kappa:.4f}")
    print(f"Interpretation: {high_kappa.interpretation}")

    # =========================================================================
    # 7. Low Agreement Example
    # =========================================================================
    print("\n7. Low Agreement Example")
    print("-" * 70)

    # Judges disagree significantly
    low_agreement_verdicts = {
        "judge_1": [
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.9, [], "NEI"),
        ],
        "judge_2": [
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.9, [], "NEI"),
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
        ],
        "judge_3": [
            Verdict(VerdictLabel.NOT_ENOUGH_INFO, 0.9, [], "NEI"),
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
            Verdict(VerdictLabel.REFUTED, 0.9, [], "Refuted"),
        ],
    }

    low_kappa = calculator.compute_fleiss_kappa_from_verdicts(low_agreement_verdicts)
    print(f"Fleiss' Kappa: {low_kappa.kappa:.4f}")
    print(f"Interpretation: {low_kappa.interpretation}")

    # =========================================================================
    # 8. Edge Case: Fewer than 2 Judges
    # =========================================================================
    print("\n8. Edge Case: Fewer than 2 Judges")
    print("-" * 70)

    single_judge_verdicts = {
        "judge_1": [
            Verdict(VerdictLabel.SUPPORTED, 0.9, [], "Supported"),
        ],
    }

    single_kappa = calculator.compute_fleiss_kappa_from_verdicts(single_judge_verdicts)
    print(f"Is undefined: {single_kappa.is_undefined}")
    print(f"Error message: {single_kappa.error_message}")

    # =========================================================================
    # 9. Using Raw Rating Matrix
    # =========================================================================
    print("\n9. Using Raw Rating Matrix")
    print("-" * 70)
    print("Format: ratings[item][category] = count of raters")
    print()

    # 4 items, 3 categories (SUPPORTED=0, REFUTED=1, NEI=2), 5 raters
    # Each row sums to 5 (number of raters)
    raw_ratings = [
        [5, 0, 0],  # Item 1: All 5 raters say SUPPORTED
        [4, 1, 0],  # Item 2: 4 say SUPPORTED, 1 says REFUTED
        [2, 2, 1],  # Item 3: Mixed opinions
        [0, 0, 5],  # Item 4: All 5 raters say NEI
    ]

    raw_kappa = calculator.compute_fleiss_kappa(raw_ratings, num_categories=3)
    print(f"Fleiss' Kappa: {raw_kappa.kappa:.4f}")
    print(f"Interpretation: {raw_kappa.interpretation}")
    print(f"Observed agreement: {raw_kappa.observed_agreement:.4f}")
    print(f"Expected agreement: {raw_kappa.expected_agreement:.4f}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
