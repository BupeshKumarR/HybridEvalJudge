"""
Example usage of the ReliabilityValidator component.

This script demonstrates how to use the ReliabilityValidator to:
1. Check evaluation consistency across multiple runs
2. Calculate inter-model agreement using Cohen's kappa
3. Validate pairwise rankings using Kendall's Tau and Spearman's rho
"""

from llm_judge_auditor.components.reliability_validator import ReliabilityValidator


def example_consistency_checking():
    """Example: Check evaluation consistency."""
    print("=" * 80)
    print("Example 1: Evaluation Consistency Checking")
    print("=" * 80)
    
    validator = ReliabilityValidator(consistency_threshold=5.0)
    
    # Simulate multiple evaluations of the same input
    print("\nScenario A: Consistent evaluations")
    consistent_scores = [85.0, 87.0, 84.5, 86.0, 85.5, 86.5]
    report = validator.check_consistency(consistent_scores)
    
    print(f"  Scores: {consistent_scores}")
    print(f"  Mean: {report.mean_score:.2f}")
    print(f"  Variance: {report.variance:.2f}")
    print(f"  Std Deviation: {report.std_deviation:.2f}")
    print(f"  Is Consistent: {report.is_consistent}")
    print(f"  ✓ Evaluation is consistent (variance < 5.0)" if report.is_consistent else "  ✗ Evaluation is inconsistent")
    
    print("\nScenario B: Inconsistent evaluations")
    inconsistent_scores = [50.0, 80.0, 30.0, 90.0, 20.0, 95.0]
    report = validator.check_consistency(inconsistent_scores)
    
    print(f"  Scores: {inconsistent_scores}")
    print(f"  Mean: {report.mean_score:.2f}")
    print(f"  Variance: {report.variance:.2f}")
    print(f"  Std Deviation: {report.std_deviation:.2f}")
    print(f"  Is Consistent: {report.is_consistent}")
    print(f"  ✓ Evaluation is consistent (variance < 5.0)" if report.is_consistent else "  ✗ Evaluation is inconsistent")


def example_inter_model_agreement():
    """Example: Calculate inter-model agreement."""
    print("\n" + "=" * 80)
    print("Example 2: Inter-Model Agreement (Cohen's Kappa)")
    print("=" * 80)
    
    validator = ReliabilityValidator()
    
    # Simulate scores from three judge models on the same test cases
    print("\nScenario A: High agreement between judges")
    judge_scores_high = {
        "llama-3-8b": [85.0, 45.0, 90.0, 30.0, 75.0, 88.0],
        "mistral-7b": [80.0, 40.0, 88.0, 35.0, 70.0, 85.0],
        "phi-3-mini": [82.0, 48.0, 92.0, 32.0, 73.0, 87.0],
    }
    
    report = validator.calculate_inter_model_agreement(judge_scores_high, threshold=50.0)
    
    print(f"  Number of judges: {report.num_models}")
    print(f"  Cohen's Kappa: {report.cohens_kappa:.3f}")
    print(f"  Agreement Level: {report.agreement_level}")
    print(f"\n  Pairwise agreements:")
    for (judge_a, judge_b), kappa in report.pairwise_agreements.items():
        print(f"    {judge_a} vs {judge_b}: κ = {kappa:.3f}")
    
    print("\nScenario B: Low agreement between judges")
    judge_scores_low = {
        "llama-3-8b": [85.0, 45.0, 90.0, 30.0, 75.0, 88.0],
        "mistral-7b": [45.0, 85.0, 30.0, 90.0, 25.0, 35.0],  # Opposite pattern
        "phi-3-mini": [50.0, 50.0, 50.0, 50.0, 50.0, 50.0],  # All at threshold
    }
    
    report = validator.calculate_inter_model_agreement(judge_scores_low, threshold=50.0)
    
    print(f"  Number of judges: {report.num_models}")
    print(f"  Cohen's Kappa: {report.cohens_kappa:.3f}")
    print(f"  Agreement Level: {report.agreement_level}")
    print(f"\n  Pairwise agreements:")
    for (judge_a, judge_b), kappa in report.pairwise_agreements.items():
        print(f"    {judge_a} vs {judge_b}: κ = {kappa:.3f}")


def example_ranking_correlation():
    """Example: Calculate ranking correlation."""
    print("\n" + "=" * 80)
    print("Example 3: Ranking Correlation (Kendall's Tau & Spearman's Rho)")
    print("=" * 80)
    
    validator = ReliabilityValidator()
    
    # Simulate pairwise rankings
    print("\nScenario A: Perfect agreement with ground truth")
    predicted_perfect = [
        ("Model_A", "Model_B"),
        ("Model_C", "Model_D"),
        ("Model_E", "Model_F"),
        ("Model_A", "Model_C"),
    ]
    ground_truth_perfect = [
        ("Model_A", "Model_B"),
        ("Model_C", "Model_D"),
        ("Model_E", "Model_F"),
        ("Model_A", "Model_C"),
    ]
    
    report = validator.calculate_ranking_correlation(predicted_perfect, ground_truth_perfect)
    
    print(f"  Number of pairs: {report.num_pairs}")
    print(f"  Kendall's Tau: {report.kendalls_tau:.3f} (p={report.kendalls_tau_p_value:.4f})")
    print(f"  Spearman's Rho: {report.spearmans_rho:.3f} (p={report.spearmans_rho_p_value:.4f})")
    print(f"  Statistically Significant: {report.is_significant}")
    
    print("\nScenario B: Partial agreement with ground truth")
    predicted_partial = [
        ("Model_A", "Model_B"),
        ("Model_C", "Model_D"),
        ("Model_E", "Model_F"),
        ("Model_A", "Model_C"),
    ]
    ground_truth_partial = [
        ("Model_A", "Model_B"),  # Match
        ("Model_D", "Model_C"),  # Reversed
        ("Model_E", "Model_F"),  # Match
        ("Model_C", "Model_A"),  # Reversed
    ]
    
    report = validator.calculate_ranking_correlation(predicted_partial, ground_truth_partial)
    
    print(f"  Number of pairs: {report.num_pairs}")
    print(f"  Kendall's Tau: {report.kendalls_tau:.3f} (p={report.kendalls_tau_p_value:.4f})")
    print(f"  Spearman's Rho: {report.spearmans_rho:.3f} (p={report.spearmans_rho_p_value:.4f})")
    print(f"  Statistically Significant: {report.is_significant}")


def example_comprehensive_validation():
    """Example: Comprehensive reliability validation."""
    print("\n" + "=" * 80)
    print("Example 4: Comprehensive Reliability Validation")
    print("=" * 80)
    
    validator = ReliabilityValidator()
    
    print("\nValidating a complete evaluation system:")
    print("-" * 80)
    
    # 1. Check consistency
    print("\n1. Consistency Check:")
    scores = [84.5, 86.0, 85.5, 87.0, 85.0]
    consistency_report = validator.check_consistency(scores)
    print(f"   Mean Score: {consistency_report.mean_score:.2f}")
    print(f"   Variance: {consistency_report.variance:.2f}")
    print(f"   Status: {'✓ PASS' if consistency_report.is_consistent else '✗ FAIL'}")
    
    # 2. Check inter-model agreement
    print("\n2. Inter-Model Agreement:")
    judge_scores = {
        "judge_1": [85.0, 45.0, 90.0, 30.0, 75.0],
        "judge_2": [80.0, 40.0, 88.0, 35.0, 70.0],
        "judge_3": [82.0, 48.0, 92.0, 32.0, 73.0],
    }
    agreement_report = validator.calculate_inter_model_agreement(judge_scores)
    print(f"   Cohen's Kappa: {agreement_report.cohens_kappa:.3f}")
    print(f"   Agreement Level: {agreement_report.agreement_level}")
    print(f"   Status: {'✓ PASS' if agreement_report.cohens_kappa > 0.6 else '✗ FAIL'} (threshold: κ > 0.6)")
    
    # 3. Check ranking correlation
    print("\n3. Ranking Correlation:")
    predicted = [("A", "B"), ("C", "D"), ("E", "F")]
    ground_truth = [("A", "B"), ("C", "D"), ("E", "F")]
    ranking_report = validator.calculate_ranking_correlation(predicted, ground_truth)
    print(f"   Kendall's Tau: {ranking_report.kendalls_tau:.3f}")
    print(f"   Spearman's Rho: {ranking_report.spearmans_rho:.3f}")
    print(f"   Status: {'✓ PASS' if ranking_report.spearmans_rho > 0.7 else '✗ FAIL'} (threshold: ρ > 0.7)")
    
    # Overall assessment
    print("\n" + "-" * 80)
    print("Overall Reliability Assessment:")
    all_pass = (
        consistency_report.is_consistent and
        agreement_report.cohens_kappa > 0.6 and
        ranking_report.spearmans_rho > 0.7
    )
    print(f"  {'✓ SYSTEM RELIABLE' if all_pass else '✗ SYSTEM NEEDS IMPROVEMENT'}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RELIABILITY VALIDATOR EXAMPLES")
    print("=" * 80)
    
    example_consistency_checking()
    example_inter_model_agreement()
    example_ranking_correlation()
    example_comprehensive_validation()
    
    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)
