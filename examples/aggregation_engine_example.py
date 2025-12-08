"""
Example demonstrating the AggregationEngine component.

This script shows how to use the AggregationEngine to combine results
from multiple judge models and specialized verifiers using different
aggregation strategies.
"""

from llm_judge_auditor.components.aggregation_engine import (
    AggregationEngine,
    AggregationStrategy,
)
from llm_judge_auditor.models import (
    Issue,
    IssueType,
    IssueSeverity,
    JudgeResult,
    Verdict,
    VerdictLabel,
)


def main():
    """Demonstrate AggregationEngine functionality."""
    print("=" * 70)
    print("AggregationEngine Example")
    print("=" * 70)

    # Create sample judge results
    judge_results = [
        JudgeResult(
            model_name="llama-3-8b",
            score=85.0,
            reasoning="The candidate output is mostly accurate with minor issues.",
            flagged_issues=[
                Issue(
                    type=IssueType.HALLUCINATION,
                    severity=IssueSeverity.LOW,
                    description="Minor unsupported detail about timing",
                    evidence=["Source doesn't mention specific time"],
                )
            ],
            confidence=0.9,
        ),
        JudgeResult(
            model_name="mistral-7b",
            score=78.0,
            reasoning="Good accuracy but some claims lack support.",
            flagged_issues=[
                Issue(
                    type=IssueType.UNSUPPORTED_CLAIM,
                    severity=IssueSeverity.MEDIUM,
                    description="Claim about location not in source",
                    evidence=[],
                )
            ],
            confidence=0.85,
        ),
        JudgeResult(
            model_name="phi-3-mini",
            score=82.0,
            reasoning="Generally accurate with good alignment to source.",
            flagged_issues=[],
            confidence=0.88,
        ),
    ]

    # Create sample verifier verdicts
    verifier_verdicts = [
        Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.92,
            evidence=["Source text confirms this claim"],
            reasoning="Statement is directly supported by source",
        ),
        Verdict(
            label=VerdictLabel.SUPPORTED,
            confidence=0.88,
            evidence=["Source provides evidence"],
            reasoning="Statement aligns with source information",
        ),
        Verdict(
            label=VerdictLabel.REFUTED,
            confidence=0.85,
            evidence=["Source contradicts this claim"],
            reasoning="Statement contradicts source information",
        ),
        Verdict(
            label=VerdictLabel.NOT_ENOUGH_INFO,
            confidence=0.70,
            evidence=[],
            reasoning="Insufficient information to verify",
        ),
    ]

    print("\n1. Mean Aggregation Strategy")
    print("-" * 70)
    engine_mean = AggregationEngine(strategy=AggregationStrategy.MEAN)
    score_mean, metadata_mean = engine_mean.aggregate_scores(
        judge_results, verifier_verdicts
    )
    print(f"Consensus Score: {score_mean:.2f}")
    print(f"Individual Scores: {metadata_mean.individual_scores}")
    print(f"Variance: {metadata_mean.variance:.2f}")
    print(f"Low Confidence: {metadata_mean.is_low_confidence}")

    print("\n2. Median Aggregation Strategy")
    print("-" * 70)
    engine_median = AggregationEngine(strategy=AggregationStrategy.MEDIAN)
    score_median, metadata_median = engine_median.aggregate_scores(
        judge_results, verifier_verdicts
    )
    print(f"Consensus Score: {score_median:.2f}")
    print(f"Strategy: {metadata_median.strategy}")

    print("\n3. Weighted Average Aggregation Strategy")
    print("-" * 70)
    weights = {
        "llama-3-8b": 0.5,
        "mistral-7b": 0.3,
        "phi-3-mini": 0.2,
    }
    engine_weighted = AggregationEngine(
        strategy=AggregationStrategy.WEIGHTED_AVERAGE, weights=weights
    )
    score_weighted, metadata_weighted = engine_weighted.aggregate_scores(
        judge_results, verifier_verdicts
    )
    print(f"Consensus Score: {score_weighted:.2f}")
    print(f"Weights: {metadata_weighted.weights}")

    print("\n4. Majority Vote Aggregation Strategy")
    print("-" * 70)
    engine_majority = AggregationEngine(strategy=AggregationStrategy.MAJORITY_VOTE)
    score_majority, metadata_majority = engine_majority.aggregate_scores(
        judge_results, verifier_verdicts
    )
    print(f"Consensus Score: {score_majority:.2f}")

    print("\n5. Disagreement Detection")
    print("-" * 70)
    disagreement = engine_mean.detect_disagreement(judge_results)
    print(f"Has Disagreement: {disagreement['has_disagreement']}")
    print(f"Variance: {disagreement['variance']:.2f}")
    print(f"Score Range: {disagreement['score_range']}")
    print(f"Outliers: {disagreement['outliers']}")

    print("\n6. High Disagreement Example")
    print("-" * 70)
    # Create results with high disagreement
    high_disagreement_results = [
        JudgeResult("judge1", 90.0, "Excellent", [], 0.95),
        JudgeResult("judge2", 50.0, "Average", [], 0.80),
        JudgeResult("judge3", 30.0, "Poor", [], 0.75),
    ]
    engine_high_disagreement = AggregationEngine(disagreement_threshold=20.0)
    score_high, metadata_high = engine_high_disagreement.aggregate_scores(
        high_disagreement_results
    )
    print(f"Consensus Score: {score_high:.2f}")
    print(f"Variance: {metadata_high.variance:.2f}")
    print(f"Low Confidence (High Disagreement): {metadata_high.is_low_confidence}")

    disagreement_high = engine_high_disagreement.detect_disagreement(
        high_disagreement_results
    )
    print(f"Score Range: {disagreement_high['score_range']}")

    print("\n7. Verifier Impact on Scores")
    print("-" * 70)
    # Show impact of verifier verdicts
    judge_only_results = [
        JudgeResult("judge1", 80.0, "Good", [], 0.9),
    ]

    # Without verifier
    engine_no_verifier = AggregationEngine()
    score_no_verifier, _ = engine_no_verifier.aggregate_scores(judge_only_results)
    print(f"Score without verifier: {score_no_verifier:.2f}")

    # With mostly supported verdicts
    supported_verdicts = [
        Verdict(VerdictLabel.SUPPORTED, 0.9, ["evidence"], "Supported") for _ in range(5)
    ]
    score_supported, _ = engine_no_verifier.aggregate_scores(
        judge_only_results, supported_verdicts
    )
    print(f"Score with supported verdicts: {score_supported:.2f}")

    # With mostly refuted verdicts
    refuted_verdicts = [
        Verdict(VerdictLabel.REFUTED, 0.9, ["evidence"], "Refuted") for _ in range(3)
    ]
    score_refuted, _ = engine_no_verifier.aggregate_scores(
        judge_only_results, refuted_verdicts
    )
    print(f"Score with refuted verdicts: {score_refuted:.2f}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
