"""
Example demonstrating component performance tracking.

This example shows how to:
1. Track separate metrics for verifier and judge ensemble
2. Monitor accuracy, latency, and confidence for each component
3. Log disagreements between components
4. Generate performance reports and comparative analysis
"""

from llm_judge_auditor.components.performance_tracker import PerformanceTracker
from llm_judge_auditor.models import (
    ClaimType,
    JudgeResult,
    Verdict,
    VerdictLabel,
)


def main():
    """Demonstrate performance tracking functionality."""
    print("=" * 70)
    print("Component Performance Tracking Example")
    print("=" * 70)
    print()

    # Initialize performance tracker
    tracker = PerformanceTracker()
    print("✓ Performance tracker initialized")
    print()

    # Simulate some evaluations
    print("Simulating evaluations...")
    print()

    # Example 1: Verifier and judges agree (factual claim)
    print("1. Factual claim - Agreement")
    print("-" * 40)

    tracker.start_verifier_timing()
    # Simulate verifier work
    import time
    time.sleep(0.05)
    verifier_latency = tracker.end_verifier_timing()

    verdict1 = Verdict(
        label=VerdictLabel.SUPPORTED,
        confidence=0.9,
        evidence=["The Eiffel Tower was completed in 1889."],
        reasoning="Statement is supported by historical records.",
    )

    tracker.record_verifier_result(
        verdict=verdict1,
        latency=verifier_latency,
        claim_type=ClaimType.FACTUAL,
        correct=True,
    )

    tracker.start_judge_timing()
    # Simulate judge work
    time.sleep(0.1)
    judge_latency = tracker.end_judge_timing()

    judge_results1 = [
        JudgeResult(
            model_name="llama-3-8b",
            score=85.0,
            reasoning="The statement is factually accurate.",
            confidence=0.85,
        ),
        JudgeResult(
            model_name="mistral-7b",
            score=90.0,
            reasoning="Confirmed by historical sources.",
            confidence=0.9,
        ),
    ]

    tracker.record_judge_results(
        judge_results=judge_results1,
        latency=judge_latency,
        claim_type=ClaimType.FACTUAL,
        correct=True,
    )

    tracker.log_disagreement(
        statement="The Eiffel Tower was completed in 1889.",
        verifier_verdict=verdict1,
        judge_results=judge_results1,
        judge_consensus_score=87.5,
        claim_type=ClaimType.FACTUAL,
    )

    print(f"  Verifier: {verdict1.label.value} (confidence: {verdict1.confidence:.2f})")
    print(f"  Judges: 87.5/100 (avg confidence: 0.875)")
    print(f"  Agreement: ✓")
    print()

    # Example 2: Disagreement - verifier says REFUTED but judges give high score
    print("2. Temporal claim - Disagreement")
    print("-" * 40)

    tracker.start_verifier_timing()
    time.sleep(0.04)
    verifier_latency = tracker.end_verifier_timing()

    verdict2 = Verdict(
        label=VerdictLabel.REFUTED,
        confidence=0.85,
        evidence=["The moon landing was in 1969, not 1968."],
        reasoning="Statement contradicts historical records.",
    )

    tracker.record_verifier_result(
        verdict=verdict2,
        latency=verifier_latency,
        claim_type=ClaimType.TEMPORAL,
        correct=True,
    )

    tracker.start_judge_timing()
    time.sleep(0.12)
    judge_latency = tracker.end_judge_timing()

    judge_results2 = [
        JudgeResult(
            model_name="llama-3-8b",
            score=75.0,
            reasoning="The date seems plausible.",
            confidence=0.7,
        ),
        JudgeResult(
            model_name="mistral-7b",
            score=80.0,
            reasoning="Cannot verify the exact date.",
            confidence=0.75,
        ),
    ]

    tracker.record_judge_results(
        judge_results=judge_results2,
        latency=judge_latency,
        claim_type=ClaimType.TEMPORAL,
        correct=False,
    )

    tracker.log_disagreement(
        statement="The moon landing was in 1968.",
        verifier_verdict=verdict2,
        judge_results=judge_results2,
        judge_consensus_score=77.5,
        claim_type=ClaimType.TEMPORAL,
    )

    print(f"  Verifier: {verdict2.label.value} (confidence: {verdict2.confidence:.2f})")
    print(f"  Judges: 77.5/100 (avg confidence: 0.725)")
    print(f"  Agreement: ✗ (Disagreement logged)")
    print()

    # Example 3: Another factual claim with agreement
    print("3. Numerical claim - Agreement")
    print("-" * 40)

    tracker.start_verifier_timing()
    time.sleep(0.03)
    verifier_latency = tracker.end_verifier_timing()

    verdict3 = Verdict(
        label=VerdictLabel.SUPPORTED,
        confidence=0.95,
        evidence=["Earth has one natural satellite."],
        reasoning="Statement is scientifically accurate.",
    )

    tracker.record_verifier_result(
        verdict=verdict3,
        latency=verifier_latency,
        claim_type=ClaimType.NUMERICAL,
        correct=True,
    )

    tracker.start_judge_timing()
    time.sleep(0.11)
    judge_latency = tracker.end_judge_timing()

    judge_results3 = [
        JudgeResult(
            model_name="llama-3-8b",
            score=95.0,
            reasoning="Scientifically correct.",
            confidence=0.95,
        ),
        JudgeResult(
            model_name="mistral-7b",
            score=92.0,
            reasoning="Accurate statement.",
            confidence=0.92,
        ),
    ]

    tracker.record_judge_results(
        judge_results=judge_results3,
        latency=judge_latency,
        claim_type=ClaimType.NUMERICAL,
        correct=True,
    )

    tracker.log_disagreement(
        statement="Earth has one moon.",
        verifier_verdict=verdict3,
        judge_results=judge_results3,
        judge_consensus_score=93.5,
        claim_type=ClaimType.NUMERICAL,
    )

    print(f"  Verifier: {verdict3.label.value} (confidence: {verdict3.confidence:.2f})")
    print(f"  Judges: 93.5/100 (avg confidence: 0.935)")
    print(f"  Agreement: ✓")
    print()

    # Generate performance report
    print("=" * 70)
    print("Performance Report")
    print("=" * 70)
    print()

    report = tracker.generate_report()

    # Display verifier metrics
    print("Verifier Metrics:")
    print("-" * 40)
    vm = report["verifier_metrics"]
    print(f"  Total Evaluations: {vm['total_evaluations']}")
    print(f"  Average Latency: {vm['average_latency']:.3f}s")
    print(f"  Average Confidence: {vm['average_confidence']:.2f}")
    if vm['accuracy'] is not None:
        print(f"  Accuracy: {vm['accuracy']:.2%}")
    print()

    # Display judge metrics
    print("Judge Ensemble Metrics:")
    print("-" * 40)
    jm = report["judge_metrics"]
    print(f"  Total Evaluations: {jm['total_evaluations']}")
    print(f"  Average Latency: {jm['average_latency']:.3f}s")
    print(f"  Average Confidence: {jm['average_confidence']:.2f}")
    if jm['accuracy'] is not None:
        print(f"  Accuracy: {jm['accuracy']:.2%}")
    print()

    # Display disagreement statistics
    print("Disagreement Statistics:")
    print("-" * 40)
    dis = report["disagreements"]
    print(f"  Total Disagreements: {dis['total_count']}")
    print(f"  Disagreement Rate: {dis['disagreement_rate']:.2%}")
    if dis['disagreements_by_claim_type']:
        print(f"  By Claim Type:")
        for claim_type, count in dis['disagreements_by_claim_type'].items():
            print(f"    - {claim_type}: {count}")
    print()

    # Display comparative analysis
    print("Comparative Analysis:")
    print("-" * 40)
    ca = report["comparative_analysis"]

    print(f"  Latency:")
    print(f"    - Verifier: {ca['latency_comparison']['verifier_avg']:.3f}s")
    print(f"    - Judge: {ca['latency_comparison']['judge_avg']:.3f}s")
    print(f"    - Faster: {ca['latency_comparison']['faster_component']}")
    print()

    print(f"  Confidence:")
    print(f"    - Verifier: {ca['confidence_comparison']['verifier_avg']:.2f}")
    print(f"    - Judge: {ca['confidence_comparison']['judge_avg']:.2f}")
    print(f"    - More Confident: {ca['confidence_comparison']['more_confident_component']}")
    print()

    if 'accuracy_comparison' in ca:
        print(f"  Accuracy:")
        print(f"    - Verifier: {ca['accuracy_comparison']['verifier_accuracy']:.2%}")
        print(f"    - Judge: {ca['accuracy_comparison']['judge_accuracy']:.2%}")
        print(f"    - More Accurate: {ca['accuracy_comparison']['more_accurate_component']}")
        print()

    # Display claim type performance
    if ca['claim_type_performance']:
        print("  Performance by Claim Type:")
        for claim_type, perf in ca['claim_type_performance'].items():
            print(f"    {claim_type}:")
            if perf['verifier']['accuracy'] is not None:
                print(f"      - Verifier: {perf['verifier']['accuracy']:.2%} "
                      f"({perf['verifier']['total_evaluations']} evals)")
            if perf['judge']['accuracy'] is not None:
                print(f"      - Judge: {perf['judge']['accuracy']:.2%} "
                      f"({perf['judge']['total_evaluations']} evals)")
            print(f"      - Better: {perf['better_component']}")
        print()

    # Display human-readable summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(tracker.get_summary())

    # Example: Reset tracking for new batch
    print("=" * 70)
    print("Resetting Performance Tracker")
    print("=" * 70)
    print()
    tracker.reset()
    print("✓ Performance tracker reset")
    print(f"  Verifier evaluations: {tracker.verifier_metrics.total_evaluations}")
    print(f"  Judge evaluations: {tracker.judge_metrics.total_evaluations}")
    print(f"  Disagreements: {len(tracker.disagreements)}")
    print()


if __name__ == "__main__":
    main()
